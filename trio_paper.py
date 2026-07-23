# -*- coding: utf-8 -*-
"""
PAPER TRADING DO TRIO (ETH+BTC+SOL) — prospectivo, SEM META.

Espelha, em tempo real, o que o backtest_combos.py fez: UMA posicao por vez,
saldo total, pega o primeiro sinal livre entre os 3 pares. Enquanto ha posicao
aberta (em qualquer par), IGNORA sinais dos outros.

Por que rodar isto: no backtest o trio deu edge +0,189%/trade (imune a
compostagem), t=9,49, DD 32%, e sobreviveu ao placebo (p=0,000). MAS o
%/ano composto (+1006% medio) e' otimista e nao vai se repetir no real. Este
paper trading, sobre candles que ainda nao existem, revela o numero VERDADEIRO.

SEM META: nao ha alvo de retorno, nao diz "passou". So mostra o edge por trade e
a significancia (t) crescendo. Os fatos decidem.
"""
import os
import json
import math
import threading
from datetime import datetime, timezone

import numpy as np

from estrategia import Posicao, NO_REV
import estado_remoto

VERSAO = "trio-ETH+BTC+SOL-HA-2h-400/80/15-norev-sess"
PARES = ["ETH", "BTC", "SOL"]
GIST_NOME = "estado_trio.json"
# referencia do backtest (NAO e' meta, so p/ comparar o que aparecer):
BT_EDGE, BT_T, BT_DD = 0.1887, 9.49, 32.0


class Trio:
    def __init__(self):
        self.lock = threading.Lock()
        self.pos = None
        self.pos_ativo = None
        self.trades = []
        self.ultimo = {a: {"bid": None, "ask": None, "preco": None} for a in PARES}
        self.inicio = datetime.now(timezone.utc).isoformat()
        self.ultimo_candle = {a: None for a in PARES}
        self._carregar()

    # ---------------------------------------------------- persistencia
    def _path(self):
        d = os.environ.get("ESTADO_DIR", os.path.dirname(os.path.abspath(__file__)))
        return os.path.join(d, "estado_trio.json")

    def _carregar(self):
        z = estado_remoto.carregar(GIST_NOME)
        if z is None and os.path.exists(self._path()):
            try:
                z = json.load(open(self._path(), encoding="utf-8"))
            except Exception:
                z = None
        if not z or z.get("versao") != VERSAO:
            if z:
                print("[trio] estado de outra versao -> comeca limpo", flush=True)
            return
        self.trades = z.get("trades", [])
        self.inicio = z.get("inicio", self.inicio)
        self.ultimo_candle = z.get("ultimo_candle", self.ultimo_candle)
        p = z.get("pos")
        if p:
            self.pos = Posicao(p["pdir"], p["entry"], p["ts"])
            self.pos.peak = p["peak"]; self.pos.armed = p["armed"]
            self.pos_ativo = p["ativo"]
        print(f"[trio] carregado: {len(self.trades)} trades", flush=True)

    def _snapshot(self):
        return {"versao": VERSAO, "inicio": self.inicio, "trades": self.trades,
                "ultimo_candle": self.ultimo_candle,
                "pos": None if self.pos is None else {
                    "ativo": self.pos_ativo, "pdir": self.pos.pdir,
                    "entry": self.pos.entry, "peak": self.pos.peak,
                    "armed": self.pos.armed, "ts": self.pos.ts_abertura}}

    def salvar(self, forcar_gist=False):
        z = self._snapshot()
        try:
            tmp = self._path() + ".tmp"; json.dump(z, open(tmp, "w"))
            os.replace(tmp, self._path())
        except Exception:
            pass
        estado_remoto.salvar(GIST_NOME, z, min_intervalo=0 if forcar_gist else 120)

    # ---------------------------------------------------- operacao
    def cotacao(self, ativo, bid, ask):
        self.ultimo[ativo]["bid"] = float(bid)
        self.ultimo[ativo]["ask"] = float(ask)

    def on_tick(self, ativo, preco):
        """So gerencia a posicao se ela estiver NESTE ativo (uma posicao por vez)."""
        preco = float(preco)
        with self.lock:
            self.ultimo[ativo]["preco"] = preco
            if self.pos is None or self.pos_ativo != ativo:
                return None
            fechou, _g, motivo = self.pos.on_price(preco)
            if not fechou:
                return None
            bid = self.ultimo[ativo]["bid"] or preco
            ask = self.ultimo[ativo]["ask"] or preco
            saida = bid if self.pos.pdir == 1 else ask
            return self._fechar(saida, motivo)

    def on_candle(self, ativo, direcao, bid, ask, ts_ms):
        bid = float(bid); ask = float(ask)
        with self.lock:
            self.ultimo_candle[ativo] = int(ts_ms)
            fechada = None
            if self.pos is not None and self.pos_ativo == ativo and direcao != 0 \
                    and direcao != self.pos.pdir and not NO_REV:
                saida = bid if self.pos.pdir == 1 else ask
                fechada = self._fechar(saida, "reversao")
            # abre so se LIVRE (nenhuma posicao em nenhum dos 3)
            if self.pos is None and direcao != 0:
                entry = ask if direcao == 1 else bid
                self.pos = Posicao(direcao, entry, int(ts_ms))
                self.pos_ativo = ativo
            return fechada

    def _fechar(self, preco_saida, motivo):
        bruto = self.pos.retorno_bruto(preco_saida)
        t = {"ativo": self.pos_ativo, "dir": self.pos.pdir, "entry": self.pos.entry,
             "saida": float(preco_saida), "bruto": bruto, "motivo": motivo,
             "fechamento": datetime.now(timezone.utc).isoformat()}
        self.trades.append(t)
        self.pos = None; self.pos_ativo = None
        return t

    # ---------------------------------------------------- leitura (sem meta)
    def stats(self):
        with self.lock:
            r = np.array([t["bruto"] for t in self.trades], dtype=float)
            por = {a: np.array([t["bruto"] for t in self.trades if t["ativo"] == a])
                   for a in PARES}
        dias = max((datetime.now(timezone.utc)
                    - datetime.fromisoformat(self.inicio)).total_seconds() / 86400, 1e-9)
        out = {"versao": VERSAO, "n": len(r), "dias": round(dias, 2),
               "trades_por_dia": round(len(r) / dias, 2),
               "posicao": None if self.pos is None else {
                   "ativo": self.pos_ativo, "dir": self.pos.pdir,
                   "entry": self.pos.entry, "armed": self.pos.armed},
               "por_ativo": {a: {"n": len(por[a]),
                                 "edge_pct": float(por[a].mean() * 100) if len(por[a]) else None,
                                 "wr": float((por[a] > 0).mean() * 100) if len(por[a]) else None}
                             for a in PARES},
               "backtest_ref": {"edge_pct": BT_EDGE, "t": BT_T, "dd_pct": BT_DD}}
        if len(r) >= 2:
            mu = float(r.mean()); sd = float(r.std(ddof=1))
            eq = np.cumprod(1 + r)
            dd = float((1 - eq / np.maximum.accumulate(eq)).max() * 100)
            out.update({
                "edge_por_trade_pct": round(mu * 100, 4),
                "tstat": round(mu / (sd / math.sqrt(len(r))), 2) if sd > 0 else 0.0,
                "winrate": round(float((r > 0).mean() * 100), 1),
                "ret_acumulado_pct": round(float((eq[-1] - 1) * 100), 2),
                "dd_pct": round(dd, 1)})
        return out
