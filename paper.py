# -*- coding: utf-8 -*-
"""
Motor de PAPER TRADING da HA-2h 250/30/25 — dinheiro nenhum, so registro.

Objetivo: rodar em tempo real sobre candles que AINDA NAO EXISTEM. E o unico
juiz que nenhum vies de selecao alcanca (walk-forward, cofre e fatias todos
olham o passado; este olha o futuro).

Custos aplicados em cada trade (ida e volta):
  FEE_LADO   = taxa taker por lado (KCEX = 0,01%)
  SPREAD_RT  = spread ida-volta estimado da ordem a mercado

CRITERIO DE APROVACAO (definido ANTES de comecar, para nao mover a trave):
  t-stat = media / (desvio / sqrt(n)) >= 1.96  com  n >= 925 trades
  Referencia do cofre 2017-20: media +0,0518%/trade, desvio 0,8030%, t=3.85
"""
import os
import json
import math
import threading
from datetime import datetime, timezone

import numpy as np

from estrategia import Posicao, TF_MS

DIR = os.path.dirname(os.path.abspath(__file__))
# No Render o estado vai para o disco persistente (ESTADO_DIR), senao some a
# cada deploy e voce perde meses de teste. Local, fica na propria pasta.
ESTADO_DIR = os.environ.get("ESTADO_DIR", DIR)
os.makedirs(ESTADO_DIR, exist_ok=True)
ESTADO = os.path.join(ESTADO_DIR, "estado_paper.json")

# ---- custos (ajuste SPREAD_RT quando medir o spread real da corretora) ----
FEE_LADO = 1e-4        # 0,01% por lado — taker do KCEX
SPREAD_RT = 2e-4       # 0,02% ida-volta — ESTIMATIVA. Meça o real!

# ---- criterio de aprovacao, travado ----
N_MIN = 925            # trades para 95% de confianca
T_MIN = 1.96

# ---- referencia do backtest (cofre virgem 2017-20) ----
REF_MEDIA = 0.000518
REF_DESVIO = 0.008030


class PaperTrader:
    def __init__(self):
        self.lock = threading.Lock()
        self.pos = None
        self.trades = []            # lista de dicts
        self.ultimo_preco = None
        self.ultimo_preco_ts = None
        self.ultimo_candle_ts = None
        self.sinal_pendente = 0
        self.inicio = datetime.now(timezone.utc).isoformat()
        self.contadores = {"hb13": 0, "preco21": 0, "estado30": 0, "externos": 0}
        self._carregar()

    # ------------------------------------------------ persistencia
    def _carregar(self):
        if not os.path.exists(ESTADO):
            return
        try:
            with open(ESTADO, "r", encoding="utf-8") as f:
                z = json.load(f)
            self.trades = z.get("trades", [])
            self.inicio = z.get("inicio", self.inicio)
            self.contadores.update(z.get("contadores", {}))
            self.ultimo_candle_ts = z.get("ultimo_candle_ts")
            p = z.get("pos")
            if p:
                self.pos = Posicao(p["pdir"], p["entry"], p["ts_abertura"])
                self.pos.peak = p["peak"]; self.pos.armed = p["armed"]
        except Exception as e:
            print(f"[paper] estado corrompido, comecando limpo: {e}")

    def salvar(self):
        with self.lock:
            z = {
                "inicio": self.inicio,
                "trades": self.trades,
                "contadores": self.contadores,
                "ultimo_candle_ts": self.ultimo_candle_ts,
                "pos": None if self.pos is None else {
                    "pdir": self.pos.pdir, "entry": self.pos.entry,
                    "peak": self.pos.peak, "armed": self.pos.armed,
                    "ts_abertura": self.pos.ts_abertura},
            }
        tmp = ESTADO + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(z, f)
        os.replace(tmp, ESTADO)      # atomico: nunca corrompe (licao do .npz)

    # ------------------------------------------------ operacao
    def on_preco(self, preco, ts_ms):
        """Chamado a cada leitura de preco (sinal interno de 21s)."""
        with self.lock:
            self.ultimo_preco = float(preco)
            self.ultimo_preco_ts = int(ts_ms)
            if self.pos is None:
                return None
            fechou, saida, motivo = self.pos.on_price(preco)
            if not fechou:
                return None
            return self._fechar(saida, motivo, ts_ms)

    def _fechar(self, preco_saida, motivo, ts_ms):
        bruto = self.pos.retorno_bruto(preco_saida)
        liquido = bruto - 2 * FEE_LADO - SPREAD_RT
        t = {
            "abertura": self.pos.ts_abertura, "fechamento": int(ts_ms),
            "dir": self.pos.pdir, "entry": self.pos.entry,
            "saida": float(preco_saida), "motivo": motivo,
            "bruto": bruto, "ret": liquido,
        }
        self.trades.append(t)
        self.pos = None
        return t

    def on_candle_fechado(self, direcao, preco_abertura, ts_ms):
        """Chamado quando fecha um candle de 2h e temos a direcao do proximo."""
        with self.lock:
            fechada = None
            # reversao: sinal inverteu e temos posicao aberta (norev=0)
            if self.pos is not None and direcao != 0 and direcao != self.pos.pdir:
                fechada = self._fechar(preco_abertura, "reversao", ts_ms)
            if self.pos is None and direcao != 0:
                self.pos = Posicao(direcao, preco_abertura, int(ts_ms))
            self.ultimo_candle_ts = int(ts_ms)
            return fechada

    # ------------------------------------------------ veredito
    def stats(self):
        with self.lock:
            r = np.array([t["ret"] for t in self.trades], dtype=float)
        n = len(r)
        base = {"n": n, "n_min": N_MIN, "t_min": T_MIN,
                "ref_media_pct": REF_MEDIA * 100, "ref_desvio_pct": REF_DESVIO * 100}
        if n < 2:
            base.update({"veredito": "coletando", "falta": N_MIN - n})
            return base
        mu = float(r.mean()); sd = float(r.std(ddof=1))
        tstat = mu / (sd / math.sqrt(n)) if sd > 0 else 0.0
        eq = np.cumprod(1 + r)
        dd = float((1 - eq / np.maximum.accumulate(eq)).max() * 100)
        eq2 = np.cumprod(1 + np.maximum(2 * r, -0.999))
        dd2 = float((1 - eq2 / np.maximum.accumulate(eq2)).max() * 100)
        dias = max((datetime.now(timezone.utc) -
                    datetime.fromisoformat(self.inicio)).total_seconds() / 86400, 1e-9)
        tpd = n / dias
        # quantos trades ainda faltam para o criterio, dado o edge observado
        n_nec = int((T_MIN * sd / mu) ** 2) if mu > 0 else -1
        if n >= N_MIN and tstat >= T_MIN and mu > 0:
            veredito = "PASSOU"
        elif n >= N_MIN:
            veredito = "REPROVOU"
        else:
            veredito = "coletando"
        base.update({
            "media_pct": mu * 100, "desvio_pct": sd * 100, "tstat": tstat,
            "winrate": float((r > 0).mean() * 100),
            "ret_total_1x_pct": float((eq[-1] - 1) * 100),
            "ret_total_2x_pct": float((eq2[-1] - 1) * 100),
            "dd_1x_pct": dd, "dd_2x_pct": dd2,
            "dias_rodando": dias, "trades_por_dia": tpd,
            "trades_necessarios": n_nec,
            "dias_restantes": (max(n_nec - n, 0) / tpd) if (tpd > 0 and n_nec > 0) else None,
            "veredito": veredito, "falta": max(N_MIN - n, 0),
        })
        return base
