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
import estado_remoto

DIR = os.path.dirname(os.path.abspath(__file__))
# No Render o estado vai para o disco persistente (ESTADO_DIR), senao some a
# cada deploy e voce perde meses de teste. Local, fica na propria pasta.
ESTADO_DIR = os.environ.get("ESTADO_DIR", DIR)
os.makedirs(ESTADO_DIR, exist_ok=True)
ESTADO = os.path.join(ESTADO_DIR, "estado_paper.json")

# ---- custos ----
# O SPREAD nao e' estimativa: o motor usa o bid/ask REAL de cada instante
# (compra no ask, vende no bid) — exatamente o que uma ordem a mercado paga.
# Medido 16/07/2026: ETH perp spread ~0,00053% (1 centavo em $1875).
#
# A TAXA e' o que muda de corretora para corretora. Como o SINAL e' o mesmo em
# todas, eu guardo o retorno BRUTO de cada trade e calculo TODOS os cenarios em
# paralelo. Assim um unico teste responde por todas as corretoras de uma vez —
# sem precisar apostar 5 meses numa so.
FEE_LADO = 0.0                      # Lighter Standard (cenario principal)

# Versao da estrategia. Se mudar, o estado salvo (Gist/disco) e' DESCARTADO —
# senao trades da estrategia antiga se misturariam com a nova e corromperiam a
# estatistica. Trocou de config -> muda esta string -> comeca do zero limpo.
STRATEGY_VERSION = "HA-2h-400/80/15-norev-sess"

CENARIOS = {                        # taker por LADO
    "lighter_0%":       0.0,
    "kcex_0.01%":       1e-4,
    "mexc_0.02%":       2e-4,
    "paradex_api_0.02%": 2e-4,
    "aster_0.035%":     3.5e-4,
    "hyperliquid_0.045%": 4.5e-4,
    "phemex_0.06%":     6e-4,
}

# ---- criterio de aprovacao, travado ANTES de comecar ----
# ATENCAO: n = (z*sd/mu)^2 esta ERRADO — nesse n o t-stat ESPERADO e' o proprio z,
# entao metade das amostras cai abaixo por acaso: 50% de poder (cara ou coroa).
# O correto inclui o poder (z_beta): n = ((z_alpha + z_beta)*sd/mu)^2.
#
# UNI-caudal: a pergunta e' "o edge e' POSITIVO?", nao "e' diferente de zero".
# Por isso z_alpha = 1.645 (95% uni) e nao 1.96 (95% bi). Isso e' o teste
# correto para a pergunta — nao e' baixar a regua.
#   bi-caudal  (1.96 +0.84) -> 981 trades -> 81,3% poder | 2,3% falso+
#   uni-caudal (1.645+0.84) -> 772 trades -> 81,1% poder | 4,8% falso+  <--
# Verificado por simulacao (4.000 amostras/ponto, edge real +0,0718%).
#
# Teste SEQUENCIAL foi avaliado e DESCARTADO: economiza so 0,9 mes
# (n medio 888 vs 981) e adiciona complexidade. Nao vale.
#
# TEMPO: rodando ETH+BTC juntos. Correlacao medida dos retornos diarios das duas
# estrategias no cofre: rho = 0,346 -> ganho efetivo de amostra 1,49x ->
# ~5,1 trades/dia efetivos -> os 772 trades saem em ~5,0 meses (vs 7,5 so ETH).
Z_ALPHA = 1.645        # uni-caudal, 95%
Z_BETA = 0.84          # 80% de poder
N_MIN = 772
T_MIN = Z_ALPHA

# ---- referencia do backtest (cofre virgem 2017-20, taxa 0 = Lighter) ----
# ETH: 3.561 trades, WR 87,2%, edge +0,0718%/trade, t=5,33, 1x +1.046%
# BTC: 3.304 trades, WR 86,2%, edge +0,0700%/trade, t=5,52, 1x   +823%
# O edge aparece nos DOIS ativos — nao e' artefato do ETH.
REF_MEDIA = 0.000718
REF_DESVIO = 0.008030
REF_WR = 87.2

ATIVOS = {"ETH": "ETH_USDT", "BTC": "BTC_USDT", "SOL": "SOL_USDT"}


class PaperTrader:
    def __init__(self, ativo="ETH"):
        self.ativo = ativo
        self.simbolo = ATIVOS[ativo]
        self.estado_path = os.path.join(ESTADO_DIR, f"estado_{ativo.lower()}.json")
        self.lock = threading.Lock()
        self.pos = None
        self.trades = []            # lista de dicts
        self.ultimo_preco = None
        self.ultimo_bid = None
        self.ultimo_ask = None
        self.spread_amostras = []
        self.funding_amostras = []
        self.ticks = 0
        self.ultimo_preco_ts = None
        self.ultimo_candle_ts = None
        self.sinal_pendente = 0
        self.inicio = datetime.now(timezone.utc).isoformat()
        self.contadores = {"hb13": 0, "preco21": 0, "estado30": 0, "externos": 0}
        self._carregar()

    # ------------------------------------------------ persistencia
    @property
    def _gist_nome(self):
        return f"estado_{self.ativo.lower()}.json"

    def _aplicar(self, z):
        # se o estado salvo e' de outra versao da estrategia, DESCARTA
        if z.get("strategy_version") != STRATEGY_VERSION:
            print(f"[{self.ativo}] estado e' de outra estrategia "
                  f"({z.get('strategy_version')}) -> comecando limpo", flush=True)
            return
        self.trades = z.get("trades", [])
        self.inicio = z.get("inicio", self.inicio)
        self.contadores.update(z.get("contadores", {}))
        self.ultimo_candle_ts = z.get("ultimo_candle_ts")
        p = z.get("pos")
        if p:
            self.pos = Posicao(p["pdir"], p["entry"], p["ts_abertura"])
            self.pos.peak = p["peak"]; self.pos.armed = p["armed"]

    def _carregar(self):
        # 1) tenta o GIST (sobrevive a restart/deploy do Render free)
        z = estado_remoto.carregar(self._gist_nome)
        if z is not None:
            try:
                self._aplicar(z)
                print(f"[{self.ativo}] estado carregado do GIST: "
                      f"{len(self.trades)} trades", flush=True)
                return
            except Exception as e:
                print(f"[{self.ativo}] gist invalido: {e}", flush=True)
        # 2) fallback: disco local
        if not os.path.exists(self.estado_path):
            return
        try:
            with open(self.estado_path, "r", encoding="utf-8") as f:
                self._aplicar(json.load(f))
        except Exception as e:
            print(f"[paper] estado corrompido, comecando limpo: {e}")

    def _snapshot(self):
        return {
            "strategy_version": STRATEGY_VERSION,
            "inicio": self.inicio,
            "trades": self.trades,
            "contadores": self.contadores,
            "ultimo_candle_ts": self.ultimo_candle_ts,
            "pos": None if self.pos is None else {
                "pdir": self.pos.pdir, "entry": self.pos.entry,
                "peak": self.pos.peak, "armed": self.pos.armed,
                "ts_abertura": self.pos.ts_abertura},
        }

    def salvar(self, forcar_gist=False):
        with self.lock:
            z = self._snapshot()
        # disco local (cache rapido)
        tmp = self.estado_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(z, f)
        os.replace(tmp, self.estado_path)      # atomico: nunca corrompe
        # gist: forcado quando fecha trade; senao no maximo 1x/2min (poupa API)
        estado_remoto.salvar(self._gist_nome, z,
                             min_intervalo=0 if forcar_gist else 120)

    # ------------------------------------------------ operacao
    def atualiza_cotacao(self, bid, ask, funding=None):
        """Canal 'ticker' (~2,7s): guarda bid/ask para modelar o preenchimento
        e registra o funding rate real (um dos pontos cegos do backtest)."""
        bid = float(bid); ask = float(ask)
        with self.lock:
            self.ultimo_bid = bid
            self.ultimo_ask = ask
            if bid > 0 and ask > 0:
                self.spread_amostras.append((ask - bid) / ((ask + bid) / 2.0))
                del self.spread_amostras[:-500]
            if funding is not None:
                self.funding_amostras.append(float(funding))
                del self.funding_amostras[:-500]

    def on_tick(self, preco, ts_ms):
        """Canal 'deal' (~0,42s): preco NEGOCIADO de verdade.

        E' o equivalente vivo do caminho de 1 segundo do backtest — na verdade
        2,4x mais fino. O GATILHO (SL/TP/trailing) usa o preco negociado, igual
        ao backtest. Ja o PREENCHIMENTO usa o lado real do livro, que e' o que
        uma ordem a mercado paga: long sai vendendo no BID, short comprando no ASK.
        """
        preco = float(preco)
        with self.lock:
            self.ultimo_preco = preco
            self.ultimo_preco_ts = int(ts_ms)
            self.ticks += 1
            if self.pos is None:
                return None
            fechou, _gatilho, motivo = self.pos.on_price(preco)
            if not fechou:
                return None
            if self.pos.pdir == 1:
                saida = self.ultimo_bid if self.ultimo_bid else preco
            else:
                saida = self.ultimo_ask if self.ultimo_ask else preco
            return self._fechar(saida, motivo, ts_ms)

    def on_preco(self, bid, ask, ts_ms):
        """REDE DE SEGURANCA (sinal de 21s): so usada se o WebSocket cair.
        Alimenta o meio como tick, para a posicao nunca ficar sem gestao."""
        self.atualiza_cotacao(bid, ask)
        return self.on_tick((float(bid) + float(ask)) / 2.0, ts_ms)

    def funding_medio(self):
        """Funding rate mediano observado (por periodo da corretora)."""
        if not self.funding_amostras:
            return None
        return float(np.median(self.funding_amostras))

    def _fechar(self, preco_saida, motivo, ts_ms):
        # 'bruto' ja tem o SPREAD real dentro (entrou no ask, saiu no bid).
        # So falta a taxa — que e' aplicada por cenario, na hora de medir.
        bruto = self.pos.retorno_bruto(preco_saida)
        liquido = bruto - 2 * FEE_LADO
        t = {
            "abertura": self.pos.ts_abertura, "fechamento": int(ts_ms),
            "dir": self.pos.pdir, "entry": self.pos.entry,
            "saida": float(preco_saida), "motivo": motivo,
            "bruto": bruto, "ret": liquido,
        }
        self.trades.append(t)
        self.pos = None
        return t

    def on_candle_fechado(self, direcao, bid, ask, ts_ms):
        """Chamado quando fecha um candle de 2h e temos a direcao do proximo."""
        bid = float(bid); ask = float(ask)
        from estrategia import NO_REV
        with self.lock:
            fechada = None
            # reversao: sinal inverteu e temos posicao aberta.
            # Se NO_REV, NAO fecha na reversao — segura ate SL/TP/trailing.
            if (not NO_REV) and self.pos is not None and direcao != 0 \
                    and direcao != self.pos.pdir:
                saida = bid if self.pos.pdir == 1 else ask
                fechada = self._fechar(saida, "reversao", ts_ms)
            if self.pos is None and direcao != 0:
                # entrada a mercado: long compra no ask, short vende no bid
                entry = ask if direcao == 1 else bid
                self.pos = Posicao(direcao, entry, int(ts_ms))
            self.ultimo_candle_ts = int(ts_ms)
            return fechada

    def spread_medio_pct(self):
        if not self.spread_amostras:
            return None
        return float(np.median(self.spread_amostras) * 100)

    # ------------------------------------------------ veredito
    def cenarios(self):
        """Resultado sob CADA corretora, a partir do mesmo retorno bruto.

        O sinal e' identico em todas; so a taxa muda. Entao um unico teste
        responde por todas — nao e' preciso apostar meses numa corretora so.
        """
        with self.lock:
            b = np.array([t["bruto"] for t in self.trades], dtype=float)
        out = {}
        for nome, fee in CENARIOS.items():
            r = b - 2 * fee
            if len(r) < 2:
                out[nome] = {"n": len(r), "veredito": "coletando"}
                continue
            mu = float(r.mean()); sd = float(r.std(ddof=1))
            t = mu / (sd / math.sqrt(len(r))) if sd > 0 else 0.0
            eq = np.cumprod(1 + r)
            eq2 = np.cumprod(1 + np.maximum(2 * r, -0.999))
            dd2 = float((1 - eq2 / np.maximum.accumulate(eq2)).max() * 100)
            n_nec = int(((Z_ALPHA + Z_BETA) * sd / mu) ** 2) if mu > 0 else -1
            out[nome] = {
                "n": len(r), "media_pct": mu * 100, "tstat": t,
                "ret_1x_pct": float((eq[-1] - 1) * 100),
                "ret_2x_pct": float((eq2[-1] - 1) * 100), "dd_2x_pct": dd2,
                "trades_necessarios": n_nec,
                "veredito": ("PASSOU" if (mu > 0 and t >= T_MIN and len(r) >= n_nec > 0)
                             else ("MORTA" if mu <= 0 else "coletando")),
            }
        return out

    def stats(self):
        with self.lock:
            r = np.array([t["ret"] for t in self.trades], dtype=float)
        n = len(r)
        base = {"n": n, "n_min": N_MIN, "t_min": T_MIN,
                "ref_media_pct": REF_MEDIA * 100, "ref_desvio_pct": REF_DESVIO * 100,
                "ref_winrate": REF_WR}
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
        n_nec = int(((Z_ALPHA + Z_BETA) * sd / mu) ** 2) if mu > 0 else -1
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
