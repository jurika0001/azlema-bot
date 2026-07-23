# -*- coding: utf-8 -*-
"""
HA-2h 250/30/25 (sess=1, mm=0, norev=0) — PORTE FIEL do backtest validado.

Origem: legado_quant.py / criativo3.py / criativo5.py / azlema.py
Config exata: {'tf':'2h','fam':'HA','sess':1,'mm':0,'azc':0,
               'sl':250,'tp':30,'tr':25,'norev':0}

Regras (1 ponto = 1e-4 = 0,01%):
  SL   = 250 pontos = 2,50%   (so vale ANTES de armar)
  TP   =  30 pontos = 0,30%   (NAO fecha — ARMA o trailing)
  TR   =  25 pontos = 0,25%   (trailing, so depois de armar)
  Reversao: se o sinal inverte, fecha na abertura do candle seguinte (norev=0)

Sinal do candle i (causal — usa APENAS dados ate i-1):
  hdir[i] = sign((hac[i-1] - hao[i-1]) / open[i-1])      (Heikin-Ashi)
  az[i]   = sign(EC[i-1] - EMA[i-1])                     (AzLema / Ehlers)
  sess[i] = hora UTC do candle i em [12, 22)
  d[i]    = hdir[i]  se  hdir[i] != 0 e hdir[i] == az[i] e sess[i]  senao 0
"""
import math
import numpy as np

PI = 3.14159265359

# ---- parametros travados (nao mexer: sao os validados no cofre 2017-20) ----
# HA-2h 400/80/15 norev sess — melhor no BTC (fill realista, taxa 0):
# +114%/ano a 1x, DD 32%, t=3,58. Passa em 2 ativos (BTC forte, ETH ok).
SL_PTS = 400.0
TP_PTS = 80.0
TR_PTS = 15.0
MM = 0.0
SESS_ON = True
NO_REV = True         # sinal invertendo NAO fecha; segura ate SL/TP/trailing
TF_MS = 2 * 3600 * 1000          # candle de 2 horas


def azlema_core(src, mode=0, fixed_period=20.0, gain_limit=900, rng=50):
    """Porte fiel do azlema.py (Ehlers Adaptive Zero-Lag EMA).
    A varredura de ganho foi vetorizada em numpy (mesmo resultado, sem numba)."""
    n = len(src)
    v1 = np.zeros(n); deltaC = np.zeros(n)
    EMA = np.zeros(n); EC = np.zeros(n)
    s2 = 0.0; s3 = 0.0; lenC = 0.0; instC_prev = 0.0
    gains = np.arange(-gain_limit, gain_limit + 1, dtype=np.float64) / 10.0
    for t in range(n):
        s = float(src[t])
        v1t = s - (float(src[t - 7]) if t >= 7 else s)
        v1[t] = v1t
        v1p = v1[t - 1] if t >= 1 else 0.0
        s2 = 0.2 * (v1p + v1t) ** 2 + 0.8 * s2
        s3 = 0.2 * (v1p - v1t) ** 2 + 0.8 * s3
        v2 = math.sqrt(s3 / s2) if s2 != 0.0 else 0.0
        dC = 2.0 * math.atan(v2) if s3 != 0.0 else 0.0
        deltaC[t] = dC
        # periodo instantaneo (Cosine IFM)
        instC = 0.0; v4 = 0.0
        for i in range(0, rng + 1):
            if t - i >= 0:
                v4 += deltaC[t - i]
            if v4 > 2.0 * PI and instC == 0.0:
                instC = i - 1
        if instC == 0.0:
            instC = instC_prev
        instC_prev = instC
        lenC = 0.25 * instC + 0.75 * lenC
        per = round(lenC) if mode == 0 else float(fixed_period)
        if per < 1.0:
            per = 1.0
        alpha = 2.0 / (per + 1.0)
        EMAt = alpha * s + (1.0 - alpha) * (EMA[t - 1] if t >= 1 else s)
        EMA[t] = EMAt
        # Zero-Lag EMA: varredura de ganho que minimiza o erro
        ECprev = EC[t - 1] if t >= 1 else s
        ec_all = alpha * (EMAt + gains * (s - ECprev)) + (1.0 - alpha) * ECprev
        best = gains[int(np.argmin(np.abs(s - ec_all)))]
        EC[t] = alpha * (EMAt + best * (s - ECprev)) + (1.0 - alpha) * ECprev
    return EC, EMA


def build_dirs(o, h, l, c, ts_ms, mm=MM, sess_on=SESS_ON):
    """Retorna d[i] em {-1,0,+1} — a direcao para operar o candle i.
    Causal: d[i] usa somente informacao ate o candle i-1."""
    o = np.asarray(o, dtype=np.float64); h = np.asarray(h, dtype=np.float64)
    l = np.asarray(l, dtype=np.float64); c = np.asarray(c, dtype=np.float64)
    ts_ms = np.asarray(ts_ms, dtype=np.int64)
    n = len(c)
    if n < 2:
        return np.zeros(n, np.int8)
    # Heikin-Ashi
    hac = (o + h + l + c) / 4.0
    hao = np.empty(n); hao[0] = o[0]
    for i in range(1, n):
        hao[i] = 0.5 * (hao[i - 1] + hac[i - 1])
    hmom = np.zeros(n); hmom[1:] = (hac[:-1] - hao[:-1]) / o[:-1]
    hdir = np.sign(hmom); hmag = np.abs(hmom) / 1e-4
    # AzLema
    EC, EMA = azlema_core(np.ascontiguousarray(c), 0, 20.0, 900, 50)
    az = np.zeros(n); az[1:] = np.sign(EC[:-1] - EMA[:-1])
    # sessao 12-22 UTC
    hrs = (ts_ms // 3_600_000) % 24
    sess = ((hrs >= 12) & (hrs < 22)) if sess_on else np.ones(n, bool)
    d = np.where((hdir != 0) & (hdir == az) & sess & (hmag >= mm), hdir, 0)
    return d.astype(np.int8)


def sinal_atual(o, h, l, c, ts_ms):
    """Direcao para o PROXIMO candle, dado o historico fechado ate agora.
    Devolve -1 (short), 0 (nada) ou +1 (long)."""
    n = len(c)
    # replica d[i] para i = n (o candle que vai abrir agora), usando i-1 = n-1
    hac = (np.asarray(o, float) + np.asarray(h, float) +
           np.asarray(l, float) + np.asarray(c, float)) / 4.0
    hao = np.empty(n); hao[0] = o[0]
    for i in range(1, n):
        hao[i] = 0.5 * (hao[i - 1] + hac[i - 1])
    hmom = (hac[-1] - hao[-1]) / o[-1]
    hdir = np.sign(hmom); hmag = abs(hmom) / 1e-4
    EC, EMA = azlema_core(np.ascontiguousarray(np.asarray(c, float)), 0, 20.0, 900, 50)
    az = np.sign(EC[-1] - EMA[-1])
    prox_ts = (int(ts_ms[-1]) // TF_MS) * TF_MS + TF_MS
    hr = (prox_ts // 3_600_000) % 24
    sess = (12 <= hr < 22) if SESS_ON else True
    if hdir != 0 and hdir == az and sess and hmag >= MM:
        return int(hdir)
    return 0


class Posicao:
    """Gerencia UMA posicao com as regras exatas do sim_flex do backtest."""

    def __init__(self, pdir, entry, ts_abertura):
        self.pdir = int(pdir)              # +1 long, -1 short
        self.entry = float(entry)
        self.peak = float(entry)
        self.armed = False                 # trailing so liga depois do TP
        self.ts_abertura = ts_abertura
        self.sl = entry * (1.0 - pdir * SL_PTS * 1e-4)
        self.arm_level = entry * (1.0 + pdir * TP_PTS * 1e-4)
        self.dist = entry * TR_PTS * 1e-4

    def on_price(self, p):
        """Alimenta um preco. Devolve (fechou, preco_saida, motivo)."""
        p = float(p)
        if self.pdir == 1:
            if not self.armed:
                if p >= self.arm_level:
                    self.armed = True; self.peak = p
                elif p <= self.sl:
                    return True, self.sl, "stop_loss"
            else:
                if p > self.peak:
                    self.peak = p
                else:
                    stop = self.peak - self.dist
                    if p <= stop:
                        return True, stop, "trailing"
        else:
            if not self.armed:
                if p <= self.arm_level:
                    self.armed = True; self.peak = p
                elif p >= self.sl:
                    return True, self.sl, "stop_loss"
            else:
                if p < self.peak:
                    self.peak = p
                else:
                    stop = self.peak + self.dist
                    if p >= stop:
                        return True, stop, "trailing"
        return False, 0.0, ""

    def retorno_bruto(self, preco_saida):
        """Retorno da trade SEM custos (a taxa/spread entra em paper.py)."""
        return self.pdir * (float(preco_saida) / self.entry - 1.0)
