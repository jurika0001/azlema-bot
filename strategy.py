import numpy as np
import logging

logger = logging.getLogger(__name__)

PI         = 3.14159265359
IFM_RANGE  = 50
GAIN_LIMIT = 55          # the user's TradingView chart "Gain Limit" input
THRESHOLD  = 0.0
DEF_PERIOD = 20


def _compute(arr: np.ndarray):
    """
    Single causal pass that matches the Pine "Adaptive Zero Lag EMA v2" EXACTLY:
    a PER-BAR adaptive period (Cos-IFM) drives a per-bar alpha for the Zero-Lag
    EMA (EMA + error-correcting EC with the per-bar best gain). Returns
    (ema, ec, period) arrays; the value at index i depends only on arr[:i+1].

    The old code recomputed EMA/EC over the whole window with ONE fixed period
    (the last bar's) — that does NOT match Pine, where alpha changes every bar.
    That was the cause of the wrong signals / ~50% winrate.
    """
    n      = len(arr)
    ema    = np.zeros(n); ec = np.zeros(n); period = np.zeros(n, dtype=int)
    v1     = np.zeros(n); s2 = np.zeros(n); s3 = np.zeros(n); deltaC = np.zeros(n)
    lenC   = np.zeros(n)
    gains  = np.arange(-GAIN_LIMIT, GAIN_LIMIT + 1, dtype=np.float64) / 10.0
    inst_prev = 0.0

    for i in range(n):
        # ── Cos-IFM adaptive period (Pine: lenC / Period) ──
        if i >= 7:
            v1[i] = arr[i] - arr[i - 7]
        if i >= 1:
            s2[i] = 0.2 * (v1[i-1] + v1[i]) ** 2 + 0.8 * s2[i-1]
            s3[i] = 0.2 * (v1[i-1] - v1[i]) ** 2 + 0.8 * s3[i-1]
        v2 = np.sqrt(s3[i] / s2[i]) if s2[i] != 0 else 0.0
        if s3[i] != 0:
            deltaC[i] = 2.0 * np.arctan(v2)
        v4 = 0.0; inst = 0.0
        for j in range(IFM_RANGE + 1):
            if i - j >= 0:
                v4 += deltaC[i - j]
            if v4 > 2 * PI and inst == 0.0:
                inst = j - 1
        if inst == 0.0:
            inst = inst_prev
        inst_prev = inst
        lenC[i] = 0.25 * inst + 0.75 * (lenC[i-1] if i >= 1 else 0.0)
        p = int(round(lenC[i]))
        period[i] = p if p >= 2 else DEF_PERIOD

        # ── Zero-Lag EMA with the PER-BAR alpha ──
        alpha = 2.0 / (period[i] + 1)
        pe = ema[i-1] if i >= 1 else 0.0
        pc = ec[i-1]  if i >= 1 else 0.0
        ema[i] = alpha * arr[i] + (1 - alpha) * pe
        trials = alpha * (ema[i] + gains * (arr[i] - pc)) + (1 - alpha) * pc
        ec[i]  = trials[int(np.argmin(np.abs(arr[i] - trials)))]

    return ema, ec, period


def calculate(closes: list) -> dict:
    """Signal at the LAST closed candle (for the live bot). EC>EMA → LONG, EC<EMA → SHORT."""
    if len(closes) < 60:
        return {"signal": None, "ema": None, "ec": None, "period": DEF_PERIOD}
    arr = np.array(closes, dtype=np.float64)
    ema, ec, period = _compute(arr)
    le, lc, per = float(ema[-1]), float(ec[-1]), int(period[-1])
    signal = "LONG" if lc > le else ("SHORT" if lc < le else None)
    logger.info(f"[STRATEGY] period={per} ema={le:.4f} ec={lc:.4f} signal={signal}")
    return {"signal": signal, "ema": le, "ec": lc, "period": per}


def calculate_series(closes: list) -> list:
    """Signal for EVERY candle (for the backtest) — one causal pass, same as the live bot."""
    n = len(closes); sig = [None] * n
    if n < 60:
        return sig
    ema, ec, _ = _compute(np.array(closes, dtype=np.float64))
    for i in range(60, n):
        if   ec[i] > ema[i]: sig[i] = "LONG"
        elif ec[i] < ema[i]: sig[i] = "SHORT"
    return sig
