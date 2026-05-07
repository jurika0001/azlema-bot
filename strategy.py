import numpy as np
import logging

logger = logging.getLogger(__name__)

PI = np.pi
IFM_RANGE   = 50
GAIN_LIMIT  = 900
THRESHOLD   = 0.0
DEF_PERIOD  = 20


# ── Cosine IFM ────────────────────────────────────────────────────────────────
def cos_ifm(closes: np.ndarray) -> np.ndarray:
    n       = len(closes)
    v1      = np.zeros(n)
    s2      = np.zeros(n)
    s3      = np.zeros(n)
    v2      = np.zeros(n)
    delta   = np.zeros(n)
    len_c   = np.zeros(n)

    for i in range(7, n):
        v1[i] = closes[i] - closes[i - 7]
    for i in range(8, n):
        s2[i] = 0.2 * (v1[i-1] + v1[i])**2 + 0.8 * s2[i-1]
        s3[i] = 0.2 * (v1[i-1] - v1[i])**2 + 0.8 * s3[i-1]
        if s2[i] != 0:
            v2[i] = np.sqrt(s3[i] / s2[i])
        if s3[i] != 0:
            delta[i] = 2.0 * np.arctan(v2[i])

    prev = 0.0
    for i in range(IFM_RANGE + 1, n):
        v4, inst = 0.0, 0.0
        for j in range(IFM_RANGE + 1):
            if i - j >= 0:
                v4 += delta[i - j]
            if v4 > 2 * PI and inst == 0.0:
                inst = float(j - 1)
                break
        if inst == 0.0:
            inst = prev
        else:
            prev = inst
        len_c[i] = 0.25 * inst + 0.75 * len_c[i - 1]
    return len_c


# ── Zero Lag EMA ──────────────────────────────────────────────────────────────
def zlema(closes: np.ndarray, period: int) -> tuple:
    n     = len(closes)
    alpha = 2.0 / (period + 1)
    ema   = np.zeros(n)
    ec    = np.zeros(n)
    lerr  = np.zeros(n)
    gains = np.arange(-GAIN_LIMIT, GAIN_LIMIT + 1, dtype=np.float64) / 10.0

    for i in range(1, n):
        ema[i]   = alpha * closes[i] + (1.0 - alpha) * ema[i - 1]
        prev_ec  = ec[i - 1]
        trials   = alpha * (ema[i] + gains * (closes[i] - prev_ec)) + (1.0 - alpha) * prev_ec
        errors   = np.abs(closes[i] - trials)
        best_idx = int(np.argmin(errors))
        ec[i]    = trials[best_idx]
        lerr[i]  = errors[best_idx]

    return ema, ec, lerr


# ── Main strategy entry point ─────────────────────────────────────────────────
def calculate(closes: list) -> dict:
    """
    Receives list of closed-candle closes (30 m).
    Returns { signal: 'LONG'|'SHORT'|None, ema, ec, period, least_error }
    EC > EMA  →  LONG   (trade every candle)
    EC < EMA  →  SHORT
    """
    if len(closes) < 60:
        return {"signal": None, "ema": None, "ec": None, "period": DEF_PERIOD}

    arr    = np.array(closes, dtype=np.float64)
    lc     = cos_ifm(arr)
    period = int(round(lc[-1])) if lc[-1] > 0 else DEF_PERIOD
    period = max(2, period)

    ema, ec, lerr = zlema(arr, period)

    last_ema  = float(ema[-1])
    last_ec   = float(ec[-1])
    last_err  = float(lerr[-1])
    last_src  = float(arr[-1])

    threshold_ok = (100.0 * last_err / last_src > THRESHOLD) if last_src else False

    if not threshold_ok:
        signal = None
    elif last_ec > last_ema:
        signal = "LONG"
    elif last_ec < last_ema:
        signal = "SHORT"
    else:
        signal = None

    logger.info(f"[STRATEGY] period={period} ema={last_ema:.4f} ec={last_ec:.4f} signal={signal}")
    return {"signal": signal, "ema": last_ema, "ec": last_ec, "period": period, "least_error": last_err}
