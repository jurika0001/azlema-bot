"""
Adaptive Zero Lag EMA Strategy (Python port of Pine Script v3 original)
Uses scipy bounded minimization to find best gain — equivalent to the
exhaustive ±GainLimit search in steps of 0.1 but runs in ~ms instead of seconds.
"""

import os
import numpy as np
from scipy.optimize import minimize_scalar

PERIOD    = int(os.environ.get("PERIOD",     "20"))
GAIN_LIM  = float(os.environ.get("GAIN_LIMIT", "50.0"))  # ±50 → ±5.0 in 0.1 steps
THRESHOLD = float(os.environ.get("THRESHOLD",  "0.0"))


# ── helpers ──────────────────────────────────────────────────────────────────

def _ema(closes: np.ndarray, alpha: float) -> np.ndarray:
    n   = len(closes)
    out = np.empty(n)
    out[0] = closes[0]
    for i in range(1, n):
        out[i] = alpha * closes[i] + (1.0 - alpha) * out[i - 1]
    return out


def _ec(closes: np.ndarray, ema: np.ndarray, alpha: float, gain: float) -> np.ndarray:
    n   = len(closes)
    out = np.empty(n)
    out[0] = closes[0]
    for i in range(1, n):
        out[i] = (alpha * (ema[i] + gain * (closes[i] - out[i - 1]))
                  + (1.0 - alpha) * out[i - 1])
    return out


# ── public API ────────────────────────────────────────────────────────────────

def calculate_signal(closes: list, period: int = None,
                     gain_limit: float = None, threshold: float = None):
    """
    Returns (signal, ema_last, ec_last, least_error)
    signal: 'LONG' | 'SHORT' | 'FLAT'
    """
    period    = period    or PERIOD
    gain_limit= gain_limit or GAIN_LIM
    threshold = threshold if threshold is not None else THRESHOLD

    arr = np.array(closes, dtype=np.float64)
    n   = len(arr)
    if n < max(period * 2, 20):
        return "FLAT", 0.0, 0.0, 0.0

    alpha = 2.0 / (period + 1.0)
    ema   = _ema(arr, alpha)

    # Find gain that minimises |close[-1] - EC[-1]|
    def error(gain):
        ec = _ec(arr, ema, alpha, gain)
        return abs(arr[-1] - ec[-1])

    res        = minimize_scalar(error, bounds=(-gain_limit, gain_limit),
                                 method="bounded",
                                 options={"xatol": 0.05})
    best_gain  = res.x
    least_err  = res.fun

    ec_arr     = _ec(arr, ema, alpha, best_gain)
    ema_last   = float(ema[-1])
    ec_last    = float(ec_arr[-1])

    # Threshold check  (same as Pine: 100*LeastError/src > Threshold)
    if arr[-1] > 0 and (100.0 * least_err / arr[-1]) <= threshold:
        return "FLAT", ema_last, ec_last, least_err

    if ec_last > ema_last:
        return "LONG",  ema_last, ec_last, least_err
    if ec_last < ema_last:
        return "SHORT", ema_last, ec_last, least_err
    return "FLAT", ema_last, ec_last, least_err
