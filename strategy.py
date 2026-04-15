import numpy as np

_PI = np.pi


def _cosine_ifm(closes: np.ndarray, rng: int = 50) -> np.ndarray:
    """Cosine Instantaneous Frequency Measurement → adaptive period array."""
    n = len(closes)
    s2 = np.zeros(n)
    s3 = np.zeros(n)
    dc = np.zeros(n)
    lc = np.zeros(n)

    for i in range(8, n):
        v1  = closes[i]     - closes[i - 7]
        v1p = closes[i - 1] - closes[i - 8]

        s2[i] = 0.2 * (v1p + v1) ** 2 + 0.8 * s2[i - 1]
        s3[i] = 0.2 * (v1p - v1) ** 2 + 0.8 * s3[i - 1]

        v2    = np.sqrt(s3[i] / s2[i]) if s2[i] else 0.0
        dc[i] = 2.0 * np.arctan(v2)    if s3[i] else 0.0

        inst = v4 = 0.0
        for j in range(min(rng + 1, i + 1)):
            v4 += dc[i - j]
            if v4 > 2.0 * _PI and inst == 0.0:
                inst = float(j - 1)
        if inst == 0.0:
            inst = lc[i - 1]
        lc[i] = 0.25 * inst + 0.75 * lc[i - 1]

    return lc


def compute(closes: np.ndarray,
            period: int = 20,
            gain_limit: int = 50,
            adaptive: str = "Cos IFM") -> tuple:
    """
    Adaptive Zero-Lag EMA.
    Returns: ec (ndarray), ema (ndarray), least_error (ndarray)
    """
    c = np.asarray(closes, dtype=float)
    n = len(c)

    if adaptive == "Cos IFM":
        raw = np.clip(np.round(_cosine_ifm(c)).astype(int), 2, 200)
    else:
        raw = np.full(n, max(period, 2), dtype=int)

    # gain candidates → i/10 for i in [-gain_limit, gain_limit]
    gains = np.arange(-gain_limit, gain_limit + 1, dtype=float) / 10.0

    ema = np.zeros(n)
    ec  = np.zeros(n)
    err = np.zeros(n)
    ema[0] = ec[0] = c[0]

    for i in range(1, n):
        a      = 2.0 / (int(raw[i]) + 1)
        ema[i] = a * c[i] + (1.0 - a) * ema[i - 1]

        # vectorised gain search (fast numpy)
        cands  = a * (ema[i] + gains * (c[i] - ec[i - 1])) + (1.0 - a) * ec[i - 1]
        errs   = np.abs(c[i] - cands)
        best   = int(np.argmin(errs))
        ec[i]  = cands[best]
        err[i] = errs[best]

    return ec, ema, err


def signal(closes: np.ndarray,
           period: int = 20,
           gain_limit: int = 50,
           threshold: float = 0.0,
           adaptive: str = "Cos IFM") -> tuple:
    """
    LONG | SHORT | NEUTRAL based on last CONFIRMED (closed) candle.
    Returns: (direction, ec, ema, least_error)
    """
    if len(closes) < 20:
        return "NEUTRAL", 0.0, 0.0, 0.0

    confirmed = closes[:-1]           # exclude live/incomplete candle
    ec, ema, err = compute(confirmed, period, gain_limit, adaptive)

    src   = float(confirmed[-1])
    ec_v  = float(ec[-1])
    ema_v = float(ema[-1])
    err_v = float(err[-1])

    # threshold filter (matches Pine Script: 100*LeastError/src > Threshold)
    if src and (100.0 * err_v / src) <= threshold:
        return "NEUTRAL", ec_v, ema_v, err_v

    if ec_v > ema_v:
        return "LONG",  ec_v, ema_v, err_v
    if ec_v < ema_v:
        return "SHORT", ec_v, ema_v, err_v
    return "NEUTRAL", ec_v, ema_v, err_v
