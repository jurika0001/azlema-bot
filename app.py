import os, time as time_mod, threading, logging, json, asyncio
import requests as req
from datetime import datetime, timezone
from flask import Flask, jsonify, request

import ccxt
# ccxt.pro (bundled since ccxt 4) gives a real-time websocket tick stream.
# Guarded so the bot still runs (on the REST fallback) if it is unavailable.
try:
    import ccxt.pro as ccxtpro
    _HAS_WS = True
except Exception as _ws_imp_err:                       # pragma: no cover
    ccxtpro, _HAS_WS = None, False
    logging.getLogger(__name__).warning(
        f"[WS] ccxt.pro import failed ({_ws_imp_err}); REST polling only")
import strategy as strat

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CONFIG  (matching Pine Script defaults)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ── Tick size (mintick) — ONE value for ALL exit distances, like Pine ────────
# Pine uses a SINGLE syminfo.mintick for loss / trail_points / trail_offset. So
# here ONE mintick multiplies EVERY exit distance (no SL-vs-TP split — that split
# broke the $ math vs the broker). With the defaults (mintick 0.01):
#   SL distance        = 2000 × 0.01 = $20.0
#   trail ACTIVATION   =   55 × 0.01 = $0.55
#   trail OFFSET       =   15 × 0.01 = $0.15
# mintick is editable in the dashboard config (env MINTICK_OVERRIDE sets default).
MINTICK_DEFAULT = float(os.environ.get("MINTICK_OVERRIDE", 0.01))
FIXED_SL     = 2000    # Pine fixedSL       (loss)         → SL distance      = ×mintick
FIXED_TP     = 55      # Pine fixedTP       (trail_points) → trail ACTIVATION = ×mintick
TRAIL_OFFSET = 15      # Pine trail_offset  (literal 15)   → trail OFFSET      = ×mintick

# ── Position sizing ───────────────────────────────────────────────────────
# Position notional = the FULL balance (~1x): qty = balance / price. So every
# trade compounds the account by the SAME % it shows in the history, and the
# total profit finally AGREES with the trades. This is DECOUPLED from the
# distance MINTICK on purpose: raising MINTICK to widen the trailing must NOT
# shrink the position (that coupling made the total ~10× below the per-trade
# %s). See _calc_qty.

# ── Editable config (persisted to config.json) ──────────────────────────────
# POINT counts (sl/tp/tr) + the SINGLE mintick + the Gain Limit. Every exit
# distance = its point count × the ONE mintick (Pine-faithful). Defaults
# 2000/55/15 pts × 0.01 → SL=$20, TP-activation=$0.55, trail=$0.15.
CONFIG_FILE = "config.json"
def _default_config():
    # User-chosen defaults (2026-06-28): SL 2000pt·$200, TP-activation 10pt·$1.00,
    # trail 1pt·$0.10, Gain Limit 1, mintick 0.1.
    return {"sl_pts": 2000, "tp_pts": 10, "tr_pts": 1,
            "mintick": 0.1,
            "gain_limit": 1}
def _load_config():
    try:
        with open(CONFIG_FILE) as f:
            d = json.load(f); base = _default_config(); base.update(d); return base
    except Exception:
        return _default_config()
def _save_config():
    try:
        with open(CONFIG_FILE, "w") as f: json.dump(config, f)
    except Exception as e: logging.getLogger(__name__).warning(f"[CONFIG] {e}")

config = _load_config()

# SL_DIST / TP_DIST / TR_DIST ($) are what the exit logic reads. _apply_config
# converts the editable POINT counts → $ distances using the ONE shared mintick
# (called on load + every save). MINTICK mirrors the active tick for display.
SL_DIST = TP_DIST = TR_DIST = 0.0
MINTICK = MINTICK_DEFAULT
def _apply_config():
    global SL_DIST, TP_DIST, TR_DIST, MINTICK
    MINTICK = float(config["mintick"])
    SL_DIST = round(float(config["sl_pts"]) * MINTICK, 8)
    TP_DIST = round(float(config["tp_pts"]) * MINTICK, 8)
    TR_DIST = round(float(config["tr_pts"]) * MINTICK, 8)
_apply_config()

# ── Exit evaluation mode ─────────────────────────────────────────────────
# THE USER WANTS INTRA-CANDLE EXITS: the trade closes LIVE, during the candle, at
# the real price the moment it retraces trail_offset (15 ticks) from the running
# peak — NOT snapped to the candle's O/H/L/C extreme at candle close. ("Não fechar
# em extremos, porque é intra-candle.")
#
# So REALTIME_EXITS defaults ON and _check_sl_tp (real-time peak + trailing) owns
# exits, fed by the websocket TICK stream. The candle-close stop check is SKIPPED
# for exits in this mode (see section 2a), so nothing is ever closed "at the
# extreme". The trail still ratchets up with each new live peak and only closes on
# a real 15-tick pullback.
#
# REALTIME_EXITS=0 switches to the candle model (exits at candle close from the
# O/H/L/C). Kept ON per the user: exits must be intra-candle / live.
REALTIME_EXITS  = os.environ.get("REALTIME_EXITS", "1") == "1"

RISK            = 0.01
MAX_LOTS        = 100
CANDLES         = 200
PAPER_START_BAL = 1_000.0   # matches Pine initial_capital=1000
PORT            = int(os.environ.get("PORT", 10000))

# Configurable trade fees (persisted to disk)
FEES_FILE = "fees.json"
def _load_fees():
    # Pine has commission_value=0, so default to 0 to match the TradingView
    # backtest. Change these in the UI if you want to simulate real exchange fees.
    try:
        with open(FEES_FILE) as f: return json.load(f)
    except Exception: return {"entry": 0.0, "exit": 0.0}
def _save_fees():
    try:
        with open(FEES_FILE, "w") as f: json.dump(fees, f)
    except Exception: pass

fees = _load_fees()
logger.info(f"[FEES] entry={fees['entry']}%  exit={fees['exit']}%")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# EXCHANGE  (Phemex live — OHLCV + ticker, no auth needed)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# live_ex   : used for OHLCV (rate-limited — fine, only polled every 15 s)
# ticker_ex : SEPARATE instance, rate-limit disabled, used ONLY for the 1-s
#             price poll. ccxt's own price-scale handling is battle-tested —
#             we do NOT guess/derive the scale manually (that was the bug:
#             a one-off guessed ratio silently drifted wrong over time and
#             corrupted every subsequent SL/TP price).
live_ex = ccxt.phemex({"options": {"defaultType": "swap"}, "enableRateLimit": True})
# ticker_ex: dedicated instance with a LIGHT custom rate limit — fast enough
# for ~1 req/s, but still self-throttled so Phemex never bans/blocks the IP
# (disabling rate-limit entirely caused exactly that — silent total failure).
ticker_ex = ccxt.phemex({
    "options": {"defaultType": "swap"},
    "enableRateLimit": True,
    "rateLimit": 250,     # ms between requests — safe, still ~4 req/s ceiling
    "timeout": 6000,
})
# bt_ex: SEPARATE instance for the backtest's bulk OHLCV fetches, so they never
# run concurrently with the strategy loop on the same ccxt session (a requests
# Session is not thread-safe for concurrent calls).
bt_ex = ccxt.phemex({"options": {"defaultType": "swap"}, "enableRateLimit": True})
SYMBOL = None

_last_good_price   = 0.0
_ticker_fail_count = 0
_jump_reject_count = 0     # consecutive jump-filter rejects → resync after a few
_MAX_TICK_JUMP_PCT = 1.5   # ETH/USDT should not move >1.5% in a single second

# ── Real-time concurrency ────────────────────────────────────────────────
# trade_lock serializes EVERY mutation of the `paper` position so the
# websocket exit thread and the main candle/entry loop can never race (double
# close, half-applied entry, corrupted balance).
# _ws_active is True only while the websocket tick stream is delivering prices;
# when True the main loop stops its REST poll and lets the WS drive exits.
# _last_candle_str mirrors the loop's candle label so WS-driven exits can tag
# the trade history with the right candle.
trade_lock       = threading.Lock()
_ws_active       = False
_last_candle_str = "—"
_last_price_ts   = 0.0     # time.time() of the last price update (WS or REST)
_last_ws_tick_ts = 0.0     # time.time() of the last WEBSOCKET tick specifically
_ws_tick_count   = 0       # how many websocket ticks have been processed

def init_markets():
    global SYMBOL
    mkts = live_ex.load_markets()
    ticker_ex.load_markets()
    for sym in ["ETH/USDT:USDT", "ETH/USD:ETH"]:
        if sym in mkts:
            SYMBOL = sym; break
    if not SYMBOL:
        for k, v in mkts.items():
            if "ETH" in k and v.get("type") in ("swap", "future"):
                SYMBOL = k; break
    if not SYMBOL:
        raise RuntimeError("No ETH perpetual found on Phemex")

    # ── SL/TP/trail distances come from the editable config (config.json / UI) ──
    _apply_config()
    logger.info(f"[INIT] SL=${SL_DIST}  trail-activation=${TP_DIST}  "
                f"trail-offset=${TR_DIST}  (editable in the dashboard)")
    # Warm up ticker_ex once at init so the first loop iteration isn't 0.0
    try:
        warm = float(ticker_ex.fetch_ticker(SYMBOL)["last"])
        if warm > 0:
            global _last_good_price
            _last_good_price = warm
            logger.info(f"[INIT] ticker warm-up OK price={warm:.2f}")
    except Exception as e:
        logger.warning(f"[INIT] ticker warm-up failed: {type(e).__name__}: {e}")
    logger.info(f"[INIT] symbol={SYMBOL}")

def get_live_price() -> float:
    """
    Real-time price via ccxt with a light, safe rate limit (avoids exchange
    bans). Sanity filter rejects single-tick jumps > _MAX_TICK_JUMP_PCT,
    which would indicate a corrupted read rather than real price movement.
    Logs the actual exception type/message on failure so root cause is
    visible in Render logs instead of failing silently.
    """
    global _last_good_price, _ticker_fail_count, _jump_reject_count, _last_price_ts
    try:
        price = float(ticker_ex.fetch_ticker(SYMBOL or "ETH/USDT:USDT")["last"])
        if price <= 0:
            _ticker_fail_count += 1
            if _ticker_fail_count % 10 == 1:
                logger.warning(f"[TICKER] got non-positive price ({price})")
            return _last_good_price

        if _last_good_price > 0:
            jump_pct = abs(price - _last_good_price) / _last_good_price * 100
            if jump_pct > _MAX_TICK_JUMP_PCT:
                _jump_reject_count += 1
                # A one-off corrupted read is rejected; but a REAL sustained move
                # keeps "jumping" vs the now-stale price, so after a few rejects in
                # a row we ACCEPT it and resync — otherwise the price freezes
                # forever (this was the "not receiving live prices" bug).
                if _jump_reject_count < 3:
                    logger.warning(f"[TICKER] rejected jump {_last_good_price:.2f} "
                                   f"→ {price:.2f} ({jump_pct:.2f}%) "
                                   f"#{_jump_reject_count}")
                    return _last_good_price
                logger.warning(f"[TICKER] resync after {_jump_reject_count} "
                               f"rejected jumps → {price:.2f}")

        _last_good_price   = price
        _last_price_ts     = time_mod.time()
        _ticker_fail_count = 0
        _jump_reject_count = 0
        return price
    except Exception as e:
        _ticker_fail_count += 1
        # Log every failure for the first 5, then every 10th — visible in
        # Render logs without flooding them.
        if _ticker_fail_count <= 5 or _ticker_fail_count % 10 == 0:
            logger.warning(f"[TICKER] fail #{_ticker_fail_count} "
                            f"{type(e).__name__}: {e}")
        return _last_good_price

def fetch_candles(limit=CANDLES):
    try:
        since = int((time_mod.time() - limit * 1800) * 1000)
        rows  = live_ex.fetch_ohlcv(SYMBOL or "ETH/USDT:USDT",
                                     timeframe="30m", since=since, limit=limit)
        if rows and len(rows) >= 70:
            return rows
        logger.warning(f"[OHLCV] only {len(rows) if rows else 0} candles")
    except Exception as e:
        logger.error(f"[OHLCV] {e}")
    return []

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PAPER TRADING STATE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
paper = {
    "balance":      PAPER_START_BAL,
    "side":         "none",
    "entry_price":  0.0,
    "qty":          0.0,
    "peak":         0.0,
    "trail_active": False,
    "trail_stop":   0.0,
    "unrealized":   0.0,
    "mfe_pct":      0.0,   # max favorable excursion (best unrealized %) this trade
    "mae_pct":      0.0,   # max adverse excursion (worst unrealized %) this trade
    "total_pnl":    0.0,
    "total_pnl_pct": 0.0,
}

def _pnl_pct(raw_pnl_usd, entry_price, qty):
    """Net PnL % after fees."""
    position_value = entry_price * qty
    if position_value == 0:
        return 0.0
    fee_cost_pct = fees["entry"] + fees["exit"]
    raw_pct      = (raw_pnl_usd / position_value) * 100
    return round(raw_pct - fee_cost_pct, 4)

def _unrealized_pct(cur_price):
    if paper["side"] == "none" or paper["entry_price"] == 0:
        return 0.0
    diff = cur_price - paper["entry_price"]
    if paper["side"] == "short":
        diff = -diff
    raw_pct = (diff / paper["entry_price"]) * 100
    return round(raw_pct - fees["entry"], 4)   # subtract entry fee already paid

def _open_position(side, qty, price):
    paper["side"]         = "long" if side == "LONG" else "short"
    paper["entry_price"]  = price
    paper["qty"]          = qty
    paper["peak"]         = price
    paper["trail_active"] = False
    paper["trail_stop"]   = 0.0
    paper["unrealized"]   = 0.0
    paper["mfe_pct"]      = 0.0   # reset excursions for the new trade
    paper["mae_pct"]      = 0.0
    oid = f"SIM-{int(time_mod.time())}"
    logger.info(f"[PAPER] OPEN {side}  qty={qty}  @ {price:.2f}  id={oid}")
    return {"id": oid}

def _close_position(price, reason):
    entry = paper["entry_price"]
    qty   = paper["qty"]
    raw   = (price - entry) * qty if paper["side"] == "long" else (entry - price) * qty
    pct   = _pnl_pct(raw, entry, qty)
    paper["balance"]       += raw - (entry * qty * fees["entry"] / 100) \
                                  - (price  * qty * fees["exit"]  / 100)
    paper["total_pnl"]     += raw
    paper["total_pnl_pct"]  = round(
        (paper["balance"] - PAPER_START_BAL) / PAPER_START_BAL * 100, 2)
    paper["unrealized"]     = 0.0
    logger.info(f"[PAPER] CLOSE {paper['side'].upper()} ({reason})  @ {price:.2f}  "
                f"pnl={pct:+.2f}%  bal={paper['balance']:.2f}")
    paper["side"]         = "none"
    paper["entry_price"]  = 0.0
    paper["qty"]          = 0.0
    paper["peak"]         = 0.0
    paper["trail_active"] = False
    paper["trail_stop"]   = 0.0
    return raw, pct

def _calc_qty(price, balance):
    # Position notional = full balance (1x): qty = balance / price. On the
    # TradingView run the position tracked the account exactly — trade #976 held
    # 5.62 ETH and 5.62 = balance / price. So each trade compounds the balance by
    # the SAME % it shows, and the total finally agrees with the trades (the old
    # risk/(fixedSL·mintick) sizing shrank the position, so the total grew far
    # slower than the per-trade %s). Capped at MAX_LOTS like Pine.
    qty = (balance / price) if price else 0.0
    return round(min(qty, MAX_LOTS), 4)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SL / TRAILING TP  — two variants
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def _check_sl_tp(pos, cur_price: float):
    """
    Intra-candle exit math — the SINGLE source of truth used by BOTH live paper
    trading (pos = the global `paper`) AND the backtest (pos = a local dict), so
    the two can never diverge again. Mutates pos["peak"]/["trail_active"]/
    ["trail_stop"] and returns (exit, exit_price, reason).

    Long  : peak = highest price since entry; trail arms once peak-entry >= TP_DIST
            then sits TR_DIST below peak. SL is FIXED at entry-SL_DIST (Pine `loss`).
    Short : symmetric (peak = lowest price; trail sits TR_DIST above it).
    """
    if pos["side"] == "none":
        return False, 0.0, ""
    entry = pos["entry_price"]

    if pos["side"] == "long":
        if cur_price > pos["peak"]:
            pos["peak"] = cur_price
        if pos["peak"] - entry >= TP_DIST:
            nt = round(pos["peak"] - TR_DIST, 8)
            if not pos["trail_active"] or nt > pos["trail_stop"]:
                pos["trail_active"] = True
                pos["trail_stop"]   = nt
        if pos["trail_active"] and cur_price <= pos["trail_stop"]:
            return True, pos["trail_stop"], "TRAIL_TP"
        sl = round(entry - SL_DIST, 8)               # FIXED stop-loss (Pine `loss`)
        if cur_price <= sl:
            return True, sl, "SL"
    else:
        if pos["peak"] == 0 or cur_price < pos["peak"]:
            pos["peak"] = cur_price
        if entry - pos["peak"] >= TP_DIST:
            nt = round(pos["peak"] + TR_DIST, 8)
            if not pos["trail_active"] or nt < pos["trail_stop"]:
                pos["trail_active"] = True
                pos["trail_stop"]   = nt
        if pos["trail_active"] and cur_price >= pos["trail_stop"]:
            return True, pos["trail_stop"], "TRAIL_TP"
        sl = round(entry + SL_DIST, 8)               # FIXED stop-loss (Pine `loss`)
        if cur_price >= sl:
            return True, sl, "SL"

    return False, 0.0, ""

def _check_sl_tp_candle(o: float, h: float, l: float, c: float):
    """
    Closed-candle exit check — mirrors the TradingView broker emulator with
    calc_on_every_tick=false (no bar magnifier).

    Two things matter for parity:

    1. Which stop is live. Pine generates the fixed stop (loss) AND the trailing
       stop but only ever uses the one closest to price. So before the trail
       arms (peak profit < trail-activation) the live stop is the fixed SL; once
       it arms, the live stop is the trailing stop (which sits nearer to price).

    2. Intra-bar order. Without a bar magnifier TradingView assumes:
         up bar   (close >= open):  open -> low  -> high -> close
         down bar (close <  open):  open -> high -> low  -> close
       i.e. it processes the adverse extreme first on up bars and the favorable
       extreme first on down bars. We replicate that ordering so SL vs trail
       fire in the same sequence TradingView would use.
    """
    if paper["side"] == "none":
        return False, 0.0, ""

    entry  = paper["entry_price"]
    is_up  = c >= o

    if paper["side"] == "long":
        sl = round(entry - SL_DIST, 8)

        def hit_low():
            # Check whichever stop is currently live against this bar's low.
            if paper["trail_active"]:
                if l <= paper["trail_stop"]:
                    return True, paper["trail_stop"], "TRAIL_TP"
            elif l <= sl:
                return True, sl, "SL"
            return False, 0.0, ""

        def advance_high():
            # Update peak with the bar high and arm/raise the trailing stop.
            if h > paper["peak"]:
                paper["peak"] = h
            if paper["peak"] - entry >= TP_DIST:
                nt = round(paper["peak"] - TR_DIST, 8)
                if not paper["trail_active"] or nt > paper["trail_stop"]:
                    paper["trail_active"] = True
                    paper["trail_stop"]   = nt

        if is_up:                       # open -> low -> high -> close
            ex = hit_low()              # adverse leg (open->low) first
            if ex[0]:
                return ex
            advance_high()              # favorable leg (low->high): ride peak, arm trail
            # retrace leg (high->close): if the price falls trail_offset below the
            # new peak on its way to the close, the trailing stop fires in THIS
            # same candle — this is how a winner exits at extreme ± offset on the
            # very candle it ran (e.g. the screenshots' same-candle exits).
            if paper["trail_active"] and c <= paper["trail_stop"]:
                return True, paper["trail_stop"], "TRAIL_TP"
        else:                           # open -> high -> low
            advance_high()
            ex = hit_low()
            if ex[0]:
                return ex

    else:  # short
        sl = round(entry + SL_DIST, 8)

        def hit_high():
            if paper["trail_active"]:
                if h >= paper["trail_stop"]:
                    return True, paper["trail_stop"], "TRAIL_TP"
            elif h >= sl:
                return True, sl, "SL"
            return False, 0.0, ""

        def advance_low():
            if paper["peak"] == 0 or l < paper["peak"]:
                paper["peak"] = l
            if entry - paper["peak"] >= TP_DIST:
                nt = round(paper["peak"] + TR_DIST, 8)
                if not paper["trail_active"] or nt < paper["trail_stop"]:
                    paper["trail_active"] = True
                    paper["trail_stop"]   = nt

        if is_up:                       # open -> low -> high : favorable first
            advance_low()
            ex = hit_high()
            if ex[0]:
                return ex
        else:                           # open -> high -> low -> close : adverse first
            ex = hit_high()
            if ex[0]:
                return ex
            advance_low()               # favorable leg (high->low): ride peak, arm trail
            # retrace leg (low->close): if the price rises trail_offset above the
            # new low on its way to the close, the trailing stop fires in THIS
            # same candle (the #976-style same-candle short exit).
            if paper["trail_active"] and c >= paper["trail_stop"]:
                return True, paper["trail_stop"], "TRAIL_TP"

    return False, 0.0, ""

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TRADE HISTORY
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TRADES_FILE = "trades.json"

def _load_history():
    try:
        with open(TRADES_FILE) as f: return json.load(f)
    except Exception: return []

def _save_history():
    try:
        with open(TRADES_FILE, "w") as f: json.dump(trade_history, f)
    except Exception as e: logger.warning(f"[HISTORY] {e}")

def _record_exit(side_str, price, reason, pnl_usd, pnl_pct_val, candle_str):
    # paper["mfe_pct"]/["mae_pct"] still hold THIS trade's excursions here:
    # _close_position does not reset them (only _open_position does), so they are
    # the max profit / max loss % the trade reached while it was open.
    trade_history.insert(0, {
        "time":     datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "candle":   candle_str,
        "side":     f"CLOSE {side_str} ({reason})",
        "price":    round(price, 4),
        "qty":      0,
        "order_id": f"EXIT-{reason}",
        "balance":  f"{paper['balance']:.2f}",
        "pnl_pct":  f"{pnl_pct_val:+.2f}",
        "mfe":      f"{paper['mfe_pct']:+.2f}",
        "mae":      f"{paper['mae_pct']:+.2f}",
    })
    _save_history()

def _record_entry(side_str, price, qty, candle_str, order_id):
    trade_history.insert(0, {
        "time":     datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "candle":   candle_str,
        "side":     side_str,
        "price":    round(price, 4),
        "qty":      qty,
        "order_id": order_id,
        "balance":  f"{paper['balance']:.2f}",
        "pnl_pct":  "—",
        "mfe":      "—",
        "mae":      "—",
    })
    _save_history()

def _winrate():
    closed = [t for t in trade_history
              if isinstance(t.get("pnl_pct"), str) and t["pnl_pct"] != "—"]
    if not closed: return "—"
    wins = sum(1 for t in closed if float(t["pnl_pct"]) > 0)
    return f"{wins/len(closed)*100:.1f}%  ({wins}/{len(closed)})"

trade_history    = _load_history()
_strategy_thread = None
logger.info(f"[HISTORY] {len(trade_history)} trades loaded")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SHARED STATE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
state = {
    "running":     False, "status":     "Stopped",
    "signal":      "—",   "position":   "None",
    "ema":         "—",   "ec":         "—",    "period":    "—",
    "last_candle": "—",   "updated":    "—",
    "balance":     f"{PAPER_START_BAL:.2f}",
    "total_pnl_pct": "0.00",
    "unrealized":  "0.00",
    "mfe":         "0.00",
    "mae":         "0.00",
    "trail_stop":  "—",
    "winrate":     "—",
    "price":       "—",
}

def _resolve_candles(ohlcv):
    """
    Returns (last_closed, sig_closes).
    A candle with timestamp T covers T … T+1800 s.
    If T+1800 > now  →  still forming  →  skip it.
    """
    TF_MS   = 1800 * 1000
    now_ms  = int(time_mod.time() * 1000)
    if ohlcv[-1][0] + TF_MS > now_ms:
        # Last bar is still forming — use second-to-last as closed
        return ohlcv[-2], ohlcv[:-1]
    else:
        # All bars are closed — last bar is the most recent closed
        return ohlcv[-1], ohlcv


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STRATEGY LOOP
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def _try_realtime_exit(price: float, candle_str: str):
    """
    Run the tick-by-tick SL/trailing check and close the position if it fires.
    Thread-safe: called from the websocket tick handler AND the REST fallback,
    so all position mutation happens under trade_lock. Returns True if it closed.
    """
    with trade_lock:
        if paper["side"] == "none":
            return False
        ex, ex_px, ex_rsn = _check_sl_tp(paper, price)
        if not ex:
            return False
        ex_side  = paper["side"]
        raw, pct = _close_position(ex_px, ex_rsn)
        _record_exit(ex_side.upper(), ex_px, ex_rsn, raw, pct, candle_str)
        state.update({
            "position": "none", "unrealized": "0.00",
            "balance":  f"{paper['balance']:.2f}",
            "total_pnl_pct": f"{paper['total_pnl_pct']:+.2f}",
            "trail_stop": "—",
            "winrate":  _winrate(),
        })
        logger.info(f"[EXIT] {ex_rsn} @ {ex_px:.2f}  pnl={pct:+.2f}%")
        return True


def _process_1m_exit(o, h, l, c):
    """
    Replay one CLOSED 1-minute candle through _check_sl_tp on the LIVE position —
    IDENTICAL to the backtest's per-1m processing (same _check_sl_tp, same
    _second_path), so live exits match the backtest instead of firing on raw tick
    noise. Updates MFE/MAE and closes the position if the trailing/SL fires.
    """
    with trade_lock:
        if paper["side"] == "none":
            return
        entry = paper["entry_price"]
        for px in _second_path(float(o), float(h), float(l), float(c)):
            d   = (px - entry) if paper["side"] == "long" else (entry - px)
            pct = (d / entry * 100) if entry else 0.0
            if pct > paper["mfe_pct"]: paper["mfe_pct"] = pct
            if pct < paper["mae_pct"]: paper["mae_pct"] = pct
            ex, ex_px, ex_rsn = _check_sl_tp(paper, px)
            if ex:
                ex_side  = paper["side"]
                raw, p   = _close_position(ex_px, ex_rsn)
                _record_exit(ex_side.upper(), ex_px, ex_rsn, raw, p, _last_candle_str)
                state.update({
                    "position": "none", "unrealized": "0.00",
                    "balance":  f"{paper['balance']:.2f}",
                    "total_pnl_pct": f"{paper['total_pnl_pct']:+.2f}",
                    "trail_stop": "—", "winrate": _winrate(),
                })
                logger.info(f"[EXIT-1m] {ex_rsn} @ {ex_px:.2f}  pnl={p:+.2f}%")
                return


def on_live_price(price: float, from_ws: bool = False):
    """
    Updates the live price + unrealized PnL for the UI ONLY. Exits are NOT driven
    here anymore — they run on CLOSED 1-minute candles (see _process_1m_exit),
    exactly like the backtest, so live results match the backtest instead of
    firing on raw tick noise (the $0.10 trailing would close on every tick wiggle).
    """
    global _last_good_price, _last_price_ts, _last_ws_tick_ts, _jump_reject_count
    if price <= 0:
        return
    if _last_good_price > 0:
        jump = abs(price - _last_good_price) / _last_good_price * 100
        if jump > _MAX_TICK_JUMP_PCT:
            _jump_reject_count += 1
            if _jump_reject_count < 3:
                return                          # reject a one-off corrupted tick
    _last_good_price = price
    now = time_mod.time()
    _last_price_ts   = now
    _jump_reject_count = 0
    if from_ws:
        _last_ws_tick_ts = now

    paper["unrealized"] = _unrealized_pct(price)
    open_pos = paper["side"] != "none"
    trail_disp = (f"${paper['trail_stop']:.2f}" if paper["trail_active"]
                  else ("armed" if open_pos else "—"))
    state.update({
        "price":         f"{price:.2f}",
        "unrealized":    f"{paper['unrealized']:+.2f}",
        "mfe":           f"{paper['mfe_pct']:+.2f}" if open_pos else "0.00",
        "mae":           f"{paper['mae_pct']:+.2f}" if open_pos else "0.00",
        "balance":       f"{paper['balance']:.2f}",
        "total_pnl_pct": f"{paper['total_pnl_pct']:+.2f}",
        "trail_stop":    trail_disp,
        "position":      paper["side"],
        "status":        "Running",
    })


def price_ws_loop():
    """
    Background websocket tick feed. Subscribes to Phemex's real-time ticker and
    pushes every price into on_live_price(), so the SL/trailing exit fires the
    instant the move happens — no REST round-trip, no 1 s sampling delay. ccxt.pro
    normalizes the price exactly like the REST client (so we never hand-roll the
    price scale — that was a past bug). Auto-reconnects; if the WS is down,
    _ws_active stays False and the main loop falls back to its REST poll.
    """
    global _ws_active
    if not _HAS_WS:
        logger.warning("[WS] ccxt.pro unavailable — staying on REST polling")
        return

    async def run():
        global _ws_active, _ws_tick_count
        ws  = ccxtpro.phemex({"options": {"defaultType": "swap"},
                              "enableRateLimit": True})
        sym = SYMBOL or "ETH/USDT:USDT"
        try:
            await ws.load_markets()
            logger.info(f"[WS] connecting ticker stream for {sym}")
            fails = 0
            while state["running"]:
                try:
                    t  = await ws.watch_ticker(sym)
                    px = t.get("last") or t.get("close")
                    if px:
                        if not _ws_active:
                            logger.info("[WS] tick stream LIVE — exits now real-time")
                        _ws_active = True
                        _ws_tick_count += 1
                        fails = 0
                        on_live_price(float(px), from_ws=True)
                except Exception as e:
                    fails += 1
                    _ws_active = False
                    if fails <= 5 or fails % 20 == 0:
                        logger.warning(f"[WS] watch fail #{fails} "
                                       f"{type(e).__name__}: {e}")
                    await asyncio.sleep(min(fails, 5))
        finally:
            try:
                await ws.close()
            except Exception:
                pass

    try:
        asyncio.run(run())
    except Exception as e:
        logger.error(f"[WS] loop crashed: {type(e).__name__}: {e}")
    finally:
        _ws_active = False


def strategy_loop():
    global state, trade_history, _last_candle_str

    try:
        init_markets()
    except Exception as e:
        logger.error(f"[INIT] {e}")
        state["status"] = f"Error: {e}"
        state["running"] = False
        return

    last_candle_ts   = None
    last_candle_str  = "—"
    last_ohlcv_check = 0.0   # throttle 30m ohlcv fetching to every 15 s
    last_1m_check    = 0.0   # throttle 1m ohlcv fetching to every 5 s
    last_1m_ts       = 0     # timestamp of the last CLOSED 1m candle processed for exits

    # Startup sync: LOCK to the current last-closed candle so the first ENTRY
    # only happens at the NEXT candle boundary (:00 / :30) — entries must be at a
    # candle open, at that candle's price. (Entering mid-candle on Start used a
    # stale candle-open price and was wrong.) The trade-off is the first trade can
    # take up to 30 min; that wait is normal, not a failure.
    init_ohlcv = fetch_candles()
    if init_ohlcv and len(init_ohlcv) >= 70:
        _lc, _ = _resolve_candles(init_ohlcv)
        last_candle_ts = _lc[0]
        logger.info(f"[SYNC] locked to candle "
                    f"{datetime.fromtimestamp(last_candle_ts/1000,tz=timezone.utc)}"
                    f" — first entry at the next candle close")

    logger.info(f"[CFG] REALTIME_EXITS={REALTIME_EXITS}  ws_available={_HAS_WS}  "
                f"(if REALTIME_EXITS is False, exits only happen at candle close — "
                f"check the Render env var)")

    # Start the real-time websocket tick feed (delay-free exits). Daemon thread;
    # if it can't connect, the ~1 s REST poll below keeps managing exits.
    threading.Thread(target=price_ws_loop, daemon=True).start()

    while state["running"]:
        try:
            # ── 1a. LIVE PRICE (UI display only — does NOT drive exits) ──────
            live_price = get_live_price()
            if live_price > 0:
                on_live_price(live_price)

            # ── 1b. EXITS run on CLOSED 1-MINUTE candles (every ~5 s), exactly
            #        like the backtest, so live results MATCH the backtest instead
            #        of firing on raw tick noise. ────────────────────────────────
            now = time_mod.time()
            if paper["side"] != "none" and now - last_1m_check >= 5:
                last_1m_check = now
                try:
                    rows1m = live_ex.fetch_ohlcv(SYMBOL or "ETH/USDT:USDT",
                                                 timeframe="1m", limit=6)
                    now_ms = int(now * 1000)
                    for c1 in (rows1m or []):
                        if c1[0] <= last_1m_ts or c1[0] + 60_000 > now_ms:
                            continue                       # done already / still forming
                        last_1m_ts = c1[0]
                        _process_1m_exit(c1[1], c1[2], c1[3], c1[4])
                        if paper["side"] == "none":
                            break
                except Exception as e:
                    logger.warning(f"[1M] {type(e).__name__}: {e}")

            # ── 2. CANDLE CLOSE  (every 15 s — this is where trades happen) ──
            if now - last_ohlcv_check >= 15:
                last_ohlcv_check = now
                ohlcv = fetch_candles()

                if ohlcv and len(ohlcv) >= 70:
                    last_closed, sig_closes = _resolve_candles(ohlcv)
                    candle_ts = last_closed[0]
                    last_candle_str = datetime.fromtimestamp(
                        candle_ts / 1000, tz=timezone.utc
                    ).strftime("%Y-%m-%d %H:%M UTC")

                    if candle_ts != last_candle_ts:
                        last_candle_ts   = candle_ts
                        _last_candle_str = last_candle_str   # share with WS exits
                        # process the NEW (forming) 30m candle's 1m sub-candles from
                        # its open (= candle_ts + 30 min) onward for the new trade.
                        last_1m_ts       = candle_ts + 1_800_000 - 1

                        # Hold trade_lock across the whole bar transition so a
                        # websocket-driven exit can't race the bar-close exit /
                        # entry below (no double close, no half-applied entry).
                        with trade_lock:
                            # (a) Candle-close stop check — ONLY in candle mode
                            #     (REALTIME_EXITS=0). In intra-candle mode the live
                            #     tick trailing already owns exits and the user does
                            #     NOT want trades closed at the candle's O/H/L/C
                            #     extreme, so this is skipped entirely.
                            if not REALTIME_EXITS:
                                cl_ex, cl_px, cl_rsn = _check_sl_tp_candle(
                                    float(last_closed[1]), float(last_closed[2]),
                                    float(last_closed[3]), float(last_closed[4]))
                                if cl_ex and paper["side"] != "none":
                                    cl_side = paper["side"]
                                    raw, pct = _close_position(cl_px, cl_rsn)
                                    _record_exit(cl_side.upper(), cl_px, cl_rsn,
                                                 raw, pct, last_candle_str)
                                    state.update({
                                        "position": "none", "unrealized": "0.00",
                                        "balance":  f"{paper['balance']:.2f}",
                                        "total_pnl_pct": f"{paper['total_pnl_pct']:+.2f}",
                                        "trail_stop": "—",
                                        "winrate":  _winrate(),
                                    })

                            # (a2) TIME EXIT (intra-candle mode): if the trade did
                            #      NOT hit the trailing-TP or the SL during the
                            #      candle, close it NOW at the candle's CLOSE price
                            #      (the user's rule: "fecha no fim do candle se não
                            #      bater trailing nem SL"). The intra-candle
                            #      trailing/SL in _check_sl_tp handle the live exits;
                            #      this guarantees nothing stays open past a candle.
                            elif paper["side"] != "none":
                                ce_px    = float(last_closed[4])
                                ce_side  = paper["side"]
                                raw, pct = _close_position(ce_px, "CANDLE_END")
                                _record_exit(ce_side.upper(), ce_px, "CANDLE_END",
                                             raw, pct, last_candle_str)
                                state.update({
                                    "position": "none", "unrealized": "0.00",
                                    "balance":  f"{paper['balance']:.2f}",
                                    "total_pnl_pct": f"{paper['total_pnl_pct']:+.2f}",
                                    "trail_stop": "—",
                                    "winrate":  _winrate(),
                                })

                            # (b) Recompute the signal on the closed bar.
                            closes = [c[4] for c in sig_closes]
                            result = strat.calculate(closes, int(config.get("gain_limit", 900)))
                            signal = result["signal"]
                            state.update({
                                "signal":      signal or "—",
                                "ema":         f"{result['ema']:.4f}" if result["ema"] else "—",
                                "ec":          f"{result['ec']:.4f}"  if result["ec"]  else "—",
                                "period":      str(result["period"]),
                                "last_candle": last_candle_str,
                                "updated":     datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                "winrate":     _winrate(),
                            })

                            # (c) Entry / reversal. Pine fires strategy.entry on EVERY
                            #     bar the condition holds, with pyramiding=1 — so the
                            #     net rule is simply: hold the side the signal points
                            #     to. Crucially this re-enters right after a stop if
                            #     the condition still holds (the old prev_signal guard
                            #     wrongly blocked that). Entry fills at the signal
                            #     bar's close = next bar's open in 24/7 crypto (no gap),
                            #     the price a TradingView market order would get.
                            if signal:
                                entry_px = float(last_closed[4])
                                qty      = _calc_qty(entry_px, paper["balance"])
                                cur_pos  = paper["side"]

                                if signal == "LONG" and cur_pos != "long":
                                    if cur_pos == "short":
                                        raw, pct = _close_position(entry_px, "SIGNAL_REVERSE")
                                        _record_exit("SHORT", entry_px, "SIGNAL_REVERSE",
                                                     raw, pct, last_candle_str)
                                    order = _open_position("LONG", qty, entry_px)
                                    _record_entry("LONG", entry_px, qty,
                                                  last_candle_str, order["id"])

                                elif signal == "SHORT" and cur_pos != "short":
                                    if cur_pos == "long":
                                        raw, pct = _close_position(entry_px, "SIGNAL_REVERSE")
                                        _record_exit("LONG", entry_px, "SIGNAL_REVERSE",
                                                     raw, pct, last_candle_str)
                                    order = _open_position("SHORT", qty, entry_px)
                                    _record_entry("SHORT", entry_px, qty,
                                                  last_candle_str, order["id"])

            # ── 3. SLEEP 1 SECOND ─────────────────────────────────────────────
            time_mod.sleep(1)

        except Exception as e:
            logger.error(f"[LOOP] {e}")
            state["status"] = f"Error: {str(e)[:80]}"
            time_mod.sleep(5)

    state["status"] = "Stopped"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# KEEPALIVE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def keepalive(interval):
    time_mod.sleep(12)
    while True:
        try: req.get(f"http://localhost:{PORT}/ping", timeout=4)
        except Exception: pass
        time_mod.sleep(interval)

for _iv in [8, 15, 23]:
    threading.Thread(target=keepalive, args=(_iv,), daemon=True).start()

# ── Always-on live-price feed ─────────────────────────────────────────────
# Keeps _last_good_price fresh for the UI even when the strategy is STOPPED, so
# the live price is always visible. When the strategy is running its own loop +
# websocket already refresh the price, so this only fills the gaps.
def price_poller():
    try:
        ticker_ex.load_markets()
    except Exception as e:
        logger.warning(f"[PRICE] poller load_markets failed: {type(e).__name__}: {e}")
    while True:
        try:
            get_live_price()              # refreshes _last_good_price + _last_price_ts
        except Exception:
            pass
        time_mod.sleep(2)

threading.Thread(target=price_poller, daemon=True).start()

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# BACKTEST  (intra-candle replay using 1-minute sub-candles)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Each 30m candle is NOT read as just its O/H/L/C — we replay the real intra-candle
# price path from its 1-minute sub-candles and run the SAME trailing/SL/candle-end
# exit rules as live trading, so the backtest is as close to real trades as REST
# data allows. Runs in a background thread; the UI polls /backtest-result.
_backtest = {"running": False, "progress": "", "result": None, "error": None}

def _fetch_30m_candles(n):
    """Paginated 30m fetch — supports UNLIMITED n (e.g. a year = 17 520 candles),
    since Phemex caps each request at ~1000."""
    now_ms = int(time_mod.time() * 1000)
    out, cur = [], now_ms - int(n) * 1_800_000
    for _ in range(400):                       # safety cap on pages (~400k candles)
        try:
            rows = bt_ex.fetch_ohlcv(SYMBOL or "ETH/USDT:USDT", timeframe="30m",
                                     since=cur, limit=1000)
        except Exception as e:
            logger.warning(f"[BT] 30m fetch: {type(e).__name__}: {e}"); break
        if not rows: break
        out.extend(rows)
        last = rows[-1][0]
        if last <= cur or last >= now_ms: break
        cur = last + 1_800_000
        _backtest["progress"] = f"buscando candles de 30m… ({len(out)})"
    seen = {r[0]: r for r in out}
    return [seen[k] for k in sorted(seen)]

def _fetch_1m_candles(since_ms, until_ms):
    out, cur = [], int(since_ms)
    for _ in range(2000):                      # safety cap on pages (~2M minutes ≈ 3.8 yr)
        try:
            rows = bt_ex.fetch_ohlcv(SYMBOL or "ETH/USDT:USDT", timeframe="1m",
                                     since=cur, limit=1000)
        except Exception as e:
            logger.warning(f"[BT] 1m fetch: {type(e).__name__}: {e}"); break
        if not rows: break
        out.extend(rows)
        last = rows[-1][0]
        if last <= cur or last >= until_ms: break   # reached the end / no progress
        cur = last + 60_000
        if len(out) % 10000 < 1000:
            _backtest["progress"] = f"buscando candles de 1m… ({len(out)})"
    seen = {r[0]: r for r in out}
    return [seen[k] for k in sorted(seen)]

def _second_path(o, h, l, c, steps=60):
    # Read each 1-minute candle SECOND BY SECOND (~`steps` points), walking the
    # broker path O→L→H→C (up bar) / O→H→L→C (down bar). Real 1-second history
    # isn't available via REST, so we step through each minute's OHLC at 1s
    # resolution — the backtest "sees" prices ~like the live 1s poll.
    pts = [o, l, h, c] if c >= o else [o, h, l, c]
    out = []; seg = max(1, steps // 3)
    for k in range(3):
        a, b = pts[k], pts[k + 1]
        for s in range(seg):
            out.append(a + (b - a) * (s / seg))
    out.append(pts[3])
    return out

def _simulate_trade(side, entry, subs, candle_close):
    """Replay one trade through its 1m sub-candles at 1s resolution, using the
    EXACT SAME _check_sl_tp as live paper trading (single source of truth, so the
    backtest result matches live). Returns (exit_px, reason, mfe%, mae%)."""
    pos = {"side": "long" if side == "LONG" else "short", "entry_price": entry,
           "peak": entry, "trail_active": False, "trail_stop": 0.0}
    mfe = 0.0; mae = 0.0
    for s in subs:
        for px in _second_path(float(s[1]), float(s[2]), float(s[3]), float(s[4])):
            d = (px - entry) if side == "LONG" else (entry - px)
            pct = (d / entry * 100) if entry else 0.0
            if pct > mfe: mfe = pct
            if pct < mae: mae = pct
            ex, ex_px, reason = _check_sl_tp(pos, px)     # same logic as live
            if ex:
                return ex_px, reason, mfe, mae
    return candle_close, "CANDLE_END", mfe, mae   # never hit TP/SL → close at candle end

def run_backtest(n_candles=300, fee_entry=0.0, fee_exit=0.0):
    _backtest.update({"running": True, "result": None, "error": None,
                      "progress": "buscando candles de 30m…"})
    try:
        try:
            bt_ex.load_markets()
        except Exception as e:
            logger.warning(f"[BT] load_markets: {type(e).__name__}: {e}")
        warm = 200                                   # warmup bars so EC/EMA/period converge like TV
        c30  = _fetch_30m_candles(n_candles + warm)   # paginated → unlimited length
        if not c30 or len(c30) < 70:
            raise RuntimeError("poucos candles de 30m da Phemex")
        c30 = c30[-(n_candles + warm):]
        # only the LAST n_candles are traded; the earlier ones are warmup-only
        start_i = max(60, len(c30) - 1 - n_candles)
        _backtest["progress"] = "buscando candles de 1m (variações intra-candle)…"
        # 1m only for the TRADED window (keeps it fast; warmup needs only 30m closes)
        c1m = _fetch_1m_candles(c30[start_i][0], c30[-1][0] + 1_800_000)
        buckets = {}
        for r in c1m:
            buckets.setdefault((r[0] // 1_800_000) * 1_800_000, []).append(r)
        for b in buckets: buckets[b].sort(key=lambda x: x[0])

        _backtest["progress"] = "calculando sinais (EC×EMA por candle)…"
        signals = strat.calculate_series([c[4] for c in c30], int(config.get("gain_limit", 900)))

        _backtest["progress"] = "simulando trades (replay 1m)…"
        sl, tp, tr = SL_DIST, TP_DIST, TR_DIST
        bal = PAPER_START_BAL; trades = []; wins = 0
        for i in range(start_i, len(c30) - 1):
            sig = signals[i]
            if not sig: continue
            entry = float(c30[i][4])                 # candle i close = candle i+1 open
            nxt   = c30[i + 1]
            subs  = buckets.get((nxt[0] // 1_800_000) * 1_800_000) or [nxt]
            ex_px, reason, mfe, mae = _simulate_trade(sig, entry, subs, float(nxt[4]))
            qty = (bal / entry) if entry else 0.0
            raw = (ex_px - entry) * qty if sig == "LONG" else (entry - ex_px) * qty
            fee = (entry * qty * fee_entry / 100) + (ex_px * qty * fee_exit / 100)
            raw -= fee                                   # net of entry + exit fees
            bal += raw
            pct = (raw / (entry * qty) * 100) if (entry * qty) else 0.0
            if pct > 0: wins += 1
            trades.append({
                "candle": datetime.fromtimestamp(nxt[0] / 1000, tz=timezone.utc)
                          .strftime("%Y-%m-%d %H:%M UTC"),
                "side": sig, "entry": round(entry, 2), "exit": round(ex_px, 2),
                "reason": reason, "pnl_pct": round(pct, 3),
                "mfe": round(mfe, 3), "mae": round(mae, 3), "balance": round(bal, 2),
            })
        n = len(trades)
        rets = [t["pnl_pct"] for t in trades]
        summary = {
            "trades": n,
            "winrate": round(wins / n * 100, 1) if n else 0.0,
            "final_balance": round(bal, 2),
            "total_pnl_pct": round((bal - PAPER_START_BAL) / PAPER_START_BAL * 100, 2),
            "max_profit": round(max(rets), 3) if rets else 0.0,   # biggest single-trade win %
            "max_loss":   round(min(rets), 3) if rets else 0.0,   # biggest single-trade loss %
            "avg_pct":    round(sum(rets) / n, 3) if n else 0.0,  # average % per trade
            "fee_entry": fee_entry, "fee_exit": fee_exit,
            "candles": n_candles, "sl": sl, "tp": tp, "tr": tr,
            "intrabar_minutes_loaded": len(c1m),
        }
        _backtest.update({"running": False, "progress": "concluído",
                          "result": {"summary": summary, "trades": trades[::-1]}})
        logger.info(f"[BT] done: {n} trades, bal={bal:.2f}, "
                    f"pnl={summary['total_pnl_pct']}%")
    except Exception as e:
        logger.error(f"[BT] {type(e).__name__}: {e}")
        _backtest.update({"running": False, "progress": "erro",
                          "error": f"{type(e).__name__}: {e}"})

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# FLASK
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
app = Flask(__name__)

HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>AZLEMA Bot</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{background:#0d1117;color:#e6edf3;font-family:'Segoe UI',sans-serif}
header{background:#161b22;border-bottom:1px solid #30363d;padding:16px 28px;
       display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:8px}
header h1{font-size:1.15rem;color:#58a6ff}
#clock{font-size:.78rem;color:#8b949e}
.container{max-width:1120px;margin:24px auto;padding:0 18px}
.top-row{display:flex;gap:14px;align-items:flex-start;flex-wrap:wrap;margin-bottom:20px}
.controls{display:flex;gap:10px}
button{padding:9px 24px;border:none;border-radius:8px;font-size:.9rem;font-weight:600;cursor:pointer}
#btn-start{background:#238636;color:#fff}#btn-start:hover:not(:disabled){background:#2ea043}
#btn-stop{background:#da3633;color:#fff}#btn-stop:hover:not(:disabled){background:#f85149}
button:disabled{opacity:.35;cursor:not-allowed}
.fee-box{background:#161b22;border:1px solid #30363d;border-radius:10px;padding:14px 18px;
         display:flex;gap:16px;align-items:center;flex-wrap:wrap}
.fee-box label{font-size:.72rem;color:#8b949e;text-transform:uppercase;letter-spacing:.05em}
.fee-box input{width:72px;background:#0d1117;border:1px solid #30363d;border-radius:6px;
               color:#e6edf3;font-size:.88rem;padding:5px 8px;text-align:center}
.fee-box input:focus{outline:none;border-color:#58a6ff}
#btn-fee{padding:7px 16px;font-size:.82rem;background:#1f6feb;color:#fff;border:none;
         border-radius:6px;cursor:pointer}#btn-fee:hover{background:#388bfd}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(142px,1fr));gap:11px;margin-bottom:22px}
.card{background:#161b22;border:1px solid #30363d;border-radius:10px;padding:14px 16px}
.lbl{font-size:.67rem;color:#8b949e;text-transform:uppercase;letter-spacing:.06em;margin-bottom:4px}
.val{font-size:1.05rem;font-weight:600}
.long{color:#3fb950}.short{color:#f85149}
.running{color:#3fb950}.stopped{color:#8b949e}
.err{color:#e3b341;font-size:.74rem;word-break:break-all}
.pos-num{color:#3fb950}.neg-num{color:#f85149}
.st{font-size:.76rem;color:#8b949e;text-transform:uppercase;letter-spacing:.06em;margin-bottom:8px}
table{width:100%;border-collapse:collapse;background:#161b22;border:1px solid #30363d;
      border-radius:10px;overflow:hidden}
th{background:#21262d;color:#8b949e;font-size:.72rem;text-transform:uppercase;
   padding:10px 12px;text-align:left}
td{padding:9px 12px;font-size:.83rem;border-top:1px solid #21262d}
tr:hover td{background:#1c2128}
.badge{display:inline-block;padding:2px 8px;border-radius:20px;font-size:.72rem;font-weight:700}
.badge.long{background:#0d4429;color:#3fb950}
.badge.short{background:#3d1010;color:#f85149}
.badge.close{background:#1e2533;color:#8b949e}
#toast{position:fixed;bottom:18px;right:18px;padding:10px 16px;border-radius:8px;
       font-size:.84rem;display:none;z-index:99;color:#fff}
footer{text-align:center;color:#484f58;font-size:.71rem;margin:28px 0 14px}
.head-right{display:flex;align-items:center;gap:22px}
#liveprice{text-align:right;line-height:1.1}
#liveprice .lbl2{font-size:.6rem;color:#8b949e;text-transform:uppercase;letter-spacing:.05em;display:block;margin-bottom:2px}
#lp-val{font-size:1.4rem;font-weight:700;color:#3fb950}
#lp-val.stale{color:#e3b341}
.live-dot{display:inline-block;width:8px;height:8px;border-radius:50%;background:#3fb950;
          margin-right:7px;vertical-align:middle;box-shadow:0 0 6px #3fb950}
.live-dot.stale{background:#e3b341;box-shadow:none}
.tabs{display:flex;gap:8px;margin-bottom:16px;border-bottom:1px solid #30363d}
.tabbtn{background:none;border:none;color:#8b949e;padding:10px 18px;font-size:.92rem;
        font-weight:600;cursor:pointer;border-bottom:2px solid transparent;border-radius:0}
.tabbtn.active{color:#58a6ff;border-bottom-color:#58a6ff}
.cfg-box{background:#161b22;border:1px solid #30363d;border-radius:10px;padding:14px 18px;
         display:flex;gap:16px;align-items:flex-end;flex-wrap:wrap;margin-bottom:18px}
.cfg-box label{font-size:.7rem;color:#8b949e;text-transform:uppercase;letter-spacing:.04em}
.cfg-box input{width:90px;background:#0d1117;border:1px solid #30363d;border-radius:6px;
               color:#e6edf3;font-size:.88rem;padding:6px 8px;text-align:center}
.cfg-box input:focus{outline:none;border-color:#58a6ff}
#btn-cfg,#btn-bt{padding:7px 16px;font-size:.82rem;background:#1f6feb;color:#fff;border:none;
                 border-radius:6px;cursor:pointer}#btn-cfg:hover,#btn-bt:hover{background:#388bfd}
.cfg-hint{font-size:.72rem;color:#8b949e}
.bt-ctrl{display:flex;gap:12px;align-items:flex-end;flex-wrap:wrap}
.bt-ctrl input{width:90px;background:#0d1117;border:1px solid #30363d;border-radius:6px;
               color:#e6edf3;font-size:.88rem;padding:6px 8px;text-align:center}
.badge.tp{background:#0d4429;color:#3fb950}.badge.slr{background:#3d1010;color:#f85149}
.badge.ce{background:#1e2533;color:#8b949e}
</style>
</head>
<body>
<header>
  <h1>⚡ AZLEMA Bot — Paper Trading</h1>
  <div class="head-right">
    <div id="liveprice">
      <span class="lbl2">ETH/USDT ao vivo</span>
      <span class="live-dot" id="lp-dot"></span><span id="lp-val">—</span>
    </div>
    <span id="clock"></span>
  </div>
</header>
<div class="container">
  <div class="tabs">
    <button class="tabbtn active" id="tb-paper" onclick="showTab('paper')">📈 Paper Trading</button>
    <button class="tabbtn" id="tb-backtest" onclick="showTab('backtest')">🧪 Backtesting</button>
  </div>

  <div class="cfg-box">
    <div><label>Stop Loss (pontos)</label><br><input id="cfg-sl" type="number" step="1" min="1"></div>
    <div><label>TP ativação (pontos)</label><br><input id="cfg-tp" type="number" step="1" min="1"></div>
    <div><label>Trailing (pontos)</label><br><input id="cfg-tr" type="number" step="1" min="1"></div>
    <div><label>Gain Limit</label><br><input id="cfg-gl" type="number" step="1" min="1"></div>
    <div><label>Mintick (tick)</label><br><input id="cfg-mt" type="number" step="0.001" min="0.0001"></div>
    <button id="btn-cfg" onclick="saveConfig()">Salvar Config</button>
    <span class="cfg-hint" id="cfg-eq">vale pro bot ao vivo E pro backtest</span>
  </div>

  <div id="tab-paper">
  <div class="top-row">
    <div class="controls">
      <button id="btn-start" onclick="ctrl('start')">▶ Start</button>
      <button id="btn-stop"  onclick="ctrl('stop')" disabled>⏹ Stop</button>
    </div>
    <div class="fee-box">
      <div>
        <label>Fee Entrada %</label><br>
        <input id="fee-entry" type="number" step="0.01" min="0" value="0">
      </div>
      <div>
        <label>Fee Saída %</label><br>
        <input id="fee-exit" type="number" step="0.01" min="0" value="0">
      </div>
      <button id="btn-fee" onclick="saveFees()">Salvar Fees</button>
    </div>
  </div>

  <div class="cards">
    <div class="card"><div class="lbl">Status</div><div class="val" id="c-status">—</div></div>
    <div class="card"><div class="lbl">Sinal</div><div class="val" id="c-signal">—</div></div>
    <div class="card"><div class="lbl">Posição</div><div class="val" id="c-pos">—</div></div>
    <div class="card"><div class="lbl">Saldo (USDT)</div><div class="val" id="c-bal">—</div></div>
    <div class="card"><div class="lbl">PnL Total %</div><div class="val" id="c-pnl">—</div></div>
    <div class="card"><div class="lbl">Não Realizado %</div><div class="val" id="c-unr">—</div></div>
    <div class="card"><div class="lbl">Lucro Máx % (aberta)</div><div class="val" id="c-mfe">—</div></div>
    <div class="card"><div class="lbl">Perda Máx % (aberta)</div><div class="val" id="c-mae">—</div></div>
    <div class="card"><div class="lbl">Winrate</div><div class="val" id="c-win">—</div></div>
    <div class="card"><div class="lbl">Trail Stop</div><div class="val" id="c-trail">—</div></div>
    <div class="card"><div class="lbl">EMA</div><div class="val" id="c-ema">—</div></div>
    <div class="card"><div class="lbl">EC</div><div class="val" id="c-ec">—</div></div>
    <div class="card"><div class="lbl">Período</div><div class="val" id="c-period">—</div></div>
    <div class="card"><div class="lbl">Último Candle</div>
      <div class="val" style="font-size:.8rem" id="c-candle">—</div></div>
    <div class="card"><div class="lbl">Próximo Candle (entrada)</div>
      <div class="val" style="font-size:.85rem" id="c-next">—</div></div>
  </div>

  <div class="st">Histórico de Trades</div>
  <table>
    <thead>
      <tr><th>Hora</th><th>Candle</th><th>Lado</th>
          <th>Preço</th><th>Qtd</th><th>PnL %</th>
          <th>Lucro Máx %</th><th>Perda Máx %</th><th>Saldo</th><th>ID</th></tr>
    </thead>
    <tbody id="tbody">
      <tr><td colspan="10" style="text-align:center;color:#484f58;padding:20px">
        Nenhuma trade ainda</td></tr>
    </tbody>
  </table>
  </div><!-- /tab-paper -->

  <div id="tab-backtest" style="display:none">
    <div class="top-row">
      <div class="bt-ctrl">
        <div><label>Candles de 30m</label><br>
          <input id="bt-candles" type="number" step="10" min="50" max="35040" value="300"></div>
        <span class="cfg-hint">300 ≈ 6 dias · 1440 ≈ 1 mês · 17520 ≈ 1 ano (longo demora minutos)</span>
        <div><label>Fee Entrada %</label><br>
          <input id="bt-fee-entry" type="number" step="0.01" min="0" value="0"></div>
        <div><label>Fee Saída %</label><br>
          <input id="bt-fee-exit" type="number" step="0.01" min="0" value="0"></div>
        <button id="btn-bt" onclick="runBacktest()">▶ Rodar Backtest</button>
        <span id="bt-progress" class="cfg-hint"></span>
      </div>
    </div>
    <div class="cards">
      <div class="card"><div class="lbl">Trades</div><div class="val" id="bt-trades">—</div></div>
      <div class="card"><div class="lbl">Winrate</div><div class="val" id="bt-win">—</div></div>
      <div class="card"><div class="lbl">PnL Total %</div><div class="val" id="bt-pnl">—</div></div>
      <div class="card"><div class="lbl">Lucro Médio %/trade</div><div class="val" id="bt-avg">—</div></div>
      <div class="card"><div class="lbl">Maior Lucro %</div><div class="val" id="bt-max">—</div></div>
      <div class="card"><div class="lbl">Maior Perda %</div><div class="val" id="bt-min">—</div></div>
      <div class="card"><div class="lbl">Saldo Final</div><div class="val" id="bt-bal">—</div></div>
      <div class="card"><div class="lbl">Candles 1m</div><div class="val" id="bt-1m">—</div></div>
    </div>
    <div class="st">Trades do Backtest — replay intra-candle (1m), regras iguais às do bot</div>
    <table>
      <thead><tr><th>Candle</th><th>Lado</th><th>Entrada</th><th>Saída</th>
        <th>Motivo</th><th>PnL %</th><th>Lucro Máx %</th><th>Perda Máx %</th><th>Saldo</th></tr></thead>
      <tbody id="bt-tbody">
        <tr><td colspan="9" style="text-align:center;color:#484f58;padding:20px">
          Rode um backtest para ver as trades</td></tr>
      </tbody>
    </table>
  </div><!-- /tab-backtest -->
</div>
<div id="toast"></div>
<footer>AZLEMA · Paper Trading Interno · 30 m · ETH/USDT · Preços Phemex Live</footer>
<script>
function p(n){return String(n).padStart(2,'0')}
function tick(){const d=new Date();document.getElementById('clock').textContent=
  `${d.getUTCFullYear()}-${p(d.getUTCMonth()+1)}-${p(d.getUTCDate())} `+
  `${p(d.getUTCHours())}:${p(d.getUTCMinutes())}:${p(d.getUTCSeconds())} UTC`;
  // next 30m candle boundary (UTC) + countdown → when the next entry happens
  const nx=new Date(d);
  if(d.getUTCMinutes()<30){nx.setUTCMinutes(30,0,0);}
  else{nx.setUTCHours(d.getUTCHours()+1,0,0,0);}
  const secs=Math.max(0,Math.round((nx-d)/1000)),mm=Math.floor(secs/60),ss=secs%60;
  const el=document.getElementById('c-next');
  if(el)el.textContent=`${p(nx.getUTCHours())}:${p(nx.getUTCMinutes())} UTC (em ${mm}:${p(ss)})`;
}
setInterval(tick,1000);tick();

async function pollPrice(){
  try{
    const r=await fetch('/price');const d=await r.json();
    const v=document.getElementById('lp-val'),dot=document.getElementById('lp-dot');
    if(d.price>0){
      v.textContent='$'+Number(d.price).toLocaleString('en-US',
        {minimumFractionDigits:2,maximumFractionDigits:2});
      const stale=(d.age_s==null||d.age_s>6);
      v.classList.toggle('stale',stale);dot.classList.toggle('stale',stale);
    }
  }catch(e){}
}
setInterval(pollPrice,1000);pollPrice();

// ── Tabs ──
function showTab(t){
  document.getElementById('tab-paper').style.display    = (t==='paper')?'':'none';
  document.getElementById('tab-backtest').style.display = (t==='backtest')?'':'none';
  document.getElementById('tb-paper').classList.toggle('active', t==='paper');
  document.getElementById('tb-backtest').classList.toggle('active', t==='backtest');
}

// ── Config (SL / TP / trailing — vale pro bot e pro backtest) ──
function loadConfig(){
  fetch('/config').then(r=>r.json()).then(d=>{
    document.getElementById('cfg-sl').value=d.sl_pts;
    document.getElementById('cfg-tp').value=d.tp_pts;
    document.getElementById('cfg-tr').value=d.tr_pts;
    document.getElementById('cfg-gl').value=d.gain_limit;
    document.getElementById('cfg-mt').value=d.mintick;
    document.getElementById('cfg-eq').textContent=
      `mintick ${d.mintick} → SL $${d.sl_dist} · TP $${d.tp_dist} · Trail $${d.tr_dist}  (vale pro bot e backtest)`;
  });
}
async function saveConfig(){
  const body={sl_pts:parseFloat(document.getElementById('cfg-sl').value),
              tp_pts:parseFloat(document.getElementById('cfg-tp').value),
              tr_pts:parseFloat(document.getElementById('cfg-tr').value),
              mintick:parseFloat(document.getElementById('cfg-mt').value),
              gain_limit:parseInt(document.getElementById('cfg-gl').value)};
  const r=await fetch('/config',{method:'POST',headers:{'Content-Type':'application/json'},
                                 body:JSON.stringify(body)});
  const d=await r.json();toast(d.message,d.ok);loadConfig();
}

// ── Backtest ──
let btTimer=null;
function btBadge(r){
  if(r==='TRAIL_TP')return '<span class="badge tp">TRAIL_TP</span>';
  if(r==='SL')return '<span class="badge slr">SL</span>';
  return '<span class="badge ce">CANDLE_END</span>';
}
async function runBacktest(){
  const n=parseInt(document.getElementById('bt-candles').value)||300;
  const fe=parseFloat(document.getElementById('bt-fee-entry').value)||0;
  const fx=parseFloat(document.getElementById('bt-fee-exit').value)||0;
  document.getElementById('btn-bt').disabled=true;
  const r=await fetch('/backtest',{method:'POST',headers:{'Content-Type':'application/json'},
                                   body:JSON.stringify({candles:n,fee_entry:fe,fee_exit:fx})});
  const d=await r.json();toast(d.message,d.ok);
  if(!d.ok){document.getElementById('btn-bt').disabled=false;return;}
  if(btTimer)clearInterval(btTimer);
  btTimer=setInterval(pollBacktest,1500);pollBacktest();
}
async function pollBacktest(){
  try{
    const r=await fetch('/backtest-result');const d=await r.json();
    const pr=document.getElementById('bt-progress');
    if(d.running){pr.textContent='⏳ '+(d.progress||'rodando…');return;}
    if(btTimer){clearInterval(btTimer);btTimer=null;}
    document.getElementById('btn-bt').disabled=false;
    if(d.error){pr.textContent='❌ '+d.error;return;}
    if(d.result){pr.textContent='✅ '+(d.progress||'concluído');renderBacktest(d.result);}
  }catch(e){}
}
function renderBacktest(res){
  const s=res.summary;
  document.getElementById('bt-trades').textContent=s.trades;
  document.getElementById('bt-win').textContent=s.winrate+'%';
  const pl=document.getElementById('bt-pnl');
  pl.textContent=s.total_pnl_pct+'%';pl.className='val '+(s.total_pnl_pct>=0?'pos-num':'neg-num');
  const avg=document.getElementById('bt-avg');
  avg.textContent=s.avg_pct+'%';avg.className='val '+(s.avg_pct>=0?'pos-num':'neg-num');
  document.getElementById('bt-max').textContent='+'+s.max_profit+'%';
  document.getElementById('bt-max').className='val pos-num';
  document.getElementById('bt-min').textContent=s.max_loss+'%';
  document.getElementById('bt-min').className='val neg-num';
  document.getElementById('bt-bal').textContent='$'+s.final_balance;
  document.getElementById('bt-1m').textContent=s.intrabar_minutes_loaded;
  const tb=document.getElementById('bt-tbody');
  tb.innerHTML=res.trades.length?res.trades.map(r=>`<tr>
    <td style="font-size:.73rem">${r.candle}</td>
    <td>${badge(r.side)}</td>
    <td>$${r.entry.toFixed(2)}</td>
    <td>$${r.exit.toFixed(2)}</td>
    <td>${btBadge(r.reason)}</td>
    <td class="${r.pnl_pct>=0?'pos-num':'neg-num'}">${r.pnl_pct.toFixed(3)}%</td>
    <td class="pos-num">${r.mfe.toFixed(3)}%</td>
    <td class="neg-num">${r.mae.toFixed(3)}%</td>
    <td>$${r.balance.toFixed(2)}</td></tr>`).join('')
    :'<tr><td colspan="9" style="text-align:center;color:#484f58;padding:20px">Sem trades</td></tr>';
}
loadConfig();

function toast(msg,ok){const e=document.getElementById('toast');
  e.textContent=msg;e.style.background=ok?'#238636':'#da3633';
  e.style.display='block';setTimeout(()=>e.style.display='none',3000)}

function pnlCls(v){const n=parseFloat(v);return n>0?'val pos-num':n<0?'val neg-num':'val'}

function badge(s){
  if(s==='LONG')  return '<span class="badge long">LONG</span>';
  if(s==='SHORT') return '<span class="badge short">SHORT</span>';
  return `<span class="badge close">${s}</span>`}

async function ctrl(a){
  const r=await fetch('/'+a,{method:'POST'});
  const d=await r.json();toast(d.message,d.ok);fetchAll()}

async function saveFees(){
  const entry=parseFloat(document.getElementById('fee-entry').value)||0;
  const exit =parseFloat(document.getElementById('fee-exit').value)||0;
  const r=await fetch('/settings',{method:'POST',
    headers:{'Content-Type':'application/json'},
    body:JSON.stringify({fee_entry:entry,fee_exit:exit})});
  const d=await r.json();toast(d.message,d.ok)}

async function fetchAll(){
  const[sr,hr]=await Promise.all([fetch('/status'),fetch('/history')]);
  const s=await sr.json(),t=await hr.json();

  const sv=document.getElementById('c-status');sv.textContent=s.status;
  sv.className='val '+(s.status==='Running'?'running':
    (s.status.startsWith('Error')||s.status.startsWith('Waiting'))?'err':'stopped');
  const sg=document.getElementById('c-signal');sg.textContent=s.signal;
  sg.className='val '+(s.signal==='LONG'?'long':s.signal==='SHORT'?'short':'');
  const ps=document.getElementById('c-pos');ps.textContent=s.position;
  ps.className='val '+(s.position==='long'?'long':s.position==='short'?'short':'');
  document.getElementById('c-bal').textContent='$'+s.balance;
  const pnl=document.getElementById('c-pnl');
  pnl.textContent=s.total_pnl_pct+'%';pnl.className=pnlCls(s.total_pnl_pct);
  const unr=document.getElementById('c-unr');
  unr.textContent=s.unrealized+'%';unr.className=pnlCls(s.unrealized);
  const mfe=document.getElementById('c-mfe');
  mfe.textContent=s.mfe+'%';mfe.className=pnlCls(s.mfe);
  const mae=document.getElementById('c-mae');
  mae.textContent=s.mae+'%';mae.className=pnlCls(s.mae);
  document.getElementById('c-win').textContent=s.winrate;
  document.getElementById('c-trail').textContent=s.trail_stop;
  ['ema','ec','period'].forEach(k=>document.getElementById('c-'+k).textContent=s[k]);
  document.getElementById('c-candle').textContent=s.last_candle;
  document.getElementById('btn-start').disabled=s.running;
  document.getElementById('btn-stop').disabled=!s.running;

  const tb=document.getElementById('tbody');
  tb.innerHTML=t.length?t.map(r=>`<tr>
    <td>${r.time}</td>
    <td style="font-size:.73rem">${r.candle}</td>
    <td>${badge(r.side)}</td>
    <td>$${Number(r.price).toFixed(2)}</td>
    <td>${r.qty||'—'}</td>
    <td class="${r.pnl_pct&&r.pnl_pct!=='—'?(parseFloat(r.pnl_pct)>=0?'pos-num':'neg-num'):''}">
      ${r.pnl_pct&&r.pnl_pct!=='—'?r.pnl_pct+'%':'—'}</td>
    <td class="${r.mfe&&r.mfe!=='—'?'pos-num':''}">${r.mfe&&r.mfe!=='—'?r.mfe+'%':'—'}</td>
    <td class="${r.mae&&r.mae!=='—'?'neg-num':''}">${r.mae&&r.mae!=='—'?r.mae+'%':'—'}</td>
    <td>$${r.balance}</td>
    <td style="font-size:.68rem;color:#8b949e">${r.order_id}</td></tr>`).join('')
    :'<tr><td colspan="10" style="text-align:center;color:#484f58;padding:20px">Nenhuma trade ainda</td></tr>'}
// Load saved fee values on page open
fetch('/settings').then(r=>r.json()).then(d=>{
  document.getElementById('fee-entry').value=d.fee_entry;
  document.getElementById('fee-exit').value=d.fee_exit;
});
fetchAll();setInterval(fetchAll,10000);
</script>
</body>
</html>"""

@app.route("/")
def index(): return HTML

@app.route("/ping")
@app.route("/health")
def ping(): return jsonify({"ok": True, "time": datetime.utcnow().isoformat()})

@app.route("/status")
def status(): return jsonify({**state, "running": state["running"]})

@app.route("/price")
def price():
    """Fast live-price read for the UI (polled ~every second)."""
    age = round(time_mod.time() - _last_price_ts, 1) if _last_price_ts else None
    return jsonify({
        "price":  _last_good_price,
        "age_s":  age,                 # seconds since last update; small = live
        "ws":     _ws_active,          # websocket delivering ticks?
        "symbol": SYMBOL or "ETH/USDT:USDT",
    })

@app.route("/history")
def history(): return jsonify(trade_history)

@app.route("/settings", methods=["GET"])
def get_settings():
    return jsonify({"fee_entry": fees["entry"], "fee_exit": fees["exit"]})

@app.route("/settings", methods=["POST"])
def settings():
    data = request.get_json() or {}
    if "fee_entry" in data:
        fees["entry"] = float(data["fee_entry"])
    if "fee_exit" in data:
        fees["exit"] = float(data["fee_exit"])
    _save_fees()
    logger.info(f"[FEES] entry={fees['entry']}%  exit={fees['exit']}%")
    return jsonify({"ok": True, "message": f"Fees: entrada {fees['entry']}% / saída {fees['exit']}%"})

@app.route("/config", methods=["GET"])
def get_config():
    # POINT counts (editable) + the resulting $ distances (read-only display)
    return jsonify({"sl_pts": config["sl_pts"], "tp_pts": config["tp_pts"],
                    "tr_pts": config["tr_pts"], "mintick": config.get("mintick", MINTICK_DEFAULT),
                    "gain_limit": config.get("gain_limit", 900),
                    "sl_dist": SL_DIST, "tp_dist": TP_DIST, "tr_dist": TR_DIST})

@app.route("/config", methods=["POST"])
def set_config():
    """Edit the exit distances in POINTS + the strategy Gain Limit."""
    data = request.get_json() or {}
    for k in ("sl_pts", "tp_pts", "tr_pts"):
        if k in data:
            try:
                v = float(data[k])
                if v > 0: config[k] = v
            except Exception:
                return jsonify({"ok": False, "message": f"valor inválido em {k}"})
    if "mintick" in data:
        try:
            mt = float(data["mintick"])
            if mt > 0: config["mintick"] = mt
        except Exception:
            return jsonify({"ok": False, "message": "mintick inválido"})
    if "gain_limit" in data:
        try:
            gl = int(data["gain_limit"])
            if gl >= 1: config["gain_limit"] = gl
        except Exception:
            return jsonify({"ok": False, "message": "Gain Limit inválido"})
    _save_config()
    _apply_config()
    logger.info(f"[CONFIG] SL={config['sl_pts']}pt(${SL_DIST}) "
                f"TP={config['tp_pts']}pt(${TP_DIST}) trail={config['tr_pts']}pt(${TR_DIST})")
    return jsonify({"ok": True,
                    "message": f"Config (mintick {MINTICK}): SL={config['sl_pts']}pt (${SL_DIST}) · "
                               f"TP={config['tp_pts']}pt (${TP_DIST}) · "
                               f"Trail={config['tr_pts']}pt (${TR_DIST}) · "
                               f"GainLimit={config.get('gain_limit', 900)}"})

@app.route("/backtest", methods=["POST"])
def backtest():
    if _backtest["running"]:
        return jsonify({"ok": False, "message": "Backtest já está rodando"})
    data = request.get_json() or {}
    n = 300; fe = 0.0; fx = 0.0
    # No 1000-candle cap — long backtests (e.g. 1 year = 17 520 candles) allowed.
    # 35 040 ≈ 2 years is a sanity ceiling so a typo can't hang the server forever.
    try: n  = max(50, min(35040, int(data.get("candles", 300))))
    except Exception: pass
    try: fe = max(0.0, float(data.get("fee_entry", 0)))
    except Exception: pass
    try: fx = max(0.0, float(data.get("fee_exit", 0)))
    except Exception: pass
    threading.Thread(target=run_backtest, args=(n, fe, fx), daemon=True).start()
    return jsonify({"ok": True, "message": f"Backtest de {n} candles iniciado"})

@app.route("/backtest-result")
def backtest_result():
    return jsonify(_backtest)

@app.route("/debug-ticker")
def debug_ticker():
    """
    Diagnoses the live-price feed AND exit engine. If a position is stuck open,
    open this: it shows whether prices are arriving (price_age_s should be small),
    whether the websocket is delivering ticks, and where the SL/trailing sits vs
    the current price — so you can see exactly why it has or hasn't fired.
    """
    t0 = time_mod.time()
    price = get_live_price()
    elapsed_ms = round((time_mod.time() - t0) * 1000, 1)
    age = round(time_mod.time() - _last_price_ts, 1) if _last_price_ts else None
    return jsonify({
        "live_price":        price,
        "last_good_price":   _last_good_price,
        "price_age_s":       age,            # seconds since any price update; small = healthy
        "ws_available":      _HAS_WS,        # ccxt.pro imported ok
        "ws_active":         _ws_active,     # websocket currently delivering ticks
        "ws_tick_count":     _ws_tick_count, # total ticks seen (should keep rising)
        "realtime_exits":    REALTIME_EXITS, # MUST be true or exits only at candle close
        "consecutive_fails": _ticker_fail_count,
        "fetch_time_ms":     elapsed_ms,
        "symbol":            SYMBOL,
        # ── current position / why it may not have exited ──
        "position":          paper["side"],
        "entry_price":       paper["entry_price"],
        "peak":              paper["peak"],
        "trail_active":      paper["trail_active"],
        "trail_stop":        paper["trail_stop"],
        "sl_dist":           SL_DIST,
        "tp_activation_dist": TP_DIST,
        "trail_offset_dist": TR_DIST,
        "mintick":           MINTICK,      # ONE tick for SL, TP and trailing
    })

@app.route("/start", methods=["POST"])
def start():
    global _strategy_thread
    # Only refuse if the strategy thread is actually ALIVE. If `running` got stuck
    # True but the thread died (e.g. an exception outside the loop), the old check
    # said "Já em execução" forever and nothing could restart it → "não inicia
    # trade nenhuma". Now a dead thread is restarted.
    alive = _strategy_thread is not None and _strategy_thread.is_alive()
    if state["running"] and alive:
        return jsonify({"ok": False, "message": "Já em execução"})
    state["running"] = True
    state["status"]  = "Iniciando…"
    _strategy_thread = threading.Thread(target=strategy_loop, daemon=True)
    _strategy_thread.start()
    return jsonify({"ok": True, "message": "Strategy iniciada"})

@app.route("/stop", methods=["POST"])
def stop():
    if not state["running"]:
        return jsonify({"ok": False, "message": "Já parada"})
    state["running"] = False
    return jsonify({"ok": True, "message": "Strategy parada"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT)