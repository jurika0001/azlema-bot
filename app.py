import os, time as time_mod, threading, logging, json
import requests as req
from datetime import datetime, timezone
from flask import Flask, jsonify, request

import ccxt
import strategy as strat

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CONFIG  (matching Pine Script defaults)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MINTICK      = 0.01
FIXED_SL     = 2000    # loss=2000 ticks → $20.00
FIXED_TP     = 55      # trail_points=55 → $0.55 activation
TRAIL_OFFSET = 15      # trail_offset=15 → $0.15 trail distance

SL_DIST = round(FIXED_SL    * MINTICK, 4)   # $20.00
TP_DIST = round(FIXED_TP    * MINTICK, 4)   # $0.55
TR_DIST = round(TRAIL_OFFSET * MINTICK, 4)  # $0.15

RISK            = 0.01
MAX_LOTS        = 100
CANDLES         = 200
PAPER_START_BAL = 10_000.0
PORT            = int(os.environ.get("PORT", 10000))

# Configurable trade fees (persisted to disk)
FEES_FILE = "fees.json"
def _load_fees():
    try:
        with open(FEES_FILE) as f: return json.load(f)
    except Exception: return {"entry": 0.05, "exit": 0.05}
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
SYMBOL = None

_last_good_price   = 0.0
_ticker_fail_count = 0
_MAX_TICK_JUMP_PCT = 1.5   # ETH/USDT should not move >1.5% in a single second

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
    global _last_good_price, _ticker_fail_count
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
                logger.warning(f"[TICKER] rejected implausible jump "
                                f"{_last_good_price:.2f} → {price:.2f} "
                                f"({jump_pct:.2f}%) — keeping last good price")
                return _last_good_price

        _last_good_price   = price
        _ticker_fail_count  = 0
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
    sl_usdt    = FIXED_SL * MINTICK
    risk_usdt  = RISK * balance
    qty        = (risk_usdt / sl_usdt) if sl_usdt else 0.01
    max_by_bal = (balance * 0.95 / price) if price else qty
    return round(min(qty, max_by_bal, MAX_LOTS), 4)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SL / TRAILING TP  — two variants
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def _check_sl_tp(cur_price: float):
    """
    Real-time check via ticker (every 15 s).
    NEVER updates peak — peak is only updated by closed candle H/L.
    Only checks whether current price has crossed the pre-computed trail_stop or SL.
    Matches TradingView calc_on_every_tick=false: peak from bars, check from ticks.
    """
    if paper["side"] == "none":
        return False, 0.0, ""
    entry = paper["entry_price"]
    if paper["side"] == "long":
        if paper["trail_active"] and cur_price <= paper["trail_stop"]:
            return True, paper["trail_stop"], "TRAIL_TP"
        sl = round(entry - SL_DIST, 4)
        if cur_price <= sl:
            return True, sl, "SL"
    else:
        if paper["trail_active"] and cur_price >= paper["trail_stop"]:
            return True, paper["trail_stop"], "TRAIL_TP"
        sl = round(entry + SL_DIST, 4)
        if cur_price >= sl:
            return True, sl, "SL"
    return False, 0.0, ""

def _check_sl_tp_candle(h: float, l: float):
    """
    Closed candle H/L check — matches TradingView broker emulator exactly.
    1. Updates peak using candle HIGH (long) or LOW (short).
    2. Activates/updates trail_stop when peak profit >= TP_DIST.
    3. Checks if candle range crossed trail_stop or SL.
    TradingView order of operations for a LONG bar:
      checks LOW for SL first, then HIGH for trail activation.
    For SHORT: checks HIGH for SL first, then LOW for trail activation.
    """
    if paper["side"] == "none":
        return False, 0.0, ""
    entry = paper["entry_price"]

    if paper["side"] == "long":
        sl = round(entry - SL_DIST, 4)
        # Check SL first (TradingView checks worst case first)
        if l <= sl:
            return True, sl, "SL"
        # Update peak with this bar's HIGH
        if h > paper["peak"]:
            paper["peak"] = h
        pp = paper["peak"] - entry
        # Activate / advance trail stop
        if pp >= TP_DIST:
            nt = round(paper["peak"] - TR_DIST, 4)
            if not paper["trail_active"] or nt > paper["trail_stop"]:
                paper["trail_active"] = True
                paper["trail_stop"]   = nt
        # Check trail stop
        if paper["trail_active"] and l <= paper["trail_stop"]:
            return True, paper["trail_stop"], "TRAIL_TP"

    else:  # short
        sl = round(entry + SL_DIST, 4)
        # Check SL first
        if h >= sl:
            return True, sl, "SL"
        # Update peak with this bar's LOW
        if l < paper["peak"] or paper["peak"] == 0:
            paper["peak"] = l
        pp = entry - paper["peak"]
        # Activate / advance trail stop
        if pp >= TP_DIST:
            nt = round(paper["peak"] + TR_DIST, 4)
            if not paper["trail_active"] or nt < paper["trail_stop"]:
                paper["trail_active"] = True
                paper["trail_stop"]   = nt
        # Check trail stop
        if paper["trail_active"] and h >= paper["trail_stop"]:
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
    trade_history.insert(0, {
        "time":     datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "candle":   candle_str,
        "side":     f"CLOSE {side_str} ({reason})",
        "price":    round(price, 4),
        "qty":      0,
        "order_id": f"EXIT-{reason}",
        "balance":  f"{paper['balance']:.2f}",
        "pnl_pct":  f"{pnl_pct_val:+.2f}",
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
    "trail_stop":  "—",
    "winrate":     "—",
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
def strategy_loop():
    global state, trade_history

    try:
        init_markets()
    except Exception as e:
        logger.error(f"[INIT] {e}")
        state["status"] = f"Error: {e}"
        state["running"] = False
        return

    prev_signal      = None
    last_candle_ts   = None
    last_candle_str  = "—"
    skip_entry       = False
    last_ohlcv_check = 0.0   # throttle ohlcv fetching to every 15 s

    # Startup sync
    init_ohlcv = fetch_candles()
    if init_ohlcv and len(init_ohlcv) >= 70:
        _lc, _ = _resolve_candles(init_ohlcv)
        last_candle_ts = _lc[0]
        logger.info(f"[SYNC] locked to candle "
                    f"{datetime.fromtimestamp(last_candle_ts/1000,tz=timezone.utc)}"
                    f" — waiting for next close")

    while state["running"]:
        try:
            # ── 1. REAL-TIME SL/TP CHECK  (every ~1 s, ticker only) ──────────
            live_price = get_live_price()
            if live_price > 0:
                paper["unrealized"] = _unrealized_pct(live_price)
                skip_entry = False

                ex, ex_px, ex_rsn = _check_sl_tp(live_price)
                if ex:
                    ex_side = paper["side"]
                    raw, pct = _close_position(ex_px, ex_rsn)
                    _record_exit(ex_side.upper(), ex_px, ex_rsn,
                                 raw, pct, last_candle_str)
                    skip_entry = True
                    state.update({
                        "position": "none", "unrealized": "0.00",
                        "balance":  f"{paper['balance']:.2f}",
                        "total_pnl_pct": f"{paper['total_pnl_pct']:+.2f}",
                        "trail_stop": "—",
                        "winrate":  _winrate(),
                    })
                    logger.info(f"[EXIT] {ex_rsn} @ {ex_px:.2f}  pnl={pct:+.2f}%")

                trail_disp = (f"${paper['trail_stop']:.2f}" if paper["trail_active"]
                              else ("armed" if paper["side"] != "none" else "—"))
                state.update({
                    "unrealized":    f"{paper['unrealized']:+.2f}",
                    "balance":       f"{paper['balance']:.2f}",
                    "total_pnl_pct": f"{paper['total_pnl_pct']:+.2f}",
                    "trail_stop":    trail_disp,
                    "position":      paper["side"],
                    "status":        "Running",
                })

            # ── 2. CANDLE CHECK  (every 15 s, full OHLCV) ────────────────────
            now = time_mod.time()
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
                        last_candle_ts = candle_ts
                        skip_entry = False

                        # Candle H/L catches any peak missed between 1-s polls
                        cl_ex, cl_px, cl_rsn = _check_sl_tp_candle(
                            float(last_closed[2]), float(last_closed[3]))
                        if cl_ex and paper["side"] != "none":
                            cl_side = paper["side"]
                            raw, pct = _close_position(cl_px, cl_rsn)
                            _record_exit(cl_side.upper(), cl_px, cl_rsn,
                                         raw, pct, last_candle_str)
                            skip_entry = True
                            state.update({
                                "position": "none", "unrealized": "0.00",
                                "balance":  f"{paper['balance']:.2f}",
                                "total_pnl_pct": f"{paper['total_pnl_pct']:+.2f}",
                                "trail_stop": "—",
                                "winrate":  _winrate(),
                            })

                        # Signal
                        closes = [c[4] for c in sig_closes]
                        result = strat.calculate(closes)
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

                        # Entry — never on the same candle as an exit
                        if signal and signal != prev_signal and not skip_entry:
                            entry_px = live_price if live_price > 0 else float(last_closed[4])
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

                        prev_signal = None if skip_entry else signal

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
</style>
</head>
<body>
<header>
  <h1>⚡ AZLEMA Bot — Paper Trading</h1>
  <span id="clock"></span>
</header>
<div class="container">
  <div class="top-row">
    <div class="controls">
      <button id="btn-start" onclick="ctrl('start')">▶ Start</button>
      <button id="btn-stop"  onclick="ctrl('stop')" disabled>⏹ Stop</button>
    </div>
    <div class="fee-box">
      <div>
        <label>Fee Entrada %</label><br>
        <input id="fee-entry" type="number" step="0.01" min="0" value="0.05">
      </div>
      <div>
        <label>Fee Saída %</label><br>
        <input id="fee-exit" type="number" step="0.01" min="0" value="0.05">
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
    <div class="card"><div class="lbl">Winrate</div><div class="val" id="c-win">—</div></div>
    <div class="card"><div class="lbl">Trail Stop</div><div class="val" id="c-trail">—</div></div>
    <div class="card"><div class="lbl">EMA</div><div class="val" id="c-ema">—</div></div>
    <div class="card"><div class="lbl">EC</div><div class="val" id="c-ec">—</div></div>
    <div class="card"><div class="lbl">Período</div><div class="val" id="c-period">—</div></div>
    <div class="card"><div class="lbl">Último Candle</div>
      <div class="val" style="font-size:.8rem" id="c-candle">—</div></div>
  </div>

  <div class="st">Histórico de Trades</div>
  <table>
    <thead>
      <tr><th>Hora</th><th>Candle</th><th>Lado</th>
          <th>Preço</th><th>Qtd</th><th>PnL %</th><th>Saldo</th><th>ID</th></tr>
    </thead>
    <tbody id="tbody">
      <tr><td colspan="8" style="text-align:center;color:#484f58;padding:20px">
        Nenhuma trade ainda</td></tr>
    </tbody>
  </table>
</div>
<div id="toast"></div>
<footer>AZLEMA · Paper Trading Interno · 30 m · ETH/USDT · Preços Phemex Live</footer>
<script>
function p(n){return String(n).padStart(2,'0')}
function tick(){const d=new Date();document.getElementById('clock').textContent=
  `${d.getUTCFullYear()}-${p(d.getUTCMonth()+1)}-${p(d.getUTCDate())} `+
  `${p(d.getUTCHours())}:${p(d.getUTCMinutes())}:${p(d.getUTCSeconds())} UTC`}
setInterval(tick,1000);tick();

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
    <td>$${r.balance}</td>
    <td style="font-size:.68rem;color:#8b949e">${r.order_id}</td></tr>`).join('')
    :'<tr><td colspan="8" style="text-align:center;color:#484f58;padding:20px">Nenhuma trade ainda</td></tr>'}
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

@app.route("/debug-ticker")
def debug_ticker():
    """Diagnoses the live-price feed — visit after deploy to verify it works."""
    t0 = time_mod.time()
    price = get_live_price()
    elapsed_ms = round((time_mod.time() - t0) * 1000, 1)
    return jsonify({
        "live_price":       price,
        "last_good_price":  _last_good_price,
        "consecutive_fails": _ticker_fail_count,
        "fetch_time_ms":    elapsed_ms,
        "symbol":           SYMBOL,
    })

@app.route("/start", methods=["POST"])
def start():
    global _strategy_thread
    if state["running"]:
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
