import os, time as time_mod, threading, logging, json
import requests as req
from datetime import datetime, timezone
from flask import Flask, jsonify

import ccxt
import strategy as strat

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CONFIG  (matching Pine Script defaults)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MINTICK       = 0.01          # ETH/USDT tick size on Phemex
FIXED_SL      = 2000          # loss=2000 ticks  → $20.00
FIXED_TP      = 55            # trail_points=55  → $0.55  (trail activation)
TRAIL_OFFSET  = 15            # trail_offset=15  → $0.15  (trail distance from peak)

SL_DIST = round(FIXED_SL   * MINTICK, 4)   # $20.00
TP_DIST = round(FIXED_TP   * MINTICK, 4)   # $0.55
TR_DIST = round(TRAIL_OFFSET * MINTICK, 4) # $0.15

RISK            = 0.01
MAX_LOTS        = 100
LEVERAGE        = 1
CANDLES         = 200
PAPER_START_BAL = 10_000.0
PORT            = int(os.environ.get("PORT", 10000))

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# EXCHANGE  (Phemex live — OHLCV only, no auth needed)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
live_ex = ccxt.phemex({"options": {"defaultType": "swap"}, "enableRateLimit": True})
SYMBOL  = None

def init_markets():
    global SYMBOL
    mkts = live_ex.load_markets()
    for sym in ["ETH/USDT:USDT", "ETH/USD:ETH"]:
        if sym in mkts:
            SYMBOL = sym
            break
    if not SYMBOL:
        for k, v in mkts.items():
            if "ETH" in k and v.get("type") in ("swap", "future"):
                SYMBOL = k; break
    if not SYMBOL:
        raise RuntimeError("No ETH perpetual found on Phemex")
    logger.info(f"[INIT] symbol={SYMBOL}")

def get_live_price() -> float:
    """Current price via ticker — accurate real-time, not stale OHLCV close."""
    try:
        return float(live_ex.fetch_ticker(SYMBOL or "ETH/USDT:USDT")["last"])
    except Exception as e:
        logger.warning(f"[TICKER] {e}")
        return 0.0


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
    "side":         "none",   # "long" | "short" | "none"
    "entry_price":  0.0,
    "qty":          0.0,
    "peak":         0.0,      # highest (long) or lowest (short) price seen
    "trail_active": False,
    "trail_stop":   0.0,
    "unrealized":   0.0,
    "total_pnl":    0.0,
}

def _unrealized(cur_price):
    if paper["side"] == "none": return 0.0
    d = cur_price - paper["entry_price"]
    return d * paper["qty"] if paper["side"] == "long" else -d * paper["qty"]

def _open_position(side, qty, price):
    paper["side"]         = "long" if side == "LONG" else "short"
    paper["entry_price"]  = price
    paper["qty"]          = qty
    paper["peak"]         = price
    paper["trail_active"] = False
    paper["trail_stop"]   = 0.0
    paper["unrealized"]   = 0.0
    order_id = f"SIM-{int(time_mod.time())}"
    logger.info(f"[PAPER] OPEN {side}  qty={qty}  @ {price:.2f}  id={order_id}")
    return {"id": order_id}

def _close_position(price, reason):
    entry = paper["entry_price"]
    qty   = paper["qty"]
    pnl   = (price - entry) * qty if paper["side"] == "long" else (entry - price) * qty
    paper["balance"]   += pnl
    paper["total_pnl"] += pnl
    paper["unrealized"] = 0.0
    logger.info(f"[PAPER] CLOSE {paper['side'].upper()} ({reason})  @ {price:.2f}  "
                f"pnl={pnl:+.4f}  bal={paper['balance']:.2f}")
    paper["side"]         = "none"
    paper["entry_price"]  = 0.0
    paper["qty"]          = 0.0
    paper["peak"]         = 0.0
    paper["trail_active"] = False
    paper["trail_stop"]   = 0.0
    return pnl

def _calc_qty(price, balance):
    sl_usdt    = FIXED_SL * MINTICK
    risk_usdt  = RISK * balance
    qty        = (risk_usdt / sl_usdt) if sl_usdt else 0.01
    max_by_bal = (balance * 0.95 / price) if price else qty
    return round(min(qty, max_by_bal, MAX_LOTS), 4)

# ── SL / Trailing TP check ────────────────────────────────────────────────────
# Matches Pine Script:  loss=fixedSL  trail_points=fixedTP  trail_offset=15
# Uses current market price — called every 15 s for real-time exits.
def _check_sl_tp(cur_price: float):
    """Returns (exit, exit_price, reason) or (False, 0, '')."""
    if paper["side"] == "none":
        return False, 0.0, ""

    entry = paper["entry_price"]

    if paper["side"] == "long":
        sl_price = round(entry - SL_DIST, 4)
        # Update running peak
        if cur_price > paper["peak"]:
            paper["peak"] = cur_price
        # Activate / advance trailing stop when profit >= TP_DIST ($0.55)
        peak_profit = paper["peak"] - entry
        if peak_profit >= TP_DIST:
            new_trail = round(paper["peak"] - TR_DIST, 4)
            if not paper["trail_active"] or new_trail > paper["trail_stop"]:
                paper["trail_active"] = True
                paper["trail_stop"]   = new_trail
        # Trailing stop hit?
        if paper["trail_active"] and cur_price <= paper["trail_stop"]:
            return True, paper["trail_stop"], "TRAIL_TP"
        # Hard SL hit?
        if cur_price <= sl_price:
            return True, sl_price, "SL"

    else:  # short
        sl_price = round(entry + SL_DIST, 4)
        if cur_price < paper["peak"] or paper["peak"] == 0:
            paper["peak"] = cur_price
        peak_profit = entry - paper["peak"]
        if peak_profit >= TP_DIST:
            new_trail = round(paper["peak"] + TR_DIST, 4)
            if not paper["trail_active"] or new_trail < paper["trail_stop"]:
                paper["trail_active"] = True
                paper["trail_stop"]   = new_trail
        if paper["trail_active"] and cur_price >= paper["trail_stop"]:
            return True, paper["trail_stop"], "TRAIL_TP"
        if cur_price >= sl_price:
            return True, sl_price, "SL"

    return False, 0.0, ""

def _check_sl_tp_candle(candle_h: float, candle_l: float):
    """
    SL/TP check using a CLOSED candle's H/L.
    Called once per new closed candle — matches TradingView broker emulator:
    peak updated with candle high (long) or low (short),
    then checks if candle low/high breached trail_stop or SL.
    """
    if paper["side"] == "none":
        return False, 0.0, ""

    entry = paper["entry_price"]

    if paper["side"] == "long":
        # Advance peak using candle HIGH (TradingView sees full bar range)
        if candle_h > paper["peak"]:
            paper["peak"] = candle_h
        peak_profit = paper["peak"] - entry
        if peak_profit >= TP_DIST:
            new_trail = round(paper["peak"] - TR_DIST, 4)
            if not paper["trail_active"] or new_trail > paper["trail_stop"]:
                paper["trail_active"] = True
                paper["trail_stop"]   = new_trail
        # Candle LOW crossed trail stop?
        if paper["trail_active"] and candle_l <= paper["trail_stop"]:
            return True, paper["trail_stop"], "TRAIL_TP"
        # Candle LOW crossed hard SL?
        sl_price = round(entry - SL_DIST, 4)
        if candle_l <= sl_price:
            return True, sl_price, "SL"

    else:  # short
        # Advance peak (lowest price) using candle LOW
        if candle_l < paper["peak"] or paper["peak"] == 0:
            paper["peak"] = candle_l
        peak_profit = entry - paper["peak"]
        if peak_profit >= TP_DIST:
            new_trail = round(paper["peak"] + TR_DIST, 4)
            if not paper["trail_active"] or new_trail < paper["trail_stop"]:
                paper["trail_active"] = True
                paper["trail_stop"]   = new_trail
        if paper["trail_active"] and candle_h >= paper["trail_stop"]:
            return True, paper["trail_stop"], "TRAIL_TP"
        sl_price = round(entry + SL_DIST, 4)
        if candle_h >= sl_price:
            return True, sl_price, "SL"

    return False, 0.0, ""


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TRADE HISTORY  (persisted to disk)
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

trade_history    = _load_history()
_strategy_thread = None
logger.info(f"[HISTORY] {len(trade_history)} trades loaded from disk")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SHARED STATE  (served to dashboard)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
state = {
    "running":     False,
    "status":      "Stopped",
    "signal":      "—",
    "position":    "None",
    "ema":         "—",
    "ec":          "—",
    "period":      "—",
    "last_candle": "—",
    "updated":     "—",
    "balance":     f"{PAPER_START_BAL:.2f}",
    "total_pnl":   "0.00",
    "unrealized":  "0.00",
    "trail_stop":  "—",
}

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STRATEGY LOOP
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def _record_exit(side_str, price, reason, pnl, candle_str):
    trade_history.insert(0, {
        "time":     datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "candle":   candle_str,
        "side":     f"CLOSE {side_str} ({reason})",
        "price":    round(price, 4),
        "qty":      0,
        "order_id": f"EXIT-{reason}",
        "balance":  f"{paper['balance']:.2f}",
        "pnl":      f"{pnl:+.4f}",
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
        "pnl":      "—",
    })
    _save_history()

def strategy_loop():
    global state, trade_history

    try:
        init_markets()
    except Exception as e:
        logger.error(f"[INIT] {e}")
        state["status"] = f"Error: {e}"
        state["running"] = False
        return

    # ── Startup: mark current candle as seen, enter on the NEXT close ──────
    # prev_signal=None ensures ANY signal on next candle triggers entry
    prev_signal     = None
    last_candle_ts  = None
    last_candle_str = "—"

    init = fetch_candles()
    if init and len(init) >= 70:
        last_candle_ts = init[-2][0]   # mark last closed candle as already seen
        logger.info(f"[SYNC] startup ready — will enter on next candle close")
    else:
        logger.warning("[SYNC] not enough candles for startup")

    while state["running"]:
        try:
            ohlcv = fetch_candles()
            if not ohlcv or len(ohlcv) < 70:
                state["status"] = f"Waiting ({len(ohlcv)} candles)"
                time_mod.sleep(20)
                continue

            # Auto-detect whether Phemex includes the forming candle.
            # If last bar's timestamp is inside the current 30-min window
            # → forming candle included → use ohlcv[-2] as last closed.
            # Otherwise → all closed → use ohlcv[-1] as last closed.
            _TF_MS        = 30 * 60 * 1000
            _now_ms       = int(time_mod.time() * 1000)
            _win_start    = (_now_ms // _TF_MS) * _TF_MS
            _has_forming  = (ohlcv[-1][0] >= _win_start)
            last_closed   = ohlcv[-2] if _has_forming else ohlcv[-1]
            _sig_closes   = ohlcv[:-1] if _has_forming else ohlcv

            candle_ts   = last_closed[0]
            candle_c    = float(last_closed[4])
            last_candle_str = datetime.fromtimestamp(
                candle_ts / 1000, tz=timezone.utc
            ).strftime("%Y-%m-%d %H:%M UTC")

            # Current live price via ticker (accurate — not stale OHLCV close)
            live_price = get_live_price()
            if live_price == 0.0:
                live_price = float(ohlcv[-1][4])   # fallback
            paper["unrealized"] = _unrealized(live_price)

            # ── REAL-TIME SL/TP CHECK (every 15 s, any price) ─────────────────
            # Uses live_price — no stale OHLC contamination
            ex, ex_px, ex_rsn = _check_sl_tp(live_price)
            if ex:
                ex_side = paper["side"]
                ex_pnl  = _close_position(ex_px, ex_rsn)
                _record_exit(ex_side.upper(), ex_px, ex_rsn, ex_pnl, last_candle_str)
                prev_signal = None   # re-entry allowed on next candle signal
                state.update({
                    "position":  "none",
                    "unrealized": "0.00",
                    "balance":    f"{paper['balance']:.2f}",
                    "total_pnl":  f"{paper['total_pnl']:+.2f}",
                    "trail_stop": "—",
                })

            # ── Every-15-s state refresh ───────────────────────────────────────
            trail_disp = (f"${paper['trail_stop']:.2f}" if paper["trail_active"]
                          else "armed" if paper["side"] != "none" else "—")
            state.update({
                "position":   paper["side"],
                "unrealized": f"{paper['unrealized']:+.2f}",
                "balance":    f"{paper['balance']:.2f}",
                "total_pnl":  f"{paper['total_pnl']:+.2f}",
                "trail_stop": trail_disp,
                "status":     "Running",
            })

            # ── NEW CANDLE processing (every 30 min) ──────────────────────────
            if candle_ts == last_candle_ts:
                time_mod.sleep(15)
                continue

            last_candle_ts = candle_ts

            # ── Candle-based SL/TP (H/L of closed candle — matches TradingView)
            candle_h = float(last_closed[2])
            candle_l = float(last_closed[3])
            cl_ex, cl_px, cl_rsn = _check_sl_tp_candle(candle_h, candle_l)
            if cl_ex and paper["side"] != "none":
                cl_side = paper["side"]
                cl_pnl  = _close_position(cl_px, cl_rsn)
                _record_exit(cl_side.upper(), cl_px, cl_rsn, cl_pnl, last_candle_str)
                prev_signal = None
                state.update({"position": "none", "unrealized": "0.00",
                               "balance": f"{paper['balance']:.2f}",
                               "total_pnl": f"{paper['total_pnl']:+.2f}",
                               "trail_stop": "—"})

            # Calculate signal on closed candles
            closes = [c[4] for c in _sig_closes]
            result = strat.calculate(closes)
            signal = result["signal"]

            state.update({
                "signal":      signal or "—",
                "ema":         f"{result['ema']:.4f}" if result["ema"] else "—",
                "ec":          f"{result['ec']:.4f}"  if result["ec"]  else "—",
                "period":      str(result["period"]),
                "last_candle": last_candle_str,
                "updated":     datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            })

            # Enter only when signal CHANGES (Pine crossover behaviour)
            # prev_signal=None forces re-entry after SL/TP exit
            if signal and signal != prev_signal:
                cur_pos = paper["side"]
                qty     = _calc_qty(live_price if live_price > 0 else candle_c, paper["balance"])

                # Entry at current market price (≈ next bar open)
                entry_px = live_price if live_price > 0 else candle_c

                if signal == "LONG" and cur_pos != "long":
                    if cur_pos == "short":
                        pnl = _close_position(entry_px, "SIGNAL_REVERSE")
                        _record_exit("SHORT", entry_px, "SIGNAL_REVERSE",
                                     pnl, last_candle_str)
                    order = _open_position("LONG", qty, entry_px)
                    _record_entry("LONG", entry_px, qty,
                                  last_candle_str, order["id"])

                elif signal == "SHORT" and cur_pos != "short":
                    if cur_pos == "long":
                        pnl = _close_position(entry_px, "SIGNAL_REVERSE")
                        _record_exit("LONG", entry_px, "SIGNAL_REVERSE",
                                     pnl, last_candle_str)
                    order = _open_position("SHORT", qty, entry_px)
                    _record_entry("SHORT", entry_px, qty,
                                  last_candle_str, order["id"])

            prev_signal = signal

        except Exception as e:
            logger.error(f"[LOOP] {e}")
            state["status"] = f"Error: {str(e)[:80]}"
            time_mod.sleep(20)

    state["status"] = "Stopped"

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# KEEPALIVE  (3 internal signals)
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
# FLASK APP
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
header{background:#161b22;border-bottom:1px solid #30363d;padding:18px 32px;
       display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:10px}
header h1{font-size:1.2rem;color:#58a6ff}
#clock{font-size:.8rem;color:#8b949e}
.container{max-width:1100px;margin:26px auto;padding:0 20px}
.controls{display:flex;gap:12px;margin-bottom:22px}
button{padding:10px 28px;border:none;border-radius:8px;font-size:.92rem;font-weight:600;cursor:pointer}
#btn-start{background:#238636;color:#fff}#btn-start:hover:not(:disabled){background:#2ea043}
#btn-stop{background:#da3633;color:#fff}#btn-stop:hover:not(:disabled){background:#f85149}
button:disabled{opacity:.35;cursor:not-allowed}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(148px,1fr));gap:12px;margin-bottom:24px}
.card{background:#161b22;border:1px solid #30363d;border-radius:10px;padding:15px 17px}
.lbl{font-size:.68rem;color:#8b949e;text-transform:uppercase;letter-spacing:.06em;margin-bottom:4px}
.val{font-size:1.1rem;font-weight:600}
.long{color:#3fb950}.short{color:#f85149}
.running{color:#3fb950}.stopped{color:#8b949e}
.err{color:#e3b341;font-size:.76rem;word-break:break-all}
.pos-num{color:#3fb950}.neg-num{color:#f85149}
.st{font-size:.78rem;color:#8b949e;text-transform:uppercase;letter-spacing:.06em;margin-bottom:9px}
table{width:100%;border-collapse:collapse;background:#161b22;border:1px solid #30363d;
      border-radius:10px;overflow:hidden}
th{background:#21262d;color:#8b949e;font-size:.73rem;text-transform:uppercase;
   padding:10px 13px;text-align:left}
td{padding:9px 13px;font-size:.84rem;border-top:1px solid #21262d}
tr:hover td{background:#1c2128}
.badge{display:inline-block;padding:2px 9px;border-radius:20px;font-size:.74rem;font-weight:700}
.badge.long{background:#0d4429;color:#3fb950}
.badge.short{background:#3d1010;color:#f85149}
.badge.close{background:#1e2533;color:#8b949e}
#toast{position:fixed;bottom:20px;right:20px;padding:10px 18px;border-radius:8px;
       font-size:.86rem;display:none;z-index:99;color:#fff}
footer{text-align:center;color:#484f58;font-size:.73rem;margin:32px 0 16px}
</style>
</head>
<body>
<header>
  <h1>⚡ AZLEMA Bot — Paper Trading</h1>
  <span id="clock"></span>
</header>
<div class="container">
  <div class="controls">
    <button id="btn-start" onclick="ctrl('start')">▶ Start Strategy</button>
    <button id="btn-stop"  onclick="ctrl('stop')" disabled>⏹ Stop Strategy</button>
  </div>
  <div class="cards">
    <div class="card"><div class="lbl">Status</div><div class="val" id="c-status">—</div></div>
    <div class="card"><div class="lbl">Signal</div><div class="val" id="c-signal">—</div></div>
    <div class="card"><div class="lbl">Position</div><div class="val" id="c-pos">—</div></div>
    <div class="card"><div class="lbl">Balance (USDT)</div><div class="val" id="c-bal">—</div></div>
    <div class="card"><div class="lbl">Total PnL</div><div class="val" id="c-pnl">—</div></div>
    <div class="card"><div class="lbl">Unrealized</div><div class="val" id="c-unr">—</div></div>
    <div class="card"><div class="lbl">Trail Stop</div><div class="val" id="c-trail">—</div></div>
    <div class="card"><div class="lbl">EMA</div><div class="val" id="c-ema">—</div></div>
    <div class="card"><div class="lbl">EC</div><div class="val" id="c-ec">—</div></div>
    <div class="card"><div class="lbl">Period</div><div class="val" id="c-period">—</div></div>
    <div class="card"><div class="lbl">Last Candle</div>
      <div class="val" style="font-size:.82rem" id="c-candle">—</div></div>
    <div class="card"><div class="lbl">Updated</div>
      <div class="val" style="font-size:.76rem" id="c-upd">—</div></div>
  </div>
  <div class="st">Trade History</div>
  <table>
    <thead>
      <tr><th>Time</th><th>Candle</th><th>Side</th>
          <th>Price</th><th>Qty</th><th>PnL</th><th>Balance</th><th>Order ID</th></tr>
    </thead>
    <tbody id="tbody">
      <tr><td colspan="8" style="text-align:center;color:#484f58;padding:20px">
        No trades yet</td></tr>
    </tbody>
  </table>
</div>
<div id="toast"></div>
<footer>AZLEMA · Paper Trading · 30 m · ETH/USDT · Phemex Live Prices</footer>
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
  if(s==='LONG') return '<span class="badge long">LONG</span>';
  if(s==='SHORT') return '<span class="badge short">SHORT</span>';
  return `<span class="badge close">${s}</span>`}
async function ctrl(a){const r=await fetch('/'+a,{method:'POST'});
  const d=await r.json();toast(d.message,d.ok);fetchAll()}
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
  pnl.textContent='$'+s.total_pnl;pnl.className=pnlCls(s.total_pnl);
  const unr=document.getElementById('c-unr');
  unr.textContent='$'+s.unrealized;unr.className=pnlCls(s.unrealized);
  document.getElementById('c-trail').textContent=s.trail_stop;
  ['ema','ec','period'].forEach(k=>document.getElementById('c-'+k).textContent=s[k]);
  document.getElementById('c-candle').textContent=s.last_candle;
  document.getElementById('c-upd').textContent=s.updated;
  document.getElementById('btn-start').disabled=s.running;
  document.getElementById('btn-stop').disabled=!s.running;
  const tb=document.getElementById('tbody');
  tb.innerHTML=t.length?t.map(r=>`<tr>
    <td>${r.time}</td><td style="font-size:.75rem">${r.candle}</td>
    <td>${badge(r.side)}</td>
    <td>$${Number(r.price).toFixed(2)}</td>
    <td>${r.qty||'—'}</td>
    <td class="${r.pnl&&r.pnl!=='—'?(parseFloat(r.pnl)>=0?'pos-num':'neg-num'):''}">
      ${r.pnl&&r.pnl!=='—'?'$'+r.pnl:'—'}</td>
    <td>$${r.balance}</td>
    <td style="font-size:.7rem;color:#8b949e">${r.order_id}</td></tr>`).join('')
    :'<tr><td colspan="8" style="text-align:center;color:#484f58;padding:20px">'
     +'No trades yet</td></tr>'}
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

@app.route("/start", methods=["POST"])
def start():
    global _strategy_thread
    if state["running"]:
        return jsonify({"ok": False, "message": "Already running"})
    state["running"] = True
    state["status"]  = "Starting…"
    _strategy_thread = threading.Thread(target=strategy_loop, daemon=True)
    _strategy_thread.start()
    return jsonify({"ok": True, "message": "Strategy started"})

@app.route("/stop", methods=["POST"])
def stop():
    if not state["running"]:
        return jsonify({"ok": False, "message": "Already stopped"})
    state["running"] = False
    return jsonify({"ok": True, "message": "Strategy stopped"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT)
