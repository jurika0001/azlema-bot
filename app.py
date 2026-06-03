import os, time as time_mod, threading, logging, json
import requests as req
from datetime import datetime, timezone
from flask import Flask, jsonify

import ccxt
import strategy as strat

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ── Config — matches Pine Script defaults ──────────────────────────────────────
RISK            = 0.01      # 1 % risk per trade
FIXED_SL        = 2000      # stop-loss ticks
FIXED_TP        = 55        # trailing-stop activation ticks
TRAIL_OFFSET    = 15        # trailing-stop offset ticks
MINTICK         = 0.01      # ETH/USDT price tick on Phemex
MAX_LOTS        = 100
LEVERAGE        = 1
CANDLES         = 200
PAPER_START_BAL = 10_000.0  # virtual USDT balance
PORT            = int(os.environ.get("PORT", 10000))

SL_DIST = FIXED_SL  * MINTICK   # $20.00
TP_DIST = FIXED_TP  * MINTICK   # $0.55 — trail activates after this profit
TR_DIST = TRAIL_OFFSET * MINTICK # $0.15 — trail distance from peak

# ── live_ex : Phemex live, no auth — OHLCV only ───────────────────────────────
live_ex = ccxt.phemex({
    "options": {"defaultType": "swap"},
    "enableRateLimit": True,
})

SYMBOL = None

def init_markets():
    global SYMBOL
    mkts = live_ex.load_markets()
    logger.info(f"[INIT] {len(mkts)} live markets loaded")
    for sym in ["ETH/USDT:USDT", "ETH/USD:ETH"]:
        if sym in mkts:
            SYMBOL = sym
            break
    if not SYMBOL:
        for k, v in mkts.items():
            if "ETH" in k and v.get("type") in ("swap", "future"):
                SYMBOL = k
                break
    if not SYMBOL:
        raise RuntimeError("No ETH perpetual found on Phemex")
    logger.info(f"[INIT] symbol={SYMBOL}")

# ── OHLCV ─────────────────────────────────────────────────────────────────────
def fetch_candles(limit: int = CANDLES) -> list:
    try:
        sym   = SYMBOL or "ETH/USDT:USDT"
        since = int((time_mod.time() - limit * 1800) * 1000)
        rows  = live_ex.fetch_ohlcv(sym, timeframe="30m", since=since, limit=limit)
        if rows and len(rows) >= 70:
            logger.info(f"[OHLCV] {len(rows)} candles  last={rows[-1][4]:.2f}")
            return rows
        logger.warning(f"[OHLCV] only {len(rows) if rows else 0} candles")
    except Exception as e:
        logger.error(f"[OHLCV] {e}")
    return []

# ── Internal paper trading ─────────────────────────────────────────────────────
# Mirrors Pine Script:  EC > EMA → LONG, EC < EMA → SHORT
# Exits:  SL = 2000 ticks ($20), trailing TP activates at 55 ticks ($0.55),
#         trail offset 15 ticks ($0.15)
paper = {
    "balance":      PAPER_START_BAL,
    "side":         "none",   # "long" | "short" | "none"
    "entry_price":  0.0,
    "qty":          0.0,
    "peak":         0.0,      # best price seen while in position
    "trail_active": False,    # trailing stop armed?
    "trail_stop":   0.0,      # current trailing stop price
    "unrealized":   0.0,
    "total_pnl":    0.0,
}

def _unrealized(cur_price: float) -> float:
    if paper["side"] == "none":
        return 0.0
    d = cur_price - paper["entry_price"]
    return (d if paper["side"] == "long" else -d) * paper["qty"]

def _open_position(side: str, qty: float, price: float) -> dict:
    paper["side"]         = "long" if side == "LONG" else "short"
    paper["entry_price"]  = price
    paper["qty"]          = qty
    paper["peak"]         = price
    paper["trail_active"] = False
    paper["trail_stop"]   = 0.0
    oid = f"SIM-{int(time_mod.time())}"
    logger.info(f"[PAPER] OPEN {side}  qty={qty}  @ {price:.2f}  id={oid}")
    return {"id": oid, "average": price, "amount": qty}

def _close_position(exit_price: float, reason: str) -> float:
    """Closes position at exit_price, returns realized PnL."""
    d   = exit_price - paper["entry_price"]
    pnl = (d if paper["side"] == "long" else -d) * paper["qty"]
    paper["balance"]    += pnl
    paper["total_pnl"]  += pnl
    logger.info(f"[PAPER] CLOSE {paper['side']}  @ {exit_price:.2f}  "
                f"pnl={pnl:+.4f}  reason={reason}  "
                f"balance={paper['balance']:.2f}")
    paper["side"]         = "none"
    paper["entry_price"]  = 0.0
    paper["qty"]          = 0.0
    paper["peak"]         = 0.0
    paper["trail_active"] = False
    paper["trail_stop"]   = 0.0
    paper["unrealized"]   = 0.0
    return pnl

def _check_exits(candle_open: float, candle_high: float,
                 candle_low: float, candle_close: float):
    """
    Evaluate SL and trailing TP against candle OHLC.
    Matches Pine Script calc_on_every_tick=false behaviour:
      SL hit   → exit at SL price
      Trail hit → exit at trail_stop price
    Returns (exited: bool, exit_price: float, reason: str)
    """
    if paper["side"] == "none":
        return False, 0.0, ""

    entry = paper["entry_price"]

    if paper["side"] == "long":
        sl_price = round(entry - SL_DIST, 4)

        # Update peak and arm/advance trailing stop
        if candle_high > paper["peak"]:
            paper["peak"] = candle_high
        peak_profit = paper["peak"] - entry
        if peak_profit >= TP_DIST:
            new_trail = round(paper["peak"] - TR_DIST, 4)
            if not paper["trail_active"] or new_trail > paper["trail_stop"]:
                paper["trail_active"] = True
                paper["trail_stop"]   = new_trail

        # Check trailing stop hit (before SL — trail is closer to price)
        if paper["trail_active"] and candle_low <= paper["trail_stop"]:
            return True, paper["trail_stop"], "TRAIL_TP"

        # Check stop loss
        if candle_low <= sl_price:
            return True, sl_price, "SL"

    else:  # short
        sl_price = round(entry + SL_DIST, 4)

        if candle_low < paper["peak"] or paper["peak"] == 0:
            paper["peak"] = candle_low
        peak_profit = entry - paper["peak"]
        if peak_profit >= TP_DIST:
            new_trail = round(paper["peak"] + TR_DIST, 4)
            if not paper["trail_active"] or new_trail < paper["trail_stop"]:
                paper["trail_active"] = True
                paper["trail_stop"]   = new_trail

        if paper["trail_active"] and candle_high >= paper["trail_stop"]:
            return True, paper["trail_stop"], "TRAIL_TP"

        if candle_high >= sl_price:
            return True, sl_price, "SL"

    return False, 0.0, ""

def _calc_qty(price: float, balance: float) -> float:
    sl_usdt    = SL_DIST  # $20
    risk_usdt  = RISK * balance
    qty        = (risk_usdt / sl_usdt) if sl_usdt else 0.01
    max_by_bal = (balance * 0.95 / price) if price else qty
    return round(min(qty, max_by_bal, MAX_LOTS), 4)

# ── Shared state ───────────────────────────────────────────────────────────────
state = {
    "running":    False,
    "status":     "Stopped",
    "signal":     "—",
    "position":   "None",
    "ema":        "—",
    "ec":         "—",
    "period":     "—",
    "last_candle":"—",
    "updated":    "—",
    "balance":    f"{PAPER_START_BAL:.2f}",
    "total_pnl":  "0.00",
    "unrealized": "0.00",
}
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
logger.info(f"[HISTORY] loaded {len(trade_history)} trades from disk")

# ── Strategy loop ──────────────────────────────────────────────────────────────
def strategy_loop():
    global state, trade_history
    try:
        init_markets()
    except Exception as e:
        logger.error(f"[INIT] {e}")
        state["status"]  = f"Error: {e}"
        state["running"] = False
        return

    # ── Sync: compute previous-candle signal to avoid false entry on startup ──
    # On startup we calculate the signal of candle N-1 (second-to-last closed).
    # The loop only enters a trade when signal CHANGES — identical to Pine Script
    # crossover/crossunder behaviour (strategy.entry same-ID = no re-entry).
    prev_signal    = None
    last_candle_ts = None

    init_ohlcv = fetch_candles()
    if init_ohlcv and len(init_ohlcv) >= 72:
        closes_prev = [c[4] for c in init_ohlcv[:-2]]   # up to candle N-1
        prev_signal = strat.calculate(closes_prev)["signal"]
        last_candle_ts = init_ohlcv[-2][0]               # mark candle N as seen
        logger.info(f"[SYNC] startup prev_signal={prev_signal}  "
                    f"(no trade until signal changes)")

    while state["running"]:
        try:
            ohlcv = fetch_candles()
            if not ohlcv or len(ohlcv) < 70:
                state["status"] = f"Waiting for data ({len(ohlcv)} candles)"
                time_mod.sleep(20)
                continue

            # ── Closed candle ([-2]) is what Pine Script "sees" at bar close ──
            last_closed  = ohlcv[-2]
            candle_ts    = last_closed[0]
            candle_o     = float(last_closed[1])
            candle_h     = float(last_closed[2])
            candle_l     = float(last_closed[3])
            candle_c     = float(last_closed[4])
            candle_str   = datetime.fromtimestamp(
                candle_ts / 1000, tz=timezone.utc
            ).strftime("%Y-%m-%d %H:%M UTC")

            # Update unrealized every cycle (live candle price)
            live_price          = float(ohlcv[-1][4])
            paper["unrealized"] = _unrealized(live_price)

            if candle_ts == last_candle_ts:
                state.update({
                    "unrealized": f"{paper['unrealized']:+.2f}",
                    "balance":    f"{paper['balance']:.2f}",
                })
                time_mod.sleep(15)
                continue

            last_candle_ts = candle_ts

            # ── 1. Check SL / Trailing TP on this closed candle ───────────────
            exited, exit_price, exit_reason = _check_exits(
                candle_o, candle_h, candle_l, candle_c)
            if exited:
                prev_side = paper["side"]
                pnl       = _close_position(exit_price, exit_reason)
                trade_history.insert(0, {
                    "time":     datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "candle":   candle_str,
                    "side":     f"CLOSE {prev_side.upper()} ({exit_reason})",
                    "price":    exit_price,
                    "qty":      0,
                    "order_id": f"EXIT-{exit_reason}",
                    "balance":  f"{paper['balance']:.2f}",
                    "pnl":      f"{pnl:+.4f}",
                })
                _save_history()
                # Allow re-entry on next candle (Pine Script re-enters at next
                # bar open after SL/TP if signal still valid)
                prev_signal = None

            # ── 2. Calculate indicators ───────────────────────────────────────
            closes = [c[4] for c in ohlcv[:-1]]
            result = strat.calculate(closes)
            signal = result["signal"]

            state.update({
                "signal":      signal or "—",
                "ema":         f"{result['ema']:.4f}" if result["ema"] else "—",
                "ec":          f"{result['ec']:.4f}"  if result["ec"]  else "—",
                "period":      str(result["period"]),
                "last_candle": candle_str,
                "updated":     datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "status":      "Running",
                "position":    paper["side"],
                "balance":     f"{paper['balance']:.2f}",
                "total_pnl":   f"{paper['total_pnl']:+.2f}",
                "unrealized":  f"{paper['unrealized']:+.2f}",
            })

            # ── 3. Entry only when signal CHANGES (Pine crossover behaviour) ──
            # Same as Pine Script: strategy.entry with identical ID never
            # re-enters the same direction — only acts on EC/EMA crossovers.
            if signal and signal != prev_signal:
                cur_pos = paper["side"]
                balance = paper["balance"]
                qty     = _calc_qty(candle_c, balance)

                if signal == "LONG" and cur_pos != "long":
                    if cur_pos == "short":
                        _close_position(candle_c, "SIGNAL_REVERSE")
                    order = _open_position("LONG", qty, candle_c)
                    trade_history.insert(0, {
                        "time":     datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "candle":   candle_str,
                        "side":     "LONG",
                        "price":    candle_c,
                        "qty":      qty,
                        "order_id": order["id"],
                        "balance":  f"{paper['balance']:.2f}",
                        "pnl":      "—",
                    })
                    _save_history()
                    state["position"] = "long"

                elif signal == "SHORT" and cur_pos != "short":
                    if cur_pos == "long":
                        _close_position(candle_c, "SIGNAL_REVERSE")
                    order = _open_position("SHORT", qty, candle_c)
                    trade_history.insert(0, {
                        "time":     datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "candle":   candle_str,
                        "side":     "SHORT",
                        "price":    candle_c,
                        "qty":      qty,
                        "order_id": order["id"],
                        "balance":  f"{paper['balance']:.2f}",
                        "pnl":      "—",
                    })
                    _save_history()
                    state["position"] = "short"

            prev_signal = signal

        except Exception as e:
            logger.error(f"[LOOP] {e}")
            state["status"] = f"Error: {str(e)[:80]}"
            time_mod.sleep(20)

    state["status"] = "Stopped"

# ── Keepalive — 3 internal signals ────────────────────────────────────────────
def keepalive(interval: int):
    time_mod.sleep(12)
    while True:
        try:
            req.get(f"http://localhost:{PORT}/ping", timeout=4)
        except Exception:
            pass
        time_mod.sleep(interval)

for _iv in [8, 15, 23]:
    threading.Thread(target=keepalive, args=(_iv,), daemon=True).start()

# ── Flask ──────────────────────────────────────────────────────────────────────
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
header h1{font-size:1.25rem;color:#58a6ff}
#clock{font-size:.8rem;color:#8b949e}
.container{max-width:1200px;margin:28px auto;padding:0 20px}
.controls{display:flex;gap:12px;margin-bottom:24px}
button{padding:10px 28px;border:none;border-radius:8px;font-size:.93rem;font-weight:600;cursor:pointer}
#btn-start{background:#238636;color:#fff}#btn-start:hover:not(:disabled){background:#2ea043}
#btn-stop{background:#da3633;color:#fff}#btn-stop:hover:not(:disabled){background:#f85149}
button:disabled{opacity:.35;cursor:not-allowed}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(148px,1fr));gap:12px;margin-bottom:24px}
.card{background:#161b22;border:1px solid #30363d;border-radius:10px;padding:14px 16px}
.lbl{font-size:.68rem;color:#8b949e;text-transform:uppercase;letter-spacing:.06em;margin-bottom:4px}
.val{font-size:1.1rem;font-weight:600}
.long{color:#3fb950}.short{color:#f85149}
.running{color:#3fb950}.stopped{color:#8b949e}
.err{color:#e3b341;font-size:.76rem;word-break:break-all;line-height:1.4}
.pos-num{color:#3fb950}.neg-num{color:#f85149}
.st{font-size:.78rem;color:#8b949e;text-transform:uppercase;letter-spacing:.07em;margin-bottom:10px}
table{width:100%;border-collapse:collapse;background:#161b22;border:1px solid #30363d;
      border-radius:10px;overflow:hidden}
th{background:#21262d;color:#8b949e;font-size:.72rem;text-transform:uppercase;
   padding:10px 12px;text-align:left}
td{padding:9px 12px;font-size:.83rem;border-top:1px solid #21262d}
tr:hover td{background:#1c2128}
.badge{display:inline-block;padding:2px 9px;border-radius:20px;font-size:.73rem;font-weight:700}
.badge.long{background:#0d4429;color:#3fb950}
.badge.short{background:#3d1010;color:#f85149}
.badge.close{background:#1e2533;color:#8b949e}
#toast{position:fixed;bottom:22px;right:22px;padding:11px 20px;border-radius:8px;
       font-size:.88rem;display:none;z-index:99;color:#fff}
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
    <div class="card"><div class="lbl">Balance (USDT)</div><div class="val" id="c-balance">—</div></div>
    <div class="card"><div class="lbl">Total PnL</div><div class="val" id="c-pnl">—</div></div>
    <div class="card"><div class="lbl">Unrealized</div><div class="val" id="c-unreal">—</div></div>
    <div class="card"><div class="lbl">EMA</div><div class="val" id="c-ema">—</div></div>
    <div class="card"><div class="lbl">EC</div><div class="val" id="c-ec">—</div></div>
    <div class="card"><div class="lbl">Period</div><div class="val" id="c-period">—</div></div>
    <div class="card"><div class="lbl">Last Candle</div>
      <div class="val" style="font-size:.8rem" id="c-candle">—</div></div>
    <div class="card"><div class="lbl">Updated</div>
      <div class="val" style="font-size:.75rem" id="c-upd">—</div></div>
  </div>
  <div class="st">Trade History</div>
  <table>
    <thead>
      <tr><th>Time</th><th>Candle</th><th>Side</th>
          <th>Price</th><th>Qty</th><th>PnL</th><th>Balance</th><th>ID</th></tr>
    </thead>
    <tbody id="tbody">
      <tr><td colspan="8" style="text-align:center;color:#484f58;padding:22px">
        No trades yet</td></tr>
    </tbody>
  </table>
</div>
<div id="toast"></div>
<footer>AZLEMA · Paper Trading · 30 m · ETH/USDT · SL=$20 · Trail TP=$0.55/$0.15 · Preços Phemex Live</footer>
<script>
function p(n){return String(n).padStart(2,'0')}
function tick(){const d=new Date();document.getElementById('clock').textContent=
  `${d.getUTCFullYear()}-${p(d.getUTCMonth()+1)}-${p(d.getUTCDate())} `+
  `${p(d.getUTCHours())}:${p(d.getUTCMinutes())}:${p(d.getUTCSeconds())} UTC`}
setInterval(tick,1000);tick();
function toast(msg,ok){const e=document.getElementById('toast');
  e.textContent=msg;e.style.background=ok?'#238636':'#da3633';
  e.style.display='block';setTimeout(()=>e.style.display='none',3000)}
function numClass(v){const n=parseFloat(v);return n>0?'val pos-num':n<0?'val neg-num':'val'}
async function ctrl(a){
  const r=await fetch('/'+a,{method:'POST'});
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
  document.getElementById('c-balance').textContent='$'+s.balance;
  const pnl=document.getElementById('c-pnl');
  pnl.textContent='$'+s.total_pnl;pnl.className=numClass(s.total_pnl);
  const unr=document.getElementById('c-unreal');
  unr.textContent='$'+s.unrealized;unr.className=numClass(s.unrealized);
  ['ema','ec','period'].forEach(k=>document.getElementById('c-'+k).textContent=s[k]);
  document.getElementById('c-candle').textContent=s.last_candle;
  document.getElementById('c-upd').textContent=s.updated;
  document.getElementById('btn-start').disabled=s.running;
  document.getElementById('btn-stop').disabled=!s.running;
  const tb=document.getElementById('tbody');
  const side2badge=s=>{
    if(s==='LONG') return'<span class="badge long">LONG</span>';
    if(s==='SHORT') return'<span class="badge short">SHORT</span>';
    return`<span class="badge close">${s}</span>`};
  tb.innerHTML=t.length?t.map(r=>`<tr>
    <td>${r.time}</td><td style="font-size:.76rem">${r.candle}</td>
    <td>${side2badge(r.side)}</td>
    <td>$${Number(r.price).toFixed(2)}</td>
    <td>${r.qty||'—'}</td>
    <td class="${r.pnl&&r.pnl!=='—'?(parseFloat(r.pnl)>=0?'pos-num':'neg-num'):''}">
      ${r.pnl&&r.pnl!=='—'?'$'+r.pnl:'—'}</td>
    <td>$${r.balance}</td>
    <td style="font-size:.7rem;color:#8b949e">${r.order_id}</td></tr>`).join('')
    :'<tr><td colspan="8" style="text-align:center;color:#484f58;padding:22px">'
     +'No trades yet</td></tr>'}
fetchAll();setInterval(fetchAll,10000);
</script>
</body>
</html>"""

@app.route("/")
def index():
    return HTML

@app.route("/ping")
@app.route("/health")
def ping():
    return jsonify({"ok": True, "time": datetime.utcnow().isoformat()})

@app.route("/status")
def status():
    return jsonify({**state, "running": state["running"]})

@app.route("/history")
def history():
    return jsonify(trade_history)

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
