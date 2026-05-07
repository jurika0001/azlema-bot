import os, time, threading, logging
from datetime import datetime, timezone
from flask import Flask, jsonify, request

import ccxt
import strategy as strat

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
SYMBOL     = "ETH/USDT:USDT"
TIMEFRAME  = "30m"
RISK       = 0.01        # 1 % risk per trade
FIXED_SL   = 2000        # stop-loss in ticks
FIXED_TP   = 55          # trailing-stop trigger in ticks
MAX_LOTS   = 100
LEVERAGE   = 1
CANDLES    = 300         # bars to fetch
PORT       = int(os.environ.get("PORT", 10000))

# ── Phemex paper-trading client (ccxt sandbox = testnet) ─────────────────────
def make_exchange():
    ex = ccxt.phemex({
        "apiKey": os.environ.get("PHEMEX_API_KEY", ""),
        "secret": os.environ.get("PHEMEX_API_SECRET", ""),
        "options": {"defaultType": "swap"},
    })
    ex.set_sandbox_mode(True)
    return ex

exchange = make_exchange()

# ── Shared state ──────────────────────────────────────────────────────────────
state = {
    "running":    False,
    "status":     "Stopped",
    "signal":     "—",
    "position":   "None",
    "ema":        "—",
    "ec":         "—",
    "period":     "—",
    "last_candle": "—",
    "updated":    "—",
}
trade_history = []          # [ {time, side, price, qty, order_id} ]
_strategy_thread = None


# ── Position helpers ──────────────────────────────────────────────────────────
def get_current_position() -> str:
    """Returns 'long', 'short', or 'none'."""
    try:
        positions = exchange.fetch_positions([SYMBOL])
        for p in positions:
            size = float(p.get("contracts") or 0)
            if size != 0:
                return p["side"].lower()
    except Exception as e:
        logger.warning(f"[POS] {e}")
    return "none"


def get_balance() -> float:
    try:
        bal = exchange.fetch_balance()
        return float(bal["total"].get("USDT", 0) or 0)
    except Exception as e:
        logger.warning(f"[BAL] {e}")
        return 1000.0


def calc_qty(price: float, balance: float) -> float:
    tick        = exchange.markets[SYMBOL]["precision"]["price"] if SYMBOL in exchange.markets else 0.01
    sl_usdt     = FIXED_SL * tick
    risk_usdt   = RISK * balance
    qty         = risk_usdt / sl_usdt if sl_usdt else 0.01
    max_by_bal  = (balance * 0.95) / price if price else qty
    return round(min(qty, max_by_bal, MAX_LOTS), 4)


def place_order(side: str, qty: float):
    try:
        exchange.set_leverage(LEVERAGE, SYMBOL)
        if side == "LONG":
            order = exchange.create_market_buy_order(SYMBOL, qty)
        else:
            order = exchange.create_market_sell_order(SYMBOL, qty)
        return order
    except Exception as e:
        logger.error(f"[ORDER] {e}")
        return None


def close_position(side: str):
    """Close open position before reversing."""
    try:
        positions = exchange.fetch_positions([SYMBOL])
        for p in positions:
            size = float(p.get("contracts") or 0)
            if size != 0 and p["side"].lower() == side:
                if side == "long":
                    exchange.create_market_sell_order(SYMBOL, size, {"reduceOnly": True})
                else:
                    exchange.create_market_buy_order(SYMBOL, size, {"reduceOnly": True})
    except Exception as e:
        logger.warning(f"[CLOSE] {e}")


# ── Strategy loop ─────────────────────────────────────────────────────────────
def strategy_loop():
    global state, trade_history
    last_candle_ts = None

    try:
        exchange.load_markets()
    except Exception as e:
        logger.error(f"[MARKETS] {e}")

    while state["running"]:
        try:
            ohlcv = exchange.fetch_ohlcv(SYMBOL, TIMEFRAME, limit=CANDLES)
            if not ohlcv or len(ohlcv) < 70:
                time.sleep(15)
                continue

            # Last CLOSED candle = ohlcv[-2]  (ohlcv[-1] is still forming)
            last_closed    = ohlcv[-2]
            candle_ts      = last_closed[0]
            candle_time_str = datetime.fromtimestamp(candle_ts / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

            if candle_ts == last_candle_ts:
                time.sleep(15)
                continue

            last_candle_ts = candle_ts
            closes = [c[4] for c in ohlcv[:-1]]   # all closed-candle closes

            result = strat.calculate(closes)
            signal = result["signal"]

            state["signal"]      = signal or "—"
            state["ema"]         = f"{result['ema']:.4f}" if result["ema"] else "—"
            state["ec"]          = f"{result['ec']:.4f}"  if result["ec"]  else "—"
            state["period"]      = str(result["period"])
            state["last_candle"] = candle_time_str
            state["updated"]     = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            state["status"]      = "Running"

            if not signal:
                time.sleep(15)
                continue

            current_pos = get_current_position()
            state["position"] = current_pos

            balance     = get_balance()
            current_price = float(ohlcv[-2][4])
            qty         = calc_qty(current_price, balance)

            # ── LONG ─────────────────────────────────────────────────────────
            if signal == "LONG" and current_pos != "long":
                if current_pos == "short":
                    close_position("short")
                order = place_order("LONG", qty)
                if order:
                    trade_history.insert(0, {
                        "time":     datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "side":     "LONG",
                        "price":    order.get("average") or order.get("price") or current_price,
                        "qty":      qty,
                        "order_id": order.get("id", "N/A"),
                        "candle":   candle_time_str,
                    })
                    state["position"] = "long"
                    logger.info(f"[TRADE] LONG qty={qty} price={current_price}")

            # ── SHORT ────────────────────────────────────────────────────────
            elif signal == "SHORT" and current_pos != "short":
                if current_pos == "long":
                    close_position("long")
                order = place_order("SHORT", qty)
                if order:
                    trade_history.insert(0, {
                        "time":     datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "side":     "SHORT",
                        "price":    order.get("average") or order.get("price") or current_price,
                        "qty":      qty,
                        "order_id": order.get("id", "N/A"),
                        "candle":   candle_time_str,
                    })
                    state["position"] = "short"
                    logger.info(f"[TRADE] SHORT qty={qty} price={current_price}")

        except Exception as e:
            logger.error(f"[LOOP] {e}")
            state["status"] = f"Error: {str(e)[:60]}"

        time.sleep(15)

    state["status"] = "Stopped"


# ── Keepalive threads (3 internal signals) ────────────────────────────────────
def keepalive(interval: int, name: str):
    """Pings /ping on localhost to keep the web service alive."""
    import requests as req
    time.sleep(10)   # wait for server to start
    while True:
        try:
            req.get(f"http://localhost:{PORT}/ping", timeout=4)
            logger.debug(f"[{name}] ping ok")
        except Exception:
            pass
        time.sleep(interval)

for _interval, _name in [(8, "Signal-1"), (15, "Signal-2"), (23, "Signal-3")]:
    threading.Thread(target=keepalive, args=(_interval, _name), daemon=True).start()


# ── Flask app ─────────────────────────────────────────────────────────────────
app = Flask(__name__)

HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>AZLEMA Bot – Phemex Paper</title>
<style>
  *{box-sizing:border-box;margin:0;padding:0}
  body{background:#0d1117;color:#e6edf3;font-family:'Segoe UI',sans-serif;min-height:100vh}
  header{background:#161b22;border-bottom:1px solid #30363d;padding:18px 32px;
         display:flex;align-items:center;justify-content:space-between}
  header h1{font-size:1.3rem;letter-spacing:.04em;color:#58a6ff}
  header span{font-size:.8rem;color:#8b949e}
  .container{max-width:1100px;margin:32px auto;padding:0 20px}
  .cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:16px;margin-bottom:28px}
  .card{background:#161b22;border:1px solid #30363d;border-radius:10px;padding:18px 20px}
  .card .label{font-size:.72rem;color:#8b949e;text-transform:uppercase;letter-spacing:.06em;margin-bottom:6px}
  .card .value{font-size:1.25rem;font-weight:600;color:#e6edf3}
  .value.long{color:#3fb950} .value.short{color:#f85149}
  .value.running{color:#3fb950} .value.stopped{color:#f85149} .value.error{color:#e3b341}
  .controls{display:flex;gap:12px;margin-bottom:28px}
  button{padding:10px 28px;border:none;border-radius:8px;font-size:.95rem;
         font-weight:600;cursor:pointer;transition:.15s}
  #btn-start{background:#238636;color:#fff} #btn-start:hover{background:#2ea043}
  #btn-stop {background:#da3633;color:#fff} #btn-stop:hover{background:#f85149}
  #btn-stop:disabled{opacity:.4;cursor:not-allowed}
  #btn-start:disabled{opacity:.4;cursor:not-allowed}
  .section-title{font-size:.85rem;color:#8b949e;text-transform:uppercase;
                 letter-spacing:.07em;margin-bottom:12px}
  table{width:100%;border-collapse:collapse;background:#161b22;
        border:1px solid #30363d;border-radius:10px;overflow:hidden}
  th{background:#21262d;color:#8b949e;font-size:.78rem;text-transform:uppercase;
     letter-spacing:.05em;padding:12px 16px;text-align:left}
  td{padding:11px 16px;font-size:.88rem;border-top:1px solid #21262d;color:#e6edf3}
  tr:hover td{background:#1c2128}
  .badge{display:inline-block;padding:2px 10px;border-radius:20px;font-size:.78rem;font-weight:600}
  .badge.long{background:#0d4429;color:#3fb950} .badge.short{background:#3d1010;color:#f85149}
  #toast{position:fixed;bottom:24px;right:24px;background:#238636;color:#fff;
         padding:12px 22px;border-radius:8px;font-size:.9rem;display:none;z-index:99}
  footer{text-align:center;color:#484f58;font-size:.78rem;margin:40px 0 20px}
</style>
</head>
<body>
<header>
  <h1>⚡ AZLEMA Bot — Phemex Paper Trading</h1>
  <span id="clock"></span>
</header>
<div class="container">

  <div class="controls">
    <button id="btn-start" onclick="controlBot('start')">▶ Start Strategy</button>
    <button id="btn-stop"  onclick="controlBot('stop')"  disabled>⏹ Stop Strategy</button>
  </div>

  <div class="cards">
    <div class="card">
      <div class="label">Status</div>
      <div class="value" id="c-status">—</div>
    </div>
    <div class="card">
      <div class="label">Signal</div>
      <div class="value" id="c-signal">—</div>
    </div>
    <div class="card">
      <div class="label">Position</div>
      <div class="value" id="c-position">—</div>
    </div>
    <div class="card">
      <div class="label">EMA</div>
      <div class="value" id="c-ema">—</div>
    </div>
    <div class="card">
      <div class="label">EC</div>
      <div class="value" id="c-ec">—</div>
    </div>
    <div class="card">
      <div class="label">Period</div>
      <div class="value" id="c-period">—</div>
    </div>
    <div class="card">
      <div class="label">Last Candle</div>
      <div class="value" style="font-size:.95rem" id="c-candle">—</div>
    </div>
    <div class="card">
      <div class="label">Updated</div>
      <div class="value" style="font-size:.9rem" id="c-updated">—</div>
    </div>
  </div>

  <div class="section-title">Trade History</div>
  <table>
    <thead>
      <tr>
        <th>Time</th><th>Candle</th><th>Side</th>
        <th>Price</th><th>Qty</th><th>Order ID</th>
      </tr>
    </thead>
    <tbody id="history-body">
      <tr><td colspan="6" style="text-align:center;color:#484f58;padding:24px">
        No trades yet
      </td></tr>
    </tbody>
  </table>
</div>

<div id="toast"></div>
<footer>AZLEMA · Phemex Paper · 30 m · ETH/USDT · 1× Leverage</footer>

<script>
function pad(n){return String(n).padStart(2,'0')}
function clock(){
  const d=new Date();
  document.getElementById('clock').textContent=
    `${d.getUTCFullYear()}-${pad(d.getUTCMonth()+1)}-${pad(d.getUTCDate())} `+
    `${pad(d.getUTCHours())}:${pad(d.getUTCMinutes())}:${pad(d.getUTCSeconds())} UTC`;
}
setInterval(clock,1000); clock();

function toast(msg,ok){
  const el=document.getElementById('toast');
  el.textContent=msg;
  el.style.background=ok?'#238636':'#da3633';
  el.style.display='block';
  setTimeout(()=>el.style.display='none',3000);
}

async function controlBot(action){
  const r=await fetch('/'+action,{method:'POST'});
  const d=await r.json();
  toast(d.message, d.ok);
  fetchStatus();
}

async function fetchStatus(){
  const r=await fetch('/status');
  const d=await r.json();

  const sv=document.getElementById('c-status');
  sv.textContent=d.status;
  sv.className='value '+(d.status==='Running'?'running':d.status==='Stopped'?'stopped':'error');

  const sig=document.getElementById('c-signal');
  sig.textContent=d.signal;
  sig.className='value '+(d.signal==='LONG'?'long':d.signal==='SHORT'?'short':'');

  const pos=document.getElementById('c-position');
  pos.textContent=d.position;
  pos.className='value '+(d.position==='long'?'long':d.position==='short'?'short':'');

  document.getElementById('c-ema').textContent=d.ema;
  document.getElementById('c-ec').textContent=d.ec;
  document.getElementById('c-period').textContent=d.period;
  document.getElementById('c-candle').textContent=d.last_candle;
  document.getElementById('c-updated').textContent=d.updated;

  document.getElementById('btn-start').disabled=d.running;
  document.getElementById('btn-stop').disabled=!d.running;
}

async function fetchHistory(){
  const r=await fetch('/history');
  const trades=await r.json();
  const tbody=document.getElementById('history-body');
  if(!trades.length){
    tbody.innerHTML='<tr><td colspan="6" style="text-align:center;color:#484f58;padding:24px">No trades yet</td></tr>';
    return;
  }
  tbody.innerHTML=trades.map(t=>`
    <tr>
      <td>${t.time}</td>
      <td>${t.candle}</td>
      <td><span class="badge ${t.side.toLowerCase()}">${t.side}</span></td>
      <td>${Number(t.price).toFixed(2)}</td>
      <td>${t.qty}</td>
      <td style="font-size:.78rem;color:#8b949e">${t.order_id}</td>
    </tr>`).join('');
}

fetchStatus(); fetchHistory();
setInterval(()=>{fetchStatus();fetchHistory();}, 10000);
</script>
</body>
</html>"""


@app.route("/")
def index():
    return HTML


@app.route("/ping")
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
    logger.info("[BOT] Strategy started")
    return jsonify({"ok": True, "message": "Strategy started"})


@app.route("/stop", methods=["POST"])
def stop():
    if not state["running"]:
        return jsonify({"ok": False, "message": "Already stopped"})
    state["running"] = False
    logger.info("[BOT] Strategy stopped")
    return jsonify({"ok": True, "message": "Strategy stopped"})


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT)
