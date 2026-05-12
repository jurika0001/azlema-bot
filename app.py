import os, time as time_mod, threading, logging
import requests as req
from datetime import datetime, timezone
from flask import Flask, jsonify

import ccxt
import strategy as strat

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ── Config ─────────────────────────────────────────────────────────────────────
RISK     = 0.01
FIXED_SL = 2000
MAX_LOTS = 100
LEVERAGE = 1
CANDLES  = 200
PORT     = int(os.environ.get("PORT", 10000))

# ── live_ex  : Phemex live, no auth, no sandbox — OHLCV only ──────────────────
# Created clean, no set_sandbox_mode, no URL override.
# ccxt instances are independent; paper_ex sandbox does NOT affect live_ex.
live_ex = ccxt.phemex({
    "options": {"defaultType": "swap"},
    "enableRateLimit": True,
})

# ── paper_ex : Phemex testnet, with auth — orders / balance / positions ───────
_api_key = (os.environ.get("PHEMEX_API_KEY") or "").strip()
_api_sec  = (os.environ.get("PHEMEX_API_SECRET") or "").strip()
logger.info(f"[AUTH] key={_api_key[:6]}... len_key={len(_api_key)} len_secret={len(_api_sec)}")

paper_ex = ccxt.phemex({
    "apiKey": _api_key,
    "secret": _api_sec,
    "options": {"defaultType": "swap"},
    "enableRateLimit": True,
})
# set_sandbox_mode first, then patch ALL url sub-keys to testnet
paper_ex.set_sandbox_mode(True)
_testnet  = "https://testnet-api.phemex.com"
_api_urls = paper_ex.urls.get("api", {})
if isinstance(_api_urls, dict):
    for _k in list(_api_urls.keys()):
        paper_ex.urls["api"][_k] = _testnet
else:
    paper_ex.urls["api"] = _testnet
logger.info(f"[AUTH] paper_ex api url = {paper_ex.urls.get('api')}")

SYMBOL = None

def init_markets():
    global SYMBOL
    # Carrega os mercados do Phemex live para OHLCV
    live_mkts = live_ex.load_markets()
    logger.info(f"[INIT] live={len(live_mkts)} markets")
    # Escolha explícita e estável para ETH perp do Phemex
    preferred = ["ETH/USDT:USDT", "ETH/USD:ETH"]
    for sym in preferred:
        if sym in live_mkts:
            SYMBOL = sym
            break
    # Fallback apenas dentro do próprio Phemex, sem outro exchange
    if not SYMBOL:
        for k, v in live_mkts.items():
            if "ETH" in k and v.get("type") in ("swap", "future"):
                SYMBOL = k
                break
    if not SYMBOL:
        raise RuntimeError("No ETH perpetual found on Phemex live")
    # Também carrega testnet, pois ordens/posição/balance continuam no paper_ex do Phemex
    paper_mkts = paper_ex.load_markets()
    logger.info(f"[INIT] testnet={len(paper_mkts)} markets")
    logger.info(f"[INIT] symbol={SYMBOL}")

# ── OHLCV — Phemex live via ccxt ──────────────────────────────────────────────
def fetch_candles(limit: int = CANDLES) -> list:
    try:
        if not SYMBOL:
            init_markets()
        sym = live_ex.market(SYMBOL)["symbol"]
        # Use since (ms) instead of limit — avoids Phemex code:30000 on linear perp
        since = int((time_mod.time() - limit * 1800) * 1000)
        rows  = live_ex.fetch_ohlcv(sym, timeframe="30m", since=since, limit=limit)
        if rows and len(rows) >= 70:
            logger.info(f"[OHLCV] {len(rows)} candles  last={rows[-1][4]:.2f}")
            return rows
        logger.warning(f"[OHLCV] only {len(rows) if rows else 0} candles returned")
    except Exception as e:
        logger.error(f"[OHLCV] {e}")
    return []

# ── Shared state ───────────────────────────────────────────────────────────────
state = {
    "running": False, "status": "Stopped",
    "signal": "—", "position": "None",
    "ema": "—", "ec": "—", "period": "—",
    "last_candle": "—", "updated": "—",
}
trade_history    = []
_strategy_thread = None

# ── Trading helpers ────────────────────────────────────────────────────────────
def get_position() -> str:
    try:
        for p in paper_ex.fetch_positions([SYMBOL]):
            if float(p.get("contracts") or 0) != 0:
                return p["side"].lower()
    except Exception as e:
        logger.warning(f"[POS] {e}")
    return "none"

def get_balance() -> float:
    try:
        bal = paper_ex.fetch_balance()
        for key in ("USDT", "USD", "BTC"):
            v = bal.get("total", {}).get(key)
            if v:
                return float(v)
    except Exception as e:
        logger.warning(f"[BAL] {e}")
    return 1000.0

def calc_qty(price: float, balance: float) -> float:
    try:
        tick = float(paper_ex.markets[SYMBOL]["precision"]["price"])
    except Exception:
        tick = 0.01
    sl_usdt    = FIXED_SL * tick
    risk_usdt  = RISK * balance
    qty        = (risk_usdt / sl_usdt) if sl_usdt else 0.01
    max_by_bal = (balance * 0.95 / price) if price else qty
    return round(min(qty, max_by_bal, MAX_LOTS), 4)

def place_order(side: str, qty: float):
    try:
        paper_ex.set_leverage(LEVERAGE, SYMBOL)
    except Exception:
        pass
    try:
        if side == "LONG":
            return paper_ex.create_market_buy_order(SYMBOL, qty)
        return paper_ex.create_market_sell_order(SYMBOL, qty)
    except Exception as e:
        logger.error(f"[ORDER] {e}")
        return None

def close_pos(side: str):
    try:
        for p in paper_ex.fetch_positions([SYMBOL]):
            size = float(p.get("contracts") or 0)
            if size != 0 and p["side"].lower() == side:
                params = {"reduceOnly": True}
                if side == "long":
                    paper_ex.create_market_sell_order(SYMBOL, size, params)
                else:
                    paper_ex.create_market_buy_order(SYMBOL, size, params)
    except Exception as e:
        logger.warning(f"[CLOSE] {e}")

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

    if SYMBOL not in live_ex.markets:
        err = f"Invalid Phemex symbol for OHLCV: {SYMBOL}"
        logger.error(f"[INIT] {err}")
        state["status"]  = f"Error: {err}"
        state["running"] = False
        return

    last_candle_ts = None

    while state["running"]:
        try:
            ohlcv = fetch_candles()
            if not ohlcv or len(ohlcv) < 70:
                state["status"] = f"Waiting for data ({len(ohlcv)} candles)"
                time_mod.sleep(20)
                continue

            last_closed = ohlcv[-2]
            candle_ts   = last_closed[0]
            candle_str  = datetime.fromtimestamp(
                candle_ts / 1000, tz=timezone.utc
            ).strftime("%Y-%m-%d %H:%M UTC")

            if candle_ts == last_candle_ts:
                time_mod.sleep(15)
                continue

            last_candle_ts = candle_ts
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
            })

            if not signal:
                time_mod.sleep(15)
                continue

            cur_pos   = get_position()
            state["position"] = cur_pos
            balance   = get_balance()
            cur_price = float(last_closed[4])
            qty       = calc_qty(cur_price, balance)

            if signal == "LONG" and cur_pos != "long":
                if cur_pos == "short":
                    close_pos("short")
                order = place_order("LONG", qty)
                if order:
                    trade_history.insert(0, {
                        "time":     datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "candle":   candle_str, "side": "LONG",
                        "price":    order.get("average") or order.get("price") or cur_price,
                        "qty":      qty, "order_id": order.get("id", "N/A"),
                    })
                    state["position"] = "long"
                    logger.info(f"[TRADE] LONG qty={qty} @ {cur_price}")

            elif signal == "SHORT" and cur_pos != "short":
                if cur_pos == "long":
                    close_pos("long")
                order = place_order("SHORT", qty)
                if order:
                    trade_history.insert(0, {
                        "time":     datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "candle":   candle_str, "side": "SHORT",
                        "price":    order.get("average") or order.get("price") or cur_price,
                        "qty":      qty, "order_id": order.get("id", "N/A"),
                    })
                    state["position"] = "short"
                    logger.info(f"[TRADE] SHORT qty={qty} @ {cur_price}")

        except ccxt.AuthenticationError as e:
            logger.error(f"[AUTH] {e}")
            state["status"]  = "Error: invalid API keys"
            state["running"] = False
            break
        except ccxt.NetworkError as e:
            logger.warning(f"[NET] {e}")
            state["status"] = "Warning: network error, retrying…"
            time_mod.sleep(30)
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
.container{max-width:1100px;margin:28px auto;padding:0 20px}
.controls{display:flex;gap:12px;margin-bottom:24px}
button{padding:10px 28px;border:none;border-radius:8px;font-size:.93rem;font-weight:600;cursor:pointer}
#btn-start{background:#238636;color:#fff}#btn-start:hover:not(:disabled){background:#2ea043}
#btn-stop{background:#da3633;color:#fff}#btn-stop:hover:not(:disabled){background:#f85149}
button:disabled{opacity:.35;cursor:not-allowed}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(155px,1fr));gap:14px;margin-bottom:26px}
.card{background:#161b22;border:1px solid #30363d;border-radius:10px;padding:16px 18px}
.lbl{font-size:.7rem;color:#8b949e;text-transform:uppercase;letter-spacing:.06em;margin-bottom:5px}
.val{font-size:1.2rem;font-weight:600}
.long{color:#3fb950}.short{color:#f85149}
.running{color:#3fb950}.stopped{color:#8b949e}
.err{color:#e3b341;font-size:.78rem;word-break:break-all;line-height:1.4}
.st{font-size:.8rem;color:#8b949e;text-transform:uppercase;letter-spacing:.07em;margin-bottom:10px}
table{width:100%;border-collapse:collapse;background:#161b22;border:1px solid #30363d;
      border-radius:10px;overflow:hidden}
th{background:#21262d;color:#8b949e;font-size:.75rem;text-transform:uppercase;
   padding:11px 14px;text-align:left}
td{padding:10px 14px;font-size:.86rem;border-top:1px solid #21262d}
tr:hover td{background:#1c2128}
.badge{display:inline-block;padding:2px 10px;border-radius:20px;font-size:.76rem;font-weight:700}
.badge.long{background:#0d4429;color:#3fb950}.badge.short{background:#3d1010;color:#f85149}
#toast{position:fixed;bottom:22px;right:22px;padding:11px 20px;border-radius:8px;
       font-size:.88rem;display:none;z-index:99;color:#fff}
footer{text-align:center;color:#484f58;font-size:.75rem;margin:36px 0 18px}
</style>
</head>
<body>
<header>
  <h1>⚡ AZLEMA Bot — Phemex Paper Trading</h1>
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
    <div class="card"><div class="lbl">EMA</div><div class="val" id="c-ema">—</div></div>
    <div class="card"><div class="lbl">EC</div><div class="val" id="c-ec">—</div></div>
    <div class="card"><div class="lbl">Period</div><div class="val" id="c-period">—</div></div>
    <div class="card"><div class="lbl">Last Candle</div>
      <div class="val" style="font-size:.88rem" id="c-candle">—</div></div>
    <div class="card"><div class="lbl">Updated</div>
      <div class="val" style="font-size:.82rem" id="c-upd">—</div></div>
  </div>
  <div class="st">Trade History</div>
  <table>
    <thead>
      <tr><th>Time</th><th>Candle</th><th>Side</th>
          <th>Price</th><th>Qty</th><th>Order ID</th></tr>
    </thead>
    <tbody id="tbody">
      <tr><td colspan="6" style="text-align:center;color:#484f58;padding:22px">
        No trades yet</td></tr>
    </tbody>
  </table>
</div>
<div id="toast"></div>
<footer>AZLEMA · Phemex Paper Testnet · 30 m · ETH/USDT · 1× Leverage</footer>
<script>
function p(n){return String(n).padStart(2,'0')}
function tick(){const d=new Date();document.getElementById('clock').textContent=
  `${d.getUTCFullYear()}-${p(d.getUTCMonth()+1)}-${p(d.getUTCDate())} `+
  `${p(d.getUTCHours())}:${p(d.getUTCMinutes())}:${p(d.getUTCSeconds())} UTC`}
setInterval(tick,1000);tick();
function toast(msg,ok){const e=document.getElementById('toast');
  e.textContent=msg;e.style.background=ok?'#238636':'#da3633';
  e.style.display='block';setTimeout(()=>e.style.display='none',3000)}
async function ctrl(a){
  const r=await fetch('/'+a,{method:'POST'});
  const d=await r.json();toast(d.message,d.ok);fetchAll()}
async function fetchAll(){
  const[sr,hr]=await Promise.all([fetch('/status'),fetch('/history')]);
  const s=await sr.json(),t=await hr.json();
  const sv=document.getElementById('c-status');sv.textContent=s.status;
  sv.className='val '+(s.status==='Running'?'running':
    (s.status.startsWith('Error')||s.status.startsWith('Warning')||
     s.status.startsWith('Waiting'))?'err':'stopped');
  const sg=document.getElementById('c-signal');sg.textContent=s.signal;
  sg.className='val '+(s.signal==='LONG'?'long':s.signal==='SHORT'?'short':'');
  const ps=document.getElementById('c-pos');ps.textContent=s.position;
  ps.className='val '+(s.position==='long'?'long':s.position==='short'?'short':'');
  ['ema','ec','period'].forEach(k=>document.getElementById('c-'+k).textContent=s[k]);
  document.getElementById('c-candle').textContent=s.last_candle;
  document.getElementById('c-upd').textContent=s.updated;
  document.getElementById('btn-start').disabled=s.running;
  document.getElementById('btn-stop').disabled=!s.running;
  const tb=document.getElementById('tbody');
  tb.innerHTML=t.length?t.map(r=>`<tr>
    <td>${r.time}</td><td>${r.candle}</td>
    <td><span class="badge ${r.side.toLowerCase()}">${r.side}</span></td>
    <td>${Number(r.price).toFixed(2)}</td><td>${r.qty}</td>
    <td style="font-size:.74rem;color:#8b949e">${r.order_id}</td></tr>`).join('')
    :'<tr><td colspan="6" style="text-align:center;color:#484f58;padding:22px">'
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
