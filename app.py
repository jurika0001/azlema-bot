import os, time as time_mod, threading, logging
import requests as req
from datetime import datetime, timezone
from flask import Flask, jsonify

import ccxt
import strategy as strat

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ── Config ─────────────────────────────────────────────────────────────────────
LIVE_BASE    = "https://api.phemex.com"
TESTNET_BASE = "https://testnet-api.phemex.com"
RESOLUTION   = 1800   # 30 min in seconds
RISK         = 0.01
FIXED_SL     = 2000
MAX_LOTS     = 100
LEVERAGE     = 1
CANDLES      = 300
PORT         = int(os.environ.get("PORT", 10000))

# ── Exchange for OHLCV — LIVE Phemex, no auth, no sandbox ─────────────────────
live_ex = ccxt.phemex({
    "options": {"defaultType": "swap"},
    "enableRateLimit": True,
})

# ── Exchange for orders — TESTNET Phemex, paper trading ───────────────────────
paper_ex = ccxt.phemex({
    "apiKey": os.environ.get("PHEMEX_API_KEY", ""),
    "secret": os.environ.get("PHEMEX_API_SECRET", ""),
    "options": {"defaultType": "swap"},
    "enableRateLimit": True,
})
paper_ex.set_sandbox_mode(True)

SYMBOL    = None   # unified ccxt e.g. "ETH/USDT:USDT"
SYMBOL_ID = None   # raw Phemex  e.g. "ETHUSDT"

def init_markets():
    global SYMBOL, SYMBOL_ID
    # Load markets from both exchanges
    live_markets  = live_ex.load_markets()
    paper_markets = paper_ex.load_markets()
    logger.info(f"[INIT] live={len(live_markets)} markets  "
                f"testnet={len(paper_markets)} markets")

    for sym in ["ETH/USDT:USDT", "ETH/USD:ETH", "ETHUSDT", "ETHUSD"]:
        if sym in live_markets:
            SYMBOL = sym
            break
    if not SYMBOL:
        for k, v in live_markets.items():
            if "ETH" in k and v.get("type") in ("swap", "future"):
                SYMBOL = k
                break
    if not SYMBOL:
        raise RuntimeError("No ETH perpetual market found on Phemex live")

    SYMBOL_ID = live_ex.market_id(SYMBOL)
    logger.info(f"[INIT] symbol={SYMBOL}  id={SYMBOL_ID}")

# ── OHLCV via ccxt live — correct params handled by ccxt ──────────────────────
def fetch_candles(limit: int = CANDLES) -> list:
    """
    Attempt order:
    1. ccxt live  fetch_ohlcv  (unified, correct params)
    2. direct HTTP /md/v2/kline/list  limit-only
    3. direct HTTP /md/kline          limit-only  (inverse fallback)
    """
    sym = SYMBOL or "ETH/USDT:USDT"

    # ── 1. ccxt live ───────────────────────────────────────────────────────────
    try:
        rows = live_ex.fetch_ohlcv(sym, "30m", limit=limit)
        if rows and len(rows) >= 70:
            logger.info(f"[OHLCV] ccxt-live OK — {len(rows)} candles "
                        f"last={rows[-1][4]:.2f}")
            return rows
    except Exception as e:
        logger.warning(f"[OHLCV] ccxt-live failed: {e}")

    # ── 2. direct HTTP — limit only, no from/to ────────────────────────────────
    sym_id  = SYMBOL_ID or "ETHUSDT"
    sym_inv = sym_id.replace("USDT", "USD")   # e.g. ETHUSD for inverse fallback

    for url, s_id in [
        (f"{LIVE_BASE}/md/v2/kline/list", sym_id),
        (f"{LIVE_BASE}/md/kline",          sym_id),
        (f"{LIVE_BASE}/md/kline",          sym_inv),
    ]:
        try:
            params = {"symbol": s_id, "resolution": RESOLUTION, "limit": limit}
            r      = req.get(url, params=params, timeout=10)
            logger.info(f"[OHLCV] HTTP {url} {s_id} "
                        f"status={r.status_code} body={r.text[:100]}")
            d = r.json()
            if not isinstance(d, dict):
                continue
            # Dig for rows list
            rows = (d.get("result") or {}).get("rows") or \
                   (d.get("data")   or {}).get("klines") or []
            if not isinstance(rows, list) or len(rows) < 70:
                continue
            # Auto-scale prices (Phemex multiplies by 10^priceScale)
            candles = []
            for row in rows:
                close_raw = float(row[5])
                scale     = 1.0
                while close_raw > 1_000_000:
                    close_raw /= 10.0
                    scale     *= 10.0
                candles.append([
                    int(row[0]) * 1000,
                    float(row[2]) / scale,
                    float(row[3]) / scale,
                    float(row[4]) / scale,
                    float(row[5]) / scale,
                    float(row[6]),
                ])
            if len(candles) >= 70:
                logger.info(f"[OHLCV] HTTP OK {url} {s_id} "
                            f"— {len(candles)} candles "
                            f"last={candles[-1][4]:.2f}")
                return candles
        except Exception as e:
            logger.warning(f"[OHLCV] HTTP {url} failed: {e}")

    # ── 3. aligned from+to fallback ────────────────────────────────────────────
    now     = int(time_mod.time())
    from_ts = (now - limit * RESOLUTION) // RESOLUTION * RESOLUTION  # aligned
    to_ts   = now // RESOLUTION * RESOLUTION

    for url, s_id, keys in [
        (f"{LIVE_BASE}/md/v2/kline/list", sym_id,  ["result", "rows"]),
        (f"{LIVE_BASE}/md/kline",          sym_inv, ["data",   "klines"]),
    ]:
        try:
            params = {"symbol": s_id, "resolution": RESOLUTION,
                      "from": from_ts, "to": to_ts}
            r      = req.get(url, params=params, timeout=10)
            logger.info(f"[OHLCV] aligned {url} {s_id} "
                        f"status={r.status_code} body={r.text[:100]}")
            d    = r.json()
            node = d
            for k in keys:
                node = node.get(k) if isinstance(node, dict) else None
                if node is None:
                    break
            if isinstance(node, list) and len(node) >= 70:
                candles = []
                for row in node:
                    c = float(row[5])
                    sc = 1.0
                    while c > 1_000_000:
                        c /= 10.0; sc *= 10.0
                    candles.append([int(row[0])*1000,
                                    float(row[2])/sc, float(row[3])/sc,
                                    float(row[4])/sc, float(row[5])/sc,
                                    float(row[6])])
                if len(candles) >= 70:
                    return candles
        except Exception as e:
            logger.warning(f"[OHLCV] aligned fallback failed: {e}")

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

# ── Trading helpers (paper_ex = testnet) ───────────────────────────────────────
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
        state["status"]  = f"Error: {e}"
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
a{color:#58a6ff;text-decoration:none}
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
<footer>AZLEMA · Phemex Paper Testnet · 30 m · ETH/USDT · 1× Leverage
  &nbsp;·&nbsp;<a href="/debug" target="_blank">Debug</a>
</footer>
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

@app.route("/debug")
def debug():
    """Raw probe of every Phemex OHLCV endpoint variant."""
    now     = int(time_mod.time())
    from_al = now // RESOLUTION * RESOLUTION - 5 * RESOLUTION
    to_al   = now // RESOLUTION * RESOLUTION
    sym     = SYMBOL_ID or "ETHUSDT"
    out     = {}

    probes = [
        ("ccxt-live fetch_ohlcv limit=5", None, None),
        (f"LIVE v2 {sym} limit",
         f"{LIVE_BASE}/md/v2/kline/list",
         {"symbol": sym,      "resolution": RESOLUTION, "limit": 5}),
        (f"LIVE v2 {sym} from+to aligned",
         f"{LIVE_BASE}/md/v2/kline/list",
         {"symbol": sym,      "resolution": RESOLUTION,
          "from": from_al, "to": to_al}),
        (f"LIVE v1 {sym} limit",
         f"{LIVE_BASE}/md/kline",
         {"symbol": sym,      "resolution": RESOLUTION, "limit": 5}),
        (f"LIVE v1 ETHUSD limit",
         f"{LIVE_BASE}/md/kline",
         {"symbol": "ETHUSD", "resolution": RESOLUTION, "limit": 5}),
        (f"LIVE v1 ETHUSD from+to aligned",
         f"{LIVE_BASE}/md/kline",
         {"symbol": "ETHUSD", "resolution": RESOLUTION,
          "from": from_al, "to": to_al}),
        (f"TESTNET v2 {sym} limit",
         f"{TESTNET_BASE}/md/v2/kline/list",
         {"symbol": sym,      "resolution": RESOLUTION, "limit": 5}),
    ]

    for label, url, params in probes:
        if url is None:   # ccxt probe
            try:
                rows = live_ex.fetch_ohlcv(SYMBOL or "ETH/USDT:USDT", "30m", limit=5)
                out[label] = {"ok": True, "count": len(rows),
                              "last": rows[-1] if rows else None}
            except Exception as e:
                out[label] = {"error": str(e)}
        else:
            try:
                r = req.get(url, params=params, timeout=8)
                out[label] = {"http": r.status_code, "body": r.text[:300]}
            except Exception as e:
                out[label] = {"error": str(e)}

    return jsonify(out)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT)
