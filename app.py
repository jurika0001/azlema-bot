#!/usr/bin/env python3
"""
Phemex Paper Trading Bot — Adaptive Zero Lag EMA
30-minute timeframe | Paper Futures | 1x Leverage
"""

import os
import time
import threading
import logging
from datetime import datetime, timezone
from collections import deque

import numpy as np
import requests
import ccxt
from flask import Flask, jsonify, render_template_string

# ─────────────────────────────────────────────────────────────
# SETUP
# ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

app = Flask(__name__)

# ─────────────────────────────────────────────────────────────
# ENV VARS — only these two go in Render's environment settings
# ─────────────────────────────────────────────────────────────
API_KEY    = os.environ.get("PHEMEX_API_KEY", "")
API_SECRET = os.environ.get("PHEMEX_API_SECRET", "")
PORT       = int(os.environ.get("PORT", 8080))

# ─────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────
SYMBOL     = "ETHUSDT"
TIMEFRAME  = "30m"
LEVERAGE   = 1
RISK_PCT   = 0.01
SL_TICKS   = 2000
GAIN_LIMIT = 50
PERIOD     = 20
THRESHOLD  = 0.0

# ─────────────────────────────────────────────────────────────
# SHARED STATE
# ─────────────────────────────────────────────────────────────
_lock         = threading.Lock()
trade_history = deque(maxlen=200)
ka_log        = deque(maxlen=40)

status = {
    "running":    False,
    "position":   "FLAT",
    "signal":     "—",
    "ec":         0.0,
    "ema":        0.0,
    "last_check": "—",
    "balance":    0.0,
    "errors":     deque(maxlen=10),
}

# ─────────────────────────────────────────────────────────────
# PHEMEX PAPER (TESTNET)
# ─────────────────────────────────────────────────────────────
def build_exchange():
    ex = ccxt.phemex({
        "apiKey":          API_KEY,
        "secret":          API_SECRET,
        "enableRateLimit": True,
        "options":         {"defaultType": "swap"},
    })
    ex.set_sandbox_mode(True)   # uses testnet-api.phemex.com
    return ex

# ─────────────────────────────────────────────────────────────
# AZLEMA CALCULATION
# ─────────────────────────────────────────────────────────────
def azlema(closes: list, period: int = PERIOD, gain_limit: int = GAIN_LIMIT):
    """
    Returns (ec, ema, least_error_pct).
    EC > EMA → LONG  |  EC < EMA → SHORT
    """
    arr = np.array(closes, dtype=np.float64)
    n   = len(arr)
    if n < period + 10:
        return None, None, None

    alpha = 2.0 / (period + 1)

    # EMA
    ema = np.empty(n)
    ema[0] = arr[0]
    for i in range(1, n):
        ema[i] = alpha * arr[i] + (1 - alpha) * ema[i - 1]

    # Best Gain search
    best_gain   = 0.0
    least_error = 1e18
    for g_int in range(-gain_limit * 10, gain_limit * 10 + 1):
        g  = g_int / 10.0
        ec = np.empty(n)
        ec[0] = arr[0]
        for i in range(1, n):
            ec[i] = alpha * (ema[i] + g * (arr[i] - ec[i - 1])) + (1 - alpha) * ec[i - 1]
        err = abs(arr[-1] - ec[-1])
        if err < least_error:
            least_error = err
            best_gain   = g

    # Final EC with best gain
    ec = np.empty(n)
    ec[0] = arr[0]
    for i in range(1, n):
        ec[i] = alpha * (ema[i] + best_gain * (arr[i] - ec[i - 1])) + (1 - alpha) * ec[i - 1]

    err_pct = 100.0 * least_error / arr[-1] if arr[-1] != 0 else 0.0
    return float(ec[-1]), float(ema[-1]), err_pct

# ─────────────────────────────────────────────────────────────
# TIMING — wait until next 30m candle close
# ─────────────────────────────────────────────────────────────
def seconds_to_next_30m() -> float:
    now  = datetime.now(timezone.utc)
    secs = now.minute * 60 + now.second + now.microsecond / 1e6
    wait = (1800 - secs) if secs < 1800 else (3600 - secs)
    return wait + 5   # +5 s buffer so candle is confirmed

# ─────────────────────────────────────────────────────────────
# TRADING LOOP
# ─────────────────────────────────────────────────────────────
def _record(side, price, qty, ec, ema, order):
    entry = {
        "time":     datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
        "side":     side,
        "price":    round(price, 4),
        "qty":      round(qty, 4),
        "ec":       round(ec, 4),
        "ema":      round(ema, 4),
        "order_id": (order or {}).get("id", "—"),
    }
    with _lock:
        trade_history.appendleft(entry)
    log.info(f"TRADE ▶ {side} {qty} @ {price}")


def trading_loop():
    status["running"] = True
    log.info("Trading loop started — waiting for first 30m candle")

    while True:
        wait = seconds_to_next_30m()
        log.info(f"Next candle in {wait:.0f}s")
        time.sleep(wait)

        try:
            ex = build_exchange()
            ex.load_markets()

            # Set 1x leverage — Phemex exige leverageEr em escala interna
            try:
                ex.set_leverage(LEVERAGE, SYMBOL, params={
                    "leverageEr": LEVERAGE * 10000,
                })
            except Exception as e:
                log.warning(f"set_leverage ignorado: {e}")

            # Fetch last 120 closed 30m candles (exclude live bar)
            ohlcv  = ex.fetch_ohlcv(SYMBOL, TIMEFRAME, limit=121)
            closes = [c[4] for c in ohlcv[:-1]]
            price  = float(closes[-1])

            # AZLEMA
            ec, ema, err_pct = azlema(closes)
            if ec is None:
                log.warning("Not enough bars for AZLEMA")
                continue

            signal = "LONG" if ec > ema else "SHORT"
            with _lock:
                status.update({
                    "ec":         round(ec, 4),
                    "ema":        round(ema, 4),
                    "signal":     signal,
                    "last_check": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
                })

            # Current position — comparacao flexivel de simbolo
            positions = ex.fetch_positions([SYMBOL])
            cur_side  = "FLAT"
            cur_qty   = 0.0
            for p in positions:
                sym = p.get("symbol") or ""
                if SYMBOL in sym or sym in SYMBOL:
                    contracts = float(p.get("contracts") or 0)
                    if contracts > 0:
                        cur_side = "LONG" if p.get("side") == "long" else "SHORT"
                        cur_qty  = contracts

            with _lock:
                status["position"] = cur_side

            # Balance & quantity
            bal_info = ex.fetch_balance()
            usdt     = bal_info.get("USDT") or bal_info.get("usdt") or {}
            balance  = float(usdt.get("free") or usdt.get("total") or 1000)
            with _lock:
                status["balance"] = round(balance, 2)

            market  = ex.markets.get(SYMBOL, {})
            mintick = market.get("precision", {}).get("price", 0.01)
            if isinstance(mintick, int):
                mintick = 10 ** (-mintick)
            sl_usdt = max(SL_TICKS * float(mintick), 0.01)
            qty     = max(round((RISK_PCT * balance) / sl_usdt, 3), 0.001)

            # Parametros obrigatorios para ordens no Phemex
            order_params = {"timeInForce": "ImmediateOrCancel"}

            # Execute — fecha posicao oposta antes de abrir nova (one-way mode)
            if signal == "LONG" and cur_side != "LONG":
                if cur_side == "SHORT":
                    ex.create_market_order(SYMBOL, "buy", cur_qty,
                                           params={**order_params, "reduceOnly": True})
                order = ex.create_market_order(SYMBOL, "buy", qty, params=order_params)
                _record("LONG", price, qty, ec, ema, order)
                with _lock:
                    status["position"] = "LONG"

            elif signal == "SHORT" and cur_side != "SHORT":
                if cur_side == "LONG":
                    ex.create_market_order(SYMBOL, "sell", cur_qty,
                                           params={**order_params, "reduceOnly": True})
                order = ex.create_market_order(SYMBOL, "sell", qty, params=order_params)
                _record("SHORT", price, qty, ec, ema, order)
                with _lock:
                    status["position"] = "SHORT"

            else:
                log.info(f"Hold {cur_side} | EC={ec:.4f} EMA={ema:.4f}")

        except Exception as exc:
            msg = str(exc)[:160]
            log.error(f"Bot error: {msg}")
            with _lock:
                status["errors"].appendleft({
                    "time": datetime.now().strftime("%H:%M:%S"),
                    "msg":  msg,
                })

# ─────────────────────────────────────────────────────────────
# INTERNAL KEEPALIVE — 3 threads: 8 s / 15 s / 23 s
# ─────────────────────────────────────────────────────────────
def _ka_worker(interval: int, name: str):
    time.sleep(interval + 3)    # stagger startup
    while True:
        time.sleep(interval)
        try:
            r   = requests.get(f"http://localhost:{PORT}/ping", timeout=5)
            msg = f"[{name}] OK {r.status_code} @ {datetime.now().strftime('%H:%M:%S')}"
        except Exception as e:
            msg = f"[{name}] FAIL @ {datetime.now().strftime('%H:%M:%S')}: {e}"
        with _lock:
            ka_log.appendleft(msg)

# ─────────────────────────────────────────────────────────────
# FLASK ROUTES
# ─────────────────────────────────────────────────────────────
@app.route("/ping")
def ping():
    return jsonify({"ok": True, "ts": datetime.utcnow().isoformat()})

@app.route("/health")
def health():
    return jsonify({
        "status":   "healthy",
        "running":  status["running"],
        "position": status["position"],
        "signal":   status["signal"],
    })

@app.route("/")
def dashboard():
    with _lock:
        snap          = dict(status)
        snap["errors"] = list(status["errors"])
        trades        = list(trade_history)
        ka            = list(ka_log)
    return render_template_string(_HTML, s=snap, trades=trades, ka=ka)

# ─────────────────────────────────────────────────────────────
# DASHBOARD HTML
# ─────────────────────────────────────────────────────────────
_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="refresh" content="30">
<title>AZLEMA Bot — Phemex Paper</title>
<link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet">
<style>
  body{background:#0d1117;color:#e6edf3}
  .card{background:#161b22;border:1px solid #30363d}
  .bl{background:#238636}.bs{background:#da3633}.bf{background:#6e7681}
  th{color:#8b949e;font-size:.78rem;text-transform:uppercase;font-weight:500}
  td{font-size:.85rem}
  .ka{font-size:.75rem;color:#8b949e;font-family:monospace;line-height:1.6}
  .er{font-size:.75rem;color:#f85149;font-family:monospace;line-height:1.6}
</style>
</head>
<body>
<div class="container-fluid py-3 px-4">
  <div class="d-flex align-items-center mb-3 flex-wrap gap-2">
    <h5 class="mb-0 me-2">⚡ AZLEMA Bot</h5>
    <span class="badge {{'bg-success' if s.running else 'bg-danger'}}">
      {{'RUNNING' if s.running else 'STOPPED'}}</span>
    <small class="text-secondary">Phemex Paper · ETH/USDT · 30m · 1×</small>
    <small class="text-secondary ms-auto">auto-refresh 30s</small>
  </div>

  <div class="row g-3 mb-3">
    {% for label, value, cls in [
        ('Position', s.position, 'bl' if s.position=='LONG' else ('bs' if s.position=='SHORT' else 'bf')),
        ('Signal',   s.signal,   'bl' if s.signal=='LONG'   else ('bs' if s.signal=='SHORT'   else 'bg-secondary')),
    ] %}
    <div class="col-6 col-md-2">
      <div class="card p-3 text-center h-100">
        <div class="text-secondary small mb-1">{{label}}</div>
        <span class="badge {{cls}} fs-6">{{value}}</span>
      </div>
    </div>
    {% endfor %}
    {% for label, value in [('EC', s.ec), ('EMA', s.ema), ('Balance USDT', s.balance)] %}
    <div class="col-6 col-md-2">
      <div class="card p-3 text-center h-100">
        <div class="text-secondary small mb-1">{{label}}</div>
        <div class="fw-bold">{{value}}</div>
      </div>
    </div>
    {% endfor %}
    <div class="col-12 col-md-2">
      <div class="card p-3 text-center h-100">
        <div class="text-secondary small mb-1">Last Check</div>
        <div style="font-size:.75rem">{{s.last_check}}</div>
      </div>
    </div>
  </div>

  <div class="row g-3">
    <div class="col-lg-8">
      <div class="card p-3">
        <h6 class="text-secondary mb-3">Trade History ({{trades|length}})</h6>
        {% if trades %}
        <div style="max-height:460px;overflow-y:auto">
        <table class="table table-dark table-hover table-sm mb-0">
          <thead><tr>
            <th>Time (UTC)</th><th>Side</th><th>Price</th>
            <th>Qty</th><th>EC</th><th>EMA</th><th>Order ID</th>
          </tr></thead>
          <tbody>
          {% for t in trades %}
          <tr>
            <td>{{t.time}}</td>
            <td><span class="badge {{'bl' if t.side=='LONG' else 'bs'}}">{{t.side}}</span></td>
            <td>{{t.price}}</td><td>{{t.qty}}</td>
            <td>{{t.ec}}</td><td>{{t.ema}}</td>
            <td class="text-secondary" style="font-size:.7rem">{{t.order_id}}</td>
          </tr>
          {% endfor %}
          </tbody>
        </table>
        </div>
        {% else %}
        <div class="text-secondary text-center py-5">
          No trades yet — waiting for first 30m candle close.
        </div>
        {% endif %}
      </div>
    </div>

    <div class="col-lg-4">
      <div class="card p-3 mb-3">
        <h6 class="text-secondary mb-2">Keepalive (8s / 15s / 23s)</h6>
        {% for k in ka %}<div class="ka">{{k}}</div>
        {% else %}<div class="text-secondary small">Starting...</div>{% endfor %}
      </div>
      {% if s.errors %}
      <div class="card p-3">
        <h6 class="text-danger mb-2">Errors</h6>
        {% for e in s.errors %}
        <div class="er">[{{e.time}}] {{e.msg}}</div>{% endfor %}
      </div>{% endif %}
    </div>
  </div>
</div>
</body>
</html>"""

# ─────────────────────────────────────────────────────────────
# STARTUP — runs once at module import (works with gunicorn too)
# ─────────────────────────────────────────────────────────────
_started = False

def _start_background():
    global _started
    if _started:
        return
    _started = True
    for interval, name in [(8, "KA-8s"), (15, "KA-15s"), (23, "KA-23s")]:
        threading.Thread(target=_ka_worker, args=(interval, name), daemon=True).start()
    threading.Thread(target=trading_loop, daemon=True).start()
    log.info("Background threads started (trading + 3x keepalive)")

_start_background()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT, debug=False)
