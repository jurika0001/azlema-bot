#!/usr/bin/env python3
"""
Phemex Paper Trading Bot — Adaptive Zero Lag EMA
- Dados reais de candles 30m da Phemex via API oficial
- Paper trading 100% interno (saldo simulado, sem ordens reais)
- Nenhuma ordem é enviada à exchange
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
# ENV VARS — apenas estas duas no Render
# ─────────────────────────────────────────────────────────────
API_KEY    = os.environ.get("PHEMEX_API_KEY", "")
API_SECRET = os.environ.get("PHEMEX_API_SECRET", "")
PORT       = int(os.environ.get("PORT", 8080))

# ─────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────
SYMBOL          = "ETH/USDT:USDT"   # perpetual linear da Phemex
TIMEFRAME       = "30m"
LEVERAGE        = 1
RISK_PCT        = 0.01              # 1% do saldo por trade
SL_TICKS        = 2000              # pontos de stop loss
GAIN_LIMIT      = 50                # busca ±500 em passos de 0.1
PERIOD          = 20
THRESHOLD       = 0.0
INITIAL_BALANCE = 1000.0            # USDT simulado inicial

# ─────────────────────────────────────────────────────────────
# PAPER ENGINE — saldo e posição simulados
# ─────────────────────────────────────────────────────────────
class PaperEngine:
    def __init__(self, initial_balance: float):
        self.balance       = initial_balance   # USDT livre simulado
        self.position_side = "FLAT"            # FLAT | LONG | SHORT
        self.position_qty  = 0.0               # quantidade em contratos
        self.entry_price   = 0.0
        self.realized_pnl  = 0.0
        self._lock         = threading.Lock()

    def snapshot(self):
        with self._lock:
            return {
                "balance":       round(self.balance, 4),
                "position_side": self.position_side,
                "position_qty":  round(self.position_qty, 6),
                "entry_price":   round(self.entry_price, 4),
                "realized_pnl":  round(self.realized_pnl, 4),
            }

    def unrealized_pnl(self, current_price: float) -> float:
        if self.position_side == "FLAT" or self.position_qty == 0:
            return 0.0
        if self.position_side == "LONG":
            return (current_price - self.entry_price) * self.position_qty
        return (self.entry_price - current_price) * self.position_qty

    def open_long(self, price: float, qty: float) -> dict:
        with self._lock:
            # Fecha short se houver
            if self.position_side == "SHORT":
                pnl = (self.entry_price - price) * self.position_qty
                self.realized_pnl += pnl
                self.balance      += pnl
                log.info(f"Close SHORT → PnL {pnl:+.4f} USDT")
            cost = price * qty / LEVERAGE
            self.balance      -= cost
            self.position_side = "LONG"
            self.position_qty  = qty
            self.entry_price   = price
            return self._trade_record("LONG", price, qty)

    def open_short(self, price: float, qty: float) -> dict:
        with self._lock:
            # Fecha long se houver
            if self.position_side == "LONG":
                pnl = (price - self.entry_price) * self.position_qty
                self.realized_pnl += pnl
                self.balance      += pnl
                log.info(f"Close LONG → PnL {pnl:+.4f} USDT")
            cost = price * qty / LEVERAGE
            self.balance      -= cost
            self.position_side = "SHORT"
            self.position_qty  = qty
            self.entry_price   = price
            return self._trade_record("SHORT", price, qty)

    def _trade_record(self, side, price, qty):
        return {
            "time":     datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
            "side":     side,
            "price":    round(price, 4),
            "qty":      round(qty, 6),
            "entry":    round(self.entry_price, 4),
            "pnl_real": round(self.realized_pnl, 4),
            "balance":  round(self.balance, 4),
        }


paper = PaperEngine(INITIAL_BALANCE)

# ─────────────────────────────────────────────────────────────
# SHARED STATE
# ─────────────────────────────────────────────────────────────
_lock         = threading.Lock()
trade_history = deque(maxlen=200)
ka_log        = deque(maxlen=40)

status = {
    "running":    False,
    "signal":     "—",
    "ec":         0.0,
    "ema":        0.0,
    "last_price": 0.0,
    "last_check": "—",
    "errors":     deque(maxlen=10),
}

# ─────────────────────────────────────────────────────────────
# PHEMEX — apenas leitura de candles (sem ordens)
# ─────────────────────────────────────────────────────────────
def build_exchange():
    """
    Conecta à Phemex real (não testnet) apenas para buscar OHLCV.
    Nenhuma ordem é enviada.
    """
    ex = ccxt.phemex({
        "apiKey":          API_KEY,
        "secret":          API_SECRET,
        "enableRateLimit": True,
        "options":         {"defaultType": "swap"},
    })
    # SEM sandbox — dados reais de mercado
    return ex

# ─────────────────────────────────────────────────────────────
# AZLEMA
# ─────────────────────────────────────────────────────────────
def azlema(closes: list, period: int = PERIOD, gain_limit: int = GAIN_LIMIT):
    arr = np.array(closes, dtype=np.float64)
    n   = len(arr)
    if n < period + 10:
        return None, None, None

    alpha = 2.0 / (period + 1)

    ema    = np.empty(n)
    ema[0] = arr[0]
    for i in range(1, n):
        ema[i] = alpha * arr[i] + (1 - alpha) * ema[i - 1]

    best_gain   = 0.0
    least_error = 1e18
    for g_int in range(-gain_limit * 10, gain_limit * 10 + 1):
        g    = g_int / 10.0
        ec   = np.empty(n)
        ec[0]= arr[0]
        for i in range(1, n):
            ec[i] = alpha * (ema[i] + g * (arr[i] - ec[i - 1])) + (1 - alpha) * ec[i - 1]
        err = abs(arr[-1] - ec[-1])
        if err < least_error:
            least_error = err
            best_gain   = g

    ec    = np.empty(n)
    ec[0] = arr[0]
    for i in range(1, n):
        ec[i] = alpha * (ema[i] + best_gain * (arr[i] - ec[i - 1])) + (1 - alpha) * ec[i - 1]

    err_pct = 100.0 * least_error / arr[-1] if arr[-1] != 0 else 0.0
    return float(ec[-1]), float(ema[-1]), err_pct

# ─────────────────────────────────────────────────────────────
# TIMING
# ─────────────────────────────────────────────────────────────
def seconds_to_next_30m() -> float:
    now  = datetime.now(timezone.utc)
    secs = now.minute * 60 + now.second + now.microsecond / 1e6
    wait = (1800 - secs) if secs < 1800 else (3600 - secs)
    return max(wait + 5, 1)

# ─────────────────────────────────────────────────────────────
# TRADING LOOP
# ─────────────────────────────────────────────────────────────
def trading_loop():
    status["running"] = True
    log.info("Trading loop iniciado — aguardando primeiro candle de 30m")

    while True:
        wait = seconds_to_next_30m()
        log.info(f"Próximo candle em {wait:.0f}s")
        time.sleep(wait)

        try:
            ex = build_exchange()

            # Busca 120 candles fechados (ignora o candle aberto atual [-1])
            ohlcv  = ex.fetch_ohlcv(SYMBOL, TIMEFRAME, limit=121)
            if not ohlcv or len(ohlcv) < 30:
                log.warning("Candles insuficientes recebidos da Phemex")
                continue

            closes = [c[4] for c in ohlcv[:-1]]   # exclui candle atual
            price  = float(closes[-1])

            # AZLEMA
            ec, ema, err_pct = azlema(closes)
            if ec is None:
                log.warning("Barras insuficientes para AZLEMA")
                continue

            signal = "LONG" if ec > ema else "SHORT"

            with _lock:
                status.update({
                    "ec":         round(ec, 4),
                    "ema":        round(ema, 4),
                    "signal":     signal,
                    "last_price": round(price, 4),
                    "last_check": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
                })

            snap    = paper.snapshot()
            cur_pos = snap["position_side"]
            balance = snap["balance"]

            # Calcula qty baseado em risco sobre saldo simulado
            mintick = 0.01   # ETH/USDT mintick padrão Phemex
            try:
                ex.load_markets()
                raw = ex.markets.get(SYMBOL, {}).get("precision", {}).get("price")
                if raw:
                    mintick = 10 ** (-int(raw)) if isinstance(raw, int) else float(raw)
            except Exception:
                pass

            sl_usdt = SL_TICKS * mintick
            qty     = max(round((RISK_PCT * balance) / sl_usdt, 6), 0.001)

            # Paper trade
            record = None
            if signal == "LONG" and cur_pos != "LONG":
                record = paper.open_long(price, qty)
                record["ec"]  = round(ec, 4)
                record["ema"] = round(ema, 4)
                log.info(f"PAPER LONG  {qty} @ {price}")

            elif signal == "SHORT" and cur_pos != "SHORT":
                record = paper.open_short(price, qty)
                record["ec"]  = round(ec, 4)
                record["ema"] = round(ema, 4)
                log.info(f"PAPER SHORT {qty} @ {price}")

            else:
                log.info(f"Hold {cur_pos} | EC={ec:.4f} EMA={ema:.4f} price={price}")

            if record:
                with _lock:
                    trade_history.appendleft(record)

        except Exception as exc:
            msg = str(exc)[:200]
            log.error(f"Erro no bot: {msg}")
            with _lock:
                status["errors"].appendleft({
                    "time": datetime.now(timezone.utc).strftime("%H:%M:%S"),
                    "msg":  msg,
                })
            time.sleep(30)   # back-off em caso de erro

# ─────────────────────────────────────────────────────────────
# INTERNAL KEEPALIVE — 3 threads: 8s / 15s / 23s
# ─────────────────────────────────────────────────────────────
def _ka_worker(interval: int, name: str):
    time.sleep(interval + 3)
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
    snap = paper.snapshot()
    return jsonify({
        "status":   "healthy",
        "running":  status["running"],
        "position": snap["position_side"],
        "signal":   status["signal"],
        "balance":  snap["balance"],
    })

@app.route("/")
def dashboard():
    snap = paper.snapshot()
    with _lock:
        s_copy = dict(status)
        s_copy["errors"] = list(status["errors"])
        trades = list(trade_history)
        ka     = list(ka_log)

    # P&L não realizado
    unreal = round(paper.unrealized_pnl(status["last_price"]), 4)
    equity = round(snap["balance"] + max(unreal, 0), 4)

    return render_template_string(
        _HTML,
        s=s_copy,
        p=snap,
        unreal=unreal,
        equity=equity,
        trades=trades,
        ka=ka,
    )

# ─────────────────────────────────────────────────────────────
# DASHBOARD HTML
# ─────────────────────────────────────────────────────────────
_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="refresh" content="30">
<title>AZLEMA Paper Bot — Phemex</title>
<link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet">
<style>
  body{background:#0d1117;color:#e6edf3;font-family:system-ui,sans-serif}
  .card{background:#161b22;border:1px solid #30363d;border-radius:8px}
  .bl{background:#238636;color:#fff}.bs{background:#da3633;color:#fff}
  .bf{background:#444c56;color:#fff}
  .bp{color:#3fb950}.bn{color:#f85149}.bz{color:#8b949e}
  th{color:#8b949e;font-size:.73rem;text-transform:uppercase;font-weight:500;border-color:#30363d!important}
  td{font-size:.82rem;border-color:#21262d!important;vertical-align:middle}
  .ka{font-size:.73rem;color:#8b949e;font-family:monospace;line-height:1.8}
  .er{font-size:.73rem;color:#f85149;font-family:monospace;line-height:1.8}
  .badge-pill{border-radius:20px;padding:.25em .75em}
  .table-dark{--bs-table-bg:#161b22;--bs-table-hover-bg:#1c2128}
  .mono{font-family:monospace}
</style>
</head>
<body>
<div class="container-fluid py-3 px-4">

  <!-- Header -->
  <div class="d-flex align-items-center mb-3 flex-wrap gap-2">
    <h5 class="mb-0 me-2">⚡ AZLEMA Paper Bot</h5>
    <span class="badge {{'bg-success' if s.running else 'bg-danger'}} badge-pill">
      {{'● RUNNING' if s.running else '○ STOPPED'}}</span>
    <span class="badge bg-secondary badge-pill">Phemex · ETH/USDT · 30m · 1×</span>
    <span class="badge bg-secondary badge-pill">Paper Trading</span>
    <small class="text-secondary ms-auto">auto-refresh 30s | last: {{s.last_check}}</small>
  </div>

  <!-- Stats row -->
  <div class="row g-2 mb-3">
    {% set pos_cls = 'bl' if p.position_side=='LONG' else ('bs' if p.position_side=='SHORT' else 'bf') %}
    {% set sig_cls = 'bl' if s.signal=='LONG' else ('bs' if s.signal=='SHORT' else 'bf') %}
    <div class="col-6 col-sm-4 col-md-2">
      <div class="card p-3 text-center h-100">
        <div class="text-secondary small mb-1">Position</div>
        <span class="badge {{pos_cls}} fs-6 badge-pill">{{p.position_side}}</span>
      </div>
    </div>
    <div class="col-6 col-sm-4 col-md-2">
      <div class="card p-3 text-center h-100">
        <div class="text-secondary small mb-1">Signal</div>
        <span class="badge {{sig_cls}} fs-6 badge-pill">{{s.signal}}</span>
      </div>
    </div>
    <div class="col-6 col-sm-4 col-md-2">
      <div class="card p-3 text-center h-100">
        <div class="text-secondary small mb-1">Balance (sim)</div>
        <div class="fw-bold mono">${{p.balance}}</div>
      </div>
    </div>
    <div class="col-6 col-sm-4 col-md-2">
      <div class="card p-3 text-center h-100">
        <div class="text-secondary small mb-1">Realized P&L</div>
        <div class="fw-bold mono {{'bp' if p.realized_pnl >= 0 else 'bn'}}">
          {{'+' if p.realized_pnl >= 0 else ''}}${{p.realized_pnl}}
        </div>
      </div>
    </div>
    <div class="col-6 col-sm-4 col-md-2">
      <div class="card p-3 text-center h-100">
        <div class="text-secondary small mb-1">Unrealized P&L</div>
        <div class="fw-bold mono {{'bp' if unreal >= 0 else ('bn' if unreal < 0 else 'bz')}}">
          {{'+' if unreal > 0 else ''}}${{unreal}}
        </div>
      </div>
    </div>
    <div class="col-6 col-sm-4 col-md-2">
      <div class="card p-3 text-center h-100">
        <div class="text-secondary small mb-1">Last Price</div>
        <div class="fw-bold mono">${{s.last_price}}</div>
        <div class="text-secondary" style="font-size:.7rem">
          EC {{s.ec}} / EMA {{s.ema}}
        </div>
      </div>
    </div>
  </div>

  <div class="row g-3">
    <!-- Trade history -->
    <div class="col-lg-8">
      <div class="card p-3">
        <div class="d-flex align-items-center mb-3">
          <h6 class="text-secondary mb-0">Trade History</h6>
          <span class="badge bg-secondary ms-2">{{trades|length}}</span>
          <small class="text-secondary ms-auto">simulado — sem ordens reais</small>
        </div>
        {% if trades %}
        <div style="max-height:480px;overflow-y:auto">
        <table class="table table-dark table-hover table-sm mb-0">
          <thead><tr>
            <th>Time (UTC)</th><th>Side</th><th>Price</th>
            <th>Qty</th><th>EC</th><th>EMA</th>
            <th>Realized P&L</th><th>Balance</th>
          </tr></thead>
          <tbody>
          {% for t in trades %}
          <tr>
            <td class="mono" style="font-size:.72rem">{{t.time}}</td>
            <td><span class="badge {{'bl' if t.side=='LONG' else 'bs'}} badge-pill">{{t.side}}</span></td>
            <td class="mono">${{t.price}}</td>
            <td class="mono">{{t.qty}}</td>
            <td class="mono">{{t.ec}}</td>
            <td class="mono">{{t.ema}}</td>
            <td class="mono {{'bp' if t.pnl_real >= 0 else 'bn'}}">
              {{'+' if t.pnl_real >= 0 else ''}}${{t.pnl_real}}
            </td>
            <td class="mono">${{t.balance}}</td>
          </tr>
          {% endfor %}
          </tbody>
        </table>
        </div>
        {% else %}
        <div class="text-center py-5">
          <div class="text-secondary mb-2" style="font-size:2rem">⏳</div>
          <div class="text-secondary">Aguardando primeiro fechamento de candle 30m...</div>
          <div class="text-secondary small mt-1">
            Candles buscados direto da Phemex em tempo real
          </div>
        </div>
        {% endif %}
      </div>
    </div>

    <!-- Side panel -->
    <div class="col-lg-4">
      <div class="card p-3 mb-3">
        <h6 class="text-secondary mb-2">Keepalive interno</h6>
        <div class="text-secondary small mb-2">8s · 15s · 23s</div>
        {% for k in ka %}<div class="ka">{{k}}</div>
        {% else %}<div class="text-secondary small">Iniciando...</div>{% endfor %}
      </div>

      <div class="card p-3 mb-3">
        <h6 class="text-secondary mb-2">Posição atual</h6>
        <table class="table table-dark table-sm mb-0">
          <tr><td class="text-secondary">Lado</td>
              <td class="text-end mono">{{p.position_side}}</td></tr>
          <tr><td class="text-secondary">Qty</td>
              <td class="text-end mono">{{p.position_qty}} ETH</td></tr>
          <tr><td class="text-secondary">Entry</td>
              <td class="text-end mono">${{p.entry_price}}</td></tr>
          <tr><td class="text-secondary">Saldo livre</td>
              <td class="text-end mono">${{p.balance}}</td></tr>
        </table>
      </div>

      {% if s.errors %}
      <div class="card p-3">
        <h6 class="text-danger mb-2">Erros recentes</h6>
        {% for e in s.errors %}
        <div class="er">[{{e.time}}] {{e.msg}}</div>{% endfor %}
      </div>{% endif %}
    </div>
  </div>
</div>
</body>
</html>"""

# ─────────────────────────────────────────────────────────────
# STARTUP
# ─────────────────────────────────────────────────────────────
_started = False

def _start_background():
    global _started
    if _started:
        return
    _started = True
    for interval, name in [(8, "KA-8s"), (15, "KA-15s"), (23, "KA-23s")]:
        threading.Thread(
            target=_ka_worker, args=(interval, name), daemon=True
        ).start()
    threading.Thread(target=trading_loop, daemon=True).start()
    log.info("Threads iniciadas: trading + 3x keepalive")

_start_background()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT, debug=False)
