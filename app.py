#!/usr/bin/env python3
"""
Phemex Paper Trading Bot — Adaptive Zero Lag EMA
Correção definitiva: probe automático de endpoints + limites corretos
"""

import os
import time
import threading
import logging
from datetime import datetime, timezone
from collections import deque

import numpy as np
import requests
from flask import Flask, jsonify, render_template_string

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
PHEMEX_BASE     = "https://api.phemex.com"
RESOLUTION      = 1800      # 30 min
BARS_NEEDED     = 120       # barras fechadas necessárias

LEVERAGE        = 1
RISK_PCT        = 0.01
SL_TICKS        = 2000
MINTICK         = 0.01
GAIN_LIMIT      = 50
PERIOD          = 20
INITIAL_BALANCE = 1000.0

# ─────────────────────────────────────────────────────────────
# ENDPOINTS — ordenados por prioridade
#
# REGRA CRÍTICA:
#   • Endpoints /exchange/public/... → REST API legítima
#   • Endpoints /md/...              → WebSocket proxy, NÃO usar via REST
#
# limit=121 causa code 30000. Máximo real ≈ 100.
# Fallback: from/to com /exchange/public/md/kline (confirmado funcionando)
# ─────────────────────────────────────────────────────────────
def _build_attempts():
    now = int(time.time())
    frm = now - RESOLUTION * (BARS_NEEDED + 5)

    return [
        # ── Hedged Perpetual USDT-M (preferido) ──────────────────
        {
            "label":    "v2/kline/last ETHUSDT limit=100",
            "url":      f"{PHEMEX_BASE}/exchange/public/md/v2/kline/last",
            "params":   {"symbol": "ETHUSDT", "resolution": RESOLUTION, "limit": 100},
            "scale":    1.0,
            "parser":   "rows",
        },
        {
            "label":    "v2/kline/last ETHUSDT limit=60",
            "url":      f"{PHEMEX_BASE}/exchange/public/md/v2/kline/last",
            "params":   {"symbol": "ETHUSDT", "resolution": RESOLUTION, "limit": 60},
            "scale":    1.0,
            "parser":   "rows",
        },
        {
            "label":    "v2/kline ETHUSDT limit=100",
            "url":      f"{PHEMEX_BASE}/exchange/public/md/v2/kline",
            "params":   {"symbol": "ETHUSDT", "resolution": RESOLUTION, "limit": 100},
            "scale":    1.0,
            "parser":   "rows",
        },
        # ── Inverse/Contract ETHUSD ───────────────────────────────
        {
            "label":    "v2/kline/last ETHUSD limit=100",
            "url":      f"{PHEMEX_BASE}/exchange/public/md/v2/kline/last",
            "params":   {"symbol": "ETHUSD", "resolution": RESOLUTION, "limit": 100},
            "scale":    1e-4,
            "parser":   "rows",
        },
        {
            "label":    "v2/kline ETHUSD limit=100",
            "url":      f"{PHEMEX_BASE}/exchange/public/md/v2/kline",
            "params":   {"symbol": "ETHUSD", "resolution": RESOLUTION, "limit": 100},
            "scale":    1e-4,
            "parser":   "rows",
        },
        # ── From/To legado — confirmado funcionando (sem v2) ─────
        {
            "label":    "md/kline ETHUSD from/to",
            "url":      f"{PHEMEX_BASE}/exchange/public/md/kline",
            "params":   {"symbol": "ETHUSD", "resolution": RESOLUTION,
                         "from": frm, "to": now},
            "scale":    1e-4,
            "parser":   "klines",   # formato diferente: result.klines
        },
    ]


def _parse_rows(data: dict, scale: float) -> list:
    """Parser para formato REST v2: data.rows"""
    rows = data.get("data", {}).get("rows", [])
    if not rows:
        raise ValueError("rows vazio")
    # /last já exclui candle aberto; /kline inclui — descartamos o último
    closed = rows[:-1] if len(rows) > 1 else rows
    return [float(r[6]) * scale for r in closed]


def _parse_klines(data: dict, scale: float) -> list:
    """Parser para formato legado: result.klines"""
    result = data.get("result") or {}
    klines = result.get("klines") or result.get("rows") or []
    if not klines:
        raise ValueError("klines vazio")
    closed = klines[:-1] if len(klines) > 1 else klines
    return [float(r[6]) * scale for r in closed]


_active = None      # attempt dict que funcionou


def fetch_closes() -> list:
    global _active

    if _active:
        try:
            resp = requests.get(_active["url"], params=_active["params"], timeout=10)
            if resp.status_code == 200:
                data   = resp.json()
                code   = data.get("code")
                err    = (data.get("error") or {})
                if code == 0 or (code is None and err.get("code") is None):
                    parser = _parse_rows if _active["parser"] == "rows" else _parse_klines
                    closes = parser(data, _active["scale"])
                    if len(closes) >= PERIOD + 5:
                        return closes
            log.warning(f"Endpoint ativo falhou ({_active['label']}) — redescubrindo")
        except Exception as e:
            log.warning(f"Endpoint ativo erro: {e} — redescubrindo")
        _active = None

    # Redescobre
    attempts = _build_attempts()
    for att in attempts:
        try:
            resp = requests.get(att["url"], params=att["params"], timeout=10)
            log.info(f"Probe {att['label']}: HTTP {resp.status_code} | {resp.text[:80]}")
            if resp.status_code != 200:
                continue
            data = resp.json()
            code = data.get("code")
            err  = (data.get("error") or {})
            if code not in (0, None) or err.get("code") is not None:
                continue
            parser = _parse_rows if att["parser"] == "rows" else _parse_klines
            closes = parser(data, att["scale"])
            if len(closes) < PERIOD + 5:
                log.warning(f"{att['label']}: apenas {len(closes)} barras")
                continue
            _active = att
            log.info(f"✓ Endpoint ativo: {att['label']} ({len(closes)} barras)")
            return closes
        except Exception as e:
            log.warning(f"Probe {att['label']} erro: {e}")

    raise ValueError("Nenhum endpoint da Phemex respondeu corretamente")


# ─────────────────────────────────────────────────────────────
# PAPER ENGINE
# ─────────────────────────────────────────────────────────────
class PaperEngine:
    def __init__(self, balance):
        self.balance       = balance
        self.position_side = "FLAT"
        self.position_qty  = 0.0
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

    def unrealized_pnl(self, price):
        if self.position_side == "FLAT" or self.position_qty == 0:
            return 0.0
        if self.position_side == "LONG":
            return (price - self.entry_price) * self.position_qty
        return (self.entry_price - price) * self.position_qty

    def open_long(self, price, qty):
        with self._lock:
            if self.position_side == "SHORT":
                pnl = (self.entry_price - price) * self.position_qty
                self.realized_pnl += pnl
                self.balance      += pnl
            self.balance       = max(self.balance - price * qty / LEVERAGE, 0)
            self.position_side = "LONG"
            self.position_qty  = qty
            self.entry_price   = price
            return self._rec("LONG", price, qty)

    def open_short(self, price, qty):
        with self._lock:
            if self.position_side == "LONG":
                pnl = (price - self.entry_price) * self.position_qty
                self.realized_pnl += pnl
                self.balance      += pnl
            self.balance       = max(self.balance - price * qty / LEVERAGE, 0)
            self.position_side = "SHORT"
            self.position_qty  = qty
            self.entry_price   = price
            return self._rec("SHORT", price, qty)

    def _rec(self, side, price, qty):
        return {
            "time": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
            "side": side, "price": round(price, 4), "qty": round(qty, 6),
            "pnl_real": round(self.realized_pnl, 4),
            "balance": round(self.balance, 4), "ec": 0.0, "ema": 0.0,
        }


paper = PaperEngine(INITIAL_BALANCE)

# ─────────────────────────────────────────────────────────────
# ESTADO COMPARTILHADO
# ─────────────────────────────────────────────────────────────
_lock         = threading.Lock()
trade_history = deque(maxlen=200)
ka_log        = deque(maxlen=40)
probe_log     = deque(maxlen=20)   # resultado do último probe

status = {
    "running":     False,
    "signal":      "—",
    "ec":          0.0,
    "ema":         0.0,
    "last_price":  0.0,
    "last_check":  "—",
    "active_feed": "aguardando probe...",
    "errors":      deque(maxlen=10),
}


# ─────────────────────────────────────────────────────────────
# AZLEMA
# ─────────────────────────────────────────────────────────────
def azlema(closes, period=PERIOD, gain_limit=GAIN_LIMIT):
    arr    = np.array(closes, dtype=np.float64)
    n      = len(arr)
    if n < period + 10:
        return None, None
    alpha  = 2.0 / (period + 1)
    ema    = np.empty(n); ema[0] = arr[0]
    for i in range(1, n):
        ema[i] = alpha * arr[i] + (1 - alpha) * ema[i-1]
    best_g, best_e = 0.0, 1e18
    for gi in range(-gain_limit * 10, gain_limit * 10 + 1):
        g = gi / 10.0
        ec = np.empty(n); ec[0] = arr[0]
        for i in range(1, n):
            ec[i] = alpha * (ema[i] + g * (arr[i] - ec[i-1])) + (1-alpha)*ec[i-1]
        e = abs(arr[-1] - ec[-1])
        if e < best_e:
            best_e, best_g = e, g
    ec = np.empty(n); ec[0] = arr[0]
    for i in range(1, n):
        ec[i] = alpha * (ema[i] + best_g * (arr[i] - ec[i-1])) + (1-alpha)*ec[i-1]
    return float(ec[-1]), float(ema[-1])


# ─────────────────────────────────────────────────────────────
# TIMING
# ─────────────────────────────────────────────────────────────
def seconds_to_next_30m():
    now  = datetime.now(timezone.utc)
    secs = now.minute * 60 + now.second + now.microsecond / 1e6
    wait = (1800 - secs) if secs < 1800 else (3600 - secs)
    return max(wait + 5, 1)


# ─────────────────────────────────────────────────────────────
# PROBE INICIAL — roda 30s após start, descobre endpoint
# ─────────────────────────────────────────────────────────────
def initial_probe():
    time.sleep(30)
    log.info("=== PROBE INICIAL DOS ENDPOINTS PHEMEX ===")
    results = []
    for att in _build_attempts():
        try:
            resp   = requests.get(att["url"], params=att["params"], timeout=10)
            body   = resp.json()
            code   = body.get("code")
            errc   = (body.get("error") or {}).get("code")
            rows_n = len((body.get("data") or {}).get("rows") or [])
            line   = (f"[{resp.status_code}] {att['label']} "
                      f"code={code} errc={errc} rows={rows_n}")
        except Exception as e:
            line = f"[ERR] {att['label']}: {str(e)[:60]}"
        log.info(f"  {line}")
        results.append(line)
    with _lock:
        probe_log.clear()
        for r in results:
            probe_log.append(r)
    log.info("=== FIM DO PROBE ===")


# ─────────────────────────────────────────────────────────────
# TRADING LOOP
# ─────────────────────────────────────────────────────────────
def trading_loop():
    status["running"] = True
    log.info("Trading loop iniciado")

    while True:
        wait = seconds_to_next_30m()
        log.info(f"Próximo candle em {wait:.0f}s")
        time.sleep(wait)

        try:
            closes = fetch_closes()

            feed = _active["label"] if _active else "?"
            with _lock:
                status["active_feed"] = feed

            price = closes[-1]
            ec, ema = azlema(closes)
            if ec is None:
                continue

            signal = "LONG" if ec > ema else "SHORT"
            with _lock:
                status.update({
                    "ec": round(ec, 4), "ema": round(ema, 4),
                    "signal": signal, "last_price": round(price, 4),
                    "last_check": datetime.now(timezone.utc).strftime(
                        "%Y-%m-%d %H:%M:%S UTC"),
                })

            snap    = paper.snapshot()
            cur_pos = snap["position_side"]
            qty     = max(round((RISK_PCT * snap["balance"]) / (SL_TICKS * MINTICK), 6), 0.001)

            record = None
            if signal == "LONG" and cur_pos != "LONG":
                record = paper.open_long(price, qty)
                log.info(f"PAPER LONG  {qty} @ {price}")
            elif signal == "SHORT" and cur_pos != "SHORT":
                record = paper.open_short(price, qty)
                log.info(f"PAPER SHORT {qty} @ {price}")
            else:
                log.info(f"Hold {cur_pos} | EC={ec:.2f} EMA={ema:.2f} p={price}")

            if record:
                record["ec"] = round(ec, 4); record["ema"] = round(ema, 4)
                with _lock:
                    trade_history.appendleft(record)

        except Exception as exc:
            msg = str(exc)[:200]
            log.error(f"Erro no bot: {msg}")
            with _lock:
                status["errors"].appendleft({
                    "time": datetime.now(timezone.utc).strftime("%H:%M:%S"),
                    "msg": msg,
                })
            time.sleep(30)


# ─────────────────────────────────────────────────────────────
# KEEPALIVE — 8s / 15s / 23s
# ─────────────────────────────────────────────────────────────
def _ka_worker(interval, name):
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
# FLASK
# ─────────────────────────────────────────────────────────────
@app.route("/ping")
def ping():
    return jsonify({"ok": True, "ts": datetime.utcnow().isoformat()})


@app.route("/health")
def health():
    snap = paper.snapshot()
    return jsonify({
        "status": "healthy", "running": status["running"],
        "position": snap["position_side"], "signal": status["signal"],
        "balance": snap["balance"], "active_feed": status["active_feed"],
    })


@app.route("/test-api")
def test_api():
    """Testa todos os endpoints agora e retorna JSON com resultado completo."""
    results = []
    for att in _build_attempts():
        try:
            resp  = requests.get(att["url"], params=att["params"], timeout=10)
            body  = resp.json()
            rows  = (body.get("data") or {}).get("rows") or []
            klines = ((body.get("result") or {}).get("klines") or [])
            sample_row = rows[0] if rows else (klines[0] if klines else None)
            results.append({
                "label":   att["label"],
                "url":     resp.url,
                "status":  resp.status_code,
                "code":    body.get("code"),
                "errc":    (body.get("error") or {}).get("code"),
                "rows_n":  len(rows) or len(klines),
                "sample":  sample_row,
                "close":   round(float(sample_row[6]) * att["scale"], 2) if sample_row else None,
            })
        except Exception as e:
            results.append({"label": att["label"], "error": str(e)})
    return jsonify({
        "active": _active["label"] if _active else None,
        "results": results,
    })


@app.route("/")
def dashboard():
    snap   = paper.snapshot()
    unreal = round(paper.unrealized_pnl(status["last_price"]), 4)
    with _lock:
        s      = dict(status)
        s["errors"] = list(status["errors"])
        trades = list(trade_history)
        ka     = list(ka_log)
        probe  = list(probe_log)
    return render_template_string(_HTML,
        s=s, p=snap, unreal=unreal, trades=trades, ka=ka, probe=probe)


_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="refresh" content="30">
<title>AZLEMA Paper Bot — Phemex</title>
<link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet">
<style>
  body{background:#0d1117;color:#e6edf3;font-family:system-ui,sans-serif}
  .card{background:#161b22;border:1px solid #30363d;border-radius:8px}
  .bl{background:#238636;color:#fff}.bs{background:#da3633;color:#fff}.bf{background:#444c56;color:#fff}
  .bp{color:#3fb950}.bn{color:#f85149}.bz{color:#8b949e}
  th{color:#8b949e;font-size:.73rem;text-transform:uppercase;font-weight:500;border-color:#30363d!important}
  td{font-size:.82rem;border-color:#21262d!important;vertical-align:middle}
  .ka,.pr{font-size:.72rem;font-family:monospace;line-height:1.8}
  .ka{color:#8b949e}.pr{color:#58a6ff}
  .er{font-size:.72rem;color:#f85149;font-family:monospace;line-height:1.8}
  .mono{font-family:monospace}
  .table-dark{--bs-table-bg:#161b22;--bs-table-hover-bg:#1c2128}
</style>
</head>
<body>
<div class="container-fluid py-3 px-4">

  <div class="d-flex align-items-center mb-3 flex-wrap gap-2">
    <h5 class="mb-0 me-2">⚡ AZLEMA Paper Bot</h5>
    <span class="badge {{'bg-success' if s.running else 'bg-danger'}} rounded-pill">
      {{'● RUNNING' if s.running else '○ STOPPED'}}</span>
    <span class="badge bg-secondary rounded-pill">Phemex · 30m · 1×</span>
    <span class="badge bg-info text-dark rounded-pill">Paper Trading</span>
    <small class="text-secondary ms-auto">auto-refresh 30s · {{s.last_check}}</small>
  </div>

  <div class="row g-2 mb-3">
    {% set pc = 'bl' if p.position_side=='LONG' else ('bs' if p.position_side=='SHORT' else 'bf') %}
    {% set sc = 'bl' if s.signal=='LONG' else ('bs' if s.signal=='SHORT' else 'bf') %}
    <div class="col-6 col-md-2"><div class="card p-3 text-center h-100">
      <div class="text-secondary small mb-1">Posição</div>
      <span class="badge {{pc}} fs-6 rounded-pill">{{p.position_side}}</span>
    </div></div>
    <div class="col-6 col-md-2"><div class="card p-3 text-center h-100">
      <div class="text-secondary small mb-1">Sinal</div>
      <span class="badge {{sc}} fs-6 rounded-pill">{{s.signal}}</span>
    </div></div>
    <div class="col-6 col-md-2"><div class="card p-3 text-center h-100">
      <div class="text-secondary small mb-1">Saldo</div>
      <div class="fw-bold mono">${{p.balance}}</div>
    </div></div>
    <div class="col-6 col-md-2"><div class="card p-3 text-center h-100">
      <div class="text-secondary small mb-1">P&L Realizado</div>
      <div class="fw-bold mono {{'bp' if p.realized_pnl>=0 else 'bn'}}">
        {{'+' if p.realized_pnl>=0 else ''}}${{p.realized_pnl}}</div>
    </div></div>
    <div class="col-6 col-md-2"><div class="card p-3 text-center h-100">
      <div class="text-secondary small mb-1">P&L Não Real.</div>
      <div class="fw-bold mono {{'bp' if unreal>0 else ('bn' if unreal<0 else 'bz')}}">
        {{'+' if unreal>0 else ''}}${{unreal}}</div>
    </div></div>
    <div class="col-6 col-md-2"><div class="card p-3 text-center h-100">
      <div class="text-secondary small mb-1">Preço / EC / EMA</div>
      <div class="fw-bold mono" style="font-size:.8rem">${{s.last_price}}</div>
      <div class="text-secondary" style="font-size:.68rem">{{s.ec}} / {{s.ema}}</div>
    </div></div>
  </div>

  <div class="row g-3">
    <div class="col-lg-8">
      <div class="card p-3">
        <div class="d-flex align-items-center mb-3">
          <h6 class="text-secondary mb-0">Histórico de Trades</h6>
          <span class="badge bg-secondary ms-2">{{trades|length}}</span>
          <a href="/test-api" target="_blank" class="btn btn-outline-secondary btn-sm ms-auto"
             style="font-size:.72rem">🔍 Test API</a>
        </div>
        {% if trades %}
        <div style="max-height:460px;overflow-y:auto">
        <table class="table table-dark table-hover table-sm mb-0">
          <thead><tr><th>Hora (UTC)</th><th>Lado</th><th>Preço</th>
            <th>Qty</th><th>EC</th><th>EMA</th><th>P&L</th><th>Saldo</th></tr></thead>
          <tbody>{% for t in trades %}<tr>
            <td class="mono" style="font-size:.7rem">{{t.time}}</td>
            <td><span class="badge {{'bl' if t.side=='LONG' else 'bs'}} rounded-pill">{{t.side}}</span></td>
            <td class="mono">${{t.price}}</td><td class="mono">{{t.qty}}</td>
            <td class="mono">{{t.ec}}</td><td class="mono">{{t.ema}}</td>
            <td class="mono {{'bp' if t.pnl_real>=0 else 'bn'}}">{{'+' if t.pnl_real>=0 else ''}}${{t.pnl_real}}</td>
            <td class="mono">${{t.balance}}</td>
          </tr>{% endfor %}</tbody>
        </table></div>
        {% else %}
        <div class="text-center py-5">
          <div class="text-secondary mb-2" style="font-size:2rem">⏳</div>
          <div class="text-secondary">Aguardando candle 30m...</div>
          <div class="text-secondary small mt-1">Feed: <strong>{{s.active_feed}}</strong></div>
          <a href="/test-api" target="_blank" class="btn btn-outline-info btn-sm mt-3">🔍 Verificar endpoints</a>
        </div>{% endif %}
      </div>
    </div>

    <div class="col-lg-4">
      <div class="card p-3 mb-3">
        <h6 class="text-secondary mb-2">Posição Atual</h6>
        <table class="table table-dark table-sm mb-0">
          <tr><td class="text-secondary">Lado</td><td class="text-end mono">{{p.position_side}}</td></tr>
          <tr><td class="text-secondary">Qty</td><td class="text-end mono">{{p.position_qty}} ETH</td></tr>
          <tr><td class="text-secondary">Entry</td><td class="text-end mono">${{p.entry_price}}</td></tr>
          <tr><td class="text-secondary">Saldo</td><td class="text-end mono">${{p.balance}}</td></tr>
        </table>
      </div>

      {% if probe %}
      <div class="card p-3 mb-3">
        <h6 class="text-secondary mb-2">Probe de endpoints (startup)</h6>
        {% for p2 in probe %}<div class="pr">{{p2}}</div>{% endfor %}
      </div>{% endif %}

      <div class="card p-3 mb-3">
        <h6 class="text-secondary mb-2">Keepalive (8s · 15s · 23s)</h6>
        {% for k in ka %}<div class="ka">{{k}}</div>
        {% else %}<div class="text-secondary small">Iniciando...</div>{% endfor %}
      </div>

      {% if s.errors %}
      <div class="card p-3">
        <h6 class="text-danger mb-2">Erros</h6>
        {% for e in s.errors %}<div class="er">[{{e.time}}] {{e.msg}}</div>{% endfor %}
      </div>{% endif %}
    </div>
  </div>
</div></body></html>"""


# ─────────────────────────────────────────────────────────────
# STARTUP
# ─────────────────────────────────────────────────────────────
_started = False

def _start_background():
    global _started
    if _started:
        return
    _started = True
    threading.Thread(target=initial_probe, daemon=True).start()
    for iv, nm in [(8, "KA-8s"), (15, "KA-15s"), (23, "KA-23s")]:
        threading.Thread(target=_ka_worker, args=(iv, nm), daemon=True).start()
    threading.Thread(target=trading_loop, daemon=True).start()
    log.info("Threads: probe + trading + 3x keepalive")

_start_background()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT, debug=False)
