#!/usr/bin/env python3
"""
Phemex Paper Trading Bot — Adaptive Zero Lag EMA
- Trades persistidas em disco (trades.json) — sobrevive a restarts
- Executa estratégia no startup (5s após subir)
- Botão de execução manual no dashboard
"""

import os
import json
import time
import threading
import logging
from datetime import datetime, timezone
from collections import deque
from pathlib import Path

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
RESOLUTION      = 1800
BARS_NEEDED     = 100

LEVERAGE        = 1
RISK_PCT        = 0.01
SL_TICKS        = 2000
MINTICK         = 0.01
GAIN_LIMIT      = 50
PERIOD          = 20
INITIAL_BALANCE = 1000.0

TRADES_FILE  = Path("trades.json")    # persiste entre restarts
STATE_FILE   = Path("state.json")     # persiste posição e saldo

# ─────────────────────────────────────────────────────────────
# PERSISTÊNCIA EM DISCO
# ─────────────────────────────────────────────────────────────
def _save_trades(trades: list):
    try:
        TRADES_FILE.write_text(json.dumps(trades, ensure_ascii=False))
    except Exception as e:
        log.warning(f"Falha ao salvar trades: {e}")


def _load_trades() -> list:
    try:
        if TRADES_FILE.exists():
            return json.loads(TRADES_FILE.read_text())
    except Exception as e:
        log.warning(f"Falha ao carregar trades: {e}")
    return []


def _save_state(state: dict):
    try:
        STATE_FILE.write_text(json.dumps(state, ensure_ascii=False))
    except Exception as e:
        log.warning(f"Falha ao salvar state: {e}")


def _load_state() -> dict:
    try:
        if STATE_FILE.exists():
            return json.loads(STATE_FILE.read_text())
    except Exception as e:
        log.warning(f"Falha ao carregar state: {e}")
    return {}


# ─────────────────────────────────────────────────────────────
# ENDPOINTS PHEMEX
# ─────────────────────────────────────────────────────────────
_ATTEMPTS = [
    {
        "label":  "v2/kline/last ETHUSDT",
        "url":    f"{PHEMEX_BASE}/exchange/public/md/v2/kline/last",
        "params": {"symbol": "ETHUSDT", "resolution": RESOLUTION, "limit": 100},
        "scale":  1.0,
        "fmt":    "rows",
    },
    {
        "label":  "v2/kline ETHUSDT",
        "url":    f"{PHEMEX_BASE}/exchange/public/md/v2/kline",
        "params": {"symbol": "ETHUSDT", "resolution": RESOLUTION, "limit": 100},
        "scale":  1.0,
        "fmt":    "rows",
    },
    {
        "label":  "v2/kline/last ETHUSD",
        "url":    f"{PHEMEX_BASE}/exchange/public/md/v2/kline/last",
        "params": {"symbol": "ETHUSD", "resolution": RESOLUTION, "limit": 100},
        "scale":  1e-4,
        "fmt":    "rows",
    },
    {
        "label":  "v2/kline ETHUSD",
        "url":    f"{PHEMEX_BASE}/exchange/public/md/v2/kline",
        "params": {"symbol": "ETHUSD", "resolution": RESOLUTION, "limit": 100},
        "scale":  1e-4,
        "fmt":    "rows",
    },
]

_active_ep = None


def _parse(data: dict, fmt: str, scale: float) -> list:
    if fmt == "rows":
        rows = data.get("data", {}).get("rows", [])
        if not rows:
            raise ValueError("rows vazio")
        return [float(r[6]) * scale for r in rows[:-1]]
    kl = ((data.get("result") or {}).get("klines")
          or (data.get("result") or {}).get("rows") or [])
    if not kl:
        raise ValueError("klines vazio")
    return [float(r[6]) * scale for r in kl[:-1]]


def _try_ep(att: dict) -> list:
    resp = requests.get(att["url"], params=att["params"], timeout=10)
    if resp.status_code != 200:
        raise ValueError(f"HTTP {resp.status_code} | {resp.text[:100]}")
    data = resp.json()
    code = data.get("code")
    errc = (data.get("error") or {}).get("code")
    if code not in (0, None) or errc is not None:
        raise ValueError(f"code={code} errc={errc} msg={data.get('msg','?')}")
    closes = _parse(data, att["fmt"], att["scale"])
    if len(closes) < PERIOD + 5:
        raise ValueError(f"apenas {len(closes)} barras")
    return closes


def fetch_closes() -> list:
    global _active_ep
    if _active_ep:
        try:
            return _try_ep(_active_ep)
        except Exception as e:
            log.warning(f"Endpoint ativo falhou: {e} — redescubrindo")
            _active_ep = None
    for att in _ATTEMPTS:
        try:
            closes = _try_ep(att)
            _active_ep = att
            log.info(f"✓ Endpoint ativo: {att['label']} ({len(closes)} barras)")
            return closes
        except Exception as e:
            log.warning(f"Falhou {att['label']}: {e}")
    raise ValueError("Todos os endpoints Phemex falharam")


# ─────────────────────────────────────────────────────────────
# PAPER ENGINE — com persistência automática
# ─────────────────────────────────────────────────────────────
class PaperEngine:
    def __init__(self, initial_balance: float):
        # Carrega estado do disco se existir
        saved = _load_state()
        self.balance       = saved.get("balance",       initial_balance)
        self.position_side = saved.get("position_side", "FLAT")
        self.position_qty  = saved.get("position_qty",  0.0)
        self.entry_price   = saved.get("entry_price",   0.0)
        self.realized_pnl  = saved.get("realized_pnl",  0.0)
        self._lock         = threading.Lock()
        if saved:
            log.info(f"Estado restaurado do disco: {self.position_side} "
                     f"qty={self.position_qty} entry={self.entry_price} "
                     f"balance={self.balance}")
        else:
            log.info("Estado inicial: saldo $1000 simulado")

    def _persist(self):
        _save_state({
            "balance":       self.balance,
            "position_side": self.position_side,
            "position_qty":  self.position_qty,
            "entry_price":   self.entry_price,
            "realized_pnl":  self.realized_pnl,
        })

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "balance":       round(self.balance, 4),
                "position_side": self.position_side,
                "position_qty":  round(self.position_qty, 6),
                "entry_price":   round(self.entry_price, 4),
                "realized_pnl":  round(self.realized_pnl, 4),
            }

    def unrealized_pnl(self, price: float) -> float:
        if self.position_side == "FLAT" or self.position_qty == 0:
            return 0.0
        if self.position_side == "LONG":
            return (price - self.entry_price) * self.position_qty
        return (self.entry_price - price) * self.position_qty

    def open_long(self, price: float, qty: float) -> dict:
        with self._lock:
            if self.position_side == "SHORT":
                pnl = (self.entry_price - price) * self.position_qty
                self.realized_pnl += pnl
                self.balance      += pnl
                log.info(f"Fechando SHORT → PnL {pnl:+.4f}")
            self.balance       = max(self.balance - price * qty / LEVERAGE, 0)
            self.position_side = "LONG"
            self.position_qty  = qty
            self.entry_price   = price
            self._persist()
            return self._rec("LONG", price, qty)

    def open_short(self, price: float, qty: float) -> dict:
        with self._lock:
            if self.position_side == "LONG":
                pnl = (price - self.entry_price) * self.position_qty
                self.realized_pnl += pnl
                self.balance      += pnl
                log.info(f"Fechando LONG → PnL {pnl:+.4f}")
            self.balance       = max(self.balance - price * qty / LEVERAGE, 0)
            self.position_side = "SHORT"
            self.position_qty  = qty
            self.entry_price   = price
            self._persist()
            return self._rec("SHORT", price, qty)

    def _rec(self, side: str, price: float, qty: float) -> dict:
        return {
            "time":     datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
            "side":     side,
            "price":    round(price, 4),
            "qty":      round(qty, 6),
            "pnl_real": round(self.realized_pnl, 4),
            "balance":  round(self.balance, 4),
            "ec":       0.0,
            "ema":      0.0,
        }


paper = PaperEngine(INITIAL_BALANCE)

# ─────────────────────────────────────────────────────────────
# ESTADO COMPARTILHADO — trades carregados do disco
# ─────────────────────────────────────────────────────────────
_lock = threading.Lock()

# Carrega histórico do disco
_saved_trades = _load_trades()
trade_history = deque(_saved_trades, maxlen=500)
log.info(f"Trades carregados do disco: {len(trade_history)}")

ka_log = deque(maxlen=40)

status = {
    "running":     False,
    "signal":      "—",
    "ec":          0.0,
    "ema":         0.0,
    "last_price":  0.0,
    "last_check":  "—",
    "active_feed": "inicializando...",
    "errors":      deque(maxlen=10),
}


# ─────────────────────────────────────────────────────────────
# AZLEMA
# ─────────────────────────────────────────────────────────────
def azlema(closes: list, period: int = PERIOD, gain_limit: int = GAIN_LIMIT):
    arr    = np.array(closes, dtype=np.float64)
    n      = len(arr)
    if n < period + 10:
        return None, None
    alpha  = 2.0 / (period + 1)
    ema    = np.empty(n); ema[0] = arr[0]
    for i in range(1, n):
        ema[i] = alpha * arr[i] + (1 - alpha) * ema[i - 1]
    best_g, best_e = 0.0, 1e18
    for gi in range(-gain_limit * 10, gain_limit * 10 + 1):
        g = gi / 10.0
        ec = np.empty(n); ec[0] = arr[0]
        for i in range(1, n):
            ec[i] = alpha * (ema[i] + g * (arr[i] - ec[i-1])) + (1 - alpha) * ec[i-1]
        e = abs(arr[-1] - ec[-1])
        if e < best_e:
            best_e, best_g = e, g
    ec = np.empty(n); ec[0] = arr[0]
    for i in range(1, n):
        ec[i] = alpha * (ema[i] + best_g * (arr[i] - ec[i-1])) + (1 - alpha) * ec[i-1]
    return float(ec[-1]), float(ema[-1])


# ─────────────────────────────────────────────────────────────
# ESTRATÉGIA
# ─────────────────────────────────────────────────────────────
_strategy_lock = threading.Lock()   # evita execuções simultâneas


def run_strategy(label: str = "candle"):
    if not _strategy_lock.acquire(blocking=False):
        log.info(f"[{label}] Estratégia já em execução — ignorado")
        return {"status": "busy"}

    try:
        closes = fetch_closes()

        if _active_ep:
            with _lock:
                status["active_feed"] = _active_ep["label"]

        price = closes[-1]
        ec, ema = azlema(closes)
        if ec is None:
            log.warning("AZLEMA: barras insuficientes")
            return {"status": "error", "msg": "barras insuficientes"}

        signal = "LONG" if ec > ema else "SHORT"

        with _lock:
            status.update({
                "ec":         round(ec, 4),
                "ema":        round(ema, 4),
                "signal":     signal,
                "last_price": round(price, 4),
                "last_check": datetime.now(timezone.utc).strftime(
                    "%Y-%m-%d %H:%M:%S UTC"),
            })

        snap    = paper.snapshot()
        cur_pos = snap["position_side"]
        qty     = max(
            round((RISK_PCT * snap["balance"]) / (SL_TICKS * MINTICK), 6),
            0.001
        )

        action = "hold"
        record = None

        if signal == "LONG" and cur_pos != "LONG":
            record = paper.open_long(price, qty)
            action = "LONG"
            log.info(f"[{label}] PAPER LONG  {qty} ETH @ {price}")
        elif signal == "SHORT" and cur_pos != "SHORT":
            record = paper.open_short(price, qty)
            action = "SHORT"
            log.info(f"[{label}] PAPER SHORT {qty} ETH @ {price}")
        else:
            log.info(f"[{label}] Hold {cur_pos} | EC={ec:.2f} EMA={ema:.2f} p={price}")

        if record:
            record["ec"]  = round(ec, 4)
            record["ema"] = round(ema, 4)
            with _lock:
                trade_history.appendleft(record)
                _save_trades(list(trade_history))   # persiste no disco

        return {
            "status":  "ok",
            "action":  action,
            "signal":  signal,
            "price":   price,
            "ec":      round(ec, 4),
            "ema":     round(ema, 4),
            "position": paper.snapshot()["position_side"],
        }

    except Exception as exc:
        msg = str(exc)[:200]
        log.error(f"Erro [{label}]: {msg}")
        with _lock:
            status["errors"].appendleft({
                "time": datetime.now(timezone.utc).strftime("%H:%M:%S"),
                "msg":  msg,
            })
        return {"status": "error", "msg": msg}

    finally:
        _strategy_lock.release()


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

    # Execução imediata no startup
    log.info("Startup: aguardando 5s para Flask subir...")
    time.sleep(5)
    log.info("Startup: executando estratégia agora...")
    run_strategy("startup")

    # Loop: a cada candle 30m
    while True:
        wait = seconds_to_next_30m()
        log.info(f"Próximo candle em {wait:.0f}s")
        time.sleep(wait)
        run_strategy("candle")


# ─────────────────────────────────────────────────────────────
# KEEPALIVE — 8s / 15s / 23s
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
        "status":      "healthy",
        "running":     status["running"],
        "position":    snap["position_side"],
        "signal":      status["signal"],
        "balance":     snap["balance"],
        "active_feed": status["active_feed"],
        "last_check":  status["last_check"],
        "trades":      len(trade_history),
    })


@app.route("/run-now", methods=["GET", "POST"])
def run_now():
    """Executa a estratégia manualmente — botão no dashboard."""
    result = run_strategy("manual")
    return jsonify(result)


@app.route("/test-api")
def test_api():
    results = []
    for att in _ATTEMPTS:
        try:
            resp   = requests.get(att["url"], params=att["params"], timeout=10)
            body   = resp.json()
            rows   = (body.get("data") or {}).get("rows", [])
            sample = rows[0] if rows else None
            results.append({
                "label":  att["label"],
                "url":    resp.url,
                "status": resp.status_code,
                "code":   body.get("code"),
                "errc":   (body.get("error") or {}).get("code"),
                "rows":   len(rows),
                "close":  round(float(sample[6]) * att["scale"], 2) if sample else None,
            })
        except Exception as e:
            results.append({"label": att["label"], "error": str(e)})
    return jsonify({
        "active":  _active_ep["label"] if _active_ep else None,
        "results": results,
    })


@app.route("/reset", methods=["GET", "POST"])
def reset():
    """Zera saldo, posição e histórico (útil para recomeçar)."""
    global trade_history
    paper.balance       = INITIAL_BALANCE
    paper.position_side = "FLAT"
    paper.position_qty  = 0.0
    paper.entry_price   = 0.0
    paper.realized_pnl  = 0.0
    paper._persist()
    trade_history = deque(maxlen=500)
    _save_trades([])
    log.info("Estado resetado manualmente")
    return jsonify({"status": "ok", "msg": "Estado resetado. Saldo: $1000"})


@app.route("/")
def dashboard():
    snap   = paper.snapshot()
    unreal = round(paper.unrealized_pnl(status["last_price"]), 4)
    equity = round(snap["balance"] + unreal, 4)
    with _lock:
        s_copy           = dict(status)
        s_copy["errors"] = list(status["errors"])
        trades           = list(trade_history)
        ka               = list(ka_log)
    return render_template_string(
        _HTML,
        s=s_copy, p=snap,
        unreal=unreal, equity=equity,
        trades=trades, ka=ka,
    )


# ─────────────────────────────────────────────────────────────
# DASHBOARD HTML
# ─────────────────────────────────────────────────────────────
_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="refresh" content="15">
<title>AZLEMA Paper Bot — Phemex</title>
<link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet">
<style>
  body  { background:#0d1117; color:#e6edf3; font-family:system-ui,sans-serif }
  .card { background:#161b22; border:1px solid #30363d; border-radius:8px }
  .bl   { background:#238636; color:#fff }
  .bs   { background:#da3633; color:#fff }
  .bf   { background:#444c56; color:#fff }
  .bp   { color:#3fb950 } .bn { color:#f85149 } .bz { color:#8b949e }
  th    { color:#8b949e; font-size:.73rem; text-transform:uppercase;
          font-weight:500; border-color:#30363d!important }
  td    { font-size:.82rem; border-color:#21262d!important; vertical-align:middle }
  .ka   { font-size:.73rem; color:#8b949e; font-family:monospace; line-height:1.8 }
  .er   { font-size:.73rem; color:#f85149; font-family:monospace; line-height:1.8 }
  .mono { font-family:monospace }
  .table-dark { --bs-table-bg:#161b22; --bs-table-hover-bg:#1c2128 }
  #run-btn { min-width:140px }
</style>
</head>
<body>
<div class="container-fluid py-3 px-4">

  <!-- Header -->
  <div class="d-flex align-items-center mb-3 flex-wrap gap-2">
    <h5 class="mb-0 me-2">⚡ AZLEMA Paper Bot</h5>
    <span class="badge {{'bg-success' if s.running else 'bg-danger'}} rounded-pill">
      {{'● RUNNING' if s.running else '○ STOPPED'}}</span>
    <span class="badge bg-secondary rounded-pill">Phemex · 30m · 1×</span>
    <span class="badge bg-info text-dark rounded-pill">Paper Trading</span>
    <small class="text-secondary ms-auto">auto-refresh 15s · {{s.last_check}}</small>
  </div>

  <!-- Stats -->
  <div class="row g-2 mb-3">
    {% set pc = 'bl' if p.position_side=='LONG' else ('bs' if p.position_side=='SHORT' else 'bf') %}
    {% set sc = 'bl' if s.signal=='LONG' else ('bs' if s.signal=='SHORT' else 'bf') %}

    <div class="col-6 col-sm-4 col-md-2">
      <div class="card p-3 text-center h-100">
        <div class="text-secondary small mb-1">Posição</div>
        <span class="badge {{pc}} fs-6 rounded-pill">{{p.position_side}}</span>
        {% if p.position_side != 'FLAT' %}
        <div class="text-secondary mt-1" style="font-size:.68rem">
          {{p.position_qty}} ETH @ ${{p.entry_price}}
        </div>
        {% endif %}
      </div>
    </div>

    <div class="col-6 col-sm-4 col-md-2">
      <div class="card p-3 text-center h-100">
        <div class="text-secondary small mb-1">Sinal Atual</div>
        <span class="badge {{sc}} fs-6 rounded-pill">{{s.signal}}</span>
        <div class="text-secondary mt-1" style="font-size:.68rem">
          EC {{s.ec}} · EMA {{s.ema}}
        </div>
      </div>
    </div>

    <div class="col-6 col-sm-4 col-md-2">
      <div class="card p-3 text-center h-100">
        <div class="text-secondary small mb-1">Saldo</div>
        <div class="fw-bold mono">${{p.balance}}</div>
        <div class="text-secondary" style="font-size:.68rem">inicial $1000</div>
      </div>
    </div>

    <div class="col-6 col-sm-4 col-md-2">
      <div class="card p-3 text-center h-100">
        <div class="text-secondary small mb-1">P&L Realizado</div>
        <div class="fw-bold mono {{'bp' if p.realized_pnl>=0 else 'bn'}}">
          {{'+' if p.realized_pnl>=0 else ''}}${{p.realized_pnl}}
        </div>
      </div>
    </div>

    <div class="col-6 col-sm-4 col-md-2">
      <div class="card p-3 text-center h-100">
        <div class="text-secondary small mb-1">P&L Não Real.</div>
        <div class="fw-bold mono {{'bp' if unreal>0 else ('bn' if unreal<0 else 'bz')}}">
          {{'+' if unreal>0 else ''}}${{unreal}}
        </div>
      </div>
    </div>

    <div class="col-6 col-sm-4 col-md-2">
      <div class="card p-3 text-center h-100">
        <div class="text-secondary small mb-1">Último Preço</div>
        <div class="fw-bold mono">${{s.last_price}}</div>
        <div class="text-secondary" style="font-size:.68rem">
          Equity ${{equity}}
        </div>
      </div>
    </div>
  </div>

  <div class="row g-3">

    <!-- Trade history -->
    <div class="col-lg-8">
      <div class="card p-3">
        <div class="d-flex align-items-center mb-3 flex-wrap gap-2">
          <h6 class="text-secondary mb-0">Histórico de Trades</h6>
          <span class="badge bg-secondary">{{trades|length}}</span>
          <span class="text-secondary" style="font-size:.72rem">
            · feed: <strong>{{s.active_feed}}</strong>
          </span>
          <div class="ms-auto d-flex gap-2">
            <button id="run-btn"
                    class="btn btn-success btn-sm"
                    onclick="runNow()">
              ▶ Executar agora
            </button>
            <a href="/test-api" target="_blank"
               class="btn btn-outline-secondary btn-sm">🔍 API</a>
          </div>
        </div>

        {% if trades %}
        <div style="max-height:500px; overflow-y:auto">
          <table class="table table-dark table-hover table-sm mb-0">
            <thead><tr>
              <th>Hora (UTC)</th>
              <th>Lado</th>
              <th>Preço</th>
              <th>Qty (ETH)</th>
              <th>EC</th>
              <th>EMA</th>
              <th>P&L Real.</th>
              <th>Saldo</th>
            </tr></thead>
            <tbody>
            {% for t in trades %}
            <tr>
              <td class="mono" style="font-size:.7rem">{{t.time}}</td>
              <td>
                <span class="badge {{'bl' if t.side=='LONG' else 'bs'}} rounded-pill">
                  {{t.side}}
                </span>
              </td>
              <td class="mono">${{t.price}}</td>
              <td class="mono">{{t.qty}}</td>
              <td class="mono">{{t.ec}}</td>
              <td class="mono">{{t.ema}}</td>
              <td class="mono {{'bp' if t.pnl_real>=0 else 'bn'}}">
                {{'+' if t.pnl_real>=0 else ''}}${{t.pnl_real}}
              </td>
              <td class="mono">${{t.balance}}</td>
            </tr>
            {% endfor %}
            </tbody>
          </table>
        </div>

        {% else %}
        <div class="text-center py-5">
          <div class="mb-2" style="font-size:2.5rem">📊</div>
          <div class="text-secondary mb-1">
            Posição atual: 
            <span class="badge {{pc}} rounded-pill ms-1">{{p.position_side}}</span>
          </div>
          <div class="text-secondary small mb-3">
            Nenhuma mudança de posição registrada ainda.<br>
            Trades só aparecem quando o sinal muda de direção.
          </div>
          <button class="btn btn-success btn-sm me-2" onclick="runNow()">
            ▶ Executar agora
          </button>
          <a href="/test-api" target="_blank" class="btn btn-outline-info btn-sm">
            🔍 Verificar endpoints
          </a>
        </div>
        {% endif %}
      </div>
    </div>

    <!-- Side panel -->
    <div class="col-lg-4">

      <!-- Posição atual -->
      <div class="card p-3 mb-3">
        <h6 class="text-secondary mb-2">Posição Atual</h6>
        <table class="table table-dark table-sm mb-0">
          <tr>
            <td class="text-secondary">Lado</td>
            <td class="text-end">
              <span class="badge {{pc}} rounded-pill">{{p.position_side}}</span>
            </td>
          </tr>
          <tr>
            <td class="text-secondary">Qty</td>
            <td class="text-end mono">{{p.position_qty}} ETH</td>
          </tr>
          <tr>
            <td class="text-secondary">Entry</td>
            <td class="text-end mono">${{p.entry_price}}</td>
          </tr>
          <tr>
            <td class="text-secondary">Saldo livre</td>
            <td class="text-end mono">${{p.balance}}</td>
          </tr>
          <tr>
            <td class="text-secondary">P&L não real.</td>
            <td class="text-end mono {{'bp' if unreal>0 else ('bn' if unreal<0 else 'bz')}}">
              {{'+' if unreal>0 else ''}}${{unreal}}
            </td>
          </tr>
          <tr>
            <td class="text-secondary">P&L realizado</td>
            <td class="text-end mono {{'bp' if p.realized_pnl>=0 else 'bn'}}">
              {{'+' if p.realized_pnl>=0 else ''}}${{p.realized_pnl}}
            </td>
          </tr>
        </table>
      </div>

      <!-- Keepalive -->
      <div class="card p-3 mb-3">
        <h6 class="text-secondary mb-2">Keepalive (8s · 15s · 23s)</h6>
        {% for k in ka %}
          <div class="ka">{{k}}</div>
        {% else %}
          <div class="text-secondary small">Iniciando...</div>
        {% endfor %}
      </div>

      <!-- Erros -->
      {% if s.errors %}
      <div class="card p-3">
        <h6 class="text-danger mb-2">Erros recentes</h6>
        {% for e in s.errors %}
          <div class="er">[{{e.time}}] {{e.msg}}</div>
        {% endfor %}
      </div>
      {% endif %}

    </div>
  </div>
</div>

<script>
function runNow() {
  const btn = document.getElementById('run-btn');
  if (!btn) return;
  btn.disabled = true;
  btn.textContent = '⏳ Executando...';
  fetch('/run-now', { method: 'POST' })
    .then(r => r.json())
    .then(d => {
      btn.textContent = '✓ ' + (d.action || d.status);
      btn.classList.remove('btn-success');
      btn.classList.add('btn-secondary');
      setTimeout(() => location.reload(), 1200);
    })
    .catch(() => {
      btn.textContent = '✗ Erro';
      setTimeout(() => location.reload(), 2000);
    });
}
</script>
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
    for iv, nm in [(8, "KA-8s"), (15, "KA-15s"), (23, "KA-23s")]:
        threading.Thread(target=_ka_worker, args=(iv, nm), daemon=True).start()
    threading.Thread(target=trading_loop, daemon=True).start()
    log.info("Threads iniciadas: trading + 3x keepalive")


_start_background()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT, debug=False)
