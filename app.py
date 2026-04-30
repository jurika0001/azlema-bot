#!/usr/bin/env python3
"""
Phemex Paper Trading Bot — Adaptive Zero Lag EMA
- Lista de trades montada em Python (não no template)
- /tmp para storage (sempre gravável no Render)
- Posição aberta sempre visível no topo
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
# ENV VARS
# ─────────────────────────────────────────────────────────────
API_KEY    = os.environ.get("PHEMEX_API_KEY", "")
API_SECRET = os.environ.get("PHEMEX_API_SECRET", "")
PORT       = int(os.environ.get("PORT", 8080))

# ─────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────
PHEMEX_BASE     = "https://api.phemex.com"
RESOLUTION      = 1800
LEVERAGE        = 1
RISK_PCT        = 0.01
SL_TICKS        = 2000
MINTICK         = 0.01
GAIN_LIMIT      = 50
PERIOD          = 20
INITIAL_BALANCE = 1000.0

# /tmp é sempre gravável no Render (e em qualquer Linux)
TRADES_FILE = Path("/tmp/trades.json")
STATE_FILE  = Path("/tmp/state.json")

# ─────────────────────────────────────────────────────────────
# DISCO
# ─────────────────────────────────────────────────────────────
def _save_json(path: Path, data):
    try:
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2))
    except Exception as e:
        log.warning(f"Falha ao salvar {path}: {e}")

def _load_json(path: Path, default):
    try:
        if path.exists():
            return json.loads(path.read_text())
    except Exception as e:
        log.warning(f"Falha ao carregar {path}: {e}")
    return default

# ─────────────────────────────────────────────────────────────
# ENDPOINTS PHEMEX
# ─────────────────────────────────────────────────────────────
_ATTEMPTS = [
    {
        "label":  "v2/kline/last ETHUSDT",
        "url":    f"{PHEMEX_BASE}/exchange/public/md/v2/kline/last",
        "params": {"symbol": "ETHUSDT", "resolution": RESOLUTION, "limit": 100},
        "scale":  1.0,
    },
    {
        "label":  "v2/kline ETHUSDT",
        "url":    f"{PHEMEX_BASE}/exchange/public/md/v2/kline",
        "params": {"symbol": "ETHUSDT", "resolution": RESOLUTION, "limit": 100},
        "scale":  1.0,
    },
    {
        "label":  "v2/kline/last ETHUSD",
        "url":    f"{PHEMEX_BASE}/exchange/public/md/v2/kline/last",
        "params": {"symbol": "ETHUSD", "resolution": RESOLUTION, "limit": 100},
        "scale":  1e-4,
    },
    {
        "label":  "v2/kline ETHUSD",
        "url":    f"{PHEMEX_BASE}/exchange/public/md/v2/kline",
        "params": {"symbol": "ETHUSD", "resolution": RESOLUTION, "limit": 100},
        "scale":  1e-4,
    },
]

_active_ep = None


def fetch_closes() -> list:
    global _active_ep

    def _try(att):
        r = requests.get(att["url"], params=att["params"], timeout=10)
        if r.status_code != 200:
            raise ValueError(f"HTTP {r.status_code} | {r.text[:80]}")
        d    = r.json()
        code = d.get("code")
        errc = (d.get("error") or {}).get("code")
        if code not in (0, None) or errc is not None:
            raise ValueError(f"code={code} errc={errc} msg={d.get('msg','?')}")
        rows = d.get("data", {}).get("rows", [])
        if not rows:
            raise ValueError("rows vazio")
        closes = [float(x[6]) * att["scale"] for x in rows[:-1]]
        if len(closes) < PERIOD + 5:
            raise ValueError(f"apenas {len(closes)} barras")
        return closes

    if _active_ep:
        try:
            return _try(_active_ep)
        except Exception as e:
            log.warning(f"Endpoint ativo falhou: {e} — redescubrindo")
            _active_ep = None

    for att in _ATTEMPTS:
        try:
            closes = _try(att)
            _active_ep = att
            log.info(f"✓ Endpoint: {att['label']} ({len(closes)} barras)")
            return closes
        except Exception as e:
            log.warning(f"Falhou {att['label']}: {e}")

    raise ValueError("Todos os endpoints Phemex falharam")


# ─────────────────────────────────────────────────────────────
# PAPER ENGINE
# ─────────────────────────────────────────────────────────────
class PaperEngine:
    def __init__(self):
        saved = _load_json(STATE_FILE, {})
        self.balance       = saved.get("balance",       INITIAL_BALANCE)
        self.position_side = saved.get("position_side", "FLAT")
        self.position_qty  = saved.get("position_qty",  0.0)
        self.entry_price   = saved.get("entry_price",   0.0)
        self.realized_pnl  = saved.get("realized_pnl",  0.0)
        self._lock         = threading.Lock()
        log.info(
            f"PaperEngine: pos={self.position_side} qty={self.position_qty} "
            f"entry={self.entry_price} bal={self.balance} pnl={self.realized_pnl}"
        )

    def _persist(self):
        _save_json(STATE_FILE, {
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
            "time":        datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
            "side":        side,
            "price":       round(price, 4),
            "qty":         round(qty, 6),
            "close_price": None,
            "close_time":  None,
            "pnl_trade":   None,
            "pnl_real":    round(self.realized_pnl, 4),
            "balance":     round(self.balance, 4),
            "ec":          0.0,
            "ema":         0.0,
            "status":      "ABERTA",
        }


paper = PaperEngine()

# ─────────────────────────────────────────────────────────────
# HISTÓRICO
# ─────────────────────────────────────────────────────────────
_lock = threading.Lock()

_raw  = _load_json(TRADES_FILE, [])
trade_history = deque(_raw, maxlen=500)
log.info(f"Trades carregadas: {len(trade_history)}")

# Posição aberta mas sem histórico → cria registro sintético
_s0 = paper.snapshot()
if _s0["position_side"] != "FLAT" and not any(
    t.get("status") == "ABERTA" for t in trade_history
):
    _synth = {
        "time":        "anterior ao restart",
        "side":        _s0["position_side"],
        "price":       _s0["entry_price"],
        "qty":         _s0["position_qty"],
        "close_price": None,
        "close_time":  None,
        "pnl_trade":   None,
        "pnl_real":    _s0["realized_pnl"],
        "balance":     _s0["balance"],
        "ec":          0.0,
        "ema":         0.0,
        "status":      "ABERTA",
    }
    trade_history.appendleft(_synth)
    _save_json(TRADES_FILE, list(trade_history))
    log.info(f"Registro sintético: {_s0['position_side']} @ {_s0['entry_price']}")

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
def azlema(closes: list):
    arr   = np.array(closes, dtype=np.float64)
    n     = len(arr)
    if n < PERIOD + 10:
        return None, None
    alpha = 2.0 / (PERIOD + 1)
    ema   = np.empty(n); ema[0] = arr[0]
    for i in range(1, n):
        ema[i] = alpha * arr[i] + (1 - alpha) * ema[i - 1]
    bg, be = 0.0, 1e18
    for gi in range(-GAIN_LIMIT * 10, GAIN_LIMIT * 10 + 1):
        g = gi / 10.0
        ec = np.empty(n); ec[0] = arr[0]
        for i in range(1, n):
            ec[i] = alpha * (ema[i] + g * (arr[i] - ec[i-1])) + (1-alpha)*ec[i-1]
        e = abs(arr[-1] - ec[-1])
        if e < be:
            be, bg = e, g
    ec = np.empty(n); ec[0] = arr[0]
    for i in range(1, n):
        ec[i] = alpha * (ema[i] + bg * (arr[i] - ec[i-1])) + (1-alpha)*ec[i-1]
    return float(ec[-1]), float(ema[-1])

# ─────────────────────────────────────────────────────────────
# ESTRATÉGIA
# ─────────────────────────────────────────────────────────────
_strat_lock = threading.Lock()

def run_strategy(label: str = "candle") -> dict:
    if not _strat_lock.acquire(blocking=False):
        return {"status": "busy"}
    try:
        closes = fetch_closes()
        if _active_ep:
            with _lock:
                status["active_feed"] = _active_ep["label"]

        price = closes[-1]
        ec, ema = azlema(closes)
        if ec is None:
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
            0.001,
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
            log.info(
                f"[{label}] Hold {cur_pos} | "
                f"EC={ec:.2f} EMA={ema:.2f} p={price}"
            )

        if record:
            record["ec"]  = round(ec, 4)
            record["ema"] = round(ema, 4)
            with _lock:
                # Fecha trade anterior
                if trade_history:
                    prev = dict(trade_history[0])
                    if prev.get("status") == "ABERTA":
                        ep = float(prev.get("price") or 0)
                        qp = float(prev.get("qty")   or 0)
                        pt = ((price - ep) * qp if prev["side"] == "LONG"
                              else (ep - price) * qp)
                        prev["status"]      = "FECHADA"
                        prev["close_price"] = round(price, 4)
                        prev["close_time"]  = datetime.now(timezone.utc).strftime(
                            "%Y-%m-%d %H:%M:%S")
                        prev["pnl_trade"]   = round(pt, 4)
                        trade_history[0]    = prev
                trade_history.appendleft(record)
                _save_json(TRADES_FILE, list(trade_history))

        return {
            "status":   "ok",
            "action":   action,
            "signal":   signal,
            "price":    price,
            "ec":       round(ec, 4),
            "ema":      round(ema, 4),
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
        _strat_lock.release()

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
    log.info("Startup: aguardando 5s...")
    time.sleep(5)
    log.info("Startup: executando estratégia...")
    run_strategy("startup")
    while True:
        wait = seconds_to_next_30m()
        log.info(f"Próximo candle em {wait:.0f}s")
        time.sleep(wait)
        run_strategy("candle")

# ─────────────────────────────────────────────────────────────
# KEEPALIVE
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
# HELPER — monta lista de display em Python (sem lógica no template)
# ─────────────────────────────────────────────────────────────
def build_display_trades(trades_raw: list, snap: dict, cur_price: float) -> list:
    """
    Retorna lista pronta para o template.
    Garante que a posição aberta SEMPRE aparece no topo,
    mesmo que trades_raw esteja vazio.
    """
    rows = []

    pos = snap["position_side"]

    # ── Linha da posição aberta ─────────────────────────────
    if pos != "FLAT":
        # Calcula P&L não realizado
        entry = snap["entry_price"]
        qty   = snap["position_qty"]
        if pos == "LONG":
            upnl = round((cur_price - entry) * qty, 4) if cur_price else 0.0
        else:
            upnl = round((entry - cur_price) * qty, 4) if cur_price else 0.0

        # Pega EC/EMA da trade aberta no histórico (se existir)
        ec_val  = 0.0
        ema_val = 0.0
        time_val = "—"
        for t in trades_raw:
            if t.get("status") == "ABERTA":
                ec_val   = t.get("ec",   0.0)
                ema_val  = t.get("ema",  0.0)
                time_val = t.get("time", "—")
                break

        rows.append({
            "row_type":    "open",
            "time":        time_val,
            "side":        pos,
            "price":       entry,
            "qty":         qty,
            "ec":          ec_val,
            "ema":         ema_val,
            "close_price": "—",
            "pnl_trade":   f"{'+' if upnl >= 0 else ''}{upnl} (aberta)",
            "pnl_trade_cls": "bp" if upnl >= 0 else "bn",
            "pnl_real":    snap["realized_pnl"],
            "balance":     snap["balance"],
            "status_lbl":  "● ABERTA",
            "status_cls":  "bg-warning text-dark",
        })

    # ── Trades do histórico ─────────────────────────────────
    for t in trades_raw:
        # Pula se já mostrada como "open" acima
        if t.get("status") == "ABERTA" and pos != "FLAT":
            continue

        is_long = t.get("side") == "LONG"
        pt      = t.get("pnl_trade")
        if pt is not None:
            pt_str = f"{'+' if pt >= 0 else ''}{pt}"
            pt_cls = "bp" if pt >= 0 else "bn"
        else:
            pt_str = "—"
            pt_cls = "bz"

        pr = t.get("realized_pnl", 0)
        rows.append({
            "row_type":      "hist",
            "time":          t.get("time", "—"),
            "side":          t.get("side", "—"),
            "price":         t.get("price", 0),
            "qty":           t.get("qty", 0),
            "ec":            t.get("ec", 0),
            "ema":           t.get("ema", 0),
            "close_price":   t.get("close_price") or "—",
            "pnl_trade":     pt_str,
            "pnl_trade_cls": pt_cls,
            "pnl_real":      pr,
            "pnl_real_cls":  "bp" if pr >= 0 else "bn",
            "balance":       t.get("balance", 0),
            "status_lbl":    "FECHADA" if t.get("status") == "FECHADA" else "hist.",
            "status_cls":    "bg-secondary",
            "is_long":       is_long,
        })

    return rows

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
    return jsonify(run_strategy("manual"))

@app.route("/reset", methods=["GET", "POST"])
def reset():
    global trade_history
    with paper._lock:
        paper.balance       = INITIAL_BALANCE
        paper.position_side = "FLAT"
        paper.position_qty  = 0.0
        paper.entry_price   = 0.0
        paper.realized_pnl  = 0.0
        paper._persist()
    trade_history = deque(maxlen=500)
    _save_json(TRADES_FILE, [])
    log.info("Reset manual")
    return jsonify({"status": "ok", "msg": "Resetado. Saldo: $1000"})

@app.route("/debug")
def debug():
    """Mostra estado interno raw — útil para diagnóstico."""
    snap = paper.snapshot()
    with _lock:
        trades = list(trade_history)
    return jsonify({
        "paper_state":    snap,
        "trade_history":  trades,
        "status":         {k: v for k, v in status.items() if k != "errors"},
        "trades_file":    str(TRADES_FILE),
        "trades_exists":  TRADES_FILE.exists(),
        "state_exists":   STATE_FILE.exists(),
        "active_ep":      _active_ep["label"] if _active_ep else None,
    })

@app.route("/test-api")
def test_api():
    results = []
    for att in _ATTEMPTS:
        try:
            r    = requests.get(att["url"], params=att["params"], timeout=10)
            body = r.json()
            rows = (body.get("data") or {}).get("rows", [])
            s    = rows[0] if rows else None
            results.append({
                "label":  att["label"],
                "url":    r.url,
                "status": r.status_code,
                "code":   body.get("code"),
                "rows":   len(rows),
                "close":  round(float(s[6]) * att["scale"], 2) if s else None,
            })
        except Exception as e:
            results.append({"label": att["label"], "error": str(e)})
    return jsonify({
        "active":  _active_ep["label"] if _active_ep else None,
        "results": results,
    })

@app.route("/")
def dashboard():
    snap      = paper.snapshot()
    cur_price = status["last_price"] or 0.0
    unreal    = round(paper.unrealized_pnl(cur_price), 4)
    equity    = round(snap["balance"] + unreal, 4)

    with _lock:
        s_copy           = dict(status)
        s_copy["errors"] = list(status["errors"])
        trades_raw       = list(trade_history)
        ka               = list(ka_log)

    # Monta lista de display em Python — sem lógica no template
    display_rows = build_display_trades(trades_raw, snap, cur_price)

    return render_template_string(
        _HTML,
        s=s_copy, p=snap,
        cur_price=cur_price, unreal=unreal, equity=equity,
        rows=display_rows,
        total_trades=len(trades_raw),
        ka=ka,
    )

# ─────────────────────────────────────────────────────────────
# HTML — template ultra-simples: só itera `rows` sem condicionais
# ─────────────────────────────────────────────────────────────
_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="refresh" content="15">
<title>AZLEMA Paper Bot</title>
<link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css"
      rel="stylesheet">
<style>
  body  { background:#0d1117; color:#e6edf3; font-family:system-ui,sans-serif }
  .card { background:#161b22; border:1px solid #30363d; border-radius:8px }
  .bl   { background:#238636!important; color:#fff!important }
  .bs   { background:#da3633!important; color:#fff!important }
  .bf   { background:#444c56!important; color:#fff!important }
  .bp   { color:#3fb950 }
  .bn   { color:#f85149 }
  .bz   { color:#8b949e }
  th    { color:#8b949e; font-size:.72rem; text-transform:uppercase;
          font-weight:500; border-color:#30363d!important }
  td    { font-size:.81rem; border-color:#21262d!important; vertical-align:middle }
  .ka   { font-size:.72rem; color:#8b949e; font-family:monospace; line-height:1.8 }
  .er   { font-size:.72rem; color:#f85149; font-family:monospace; line-height:1.8 }
  .mono { font-family:monospace }
  .table-dark { --bs-table-bg:#161b22; --bs-table-hover-bg:#1c2128 }
  .tr-open td { background:rgba(255,193,7,.07)!important }
</style>
</head>
<body>
<div class="container-fluid py-3 px-4">

  <!-- Header -->
  <div class="d-flex align-items-center mb-3 flex-wrap gap-2">
    <h5 class="mb-0 me-2">⚡ AZLEMA Paper Bot</h5>
    <span class="badge {{ 'bg-success' if s.running else 'bg-danger' }} rounded-pill">
      {{ '● RUNNING' if s.running else '○ STOPPED' }}</span>
    <span class="badge bg-secondary rounded-pill">Phemex · 30m · 1×</span>
    <span class="badge bg-info text-dark rounded-pill">Paper Trading</span>
    <small class="text-secondary ms-auto">refresh 15s · {{ s.last_check }}</small>
  </div>

  <!-- Stats -->
  {% set pc = 'bl' if p.position_side=='LONG' else ('bs' if p.position_side=='SHORT' else 'bf') %}
  {% set sc = 'bl' if s.signal=='LONG'        else ('bs' if s.signal=='SHORT'        else 'bf') %}
  <div class="row g-2 mb-3">
    <div class="col-6 col-sm-4 col-md-2">
      <div class="card p-3 text-center h-100">
        <div class="text-secondary small mb-1">Posição</div>
        <span class="badge {{ pc }} fs-6 rounded-pill">{{ p.position_side }}</span>
        {% if p.position_side != 'FLAT' %}
        <div class="text-secondary mt-1" style="font-size:.63rem">
          {{ p.position_qty }} ETH @ ${{ p.entry_price }}
        </div>
        {% endif %}
      </div>
    </div>
    <div class="col-6 col-sm-4 col-md-2">
      <div class="card p-3 text-center h-100">
        <div class="text-secondary small mb-1">Sinal</div>
        <span class="badge {{ sc }} fs-6 rounded-pill">{{ s.signal }}</span>
        <div class="text-secondary mt-1" style="font-size:.63rem">
          EC {{ s.ec }} · EMA {{ s.ema }}
        </div>
      </div>
    </div>
    <div class="col-6 col-sm-4 col-md-2">
      <div class="card p-3 text-center h-100">
        <div class="text-secondary small mb-1">Saldo</div>
        <div class="fw-bold mono">${{ p.balance }}</div>
        <div class="text-secondary" style="font-size:.63rem">equity ${{ equity }}</div>
      </div>
    </div>
    <div class="col-6 col-sm-4 col-md-2">
      <div class="card p-3 text-center h-100">
        <div class="text-secondary small mb-1">P&L Realizado</div>
        <div class="fw-bold mono {{ 'bp' if p.realized_pnl >= 0 else 'bn' }}">
          {{ '+' if p.realized_pnl >= 0 else '' }}${{ p.realized_pnl }}
        </div>
      </div>
    </div>
    <div class="col-6 col-sm-4 col-md-2">
      <div class="card p-3 text-center h-100">
        <div class="text-secondary small mb-1">P&L Não Real.</div>
        <div class="fw-bold mono {{ 'bp' if unreal > 0 else ('bn' if unreal < 0 else 'bz') }}">
          {{ '+' if unreal > 0 else '' }}${{ unreal }}
        </div>
        <div class="text-secondary" style="font-size:.63rem">preço ${{ cur_price }}</div>
      </div>
    </div>
    <div class="col-6 col-sm-4 col-md-2">
      <div class="card p-3 text-center h-100 d-flex flex-column justify-content-center gap-2">
        <button id="run-btn" class="btn btn-success btn-sm" onclick="runNow()">
          ▶ Executar agora
        </button>
        <a href="/debug" target="_blank" class="btn btn-outline-secondary btn-sm">
          🔍 Debug
        </a>
      </div>
    </div>
  </div>

  <div class="row g-3">
    <!-- Tabela -->
    <div class="col-lg-8">
      <div class="card p-3">
        <div class="d-flex align-items-center mb-3 gap-2 flex-wrap">
          <h6 class="text-secondary mb-0">Histórico de Trades</h6>
          <span class="badge bg-secondary">{{ total_trades }}</span>
          <small class="text-secondary">feed: <strong>{{ s.active_feed }}</strong></small>
          <a href="/test-api" target="_blank"
             class="btn btn-outline-secondary btn-sm ms-auto"
             style="font-size:.7rem">API</a>
        </div>

        {% if rows %}
        <div style="max-height:520px;overflow-y:auto">
          <table class="table table-dark table-hover table-sm mb-0">
            <thead><tr>
              <th>Hora abertura</th>
              <th>Lado</th>
              <th>Entry $</th>
              <th>Qty ETH</th>
              <th>EC</th>
              <th>EMA</th>
              <th>Fechou $</th>
              <th>P&L trade</th>
              <th>P&L acum.</th>
              <th>Saldo</th>
              <th>Status</th>
            </tr></thead>
            <tbody>
            {% for r in rows %}
            <tr class="{{ 'tr-open' if r.row_type == 'open' else '' }}">
              <td class="mono" style="font-size:.69rem">{{ r.time }}</td>
              <td>
                <span class="badge {{ 'bl' if r.side == 'LONG' else 'bs' }} rounded-pill">
                  {{ r.side }}
                </span>
              </td>
              <td class="mono">${{ r.price }}</td>
              <td class="mono">{{ r.qty }}</td>
              <td class="mono">{{ r.ec }}</td>
              <td class="mono">{{ r.ema }}</td>
              <td class="mono">{{ r.close_price }}</td>
              <td class="mono {{ r.pnl_trade_cls }}">{{ r.pnl_trade }}</td>
              <td class="mono {{ r.get('pnl_real_cls', 'bp' if r.pnl_real >= 0 else 'bn') }}">
                {{ '+' if r.pnl_real >= 0 else '' }}${{ r.pnl_real }}
              </td>
              <td class="mono">${{ r.balance }}</td>
              <td>
                <span class="badge {{ r.status_cls }}" style="font-size:.62rem">
                  {{ r.status_lbl }}
                </span>
              </td>
            </tr>
            {% endfor %}
            </tbody>
          </table>
        </div>
        {% else %}
        <div class="text-center py-5">
          <div class="mb-2" style="font-size:2.5rem">📊</div>
          <div class="text-secondary mb-3">
            Nenhuma trade ainda — bot analisa a cada 30 min.
          </div>
          <button class="btn btn-success btn-sm me-2" onclick="runNow()">
            ▶ Executar agora
          </button>
          <a href="/debug" target="_blank" class="btn btn-outline-info btn-sm">
            🔍 Debug
          </a>
        </div>
        {% endif %}
      </div>
    </div>

    <!-- Side -->
    <div class="col-lg-4">
      <div class="card p-3 mb-3">
        <h6 class="text-secondary mb-2">Posição Atual</h6>
        <table class="table table-dark table-sm mb-0">
          <tr><td class="text-secondary">Lado</td>
              <td class="text-end">
                <span class="badge {{ pc }} rounded-pill">{{ p.position_side }}</span>
              </td></tr>
          <tr><td class="text-secondary">Qty</td>
              <td class="text-end mono">{{ p.position_qty }} ETH</td></tr>
          <tr><td class="text-secondary">Entry</td>
              <td class="text-end mono">${{ p.entry_price }}</td></tr>
          <tr><td class="text-secondary">Preço</td>
              <td class="text-end mono">${{ cur_price }}</td></tr>
          <tr><td class="text-secondary">P&L não real.</td>
              <td class="text-end mono {{ 'bp' if unreal > 0 else ('bn' if unreal < 0 else 'bz') }}">
                {{ '+' if unreal > 0 else '' }}${{ unreal }}</td></tr>
          <tr><td class="text-secondary">P&L realizado</td>
              <td class="text-end mono {{ 'bp' if p.realized_pnl >= 0 else 'bn' }}">
                {{ '+' if p.realized_pnl >= 0 else '' }}${{ p.realized_pnl }}</td></tr>
          <tr><td class="text-secondary">Saldo</td>
              <td class="text-end mono">${{ p.balance }}</td></tr>
          <tr><td class="text-secondary fw-bold">Equity</td>
              <td class="text-end mono fw-bold">${{ equity }}</td></tr>
        </table>
      </div>

      <div class="card p-3 mb-3">
        <h6 class="text-secondary mb-2">Keepalive (8s · 15s · 23s)</h6>
        {% for k in ka %}
          <div class="ka">{{ k }}</div>
        {% else %}
          <div class="text-secondary small">Iniciando...</div>
        {% endfor %}
      </div>

      <div class="card p-3 mb-3">
        <h6 class="text-secondary mb-2">Ações</h6>
        <div class="d-flex flex-column gap-2">
          <button class="btn btn-success btn-sm" onclick="runNow()">
            ▶ Executar estratégia agora
          </button>
          <a href="/debug" target="_blank"
             class="btn btn-outline-secondary btn-sm">
            🔍 Ver estado interno (debug)
          </a>
          <a href="/test-api" target="_blank"
             class="btn btn-outline-secondary btn-sm">
            🔌 Testar endpoints API
          </a>
          <button class="btn btn-outline-danger btn-sm"
                  onclick="if(confirm('Zerar tudo?')) resetBot()">
            ⚠ Reset total
          </button>
        </div>
      </div>

      {% if s.errors %}
      <div class="card p-3">
        <h6 class="text-danger mb-2">Erros</h6>
        {% for e in s.errors %}
          <div class="er">[{{ e.time }}] {{ e.msg }}</div>
        {% endfor %}
      </div>
      {% endif %}
    </div>
  </div>
</div>

<script>
function runNow() {
  const btn = document.getElementById('run-btn');
  if (btn) { btn.disabled = true; btn.textContent = '⏳ Executando...'; }
  fetch('/run-now', { method: 'POST' })
    .then(r => r.json())
    .then(d => {
      if (btn) btn.textContent = '✓ ' + (d.action || d.status);
      setTimeout(() => location.reload(), 1200);
    })
    .catch(() => setTimeout(() => location.reload(), 2000));
}
function resetBot() {
  fetch('/reset', { method: 'POST' }).then(() => location.reload());
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
    log.info("Threads: trading + 3x keepalive")

_start_background()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT, debug=False)
