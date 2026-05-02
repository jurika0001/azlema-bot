#!/usr/bin/env python3
"""
Phemex Paper Trading Bot — Adaptive Zero Lag EMA
FIX DEFINITIVO: estratégia roda sincronamente no startup,
posição aberta SEMPRE visível independente de arquivo.
"""

import os, json, time, threading, logging
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

# ── ENV ──────────────────────────────────────────────────────
API_KEY    = os.environ.get("PHEMEX_API_KEY",    "")
API_SECRET = os.environ.get("PHEMEX_API_SECRET", "")
PORT       = int(os.environ.get("PORT", 8080))

# ── CONSTANTS ────────────────────────────────────────────────
PHEMEX_BASE     = "https://api.phemex.com"
RESOLUTION      = 1800
LEVERAGE        = 1
RISK_PCT        = 0.01
SL_TICKS        = 2000
MINTICK         = 0.01
GAIN_LIMIT      = 50
PERIOD          = 20
INITIAL_BALANCE = 1000.0

# ── STORAGE ──────────────────────────────────────────────────
def _writable(d: Path) -> bool:
    try:
        t = d / "._wt"; t.write_text("x"); t.unlink(); return True
    except Exception:
        return False

STORAGE     = Path(".") if _writable(Path(".")) else Path("/tmp")
STATE_FILE  = STORAGE / "bot_state.json"
TRADES_FILE = STORAGE / "bot_trades.json"
log.info(f"Storage: {STORAGE.resolve()}")

def _save(p: Path, data):
    try:
        p.write_text(json.dumps(data, ensure_ascii=False, indent=2))
    except Exception as e:
        log.warning(f"save {p.name}: {e}")

def _load(p: Path, default):
    try:
        if p.exists():
            t = p.read_text().strip()
            if t:
                return json.loads(t)
    except Exception as e:
        log.warning(f"load {p.name}: {e}")
    return default

# ── PHEMEX FETCH ─────────────────────────────────────────────
_ATTEMPTS = [
    {"label": "v2/kline/last ETHUSDT",
     "url":   f"{PHEMEX_BASE}/exchange/public/md/v2/kline/last",
     "params": {"symbol":"ETHUSDT","resolution":RESOLUTION,"limit":100},
     "scale": 1.0},
    {"label": "v2/kline ETHUSDT",
     "url":   f"{PHEMEX_BASE}/exchange/public/md/v2/kline",
     "params": {"symbol":"ETHUSDT","resolution":RESOLUTION,"limit":100},
     "scale": 1.0},
    {"label": "v2/kline/last ETHUSD",
     "url":   f"{PHEMEX_BASE}/exchange/public/md/v2/kline/last",
     "params": {"symbol":"ETHUSD","resolution":RESOLUTION,"limit":100},
     "scale": 1e-4},
    {"label": "v2/kline ETHUSD",
     "url":   f"{PHEMEX_BASE}/exchange/public/md/v2/kline",
     "params": {"symbol":"ETHUSD","resolution":RESOLUTION,"limit":100},
     "scale": 1e-4},
]
_active_ep = None

def fetch_closes() -> list:
    global _active_ep

    def _try(att):
        r = requests.get(att["url"], params=att["params"], timeout=10)
        if r.status_code != 200:
            raise ValueError(f"HTTP {r.status_code}")
        d = r.json()
        if d.get("code") not in (0, None):
            raise ValueError(f"code={d.get('code')}")
        rows = d.get("data", {}).get("rows", [])
        if not rows:
            raise ValueError("rows vazio")
        c = [float(x[6]) * att["scale"] for x in rows[:-1]]
        if len(c) < PERIOD + 5:
            raise ValueError(f"só {len(c)} barras")
        return c

    if _active_ep:
        try:
            return _try(_active_ep)
        except Exception as e:
            log.warning(f"Endpoint ativo falhou: {e}")
            _active_ep = None

    for att in _ATTEMPTS:
        try:
            c = _try(att)
            _active_ep = att
            log.info(f"✓ {att['label']} ({len(c)} barras)")
            return c
        except Exception as e:
            log.warning(f"Falhou {att['label']}: {e}")

    raise ValueError("Todos os endpoints falharam")

# ── AZLEMA ───────────────────────────────────────────────────
def azlema(closes: list):
    arr = np.array(closes, dtype=np.float64)
    n   = len(arr)
    if n < PERIOD + 10:
        return None, None
    alpha = 2.0 / (PERIOD + 1)
    ema   = np.empty(n); ema[0] = arr[0]
    for i in range(1, n):
        ema[i] = alpha * arr[i] + (1 - alpha) * ema[i-1]
    bg, be = 0.0, 1e18
    for gi in range(-GAIN_LIMIT*10, GAIN_LIMIT*10+1):
        g = gi/10.0
        ec = np.empty(n); ec[0] = arr[0]
        for i in range(1, n):
            ec[i] = alpha*(ema[i]+g*(arr[i]-ec[i-1]))+(1-alpha)*ec[i-1]
        e = abs(arr[-1]-ec[-1])
        if e < be:
            be, bg = e, g
    ec = np.empty(n); ec[0] = arr[0]
    for i in range(1, n):
        ec[i] = alpha*(ema[i]+bg*(arr[i]-ec[i-1]))+(1-alpha)*ec[i-1]
    return float(ec[-1]), float(ema[-1])

# ── PAPER ENGINE ─────────────────────────────────────────────
class PaperEngine:
    def __init__(self):
        s = _load(STATE_FILE, {})
        self.balance       = float(s.get("balance",       INITIAL_BALANCE))
        self.position_side = str  (s.get("position_side", "FLAT"))
        self.position_qty  = float(s.get("position_qty",  0.0))
        self.entry_price   = float(s.get("entry_price",   0.0))
        self.realized_pnl  = float(s.get("realized_pnl",  0.0))
        self.open_time     = str  (s.get("open_time",     "—"))
        self.open_ec       = float(s.get("open_ec",       0.0))
        self.open_ema      = float(s.get("open_ema",      0.0))
        self._lock         = threading.Lock()
        log.info(f"Engine: {self.position_side} entry={self.entry_price} "
                 f"qty={self.position_qty} bal={self.balance}")

    def _persist(self):
        _save(STATE_FILE, {
            "balance":       self.balance,      "position_side": self.position_side,
            "position_qty":  self.position_qty, "entry_price":   self.entry_price,
            "realized_pnl":  self.realized_pnl, "open_time":     self.open_time,
            "open_ec":       self.open_ec,       "open_ema":      self.open_ema,
        })

    def snapshot(self) -> dict:
        with self._lock:
            return dict(
                balance       = round(self.balance,      4),
                position_side = self.position_side,
                position_qty  = round(self.position_qty, 6),
                entry_price   = round(self.entry_price,  4),
                realized_pnl  = round(self.realized_pnl, 4),
                open_time     = self.open_time,
                open_ec       = round(self.open_ec,  4),
                open_ema      = round(self.open_ema, 4),
            )

    def unrealized_pnl(self, price: float) -> float:
        if self.position_side == "FLAT" or self.position_qty == 0:
            return 0.0
        if self.position_side == "LONG":
            return (price - self.entry_price) * self.position_qty
        return (self.entry_price - price) * self.position_qty

    def _close_current(self, close_px: float) -> dict:
        """Gera registro de fechamento da posição atual."""
        pnl = ((close_px - self.entry_price) if self.position_side == "LONG"
               else (self.entry_price - close_px)) * self.position_qty
        rec = dict(
            time        = self.open_time,
            side        = self.position_side,
            price       = round(self.entry_price, 4),
            qty         = round(self.position_qty, 6),
            ec          = round(self.open_ec,  4),
            ema         = round(self.open_ema, 4),
            close_price = round(close_px, 4),
            close_time  = _utc(),
            pnl_trade   = round(pnl, 4),
            pnl_real    = round(self.realized_pnl + pnl, 4),
            balance     = round(self.balance + pnl, 4),
        )
        self.realized_pnl += pnl
        self.balance      += pnl
        return rec

    def enter(self, side: str, price: float, qty: float,
              ec: float, ema: float) -> dict | None:
        """Abre posição. Fecha a anterior se houver. Retorna dict fechada ou None."""
        with self._lock:
            closed = None
            if self.position_side != "FLAT" and self.position_side != side:
                closed = self._close_current(price)
            self.balance       = max(self.balance - price * qty / LEVERAGE, 0)
            self.position_side = side
            self.position_qty  = qty
            self.entry_price   = price
            self.open_time     = _utc()
            self.open_ec       = ec
            self.open_ema      = ema
            self._persist()
            return closed

def _utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

paper = PaperEngine()

# ── SHARED STATE ─────────────────────────────────────────────
_lock        = threading.Lock()
_closed_hist = deque(_load(TRADES_FILE, []), maxlen=500)
_ka_log      = deque(maxlen=40)
_bot = dict(
    running     = False,
    signal      = "—",
    ec          = 0.0,
    ema         = 0.0,
    last_price  = 0.0,
    last_check  = "nunca",
    active_feed = "inicializando...",
    errors      = [],
)
log.info(f"Histórico fechadas carregado: {len(_closed_hist)}")

# ── ESTRATÉGIA ───────────────────────────────────────────────
_strat_lock = threading.Lock()

def run_strategy(label: str = "candle") -> dict:
    if not _strat_lock.acquire(blocking=False):
        return {"status": "busy"}
    try:
        closes = fetch_closes()
        if _active_ep:
            with _lock:
                _bot["active_feed"] = _active_ep["label"]

        price   = closes[-1]
        ec, ema = azlema(closes)
        if ec is None:
            return {"status": "error", "msg": "barras insuficientes"}

        signal = "LONG" if ec > ema else "SHORT"
        now    = _utc()

        with _lock:
            _bot.update(ec=round(ec,4), ema=round(ema,4), signal=signal,
                        last_price=round(price,4), last_check=now+" UTC")

        snap    = paper.snapshot()
        cur_pos = snap["position_side"]
        qty     = max(round((RISK_PCT * snap["balance"]) /
                            (SL_TICKS * MINTICK), 6), 0.001)

        action = "hold"
        closed = None

        if signal != cur_pos:          # precisa trocar de lado (ou sair do FLAT)
            closed = paper.enter(signal, price, qty, ec, ema)
            action = signal
            log.info(f"[{label}] PAPER {signal} {qty} ETH @ {price}")
        else:
            log.info(f"[{label}] Hold {cur_pos} | "
                     f"EC={ec:.2f} EMA={ema:.2f} p={price}")

        if closed is not None:
            with _lock:
                _closed_hist.appendleft(closed)
                _save(TRADES_FILE, list(_closed_hist))

        return {"status":"ok","action":action,"signal":signal,
                "price":price,"position":paper.snapshot()["position_side"]}

    except Exception as exc:
        msg = str(exc)[:200]
        log.error(f"Erro [{label}]: {msg}")
        with _lock:
            _bot["errors"] = ([{"time":_utc()[-8:],"msg":msg}]
                              + _bot["errors"])[:10]
        return {"status":"error","msg":msg}
    finally:
        _strat_lock.release()

# ── TIMING + LOOP ────────────────────────────────────────────
def _next_30m() -> float:
    now  = datetime.now(timezone.utc)
    secs = now.minute*60 + now.second + now.microsecond/1e6
    wait = (1800-secs) if secs < 1800 else (3600-secs)
    return max(wait+5, 1)

def _candle_loop():
    while True:
        wait = _next_30m()
        log.info(f"Próximo candle em {wait:.0f}s")
        time.sleep(wait)
        run_strategy("candle")

def _ka_worker(interval: int, name: str):
    time.sleep(interval+3)
    while True:
        time.sleep(interval)
        try:
            r   = requests.get(f"http://localhost:{PORT}/ping", timeout=5)
            msg = f"[{name}] OK {r.status_code} @ {datetime.now().strftime('%H:%M:%S')}"
        except Exception as e:
            msg = f"[{name}] FAIL: {e}"
        with _lock:
            _ka_log.appendleft(msg)

# ── FLASK ─────────────────────────────────────────────────────
@app.route("/ping")
def ping():
    return jsonify({"ok":True,"ts":datetime.utcnow().isoformat()})

@app.route("/health")
def health():
    s = paper.snapshot()
    return jsonify({"status":"healthy","position":s["position_side"],
                    "signal":_bot["signal"],"balance":s["balance"],
                    "trades":len(_closed_hist)})

@app.route("/run-now", methods=["GET","POST"])
def run_now():
    return jsonify(run_strategy("manual"))

@app.route("/reset", methods=["GET","POST"])
def reset():
    global _closed_hist
    with paper._lock:
        paper.balance=INITIAL_BALANCE; paper.position_side="FLAT"
        paper.position_qty=0.0; paper.entry_price=0.0
        paper.realized_pnl=0.0; paper.open_time="—"
        paper.open_ec=0.0; paper.open_ema=0.0; paper._persist()
    _closed_hist=deque(maxlen=500); _save(TRADES_FILE,[])
    return jsonify({"status":"ok","msg":"Resetado. Saldo: $1000"})

@app.route("/debug")
def debug():
    with _lock:
        hist = list(_closed_hist)
    return jsonify({"paper":paper.snapshot(),"closed":hist,
                    "bot":{k:v for k,v in _bot.items()},
                    "storage":str(STORAGE.resolve()),
                    "state_exists":STATE_FILE.exists(),
                    "trades_exists":TRADES_FILE.exists(),
                    "active_ep":_active_ep["label"] if _active_ep else None})

@app.route("/test-api")
def test_api():
    out=[]
    for att in _ATTEMPTS:
        try:
            r=requests.get(att["url"],params=att["params"],timeout=10)
            b=r.json(); rows=(b.get("data")or{}).get("rows",[])
            s=rows[0] if rows else None
            out.append({"label":att["label"],"status":r.status_code,
                        "code":b.get("code"),"rows":len(rows),
                        "close":round(float(s[6])*att["scale"],2) if s else None})
        except Exception as e:
            out.append({"label":att["label"],"error":str(e)})
    return jsonify({"active":_active_ep["label"] if _active_ep else None,"results":out})

# ── DASHBOARD ────────────────────────────────────────────────
@app.route("/")
def dashboard():
    snap      = paper.snapshot()
    cur_price = _bot["last_price"] or 0.0
    unreal    = round(paper.unrealized_pnl(cur_price), 4)
    equity    = round(snap["balance"] + unreal, 4)

    with _lock:
        closed = list(_closed_hist)
        ka     = list(_ka_log)
        bot    = dict(_bot)
        errors = list(_bot["errors"])

    # ── monta tabela 100% em Python ──────────────────────────
    def _clr(v): return "#3fb950" if v > 0 else ("#f85149" if v < 0 else "#8b949e")
    def _sgn(v): return ("+" if v >= 0 else "") + str(round(v, 4))

    rows = []
    pos  = snap["position_side"]

    # linha da posição ABERTA — vem do snapshot(), nunca de arquivo
    if pos != "FLAT":
        upnl = round(paper.unrealized_pnl(cur_price), 4)
        rows.append(dict(
            bg          = "#1b2e1b" if pos=="LONG" else "#2e1b1b",
            time        = snap["open_time"],
            side        = pos,
            side_bg     = "#238636" if pos=="LONG" else "#da3633",
            price       = snap["entry_price"],
            qty         = snap["position_qty"],
            ec          = snap["open_ec"],
            ema         = snap["open_ema"],
            close_price = "—",
            close_time  = "—",
            pnl_trade   = _sgn(upnl)+" (aberta)",
            pnl_clr     = _clr(upnl),
            pnl_real    = _sgn(snap["realized_pnl"]),
            pr_clr      = _clr(snap["realized_pnl"]),
            balance     = snap["balance"],
            status      = "● ABERTA",
            st_bg       = "#5a4a00",
            st_fg       = "#ffc107",
        ))

    # linhas das trades FECHADAS
    for t in closed:
        pt = t.get("pnl_trade", 0) or 0
        pr = t.get("pnl_real",  0) or 0
        sd = t.get("side","?")
        rows.append(dict(
            bg          = "",
            time        = t.get("time","—"),
            side        = sd,
            side_bg     = "#238636" if sd=="LONG" else "#da3633",
            price       = t.get("price",0),
            qty         = t.get("qty",0),
            ec          = t.get("ec",0),
            ema         = t.get("ema",0),
            close_price = t.get("close_price","—"),
            close_time  = t.get("close_time","—"),
            pnl_trade   = _sgn(pt),
            pnl_clr     = _clr(pt),
            pnl_real    = _sgn(pr),
            pr_clr      = _clr(pr),
            balance     = t.get("balance",0),
            status      = "fechada",
            st_bg       = "#21262d",
            st_fg       = "#8b949e",
        ))

    pc   = "#238636" if pos=="LONG" else ("#da3633" if pos=="SHORT" else "#444c56")
    sc   = "#238636" if bot["signal"]=="LONG" else ("#da3633" if bot["signal"]=="SHORT" else "#444c56")
    up_s = _sgn(unreal); up_c = _clr(unreal)
    rp_s = _sgn(snap["realized_pnl"]); rp_c = _clr(snap["realized_pnl"])

    return render_template_string(_HTML,
        rows=rows, pos=pos, pc=pc, sc=sc,
        sig=bot["signal"], ec_v=bot["ec"], ema_v=bot["ema"],
        bal=snap["balance"], equity=equity,
        entry=snap["entry_price"], qty=snap["position_qty"],
        up_s=up_s, up_c=up_c, rp_s=rp_s, rp_c=rp_c,
        cur_price=cur_price, chk=bot["last_check"],
        feed=bot["active_feed"], running=bot["running"],
        n_closed=len(closed), ka=ka, errors=errors,
    )

_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="refresh" content="10">
<title>AZLEMA Paper Bot</title>
<link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet">
<style>
  body{background:#0d1117;color:#e6edf3;font-family:system-ui,sans-serif}
  .card{background:#161b22;border:1px solid #30363d;border-radius:8px}
  th{color:#8b949e;font-size:.71rem;text-transform:uppercase;font-weight:500;border-color:#30363d!important}
  td{font-size:.80rem;border-color:#21262d!important;vertical-align:middle}
  .mono{font-family:monospace}
  .ka{font-size:.71rem;color:#8b949e;font-family:monospace;line-height:1.8}
  .er{font-size:.71rem;color:#f85149;font-family:monospace;line-height:1.8}
  .table-dark{--bs-table-bg:#161b22;--bs-table-hover-bg:#1c2128}
</style>
</head>
<body>
<div class="container-fluid py-3 px-4">

  <div class="d-flex align-items-center mb-3 flex-wrap gap-2">
    <h5 class="mb-0 me-2">⚡ AZLEMA Paper Bot</h5>
    <span class="badge rounded-pill px-3"
          style="background:{{'#238636' if running else '#da3633'}}">
      {{'● RUNNING' if running else '○ STOPPED'}}</span>
    <span class="badge bg-secondary rounded-pill">Phemex · ETH/USDT · 30m · 1×</span>
    <span class="badge bg-info text-dark rounded-pill">Paper Trading</span>
    <small class="text-secondary ms-auto">refresh 10s · {{chk}}</small>
  </div>

  <div class="row g-2 mb-3">
    <div class="col-6 col-sm-4 col-md-2">
      <div class="card p-3 text-center h-100">
        <div class="text-secondary small mb-1">Posição</div>
        <span class="badge rounded-pill fs-6 px-3" style="background:{{pc}}">{{pos}}</span>
        {% if pos != 'FLAT' %}
        <div class="text-secondary mt-1" style="font-size:.63rem">{{qty}} ETH @ ${{entry}}</div>
        {% endif %}
      </div>
    </div>
    <div class="col-6 col-sm-4 col-md-2">
      <div class="card p-3 text-center h-100">
        <div class="text-secondary small mb-1">Sinal</div>
        <span class="badge rounded-pill fs-6 px-3" style="background:{{sc}}">{{sig}}</span>
        <div class="text-secondary mt-1" style="font-size:.63rem">EC {{ec_v}} · EMA {{ema_v}}</div>
      </div>
    </div>
    <div class="col-6 col-sm-4 col-md-2">
      <div class="card p-3 text-center h-100">
        <div class="text-secondary small mb-1">Saldo</div>
        <div class="fw-bold mono">${{bal}}</div>
        <div class="text-secondary" style="font-size:.63rem">equity ${{equity}}</div>
      </div>
    </div>
    <div class="col-6 col-sm-4 col-md-2">
      <div class="card p-3 text-center h-100">
        <div class="text-secondary small mb-1">P&L Realizado</div>
        <div class="fw-bold mono" style="color:{{rp_c}}">{{rp_s}}</div>
      </div>
    </div>
    <div class="col-6 col-sm-4 col-md-2">
      <div class="card p-3 text-center h-100">
        <div class="text-secondary small mb-1">P&L Não Real.</div>
        <div class="fw-bold mono" style="color:{{up_c}}">{{up_s}}</div>
        <div class="text-secondary" style="font-size:.63rem">preço ${{cur_price}}</div>
      </div>
    </div>
    <div class="col-6 col-sm-4 col-md-2">
      <div class="card p-3 h-100 d-flex flex-column justify-content-center gap-2">
        <button id="rb" class="btn btn-sm text-white fw-bold"
                style="background:#238636" onclick="runNow()">▶ Executar agora</button>
        <a href="/debug" target="_blank" class="btn btn-outline-secondary btn-sm">🔍 Debug</a>
      </div>
    </div>
  </div>

  <div class="row g-3">
    <div class="col-lg-8">
      <div class="card p-3">
        <div class="d-flex align-items-center mb-3 gap-2 flex-wrap">
          <h6 class="text-secondary mb-0">Histórico de Trades</h6>
          <span class="badge bg-secondary">{{n_closed}} fechadas</span>
          {% if pos != 'FLAT' %}
          <span class="badge rounded-pill px-2" style="background:#5a4a00;color:#ffc107">
            + 1 aberta
          </span>
          {% endif %}
          <small class="text-secondary">feed: <strong>{{feed}}</strong></small>
          <a href="/test-api" target="_blank"
             class="btn btn-outline-secondary btn-sm ms-auto" style="font-size:.7rem">API</a>
        </div>

        <div style="max-height:540px;overflow-y:auto">
          <table class="table table-dark table-sm mb-0">
            <thead><tr>
              <th>Abertura</th><th>Lado</th><th>Entry $</th><th>Qty ETH</th>
              <th>EC</th><th>EMA</th><th>Fechou $</th><th>Fechou em</th>
              <th>P&L trade</th><th>P&L acum.</th><th>Saldo</th><th>Status</th>
            </tr></thead>
            <tbody>
            {% if rows %}
              {% for r in rows %}
              <tr style="{{'background:'+r.bg+'!important;' if r.bg else ''}}">
                <td class="mono" style="font-size:.69rem">{{r.time}}</td>
                <td><span class="badge rounded-pill px-2"
                          style="background:{{r.side_bg}}">{{r.side}}</span></td>
                <td class="mono">${{r.price}}</td>
                <td class="mono">{{r.qty}}</td>
                <td class="mono">{{r.ec}}</td>
                <td class="mono">{{r.ema}}</td>
                <td class="mono">{{r.close_price}}</td>
                <td class="mono" style="font-size:.69rem">{{r.close_time}}</td>
                <td class="mono" style="color:{{r.pnl_clr}}">{{r.pnl_trade}}</td>
                <td class="mono" style="color:{{r.pr_clr}}">{{r.pnl_real}}</td>
                <td class="mono">${{r.balance}}</td>
                <td><span class="badge px-2" style="font-size:.62rem;
                          background:{{r.st_bg}};color:{{r.st_fg}}">{{r.status}}</span></td>
              </tr>
              {% endfor %}
            {% else %}
              <tr><td colspan="12" class="text-center text-secondary py-5">
                Nenhuma trade ainda — clique em "▶ Executar agora"
              </td></tr>
            {% endif %}
            </tbody>
          </table>
        </div>
      </div>
    </div>

    <div class="col-lg-4">
      <div class="card p-3 mb-3">
        <h6 class="text-secondary mb-2">Posição Atual</h6>
        <table class="table table-dark table-sm mb-0">
          <tr><td class="text-secondary">Lado</td>
              <td class="text-end"><span class="badge rounded-pill"
                style="background:{{pc}}">{{pos}}</span></td></tr>
          <tr><td class="text-secondary">Qty</td>
              <td class="text-end mono">{{qty}} ETH</td></tr>
          <tr><td class="text-secondary">Entry</td>
              <td class="text-end mono">${{entry}}</td></tr>
          <tr><td class="text-secondary">Preço atual</td>
              <td class="text-end mono">${{cur_price}}</td></tr>
          <tr><td class="text-secondary">P&L não real.</td>
              <td class="text-end mono" style="color:{{up_c}}">{{up_s}}</td></tr>
          <tr><td class="text-secondary">P&L realizado</td>
              <td class="text-end mono" style="color:{{rp_c}}">{{rp_s}}</td></tr>
          <tr><td class="text-secondary">Saldo</td>
              <td class="text-end mono">${{bal}}</td></tr>
          <tr><td class="text-secondary fw-bold">Equity</td>
              <td class="text-end mono fw-bold">${{equity}}</td></tr>
        </table>
      </div>

      <div class="card p-3 mb-3">
        <h6 class="text-secondary mb-2">Keepalive (8s · 15s · 23s)</h6>
        {% for k in ka %}<div class="ka">{{k}}</div>
        {% else %}<div class="text-secondary small">Iniciando...</div>{% endfor %}
      </div>

      <div class="card p-3 mb-3">
        <h6 class="text-secondary mb-2">Ações</h6>
        <div class="d-flex flex-column gap-2">
          <button class="btn btn-sm text-white" style="background:#238636"
                  onclick="runNow()">▶ Executar estratégia agora</button>
          <a href="/debug" target="_blank"
             class="btn btn-outline-secondary btn-sm">🔍 Debug (estado interno)</a>
          <a href="/test-api" target="_blank"
             class="btn btn-outline-secondary btn-sm">🔌 Testar endpoints</a>
          <button class="btn btn-outline-danger btn-sm"
                  onclick="if(confirm('Zerar tudo?'))resetBot()">⚠ Reset total</button>
        </div>
      </div>

      {% if errors %}
      <div class="card p-3">
        <h6 class="text-danger mb-2">Erros</h6>
        {% for e in errors %}<div class="er">[{{e.time}}] {{e.msg}}</div>{% endfor %}
      </div>{% endif %}
    </div>
  </div>
</div>
<script>
function runNow(){
  const b=document.getElementById('rb');
  if(b){b.disabled=true;b.textContent='⏳ Executando...';}
  fetch('/run-now',{method:'POST'}).then(r=>r.json()).then(d=>{
    if(b)b.textContent='✓ '+(d.action||d.status);
    setTimeout(()=>location.reload(),1200);
  }).catch(()=>setTimeout(()=>location.reload(),2000));
}
function resetBot(){fetch('/reset',{method:'POST'}).then(()=>location.reload());}
</script>
</body>
</html>"""

# ── STARTUP — estratégia roda SINCRONAMENTE antes do Flask ───
_started = False

def _start_background():
    global _started
    if _started:
        return
    _started = True
    _bot["running"] = True

    # ── Roda estratégia AGORA, sincronamente ─────────────────
    # Garante que posição está em memória ANTES do 1º request
    log.info("=== Startup: executando estratégia sincronamente ===")
    try:
        result = run_strategy("startup")
        log.info(f"Startup resultado: {result}")
    except Exception as e:
        log.error(f"Startup erro: {e}")

    # ── Threads de fundo ──────────────────────────────────────
    for iv, nm in [(8,"KA-8s"),(15,"KA-15s"),(23,"KA-23s")]:
        threading.Thread(target=_ka_worker, args=(iv,nm), daemon=True).start()
    threading.Thread(target=_candle_loop, daemon=True).start()
    log.info("=== Startup completo ===")

_start_background()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT, debug=False)
