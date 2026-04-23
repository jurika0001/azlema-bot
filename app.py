#!/usr/bin/env python3
"""
Phemex Paper Trading Bot — Adaptive Zero Lag EMA
30-minute timeframe | Paper Futures | 1x Leverage
Usa API REST direta do Phemex testnet (sem ccxt para ordens)
"""

import os
import time
import hmac
import hashlib
import json
import uuid
import threading
import logging
from datetime import datetime, timezone
from collections import deque

import numpy as np
import requests
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
TESTNET_BASE  = "https://testnet-api.phemex.com"
RAW_SYMBOL    = "ETHUSDT"          # Phemex raw symbol
TIMEFRAME     = "30m"
RISK_PCT      = 0.01
SL_TICKS      = 2000
GAIN_LIMIT    = 50
PERIOD        = 20
THRESHOLD     = 0.0
# ETHUSDT perp no Phemex: 1 contrato = 0.001 ETH
CONTRACT_SIZE = 0.001

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
# PHEMEX TESTNET — API REST DIRETA COM ASSINATURA HMAC
# ─────────────────────────────────────────────────────────────
def _phemex_sign(path: str, query: str, expiry: str, body: str) -> str:
    # Phemex signature: HMAC-SHA256(secret, path + queryString + expiry + body)
    # query must NOT include "?"
    msg = path + query + expiry + body
    return hmac.new(
        API_SECRET.encode("utf-8"),
        msg.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()


def _phemex_headers(path: str, query: str = "", body: str = "") -> dict:
    expiry = str(int(time.time()) + 60)  # 60s window
    sig    = _phemex_sign(path, query, expiry, body)
    log.debug(f"HMAC path={path} query={query!r} expiry={expiry} sig={sig[:12]}...")
    return {
        "x-phemex-access-token":      API_KEY,
        "x-phemex-request-expiry":    expiry,
        "x-phemex-request-signature": sig,
        "Content-Type":               "application/json",
    }


def phemex_get(path: str, params: dict = None) -> dict:
    # Sort params for deterministic query string (required by Phemex)
    sorted_params = sorted((params or {}).items())
    query = "&".join(f"{k}={v}" for k, v in sorted_params)
    url   = TESTNET_BASE + path + (f"?{query}" if query else "")
    hdrs  = _phemex_headers(path, query)
    log.debug(f"GET {url}")
    resp  = requests.get(url, headers=hdrs, timeout=10)
    data  = resp.json()
    log.debug(f"GET {path} -> {str(data)[:200]}")
    return data


def phemex_post(path: str, body: dict) -> dict:
    body_str = json.dumps(body, separators=(",", ":"))
    hdrs     = _phemex_headers(path, "", body_str)
    log.info(f"POST {path} body={body_str}")
    resp     = requests.post(
        TESTNET_BASE + path,
        headers=hdrs,
        data=body_str,
        timeout=10,
    )
    data = resp.json()
    log.info(f"POST {path} -> {data}")
    return data


# ─────────────────────────────────────────────────────────────
# PHEMEX HELPERS
# ─────────────────────────────────────────────────────────────
def get_balance_and_position():
    """
    Retorna (balance_usdt, cur_side, cur_qty_contracts).
    cur_side: 'LONG' | 'SHORT' | 'FLAT'
    """
    data = phemex_get("/g-accounts/accountPositions", {"currency": "USDT"})
    log.info(f"accountPositions raw: {data}")

    # Detectar erro de autenticacao ou argumento
    err = data.get("error") or {}
    if err.get("code") or (data.get("code") and data.get("code") != 0):
        log.error(f"Erro em accountPositions: {data}")
        return 1000.0, "FLAT", 0   # fallback para nao travar o bot

    result    = data.get("data") or {}
    account   = result.get("account") or {}
    positions = result.get("positions") or []

    log.info(f"account keys: {list(account.keys())}")

    # Tentar varios campos de balance (USDT linear usa campos diferentes)
    bal_ev      = (account.get("accountBalanceEv")
                or account.get("totalBalanceEv")
                or account.get("accountBalance")
                or 0)
    # Se valor > 1e6 provavelmente esta em Ev (scaled 1e8), senao ja e USDT
    balance = (bal_ev / 1e8) if bal_ev > 1e6 else float(bal_ev)
    if balance == 0:
        balance = 1000.0  # fallback para testnet sem saldo listado

    cur_side = "FLAT"
    cur_qty  = 0
    for p in positions:
        sym = p.get("symbol") or ""
        if sym == RAW_SYMBOL:
            size = int(p.get("size") or 0)
            if size > 0:
                side_raw = p.get("side") or ""
                cur_side = "LONG" if side_raw == "Buy" else "SHORT"
                cur_qty  = size
            log.info(f"Posicao encontrada: {p}")
    return balance, cur_side, cur_qty


def place_order(side: str, qty_contracts: int, reduce_only: bool = False) -> dict:
    """
    side: 'Buy' | 'Sell'
    qty_contracts: numero inteiro de contratos
    """
    body = {
        "symbol":    RAW_SYMBOL,
        "clOrdID":   uuid.uuid4().hex[:16],
        "side":      side,
        "orderType": "Market",
        "orderQty":  qty_contracts,
    }
    if reduce_only:
        body["reduceOnly"] = True
    log.info(f"Enviando ordem: {body}")
    resp = phemex_post("/g-orders", body)
    log.info(f"Resposta ordem: {resp}")
    if resp.get("code") != 0:
        log.error(f"ERRO ordem {side}: code={resp.get('code') or (resp.get('error') or {}).get('code')} msg={resp.get('msg') or (resp.get('error') or {}).get('message')}")
    return resp


def close_position(cur_side: str, cur_qty: int):
    if cur_qty <= 0:
        return
    close_side = "Sell" if cur_side == "LONG" else "Buy"
    resp = place_order(close_side, cur_qty, reduce_only=True)
    if resp.get("code") == 0:
        log.info(f"Posicao {cur_side} fechada ({cur_qty} contratos)")
    else:
        log.warning(f"Erro ao fechar {cur_side}: {resp}")


# ─────────────────────────────────────────────────────────────
# OHLCV — Dados com Fallback Indestrutivel (Phemex + Binance)
# ─────────────────────────────────────────────────────────────
_OHLCV_ENDPOINTS = [
    # 1. Phemex Mainnet (V2 List — endpoint corrigido)
    f"https://api.phemex.com/md/v2/kline/list?symbol={RAW_SYMBOL}&resolution=1800&limit={{limit}}",
    # 2. Phemex Mainnet (V1 padrao)
    f"https://api.phemex.com/md/kline?symbol={RAW_SYMBOL}&resolution=1800&limit={{limit}}",
    # 3. Binance — bala de prata caso a Phemex caia
    f"https://api.binance.com/api/v3/klines?symbol={RAW_SYMBOL}&interval=30m&limit={{limit}}",
]
_OHLCV_RETRIES = 3
_OHLCV_BACKOFF = 3   # segundos entre tentativas


def _parse_ohlcv_rows(data) -> list:
    """Extrai linhas OHLCV suportando formato Phemex e Binance."""
    # Lista direta = formato Binance: [ts_open, open, high, low, close, vol, ...]
    if isinstance(data, list):
        return [[r[0], float(r[1]), float(r[2]), float(r[3]), float(r[4]), float(r[5])] for r in data]
    # Dicionario = formato Phemex
    inner = data.get("data") or data
    rows  = inner.get("rows") or inner.get("klines") or []
    if not rows:
        raise ValueError(f"OHLCV sem rows: {data}")
    # Phemex: [ts, interval, last, open, high, low, close, vol, turnover]
    return [[r[0], r[3], r[4], r[5], r[6], r[7]] for r in rows]


def fetch_ohlcv(limit: int = 122) -> list:
    """
    Busca candles 30m com retry, backoff e fallback automatico.
    User-Agent de browser para evitar bloqueio 500/403 do Cloudflare.
    """
    last_exc = None
    headers  = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Accept":     "application/json",
    }

    for ep_idx, endpoint_tpl in enumerate(_OHLCV_ENDPOINTS):
        url = endpoint_tpl.format(limit=limit)

        for attempt in range(1, _OHLCV_RETRIES + 1):
            try:
                log.info(f"OHLCV ep={ep_idx+1} tentativa={attempt} url={url}")
                resp = requests.get(url, headers=headers, timeout=15)

                if resp.status_code >= 500:
                    raise requests.HTTPError(
                        f"{resp.status_code} Server Error", response=resp
                    )

                resp.raise_for_status()
                candles = _parse_ohlcv_rows(resp.json())
                log.info(f"OHLCV ok: {len(candles)} barras (ep={ep_idx+1})")
                return candles

            except (requests.HTTPError, requests.Timeout,
                    requests.ConnectionError, ValueError) as exc:
                last_exc = exc
                wait = _OHLCV_BACKOFF * attempt   # 3s, 6s, 9s
                log.warning(
                    f"OHLCV falhou ep={ep_idx+1} tentativa={attempt}/{_OHLCV_RETRIES}: "
                    f"{exc} — aguardando {wait}s"
                )
                if attempt < _OHLCV_RETRIES:
                    time.sleep(wait)

        log.warning(f"OHLCV endpoint {ep_idx+1} esgotado, tentando proximo...")

    raise RuntimeError(f"OHLCV falhou em todos os endpoints. Ultimo erro: {last_exc}")


# ─────────────────────────────────────────────────────────────
# AZLEMA CALCULATION
# ─────────────────────────────────────────────────────────────
def azlema(closes: list, period: int = PERIOD, gain_limit: int = GAIN_LIMIT):
    arr = np.array(closes, dtype=np.float64)
    n   = len(arr)
    if n < period + 10:
        return None, None, None

    alpha  = 2.0 / (period + 1)
    ema    = np.empty(n)
    ema[0] = arr[0]
    for i in range(1, n):
        ema[i] = alpha * arr[i] + (1 - alpha) * ema[i - 1]

    best_gain   = 0.0
    least_error = 1e18
    for g_int in range(-gain_limit * 10, gain_limit * 10 + 1):
        g     = g_int / 10.0
        ec    = np.empty(n)
        ec[0] = arr[0]
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
    return wait + 5


# ─────────────────────────────────────────────────────────────
# TRADING LOOP
# ─────────────────────────────────────────────────────────────

def _order_ok(resp: dict) -> bool:
    """Verifica sucesso na resposta da ordem (REST Phemex pode usar code ou error)."""
    if resp.get("code") == 0:
        return True
    if "error" not in resp and resp.get("data"):
        return True   # sem campo code mas tem data = sucesso em algumas versoes
    return False


def _record(side, price, qty, ec, ema, order_resp):
    data  = order_resp.get("data") or {}
    oid   = data.get("orderID") or data.get("clOrdID") or "—"
    entry = {
        "time":     datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
        "side":     side,
        "price":    round(price, 4),
        "qty":      qty,
        "ec":       round(ec, 4),
        "ema":      round(ema, 4),
        "order_id": oid,
    }
    with _lock:
        trade_history.appendleft(entry)
    log.info(f"TRADE ▶ {side} {qty} contratos @ {price}")


def trading_loop():
    status["running"] = True
    log.info("Trading loop iniciado — aguardando proximo candle 30m")

    while True:
        wait = seconds_to_next_30m()
        log.info(f"Proximo candle em {wait:.0f}s")
        time.sleep(wait)

        try:
            # ── OHLCV direto no testnet ────────────────────────────
            # fetch_ohlcv ja faz retry+backoff+fallback de endpoint
            # Se tudo falhar lanca RuntimeError — pulamos este ciclo
            try:
                candles = fetch_ohlcv(limit=122)
            except RuntimeError as ohlcv_err:
                log.error(f"OHLCV indisponivel, pulando ciclo: {ohlcv_err}")
                with _lock:
                    status["errors"].appendleft({
                        "time": datetime.now().strftime("%H:%M:%S"),
                        "msg":  f"OHLCV falhou (testnet instavel): {str(ohlcv_err)[:120]}",
                    })
                continue

            closes = [c[4] for c in candles[:-1]]   # descarta barra atual
            price  = float(closes[-1])

            # ── AZLEMA ────────────────────────────────────────────
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
                    "last_check": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
                })

            # ── POSICAO E SALDO via API direta ────────────────────
            balance, cur_side, cur_qty = get_balance_and_position()
            with _lock:
                status["position"] = cur_side
                status["balance"]  = round(balance, 2)

            # ── QUANTIDADE ────────────────────────────────────────
            risk_usdt     = RISK_PCT * balance
            qty_eth       = risk_usdt / price if price > 0 else 0
            qty_contracts = max(int(qty_eth / CONTRACT_SIZE), 1)

            log.info(
                f"Signal={signal} | Pos={cur_side}({cur_qty}) | "
                f"EC={ec:.2f} EMA={ema:.2f} | "
                f"Bal={balance:.2f} USDT | Qty={qty_contracts} contratos"
            )

            # ── EXECUTAR ──────────────────────────────────────────
            if signal == "LONG" and cur_side != "LONG":
                if cur_side == "SHORT" and cur_qty > 0:
                    close_position("SHORT", cur_qty)
                    time.sleep(0.5)
                resp = place_order("Buy", qty_contracts)
                if _order_ok(resp):
                    _record("LONG", price, qty_contracts, ec, ema, resp)
                    with _lock:
                        status["position"] = "LONG"
                else:
                    raise Exception(f"Ordem BUY rejeitada: {resp}")

            elif signal == "SHORT" and cur_side != "SHORT":
                if cur_side == "LONG" and cur_qty > 0:
                    close_position("LONG", cur_qty)
                    time.sleep(0.5)
                resp = place_order("Sell", qty_contracts)
                if _order_ok(resp):
                    _record("SHORT", price, qty_contracts, ec, ema, resp)
                    with _lock:
                        status["position"] = "SHORT"
                else:
                    raise Exception(f"Ordem SELL rejeitada: {resp}")

            else:
                log.info(f"Mantendo {cur_side} | sem acao")

        except Exception as exc:
            import traceback
            msg = str(exc)[:200]
            log.error(f"Bot error: {msg}")
            log.error(traceback.format_exc())
            with _lock:
                status["errors"].appendleft({
                    "time": datetime.now().strftime("%H:%M:%S"),
                    "msg":  msg,
                })


# ─────────────────────────────────────────────────────────────
# INTERNAL KEEPALIVE — 3 threads: 8 s / 15 s / 23 s
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
@app.route("/test")
def test_api():
    """
    Dispara imediatamente e mostra respostas brutas da API Phemex.
    Acesse /test no browser para diagnosticar sem esperar 30 minutos.
    """
    results = {}

    # 1) OHLCV direto no testnet
    try:
        candles = fetch_ohlcv(limit=5)
        results["ohlcv"] = {"ok": True, "last_close": candles[-1][4] if candles else None, "bars": len(candles)}
    except Exception as e:
        results["ohlcv"] = {"ok": False, "error": str(e)}

    # 2) GET /g-accounts/accountPositions
    try:
        r = phemex_get("/g-accounts/accountPositions", {"currency": "USDT"})
        results["accountPositions"] = r
    except Exception as e:
        results["accountPositions"] = {"ok": False, "error": str(e)}

    # 3) GET /accounts/accountPositions (endpoint inverso — para comparar)
    try:
        r2 = phemex_get("/accounts/accountPositions", {"currency": "USDT"})
        results["accountPositions_inverse"] = r2
    except Exception as e:
        results["accountPositions_inverse"] = {"ok": False, "error": str(e)}

    # 4) Testar assinatura com /g-orders/activeList (GET, sem criar ordem)
    try:
        r3 = phemex_get("/g-orders/activeList", {"symbol": RAW_SYMBOL})
        results["activeOrders"] = r3
    except Exception as e:
        results["activeOrders"] = {"ok": False, "error": str(e)}

    results["api_key_prefix"] = API_KEY[:8] + "..." if API_KEY else "NOT SET"
    results["secret_set"] = bool(API_SECRET)
    results["testnet_base"] = TESTNET_BASE

    return jsonify(results)


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
        snap           = dict(status)
        snap["errors"] = list(status["errors"])
        trades         = list(trade_history)
        ka             = list(ka_log)
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
    <h5 class="mb-0 me-2">&#9889; AZLEMA Bot</h5>
    <span class="badge {{'bg-success' if s.running else 'bg-danger'}}">
      {{'RUNNING' if s.running else 'STOPPED'}}</span>
    <small class="text-secondary">Phemex Paper &middot; ETH/USDT &middot; 30m &middot; 1&times;</small>
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
            <th>Contratos</th><th>EC</th><th>EMA</th><th>Order ID</th>
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
          Nenhuma trade ainda &mdash; aguardando fechamento do proximo candle 30m.
        </div>
        {% endif %}
      </div>
    </div>

    <div class="col-lg-4">
      <div class="card p-3 mb-3">
        <h6 class="text-secondary mb-2">Keepalive (8s / 15s / 23s)</h6>
        {% for k in ka %}<div class="ka">{{k}}</div>
        {% else %}<div class="text-secondary small">Iniciando...</div>{% endfor %}
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
# STARTUP
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
    log.info("Background threads iniciados (trading + 3x keepalive)")


_start_background()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT, debug=False)
