import os, time as time_mod, threading, logging, json, asyncio, math
import requests as req
from datetime import datetime, timezone
from flask import Flask, jsonify, request

import ccxt
# ccxt.pro (bundled since ccxt 4) gives a real-time websocket tick stream.
# Guarded so the bot still runs (on the REST fallback) if it is unavailable.
try:
    import ccxt.pro as ccxtpro
    _HAS_WS = True
except Exception as _ws_imp_err:                       # pragma: no cover
    ccxtpro, _HAS_WS = None, False
    logging.getLogger(__name__).warning(
        f"[WS] ccxt.pro import failed ({_ws_imp_err}); REST polling only")
import strategy as strat

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CONFIG  (matching Pine Script defaults)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ── Valor do ponto (mintick) — DOIS valores separados (pedido 2026-07-01) ────
# O usuário quer controlar quanto vale 1 ponto do SL SEPARADO de quanto vale
# 1 ponto do TP (ativação) e do trailing:
#   SL distance      = sl_pts × mintick_sl
#   trail ACTIVATION = tp_pts × mintick_tp
#   trail OFFSET     = tr_pts × mintick_tp
# (No Pine os três usam o MESMO syminfo.mintick — pra reproduzir o TV é só
# deixar mintick_sl = mintick_tp.) Editável no dashboard; env MINTICK_OVERRIDE
# define o default inicial dos dois.
MINTICK_DEFAULT = float(os.environ.get("MINTICK_OVERRIDE", 0.01))
FIXED_SL     = 2000    # Pine fixedSL       (loss)         → SL distance      = ×mintick
FIXED_TP     = 55      # Pine fixedTP       (trail_points) → trail ACTIVATION = ×mintick
TRAIL_OFFSET = 15      # Pine trail_offset  (literal 15)   → trail OFFSET      = ×mintick

# ── Position sizing ───────────────────────────────────────────────────────
# Position notional = the FULL balance (~1x): qty = balance / price. So every
# trade compounds the account by the SAME % it shows in the history, and the
# total profit finally AGREES with the trades. This is DECOUPLED from the
# distance MINTICK on purpose: raising MINTICK to widen the trailing must NOT
# shrink the position (that coupling made the total ~10× below the per-trade
# %s). See _calc_qty.

# ── Editable config (persisted to config.json) ──────────────────────────────
# POINT counts (sl/tp/tr) + DOIS valores de ponto (pedido do usuário 2026-07-01):
#   mintick_sl → quanto vale 1 ponto do SL
#   mintick_tp → quanto vale 1 ponto do TP (ativação) e do trailing
# Cada distância = seus pontos × o mintick do seu grupo:
#   SL = sl_pts × mintick_sl · TP = tp_pts × mintick_tp · Trail = tr_pts × mintick_tp
CONFIG_FILE = "config.json"
def _default_config():
    # User-chosen defaults (2026-07-01): SL 2000pt, TP-activation 55pt, trail
    # 15pt, Gain Limit 1000, mintick 0.1 nos dois grupos (SL $200 · TP $5.50 ·
    # trail $1.50). Tudo editável no dashboard.
    return {"sl_pts": 2000, "tp_pts": 55, "tr_pts": 15,
            "mintick_sl": 0.1, "mintick_tp": 0.1,
            "gain_limit": 1000}
def _load_config():
    try:
        with open(CONFIG_FILE) as f:
            d = json.load(f)
        base = _default_config()
        base.update({k: d[k] for k in base if k in d})
        # migração: config antigo tinha UM "mintick" pra tudo — ele vira o
        # default dos dois valores de ponto se os novos não estiverem no arquivo
        if "mintick" in d:
            if "mintick_sl" not in d: base["mintick_sl"] = d["mintick"]
            if "mintick_tp" not in d: base["mintick_tp"] = d["mintick"]
        return base
    except Exception:
        return _default_config()
def _save_config():
    try:
        with open(CONFIG_FILE, "w") as f: json.dump(config, f)
    except Exception as e: logging.getLogger(__name__).warning(f"[CONFIG] {e}")

config = _load_config()

# SL_DIST / TP_DIST / TR_DIST ($) are what the exit logic reads. _apply_config
# converts the editable POINT counts → $ distances (called on load + every
# save): o SL usa mintick_sl; o TP (ativação) e o trailing usam mintick_tp.
SL_DIST = TP_DIST = TR_DIST = 0.0
MINTICK_SL = MINTICK_TP = MINTICK_DEFAULT
def _apply_config():
    global SL_DIST, TP_DIST, TR_DIST, MINTICK_SL, MINTICK_TP
    MINTICK_SL = float(config["mintick_sl"])
    MINTICK_TP = float(config["mintick_tp"])
    SL_DIST = round(float(config["sl_pts"]) * MINTICK_SL, 8)
    TP_DIST = round(float(config["tp_pts"]) * MINTICK_TP, 8)
    TR_DIST = round(float(config["tr_pts"]) * MINTICK_TP, 8)
_apply_config()

# ── Exit evaluation mode ─────────────────────────────────────────────────
# THE USER WANTS INTRA-CANDLE EXITS: the trade closes LIVE, during the candle, at
# the real price the moment it retraces trail_offset (15 ticks) from the running
# peak — NOT snapped to the candle's O/H/L/C extreme at candle close. ("Não fechar
# em extremos, porque é intra-candle.")
#
# So REALTIME_EXITS defaults ON and _check_sl_tp (real-time peak + trailing) owns
# exits, fed by the websocket TICK stream. The candle-close stop check is SKIPPED
# for exits in this mode (see section 2a), so nothing is ever closed "at the
# extreme". The trail still ratchets up with each new live peak and only closes on
# a real 15-tick pullback.
#
# REALTIME_EXITS=0 switches to the candle model (exits at candle close from the
# O/H/L/C). Kept ON per the user: exits must be intra-candle / live.
REALTIME_EXITS  = os.environ.get("REALTIME_EXITS", "1") == "1"

RISK            = 0.01
MAX_LOTS        = 100
CANDLES         = 200
PAPER_START_BAL = 1_000.0   # matches Pine initial_capital=1000
PORT            = int(os.environ.get("PORT", 10000))

# Configurable trade fees (persisted to disk)
FEES_FILE = "fees.json"
def _load_fees():
    # Pine has commission_value=0, so default to 0 to match the TradingView
    # backtest. Change these in the UI if you want to simulate real exchange fees.
    try:
        with open(FEES_FILE) as f: return json.load(f)
    except Exception: return {"entry": 0.0, "exit": 0.0}
def _save_fees():
    try:
        with open(FEES_FILE, "w") as f: json.dump(fees, f)
    except Exception: pass

fees = _load_fees()
logger.info(f"[FEES] entry={fees['entry']}%  exit={fees['exit']}%")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TICK RECORDER  (for a 1:1-real backtest)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Phemex has NO 1-second historical candle (smallest is 1m), so a synthetic path
# is the best we can do for HISTORICAL backtests. But the live bot already sees a
# real ~1-second price stream (the ticker websocket + the 1s REST poll). If we
# RECORD that exact stream, the backtest can REPLAY it and match paper trading
# ~1:1 for the recorded period — no interpolation, the real prices the exit engine
# actually saw. LIGADO POR PADRÃO desde 2026-07-02 (pedido do usuário — nada a
# configurar; env RECORD_TICKS=0 desliga se um dia precisar).
#
# Caveats: (1) it only works FORWARD — you cannot record the past; a gravação só
# acontece enquanto a strategy está INICIADA (Start). (2) On Render's free tier
# the disk is EPHEMERAL: ticks.csv is wiped on every redeploy/restart, so attach
# a Persistent Disk (or point TICKS_FILE at one) to keep the history.
RECORD_TICKS = os.environ.get("RECORD_TICKS", "1") == "1"
TICKS_FILE   = os.environ.get("TICKS_FILE", "ticks.csv")

_tick_buf        = []
_tick_buf_lock   = threading.Lock()
_tick_last_flush = 0.0
_tick_last_px    = 0.0

def _record_tick(ts_ms: int, price: float):
    """Append (timestamp_ms, price) to ticks.csv — buffered & flushed every ~5 s so
    it never does disk I/O on the hot path per tick. Exception-proof: recording can
    NEVER break trading. Consecutive identical prices are skipped to save space
    (the exit logic only cares about price CHANGES)."""
    if not RECORD_TICKS:
        return
    global _tick_last_flush, _tick_last_px
    try:
        lines = None
        with _tick_buf_lock:
            if price == _tick_last_px:
                return
            _tick_last_px = price
            _tick_buf.append((int(ts_ms), float(price)))
            now = time_mod.time()
            if now - _tick_last_flush >= 5:
                lines = "".join(f"{t},{p}\n" for t, p in _tick_buf)
                _tick_buf.clear()
                _tick_last_flush = now
        if lines:
            with open(TICKS_FILE, "a") as f:
                f.write(lines)
    except Exception:
        pass   # recording is best-effort; trading must never be affected

def _load_recorded_ticks():
    """Read ticks.csv → sorted list of (ts_ms, price). Empty list if none."""
    out = []
    try:
        with open(TICKS_FILE) as f:
            for ln in f:
                try:
                    t, p = ln.split(",")
                    out.append((int(t), float(p)))
                except Exception:
                    continue
    except Exception:
        return []
    out.sort(key=lambda x: x[0])
    return out

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# EXCHANGE  (Phemex live — OHLCV + ticker, no auth needed)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# live_ex   : used for OHLCV (rate-limited — fine, only polled every 15 s)
# ticker_ex : SEPARATE instance, rate-limit disabled, used ONLY for the 1-s
#             price poll. ccxt's own price-scale handling is battle-tested —
#             we do NOT guess/derive the scale manually (that was the bug:
#             a one-off guessed ratio silently drifted wrong over time and
#             corrupted every subsequent SL/TP price).
# timeout 8 s explícito: numa hospedagem onde a Phemex não responde, a conexão
# fica PENDURADA e o init/loop trava minutos em "Iniciando…" sem dizer nada —
# com timeout curto a falha vira erro visível no status em segundos.
live_ex = ccxt.phemex({"options": {"defaultType": "swap"}, "enableRateLimit": True,
                       "timeout": 8000})
# ticker_ex: dedicated instance with a LIGHT custom rate limit — fast enough
# for ~1 req/s, but still self-throttled so Phemex never bans/blocks the IP
# (disabling rate-limit entirely caused exactly that — silent total failure).
ticker_ex = ccxt.phemex({
    "options": {"defaultType": "swap"},
    "enableRateLimit": True,
    "rateLimit": 250,     # ms between requests — safe, still ~4 req/s ceiling
    "timeout": 6000,
})
# bt_ex: SEPARATE instance for the backtest's bulk OHLCV fetches, so they never
# run concurrently with the strategy loop on the same ccxt session (a requests
# Session is not thread-safe for concurrent calls).
bt_ex = ccxt.phemex({"options": {"defaultType": "swap"}, "enableRateLimit": True,
                     "timeout": 8000})
SYMBOL = None

_last_good_price   = 0.0
_ticker_fail_count = 0
_jump_reject_count = 0     # consecutive jump-filter rejects → resync after a few
_MAX_TICK_JUMP_PCT = 1.5   # ETH/USDT should not move >1.5% in a single second
_ohlcv_last_err    = None  # último erro do fetch de candles 30m — vai pro dashboard

# ── Real-time concurrency ────────────────────────────────────────────────
# trade_lock serializes EVERY mutation of the `paper` position so the
# websocket exit thread and the main candle/entry loop can never race (double
# close, half-applied entry, corrupted balance).
# _ws_active is True only while the websocket tick stream is delivering prices;
# when True the main loop stops its REST poll and lets the WS drive exits.
# _last_candle_str mirrors the loop's candle label so WS-driven exits can tag
# the trade history with the right candle.
trade_lock       = threading.Lock()
_ws_active       = False
_last_candle_str = "—"
_last_price_ts   = 0.0     # time.time() of the last price update (WS or REST)
_last_ws_tick_ts = 0.0     # time.time() of the last WEBSOCKET tick specifically
_ws_tick_count   = 0       # how many websocket ticks have been processed

def init_markets():
    global SYMBOL
    mkts = live_ex.load_markets()
    ticker_ex.load_markets()
    for sym in ["ETH/USDT:USDT", "ETH/USD:ETH"]:
        if sym in mkts:
            SYMBOL = sym; break
    if not SYMBOL:
        for k, v in mkts.items():
            if "ETH" in k and v.get("type") in ("swap", "future"):
                SYMBOL = k; break
    if not SYMBOL:
        raise RuntimeError("No ETH perpetual found on Phemex")

    # ── SL/TP/trail distances come from the editable config (config.json / UI) ──
    _apply_config()
    logger.info(f"[INIT] SL=${SL_DIST}  trail-activation=${TP_DIST}  "
                f"trail-offset=${TR_DIST}  (editable in the dashboard)")
    # Warm up ticker_ex once at init so the first loop iteration isn't 0.0
    try:
        warm = float(ticker_ex.fetch_ticker(SYMBOL)["last"])
        if warm > 0:
            global _last_good_price
            _last_good_price = warm
            logger.info(f"[INIT] ticker warm-up OK price={warm:.2f}")
    except Exception as e:
        logger.warning(f"[INIT] ticker warm-up failed: {type(e).__name__}: {e}")
    logger.info(f"[INIT] symbol={SYMBOL}")

def get_live_price() -> float:
    """
    Real-time price via ccxt with a light, safe rate limit (avoids exchange
    bans). Sanity filter rejects single-tick jumps > _MAX_TICK_JUMP_PCT,
    which would indicate a corrupted read rather than real price movement.
    Logs the actual exception type/message on failure so root cause is
    visible in Render logs instead of failing silently.
    """
    global _last_good_price, _ticker_fail_count, _jump_reject_count, _last_price_ts
    try:
        price = float(ticker_ex.fetch_ticker(SYMBOL or "ETH/USDT:USDT")["last"])
        if price <= 0:
            _ticker_fail_count += 1
            if _ticker_fail_count % 10 == 1:
                logger.warning(f"[TICKER] got non-positive price ({price})")
            return _last_good_price

        if _last_good_price > 0:
            jump_pct = abs(price - _last_good_price) / _last_good_price * 100
            if jump_pct > _MAX_TICK_JUMP_PCT:
                _jump_reject_count += 1
                # A one-off corrupted read is rejected; but a REAL sustained move
                # keeps "jumping" vs the now-stale price, so after a few rejects in
                # a row we ACCEPT it and resync — otherwise the price freezes
                # forever (this was the "not receiving live prices" bug).
                if _jump_reject_count < 3:
                    logger.warning(f"[TICKER] rejected jump {_last_good_price:.2f} "
                                   f"→ {price:.2f} ({jump_pct:.2f}%) "
                                   f"#{_jump_reject_count}")
                    return _last_good_price
                logger.warning(f"[TICKER] resync after {_jump_reject_count} "
                               f"rejected jumps → {price:.2f}")

        _last_good_price   = price
        _last_price_ts     = time_mod.time()
        _ticker_fail_count = 0
        _jump_reject_count = 0
        return price
    except Exception as e:
        _ticker_fail_count += 1
        # Log every failure for the first 5, then every 10th — visible in
        # Render logs without flooding them.
        if _ticker_fail_count <= 5 or _ticker_fail_count % 10 == 0:
            logger.warning(f"[TICKER] fail #{_ticker_fail_count} "
                            f"{type(e).__name__}: {e}")
        return _last_good_price

def fetch_candles(limit=CANDLES):
    """Candles de 30m pro bot ao vivo, com RETRY (igual ao backtest): uma falha
    transitória da Phemex não pode deixar o bot um ciclo inteiro sem sinal. O
    último erro fica em _ohlcv_last_err e aparece NO DASHBOARD (status) — sem
    candles não há trade, e antes isso ficava invisível ("Running" mudo)."""
    global _ohlcv_last_err
    since = int((time_mod.time() - limit * 1800) * 1000)
    for a in range(3):
        try:
            rows = live_ex.fetch_ohlcv(SYMBOL or "ETH/USDT:USDT",
                                       timeframe="30m", since=since, limit=limit)
            if rows and len(rows) >= 70:
                _ohlcv_last_err = None
                return rows
            _ohlcv_last_err = f"Phemex retornou só {len(rows) if rows else 0} candles"
            logger.warning(f"[OHLCV] {_ohlcv_last_err}")
        except Exception as e:
            _ohlcv_last_err = f"{type(e).__name__}: {str(e)[:200]}"
            logger.error(f"[OHLCV] tentativa {a+1}/3: {_ohlcv_last_err}")
        if a < 2:
            time_mod.sleep(1.5 * (a + 1))
    return []

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PAPER TRADING STATE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
paper = {
    "balance":      PAPER_START_BAL,
    "side":         "none",
    "entry_price":  0.0,
    "qty":          0.0,
    "peak":         0.0,
    "trail_active": False,
    "trail_stop":   0.0,
    "unrealized":   0.0,
    "mfe_pct":      0.0,   # max favorable excursion (best unrealized %) this trade
    "mae_pct":      0.0,   # max adverse excursion (worst unrealized %) this trade
    "total_pnl":    0.0,
    "total_pnl_pct": 0.0,
}

def _pnl_pct(raw_pnl_usd, entry_price, qty):
    """Net PnL % after fees."""
    position_value = entry_price * qty
    if position_value == 0:
        return 0.0
    fee_cost_pct = fees["entry"] + fees["exit"]
    raw_pct      = (raw_pnl_usd / position_value) * 100
    return round(raw_pct - fee_cost_pct, 4)

def _unrealized_pct(cur_price):
    if paper["side"] == "none" or paper["entry_price"] == 0:
        return 0.0
    diff = cur_price - paper["entry_price"]
    if paper["side"] == "short":
        diff = -diff
    raw_pct = (diff / paper["entry_price"]) * 100
    return round(raw_pct - fees["entry"], 4)   # subtract entry fee already paid

def _open_position(side, qty, price):
    paper["side"]         = "long" if side == "LONG" else "short"
    paper["entry_price"]  = price
    paper["qty"]          = qty
    paper["peak"]         = price
    paper["trail_active"] = False
    paper["trail_stop"]   = 0.0
    paper["unrealized"]   = 0.0
    paper["mfe_pct"]      = 0.0   # reset excursions for the new trade
    paper["mae_pct"]      = 0.0
    oid = f"SIM-{int(time_mod.time())}"
    logger.info(f"[PAPER] OPEN {side}  qty={qty}  @ {price:.2f}  id={oid}")
    return {"id": oid}

def _close_position(price, reason):
    entry = paper["entry_price"]
    qty   = paper["qty"]
    raw   = (price - entry) * qty if paper["side"] == "long" else (entry - price) * qty
    pct   = _pnl_pct(raw, entry, qty)
    paper["balance"]       += raw - (entry * qty * fees["entry"] / 100) \
                                  - (price  * qty * fees["exit"]  / 100)
    paper["total_pnl"]     += raw
    paper["total_pnl_pct"]  = round(
        (paper["balance"] - PAPER_START_BAL) / PAPER_START_BAL * 100, 2)
    paper["unrealized"]     = 0.0
    logger.info(f"[PAPER] CLOSE {paper['side'].upper()} ({reason})  @ {price:.2f}  "
                f"pnl={pct:+.2f}%  bal={paper['balance']:.2f}")
    paper["side"]         = "none"
    paper["entry_price"]  = 0.0
    paper["qty"]          = 0.0
    paper["peak"]         = 0.0
    paper["trail_active"] = False
    paper["trail_stop"]   = 0.0
    return raw, pct

def _calc_qty(price, balance):
    # Position notional = full balance (1x): qty = balance / price. On the
    # TradingView run the position tracked the account exactly — trade #976 held
    # 5.62 ETH and 5.62 = balance / price. So each trade compounds the balance by
    # the SAME % it shows, and the total finally agrees with the trades (the old
    # risk/(fixedSL·mintick) sizing shrank the position, so the total grew far
    # slower than the per-trade %s). Capped at MAX_LOTS like Pine.
    qty = (balance / price) if price else 0.0
    return round(min(qty, MAX_LOTS), 4)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SL / TRAILING TP  — two variants
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def _check_sl_tp(pos, cur_price: float, sl_d=None, tp_d=None, tr_d=None):
    """
    Intra-candle exit math — the SINGLE source of truth used by BOTH live paper
    trading (pos = the global `paper`) AND the backtest (pos = a local dict), so
    the two can never diverge again. Mutates pos["peak"]/["trail_active"]/
    ["trail_stop"] and returns (exit, exit_price, reason).

    sl_d/tp_d/tr_d: o bot ao vivo NÃO passa (usa o config global); um backtest
    paralelo passa as SUAS distâncias, então 5 slots podem simular configs
    diferentes ao mesmo tempo sem interferir um no outro nem no bot.

    Long  : peak = highest price since entry; trail arms once peak-entry >= tp_d
            then sits tr_d below peak. SL is FIXED at entry-sl_d (Pine `loss`).
    Short : symmetric (peak = lowest price; trail sits tr_d above it).
    """
    if sl_d is None:
        sl_d, tp_d, tr_d = SL_DIST, TP_DIST, TR_DIST
    if pos["side"] == "none":
        return False, 0.0, ""
    entry = pos["entry_price"]

    if pos["side"] == "long":
        if cur_price > pos["peak"]:
            pos["peak"] = cur_price
        if pos["peak"] - entry >= tp_d:
            nt = round(pos["peak"] - tr_d, 8)
            if not pos["trail_active"] or nt > pos["trail_stop"]:
                pos["trail_active"] = True
                pos["trail_stop"]   = nt
        if pos["trail_active"] and cur_price <= pos["trail_stop"]:
            return True, pos["trail_stop"], "TRAIL_TP"
        sl = round(entry - sl_d, 8)                  # FIXED stop-loss (Pine `loss`)
        if cur_price <= sl:
            return True, sl, "SL"
    else:
        if pos["peak"] == 0 or cur_price < pos["peak"]:
            pos["peak"] = cur_price
        if entry - pos["peak"] >= tp_d:
            nt = round(pos["peak"] + tr_d, 8)
            if not pos["trail_active"] or nt < pos["trail_stop"]:
                pos["trail_active"] = True
                pos["trail_stop"]   = nt
        if pos["trail_active"] and cur_price >= pos["trail_stop"]:
            return True, pos["trail_stop"], "TRAIL_TP"
        sl = round(entry + sl_d, 8)                  # FIXED stop-loss (Pine `loss`)
        if cur_price >= sl:
            return True, sl, "SL"

    return False, 0.0, ""

def _check_sl_tp_candle(o: float, h: float, l: float, c: float):
    """
    Closed-candle exit check — mirrors the TradingView broker emulator with
    calc_on_every_tick=false (no bar magnifier).

    Two things matter for parity:

    1. Which stop is live. Pine generates the fixed stop (loss) AND the trailing
       stop but only ever uses the one closest to price. So before the trail
       arms (peak profit < trail-activation) the live stop is the fixed SL; once
       it arms, the live stop is the trailing stop (which sits nearer to price).

    2. Intra-bar order. Without a bar magnifier TradingView assumes:
         up bar   (close >= open):  open -> low  -> high -> close
         down bar (close <  open):  open -> high -> low  -> close
       i.e. it processes the adverse extreme first on up bars and the favorable
       extreme first on down bars. We replicate that ordering so SL vs trail
       fire in the same sequence TradingView would use.
    """
    if paper["side"] == "none":
        return False, 0.0, ""

    entry  = paper["entry_price"]
    is_up  = c >= o

    if paper["side"] == "long":
        sl = round(entry - SL_DIST, 8)

        def hit_low():
            # Check whichever stop is currently live against this bar's low.
            if paper["trail_active"]:
                if l <= paper["trail_stop"]:
                    return True, paper["trail_stop"], "TRAIL_TP"
            elif l <= sl:
                return True, sl, "SL"
            return False, 0.0, ""

        def advance_high():
            # Update peak with the bar high and arm/raise the trailing stop.
            if h > paper["peak"]:
                paper["peak"] = h
            if paper["peak"] - entry >= TP_DIST:
                nt = round(paper["peak"] - TR_DIST, 8)
                if not paper["trail_active"] or nt > paper["trail_stop"]:
                    paper["trail_active"] = True
                    paper["trail_stop"]   = nt

        if is_up:                       # open -> low -> high -> close
            ex = hit_low()              # adverse leg (open->low) first
            if ex[0]:
                return ex
            advance_high()              # favorable leg (low->high): ride peak, arm trail
            # retrace leg (high->close): if the price falls trail_offset below the
            # new peak on its way to the close, the trailing stop fires in THIS
            # same candle — this is how a winner exits at extreme ± offset on the
            # very candle it ran (e.g. the screenshots' same-candle exits).
            if paper["trail_active"] and c <= paper["trail_stop"]:
                return True, paper["trail_stop"], "TRAIL_TP"
        else:                           # open -> high -> low
            advance_high()
            ex = hit_low()
            if ex[0]:
                return ex

    else:  # short
        sl = round(entry + SL_DIST, 8)

        def hit_high():
            if paper["trail_active"]:
                if h >= paper["trail_stop"]:
                    return True, paper["trail_stop"], "TRAIL_TP"
            elif h >= sl:
                return True, sl, "SL"
            return False, 0.0, ""

        def advance_low():
            if paper["peak"] == 0 or l < paper["peak"]:
                paper["peak"] = l
            if entry - paper["peak"] >= TP_DIST:
                nt = round(paper["peak"] + TR_DIST, 8)
                if not paper["trail_active"] or nt < paper["trail_stop"]:
                    paper["trail_active"] = True
                    paper["trail_stop"]   = nt

        if is_up:                       # open -> low -> high : favorable first
            advance_low()
            ex = hit_high()
            if ex[0]:
                return ex
        else:                           # open -> high -> low -> close : adverse first
            ex = hit_high()
            if ex[0]:
                return ex
            advance_low()               # favorable leg (high->low): ride peak, arm trail
            # retrace leg (low->close): if the price rises trail_offset above the
            # new low on its way to the close, the trailing stop fires in THIS
            # same candle (the #976-style same-candle short exit).
            if paper["trail_active"] and c >= paper["trail_stop"]:
                return True, paper["trail_stop"], "TRAIL_TP"

    return False, 0.0, ""

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TRADE HISTORY
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TRADES_FILE = "trades.json"

def _load_history():
    try:
        with open(TRADES_FILE) as f: return json.load(f)
    except Exception: return []

def _save_history():
    try:
        with open(TRADES_FILE, "w") as f: json.dump(trade_history, f)
    except Exception as e: logger.warning(f"[HISTORY] {e}")

def _record_exit(side_str, price, reason, pnl_usd, pnl_pct_val, candle_str):
    # paper["mfe_pct"]/["mae_pct"] still hold THIS trade's excursions here:
    # _close_position does not reset them (only _open_position does), so they are
    # the max profit / max loss % the trade reached while it was open.
    trade_history.insert(0, {
        "time":     datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "candle":   candle_str,
        "side":     f"CLOSE {side_str} ({reason})",
        "price":    round(price, 4),
        "qty":      0,
        "order_id": f"EXIT-{reason}",
        "balance":  f"{paper['balance']:.2f}",
        "pnl_pct":  f"{pnl_pct_val:+.2f}",
        "mfe":      f"{paper['mfe_pct']:+.2f}",
        "mae":      f"{paper['mae_pct']:+.2f}",
    })
    _save_history()

def _record_entry(side_str, price, qty, candle_str, order_id):
    trade_history.insert(0, {
        "time":     datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "candle":   candle_str,
        "side":     side_str,
        "price":    round(price, 4),
        "qty":      qty,
        "order_id": order_id,
        "balance":  f"{paper['balance']:.2f}",
        "pnl_pct":  "—",
        "mfe":      "—",
        "mae":      "—",
    })
    _save_history()

def _winrate():
    closed = [t for t in trade_history
              if isinstance(t.get("pnl_pct"), str) and t["pnl_pct"] != "—"]
    if not closed: return "—"
    wins = sum(1 for t in closed if float(t["pnl_pct"]) > 0)
    return f"{wins/len(closed)*100:.1f}%  ({wins}/{len(closed)})"

trade_history    = _load_history()
_strategy_thread = None
logger.info(f"[HISTORY] {len(trade_history)} trades loaded")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SHARED STATE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
state = {
    "running":     False, "status":     "Stopped",
    "signal":      "—",   "position":   "None",
    "ema":         "—",   "ec":         "—",    "period":    "—",
    "last_candle": "—",   "updated":    "—",
    "balance":     f"{PAPER_START_BAL:.2f}",
    "total_pnl_pct": "0.00",
    "unrealized":  "0.00",
    "mfe":         "0.00",
    "mae":         "0.00",
    "trail_stop":  "—",
    "winrate":     "—",
    "price":       "—",
}

def _resolve_candles(ohlcv):
    """
    Returns (last_closed, sig_closes).
    A candle with timestamp T covers T … T+1800 s.
    If T+1800 > now  →  still forming  →  skip it.
    """
    TF_MS   = 1800 * 1000
    now_ms  = int(time_mod.time() * 1000)
    if ohlcv[-1][0] + TF_MS > now_ms:
        # Last bar is still forming — use second-to-last as closed
        return ohlcv[-2], ohlcv[:-1]
    else:
        # All bars are closed — last bar is the most recent closed
        return ohlcv[-1], ohlcv


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STRATEGY LOOP
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def _try_realtime_exit(price: float, candle_str: str):
    """
    Run the tick-by-tick SL/trailing check and close the position if it fires.
    Thread-safe: called from the websocket tick handler AND the REST fallback,
    so all position mutation happens under trade_lock. Returns True if it closed.
    """
    with trade_lock:
        if paper["side"] == "none":
            return False
        ex, ex_px, ex_rsn = _check_sl_tp(paper, price)
        if not ex:
            return False
        ex_side  = paper["side"]
        raw, pct = _close_position(ex_px, ex_rsn)
        _record_exit(ex_side.upper(), ex_px, ex_rsn, raw, pct, candle_str)
        state.update({
            "position": "none", "unrealized": "0.00",
            "balance":  f"{paper['balance']:.2f}",
            "total_pnl_pct": f"{paper['total_pnl_pct']:+.2f}",
            "trail_stop": "—",
            "winrate":  _winrate(),
        })
        logger.info(f"[EXIT] {ex_rsn} @ {ex_px:.2f}  pnl={pct:+.2f}%")
        return True


def on_live_price(price: float, from_ws: bool = False):
    """
    Called on EVERY price update — each websocket tick (from_ws=True) or each
    ~1 s REST poll. Tracks the SL/trailing in REAL TIME and exits intra-candle,
    exactly like the standing strategy.exit order the user watched close live.

    Exit driving: the websocket (true ticks) drives exits whenever it is healthy;
    the coarser 1 s REST poll only drives exits when the websocket has gone
    silent (>5 s). So normal sampling jitter never closes a trade early, but a
    websocket stall can never freeze it either.
    """
    global _last_good_price, _last_price_ts, _last_ws_tick_ts, _jump_reject_count
    if price <= 0:
        return
    if _last_good_price > 0:
        jump = abs(price - _last_good_price) / _last_good_price * 100
        if jump > _MAX_TICK_JUMP_PCT:
            _jump_reject_count += 1
            if _jump_reject_count < 3:
                return                          # reject a one-off corrupted tick
            # sustained move → resync instead of freezing on a stale price
    _last_good_price = price
    now = time_mod.time()
    _last_price_ts   = now
    _jump_reject_count = 0
    if from_ws:
        _last_ws_tick_ts = now

    # Record the exact price the exit engine is about to act on, so a "ticks"
    # backtest can replay the real stream (1:1 with paper trading). No-op unless
    # RECORD_TICKS=1. Timestamped in ms to bucket into 30m candles later.
    _record_tick(int(now * 1000), price)

    paper["unrealized"] = _unrealized_pct(price)
    # Track the max favorable / adverse excursion (best & worst unrealized %)
    # the trade reached while open — BEFORE the exit check (so a trade that hits
    # its peak and then closes still records that peak).
    if paper["side"] != "none":
        u = paper["unrealized"]
        if u > paper["mfe_pct"]:
            paper["mfe_pct"] = u
        if u < paper["mae_pct"]:
            paper["mae_pct"] = u
        mfe_disp, mae_disp = f"{paper['mfe_pct']:+.2f}", f"{paper['mae_pct']:+.2f}"
    else:
        mfe_disp, mae_disp = "0.00", "0.00"

    # Check the SL/trailing on EVERY price update — every websocket tick AND the
    # 1 s loop poll. The old gate skipped the loop's check while the WS *looked*
    # healthy; if the WS then went quietly stale, exits went unchecked and trades
    # only closed at the candle end. With the trailing offset now $1.5 (not the
    # old $0.15), the coarse 1 s sampling can't trigger an early exit on jitter
    # (a real $1.5 move ≠ jitter), so always checking is safe. After a close,
    # side is 'none' → the duplicate check is a harmless no-op (no double close).
    if REALTIME_EXITS:
        _try_realtime_exit(price, _last_candle_str)

    trail_disp = (f"${paper['trail_stop']:.2f}" if paper["trail_active"]
                  else ("armed" if paper["side"] != "none" else "—"))
    state.update({
        "price":         f"{price:.2f}",
        "unrealized":    f"{paper['unrealized']:+.2f}",
        "mfe":           mfe_disp,
        "mae":           mae_disp,
        "balance":       f"{paper['balance']:.2f}",
        "total_pnl_pct": f"{paper['total_pnl_pct']:+.2f}",
        "trail_stop":    trail_disp,
        "position":      paper["side"],
        # o preço pode estar vivo com o fetch de candles morto — sem candles não
        # há sinal nem trade, então o status TEM que mostrar esse erro
        "status":        ("Running" if not _ohlcv_last_err
                          else ("Erro candles 30m: " + _ohlcv_last_err)[:110]),
    })


def price_ws_loop():
    """
    Background websocket tick feed. Subscribes to Phemex's real-time ticker and
    pushes every price into on_live_price(), so the SL/trailing exit fires the
    instant the move happens — no REST round-trip, no 1 s sampling delay. ccxt.pro
    normalizes the price exactly like the REST client (so we never hand-roll the
    price scale — that was a past bug). Auto-reconnects; if the WS is down,
    _ws_active stays False and the main loop falls back to its REST poll.
    """
    global _ws_active
    if not _HAS_WS:
        logger.warning("[WS] ccxt.pro unavailable — staying on REST polling")
        return

    async def run():
        global _ws_active, _ws_tick_count
        ws  = ccxtpro.phemex({"options": {"defaultType": "swap"},
                              "enableRateLimit": True})
        sym = SYMBOL or "ETH/USDT:USDT"
        try:
            await ws.load_markets()
            logger.info(f"[WS] connecting ticker stream for {sym}")
            fails = 0
            while state["running"]:
                try:
                    t  = await ws.watch_ticker(sym)
                    px = t.get("last") or t.get("close")
                    if px:
                        if not _ws_active:
                            logger.info("[WS] tick stream LIVE — exits now real-time")
                        _ws_active = True
                        _ws_tick_count += 1
                        fails = 0
                        on_live_price(float(px), from_ws=True)
                except Exception as e:
                    fails += 1
                    _ws_active = False
                    if fails <= 5 or fails % 20 == 0:
                        logger.warning(f"[WS] watch fail #{fails} "
                                       f"{type(e).__name__}: {e}")
                    await asyncio.sleep(min(fails, 5))
        finally:
            try:
                await ws.close()
            except Exception:
                pass

    try:
        asyncio.run(run())
    except Exception as e:
        logger.error(f"[WS] loop crashed: {type(e).__name__}: {e}")
    finally:
        _ws_active = False


def strategy_loop():
    global state, trade_history, _last_candle_str

    try:
        init_markets()
    except Exception as e:
        logger.error(f"[INIT] {e}")
        state["status"] = f"Error: {e}"
        state["running"] = False
        return

    last_candle_ts   = None
    last_candle_str  = "—"
    last_ohlcv_check = 0.0   # throttle ohlcv fetching to every 15 s

    # Startup sync: LOCK to the current last-closed candle so the first ENTRY
    # only happens at the NEXT candle boundary (:00 / :30) — entries must be at a
    # candle open, at that candle's price. (Entering mid-candle on Start used a
    # stale candle-open price and was wrong.) The trade-off is the first trade can
    # take up to 30 min; that wait is normal, not a failure.
    init_ohlcv = fetch_candles()
    if init_ohlcv and len(init_ohlcv) >= 70:
        _lc, _ = _resolve_candles(init_ohlcv)
        last_candle_ts = _lc[0]
        logger.info(f"[SYNC] locked to candle "
                    f"{datetime.fromtimestamp(last_candle_ts/1000,tz=timezone.utc)}"
                    f" — first entry at the next candle close")

    logger.info(f"[CFG] REALTIME_EXITS={REALTIME_EXITS}  ws_available={_HAS_WS}  "
                f"(if REALTIME_EXITS is False, exits only happen at candle close — "
                f"check the Render env var)")

    # Start the real-time websocket tick feed (delay-free exits). Daemon thread;
    # if it can't connect, the ~1 s REST poll below keeps managing exits.
    threading.Thread(target=price_ws_loop, daemon=True).start()

    while state["running"]:
        try:
            # ── 1. PRICE + REAL-TIME EXITS ───────────────────────────────────
            # The websocket feed (price_ws_loop → on_live_price) fires exits the
            # instant a tick arrives. This ~1 s REST poll ALWAYS runs too, as a
            # guaranteed safety net: if the websocket silently stalls (stays
            # "connected" but stops sending ticks), exits still happen within a
            # second instead of freezing. Both call the same thread-safe
            # on_live_price(); after a close, side is 'none' so the next check is
            # a harmless no-op — there is never a double close.
            live_price = get_live_price()
            if live_price > 0:
                on_live_price(live_price)

            # ── 2. CANDLE CLOSE  (every 15 s — this is where trades happen) ──
            now = time_mod.time()
            if now - last_ohlcv_check >= 15:
                last_ohlcv_check = now
                ohlcv = fetch_candles()
                if not ohlcv:
                    # sem candles = sem sinal e sem trade — mostra o motivo NO
                    # dashboard em vez de fingir que está tudo bem
                    state["status"] = ("Erro candles 30m: "
                                       + (_ohlcv_last_err or "desconhecido"))[:110]

                if ohlcv and len(ohlcv) >= 70:
                    last_closed, sig_closes = _resolve_candles(ohlcv)
                    candle_ts = last_closed[0]
                    last_candle_str = datetime.fromtimestamp(
                        candle_ts / 1000, tz=timezone.utc
                    ).strftime("%Y-%m-%d %H:%M UTC")

                    if candle_ts != last_candle_ts:
                        last_candle_ts   = candle_ts
                        _last_candle_str = last_candle_str   # share with WS exits

                        # Hold trade_lock across the whole bar transition so a
                        # websocket-driven exit can't race the bar-close exit /
                        # entry below (no double close, no half-applied entry).
                        with trade_lock:
                            # (a) Candle-close stop check — ONLY in candle mode
                            #     (REALTIME_EXITS=0). In intra-candle mode the live
                            #     tick trailing already owns exits and the user does
                            #     NOT want trades closed at the candle's O/H/L/C
                            #     extreme, so this is skipped entirely.
                            if not REALTIME_EXITS:
                                cl_ex, cl_px, cl_rsn = _check_sl_tp_candle(
                                    float(last_closed[1]), float(last_closed[2]),
                                    float(last_closed[3]), float(last_closed[4]))
                                if cl_ex and paper["side"] != "none":
                                    cl_side = paper["side"]
                                    raw, pct = _close_position(cl_px, cl_rsn)
                                    _record_exit(cl_side.upper(), cl_px, cl_rsn,
                                                 raw, pct, last_candle_str)
                                    state.update({
                                        "position": "none", "unrealized": "0.00",
                                        "balance":  f"{paper['balance']:.2f}",
                                        "total_pnl_pct": f"{paper['total_pnl_pct']:+.2f}",
                                        "trail_stop": "—",
                                        "winrate":  _winrate(),
                                    })

                            # (a2) TIME EXIT (intra-candle mode): if the trade did
                            #      NOT hit the trailing-TP or the SL during the
                            #      candle, close it NOW at the candle's CLOSE price
                            #      (the user's rule: "fecha no fim do candle se não
                            #      bater trailing nem SL"). The intra-candle
                            #      trailing/SL in _check_sl_tp handle the live exits;
                            #      this guarantees nothing stays open past a candle.
                            elif paper["side"] != "none":
                                ce_px    = float(last_closed[4])
                                ce_side  = paper["side"]
                                raw, pct = _close_position(ce_px, "CANDLE_END")
                                _record_exit(ce_side.upper(), ce_px, "CANDLE_END",
                                             raw, pct, last_candle_str)
                                state.update({
                                    "position": "none", "unrealized": "0.00",
                                    "balance":  f"{paper['balance']:.2f}",
                                    "total_pnl_pct": f"{paper['total_pnl_pct']:+.2f}",
                                    "trail_stop": "—",
                                    "winrate":  _winrate(),
                                })

                            # (b) Recompute the signal on the closed bar.
                            closes = [c[4] for c in sig_closes]
                            result = strat.calculate(closes, int(config.get("gain_limit", 900)))
                            signal = result["signal"]
                            state.update({
                                "signal":      signal or "—",
                                "ema":         f"{result['ema']:.4f}" if result["ema"] else "—",
                                "ec":          f"{result['ec']:.4f}"  if result["ec"]  else "—",
                                "period":      str(result["period"]),
                                "last_candle": last_candle_str,
                                "updated":     datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                "winrate":     _winrate(),
                            })

                            # (c) Entry / reversal. Pine fires strategy.entry on EVERY
                            #     bar the condition holds, with pyramiding=1 — so the
                            #     net rule is simply: hold the side the signal points
                            #     to. Crucially this re-enters right after a stop if
                            #     the condition still holds (the old prev_signal guard
                            #     wrongly blocked that). Entry fills at the signal
                            #     bar's close = next bar's open in 24/7 crypto (no gap),
                            #     the price a TradingView market order would get.
                            if signal:
                                entry_px = float(last_closed[4])
                                qty      = _calc_qty(entry_px, paper["balance"])
                                cur_pos  = paper["side"]

                                if signal == "LONG" and cur_pos != "long":
                                    if cur_pos == "short":
                                        raw, pct = _close_position(entry_px, "SIGNAL_REVERSE")
                                        _record_exit("SHORT", entry_px, "SIGNAL_REVERSE",
                                                     raw, pct, last_candle_str)
                                    order = _open_position("LONG", qty, entry_px)
                                    _record_entry("LONG", entry_px, qty,
                                                  last_candle_str, order["id"])

                                elif signal == "SHORT" and cur_pos != "short":
                                    if cur_pos == "long":
                                        raw, pct = _close_position(entry_px, "SIGNAL_REVERSE")
                                        _record_exit("LONG", entry_px, "SIGNAL_REVERSE",
                                                     raw, pct, last_candle_str)
                                    order = _open_position("SHORT", qty, entry_px)
                                    _record_entry("SHORT", entry_px, qty,
                                                  last_candle_str, order["id"])

            # ── 3. SLEEP 1 SECOND ─────────────────────────────────────────────
            time_mod.sleep(1)

        except Exception as e:
            logger.error(f"[LOOP] {e}")
            state["status"] = f"Error: {str(e)[:80]}"
            time_mod.sleep(5)

    state["status"] = "Stopped"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# KEEPALIVE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def keepalive(interval):
    time_mod.sleep(12)
    while True:
        try: req.get(f"http://localhost:{PORT}/ping", timeout=4)
        except Exception: pass
        time_mod.sleep(interval)

for _iv in [8, 15, 23]:
    threading.Thread(target=keepalive, args=(_iv,), daemon=True).start()

# ── Always-on live-price feed ─────────────────────────────────────────────
# Keeps _last_good_price fresh for the UI even when the strategy is STOPPED, so
# the live price is always visible. When the strategy is running its own loop +
# websocket already refresh the price, so this only fills the gaps.
def price_poller():
    try:
        ticker_ex.load_markets()
    except Exception as e:
        logger.warning(f"[PRICE] poller load_markets failed: {type(e).__name__}: {e}")
    while True:
        try:
            get_live_price()              # refreshes _last_good_price + _last_price_ts
        except Exception:
            pass
        time_mod.sleep(2)

threading.Thread(target=price_poller, daemon=True).start()

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# BACKTEST  (intra-candle replay using 1-minute sub-candles)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Each 30m candle is NOT read as just its O/H/L/C — we replay the real intra-candle
# price path from its 1-minute sub-candles and run the SAME trailing/SL/candle-end
# exit rules as live trading, so the backtest is as close to real trades as REST
# data allows. Runs in a background thread; the UI polls /backtest-result.
#
# 5 SLOTS simultâneos (pedido 2026-07-02): o otimizador roda 5 combinações em
# paralelo, cada uma com as PRÓPRIAS variáveis passadas na requisição (nunca
# mexendo no config global — mudar o /config no meio corrompia os resultados).
# O slot 0 é o legado: a UI e os modos tv/ticks usam ele.
N_BT_SLOTS = 5
_backtests = [{"running": False, "progress": "", "result": None, "error": None}
              for _ in range(N_BT_SLOTS)]
_backtest  = _backtests[0]        # alias legado (UI, tv, ticks)

# Última falha de fetch do backtest — vai pro erro mostrado NA TELA. Antes a
# causa real (rate-limit, timeout, geo-block…) ficava só no log do servidor e a
# UI dizia apenas "poucos candles de 30m", sem explicar o porquê.
_bt_fetch_err = None

def _bt_fetch(tf, since, tries=3):
    """Uma página de OHLCV com RETRY. Um rate-limit/timeout transitório da
    Phemex NÃO pode abortar a paginação inteira — era isso que fazia o backtest
    'não funcionar': a 1ª exceção quebrava o loop e sobravam poucos candles."""
    global _bt_fetch_err
    for a in range(tries):
        try:
            return bt_ex.fetch_ohlcv(SYMBOL or "ETH/USDT:USDT", timeframe=tf,
                                     since=int(since), limit=1000)
        except Exception as e:
            _bt_fetch_err = f"{type(e).__name__}: {str(e)[:200]}"
            logger.warning(f"[BT] {tf} fetch tentativa {a+1}/{tries}: {_bt_fetch_err}")
            if a < tries - 1:
                time_mod.sleep(1.5 * (a + 1))
    return None

def _few_candles_msg(got):
    m = f"poucos candles de 30m da Phemex (recebi {got})"
    if _bt_fetch_err:
        m += f" — último erro do fetch: {_bt_fetch_err}"
    return m

# ── Cache de candles + sinais (pedido 2026-07-02) ───────────────────────────
# Baixa UMA vez e reutiliza em todos os backtests seguintes da mesma janela —
# o otimizador roda milhares de backtests sobre os mesmos candles e o download
# era a maior parte do tempo de cada um. O cache de 30m expira quando vira o
# candle (bucket de 30 min); o lock impede que 5 slots concorrentes façam 5
# downloads iguais (e o bt_ex não é thread-safe pra chamadas simultâneas).
_fetch_lock = threading.Lock()
_c30_cache  = {"bucket": None, "n": 0, "rows": []}
_c1m_cache  = {"key": None, "rows": [], "buckets": None}   # 1 janela (a de 1 ano é grande)
_sig_lock   = threading.Lock()
_sig_cache  = {}                                            # (gl, ts0, ts1, len) → sinais

def _fetch_30m_cached(n, bt):
    bucket = int(time_mod.time() // 1800)
    with _fetch_lock:
        if _c30_cache["bucket"] == bucket and _c30_cache["n"] >= n:
            return _c30_cache["rows"]
        bt["progress"] = "buscando candles de 30m…"
        rows = _fetch_30m_candles(n)
        if rows:
            _c30_cache.update({"bucket": bucket, "n": n, "rows": rows})
        return rows

def _fetch_1m_cached(since_ms, until_ms, bt):
    """1m candles + os buckets por candle de 30m, cacheados juntos (montar os
    buckets de um ano ~525k candles a cada execução também custava caro)."""
    key = (int(since_ms) // 60_000, int(until_ms) // 60_000)
    with _fetch_lock:
        if _c1m_cache["key"] == key:
            return _c1m_cache["rows"], _c1m_cache["buckets"]
        bt["progress"] = "buscando candles de 1m…"
        rows = _fetch_1m_candles(since_ms, until_ms)
        buckets = {}
        for r in rows:
            buckets.setdefault((r[0] // 1_800_000) * 1_800_000, []).append(r)
        for b in buckets:
            buckets[b].sort(key=lambda x: x[0])
        if rows:
            _c1m_cache.update({"key": key, "rows": rows, "buckets": buckets})
        return rows, buckets

def _signals_cached(c30, gain_limit, bt):
    """Sinais EC×EMA cacheados por (gain_limit, janela) — com o gain_limit fixo
    o cálculo pesado da strategy roda UMA vez pra milhares de backtests."""
    key = (int(gain_limit), c30[0][0], c30[-1][0], len(c30))
    with _sig_lock:
        if key in _sig_cache:
            return _sig_cache[key]
    bt["progress"] = "calculando sinais (EC×EMA por candle)…"
    sig = strat.calculate_series([r[4] for r in c30], int(gain_limit))
    with _sig_lock:
        _sig_cache[key] = sig
        while len(_sig_cache) > 12:                 # limita memória
            _sig_cache.pop(next(iter(_sig_cache)))
    return sig

def _fetch_30m_candles(n):
    """Paginated 30m fetch — supports UNLIMITED n (e.g. a year = 17 520 candles),
    since Phemex caps each request at ~1000."""
    now_ms = int(time_mod.time() * 1000)
    out, cur = [], now_ms - int(n) * 1_800_000
    for _ in range(400):                       # safety cap on pages (~400k candles)
        rows = _bt_fetch("30m", cur)
        if not rows: break
        out.extend(rows)
        last = rows[-1][0]
        if last <= cur or last >= now_ms: break
        cur = last + 1_800_000
        _backtest["progress"] = f"buscando candles de 30m… ({len(out)})"
    seen = {r[0]: r for r in out}
    return [seen[k] for k in sorted(seen)]

def _fetch_1m_candles(since_ms, until_ms):
    out, cur = [], int(since_ms)
    for _ in range(2000):                      # safety cap on pages (~2M minutes ≈ 3.8 yr)
        rows = _bt_fetch("1m", cur)
        if not rows: break
        out.extend(rows)
        last = rows[-1][0]
        if last <= cur or last >= until_ms: break   # reached the end / no progress
        cur = last + 60_000
        if len(out) % 10000 < 1000:
            _backtest["progress"] = f"buscando candles de 1m… ({len(out)})"
    seen = {r[0]: r for r in out}
    return [seen[k] for k in sorted(seen)]

# ── Intra-minute realism knob ────────────────────────────────────────────────
# THE #1 REASON the backtest diverged from paper trading.
#
# A 1-minute candle only gives us O/H/L/C. The OLD path walked L→H in a perfectly
# STRAIGHT line, so on the way up the price NEVER pulled back. A tight trailing
# stop (e.g. trail offset 1 tick = $0.10) therefore never tripped during the
# climb — the backtest rode EVERY winner all the way to the minute's HIGH and
# exited at high−offset. Live trading is the opposite: the real websocket ticks
# wiggle constantly, so the same trailing stop gets shaken out on the FIRST small
# pullback, exiting near where it armed (≈ entry + activation). Result: the
# backtest's winners were several times bigger than the live ones → "completamente
# diferente".
#
# INTRABAR_JAG overlays a deterministic, mean-reverting zig-zag on each leg so the
# path contains sub-minute pullbacks of about INTRABAR_JAG × (this minute's H−L).
# The shared _check_sl_tp then fires the trailing stop at a realistic spot. A WIDE
# trailing offset is barely affected (the wiggle stays inside it); only the tight
# offsets — the ones the straight line mis-modelled — get corrected. The walk is
# fully deterministic (no RNG) so a backtest is reproducible, and every leg still
# starts/ends exactly on O/H/L/C (the candle's real extremes are never exceeded).
#
# 0.0  → restores the OLD straight-line path (optimistic, rides to the extreme).
# 0.4  → sensible default for ETH 30m/1m.   Env BT_INTRABAR_JAG overrides.
# NOTE: this is still a MODEL of the path, not the real ticks. For literal 1:1
# parity with paper trading, replay recorded live ticks (see notes in the answer).
INTRABAR_JAG = float(os.environ.get("BT_INTRABAR_JAG", 0.4))

def _leg_path(a, b, n, jag):
    """One O/H/L/C leg a→b as ~n points, walked with periodic pullbacks AGAINST
    the leg direction (amplitude jag×|b−a|) instead of a straight line. Stays
    strictly inside [min(a,b), max(a,b)] and ends exactly at b, so the candle's
    real high/low are touched but never overshot."""
    if n <= 1 or a == b:
        return [b]
    up    = b > a
    span  = abs(b - a)
    pull  = jag * span
    lo, hi = (a, b) if up else (b, a)
    pts = []
    for i in range(1, n + 1):
        f    = i / n
        base = a + (b - a) * f                       # monotone progress a→b
        # envelope sin(πf) is 0 at both ends; the (1−cos) term makes ~4 humps,
        # so the retraces fade out at the leg boundaries (exact endpoints).
        saw  = pull * math.sin(f * math.pi) * (0.5 - 0.5 * math.cos(f * 8.0 * math.pi))
        p    = base - saw if up else base + saw      # pull back, never past b
        pts.append(min(hi, max(lo, p)))
    pts[-1] = b
    return pts

def _second_path(o, h, l, c, steps=60):
    # Read each 1-minute candle SECOND BY SECOND (~`steps` points), walking the
    # broker path O→L→H→C (up bar) / O→H→L→C (down bar). Real 1-second history
    # isn't available via REST, so we step through each minute's OHLC at 1s
    # resolution — the backtest "sees" prices ~like the live tick stream, INCLUDING
    # the small intra-minute pullbacks (INTRABAR_JAG) that a tight trailing stop
    # reacts to. With INTRABAR_JAG=0 this is the old straight-line walk.
    pts = [o, l, h, c] if c >= o else [o, h, l, c]
    seg = max(1, steps // 3)
    out = []
    for k in range(3):
        out.extend(_leg_path(pts[k], pts[k + 1], seg, INTRABAR_JAG))
    return out

def _simulate_trade(side, entry, subs, candle_close, sl_d=None, tp_d=None, tr_d=None):
    """Replay one trade through its 1m sub-candles at 1s resolution, using the
    EXACT SAME _check_sl_tp as live paper trading (single source of truth, so the
    backtest result matches live). sl_d/tp_d/tr_d opcionais = distâncias próprias
    desta execução (slots paralelos). Returns (exit_px, reason, mfe%, mae%)."""
    pos = {"side": "long" if side == "LONG" else "short", "entry_price": entry,
           "peak": entry, "trail_active": False, "trail_stop": 0.0}
    mfe = 0.0; mae = 0.0
    for s in subs:
        for px in _second_path(float(s[1]), float(s[2]), float(s[3]), float(s[4])):
            d = (px - entry) if side == "LONG" else (entry - px)
            pct = (d / entry * 100) if entry else 0.0
            if pct > mfe: mfe = pct
            if pct < mae: mae = pct
            ex, ex_px, reason = _check_sl_tp(pos, px, sl_d, tp_d, tr_d)
            if ex:
                return ex_px, reason, mfe, mae
    return candle_close, "CANDLE_END", mfe, mae   # never hit TP/SL → close at candle end

def _fetch_range(tf: str, tf_ms: int, since_ms: int, until_ms: int):
    """Generic paginated OHLCV fetch for [since_ms, until_ms) at timeframe tf."""
    out, cur = [], int(since_ms)
    for _ in range(4000):
        rows = _bt_fetch(tf, cur)
        if not rows: break
        out.extend(rows)
        last = rows[-1][0]
        if last <= cur or last >= until_ms: break
        cur = last + tf_ms
        _backtest["progress"] = f"buscando candles de {tf}… ({len(out)})"
    seen = {r[0]: r for r in out}
    return [seen[k] for k in sorted(seen)]

def _simulate_trade_ticks(side, entry, prices, candle_close):
    """Replay one trade through the REAL recorded 1s prices of its candle, using the
    EXACT SAME _check_sl_tp as live paper trading. `prices` is the list of recorded
    prices (chronological) that fell inside the exit candle. Returns
    (exit_px, reason, mfe%, mae%)."""
    pos = {"side": "long" if side == "LONG" else "short", "entry_price": entry,
           "peak": entry, "trail_active": False, "trail_stop": 0.0}
    mfe = 0.0; mae = 0.0
    for px in prices:
        d = (px - entry) if side == "LONG" else (entry - px)
        pct = (d / entry * 100) if entry else 0.0
        if pct > mfe: mfe = pct
        if pct < mae: mae = pct
        ex, ex_px, reason = _check_sl_tp(pos, px)     # same logic as live
        if ex:
            return ex_px, reason, mfe, mae
    return candle_close, "CANDLE_END", mfe, mae

def run_backtest_ticks(fee_entry=0.0, fee_exit=0.0):
    """1:1-REAL backtest: replays the ACTUAL recorded 1-second price stream (from
    ticks.csv, written live when RECORD_TICKS=1) through the same signal + exit
    logic as paper trading. No interpolation — the real prices the bot saw. Only
    covers the period that was actually recorded."""
    _backtest.update({"running": True, "result": None, "error": None,
                      "progress": "lendo ticks gravados…"})
    global _bt_fetch_err
    _bt_fetch_err = None
    try:
        ticks = _load_recorded_ticks()
        if len(ticks) < 100:
            raise RuntimeError("poucos ticks gravados ainda — a gravação é automática "
                               "enquanto a strategy está iniciada (Start); deixe o bot "
                               "rodando um tempo e tente de novo (acompanhe em /ticks-info)")
        t0, t1 = ticks[0][0], ticks[-1][0]
        try:
            bt_ex.load_markets()
        except Exception as e:
            logger.warning(f"[BT] load_markets: {type(e).__name__}: {e}")

        # 30m candles: 200 bars of warmup BEFORE the recorded window (signals need
        # lead-in) through the end of the recorded window.
        warm_ms = 200 * 1_800_000
        _backtest["progress"] = "buscando candles de 30m (sinais)…"
        c30 = _fetch_range("30m", 1_800_000, t0 - warm_ms, t1 + 1_800_000)
        if not c30 or len(c30) < 70:
            raise RuntimeError(_few_candles_msg(len(c30) if c30 else 0))

        # Bucket the recorded prices by 30m candle boundary.
        tbuckets = {}
        for ts, px in ticks:
            tbuckets.setdefault((ts // 1_800_000) * 1_800_000, []).append(px)

        _backtest["progress"] = "calculando sinais (EC×EMA por candle)…"
        signals = strat.calculate_series([c[4] for c in c30],
                                         int(config.get("gain_limit", 900)))

        _backtest["progress"] = "simulando trades (replay de ticks reais)…"
        bal = PAPER_START_BAL; trades = []; wins = 0; skipped = 0
        for i in range(60, len(c30) - 1):
            sig = signals[i]
            if not sig: continue
            nxt = c30[i + 1]
            b   = (nxt[0] // 1_800_000) * 1_800_000
            prices = tbuckets.get(b)
            if not prices:
                # No recorded ticks for this exit candle (bot was down / outside the
                # recorded window) → skip, so we only compare the real recorded period.
                skipped += 1
                continue
            entry = float(c30[i][4])                 # candle i close = candle i+1 open
            ex_px, reason, mfe, mae = _simulate_trade_ticks(sig, entry, prices, float(nxt[4]))
            qty = (bal / entry) if entry else 0.0
            raw = (ex_px - entry) * qty if sig == "LONG" else (entry - ex_px) * qty
            fee = (entry * qty * fee_entry / 100) + (ex_px * qty * fee_exit / 100)
            raw -= fee
            bal += raw
            pct = (raw / (entry * qty) * 100) if (entry * qty) else 0.0
            if pct > 0: wins += 1
            trades.append({
                "candle": datetime.fromtimestamp(nxt[0] / 1000, tz=timezone.utc)
                          .strftime("%Y-%m-%d %H:%M UTC"),
                "side": sig, "entry": round(entry, 2), "exit": round(ex_px, 2),
                "reason": reason, "pnl_pct": round(pct, 3),
                "mfe": round(mfe - fee_entry, 3), "mae": round(mae - fee_entry, 3),
                "balance": round(bal, 2),
            })
        n = len(trades)
        rets = [t["pnl_pct"] for t in trades]
        span_h = round((t1 - t0) / 3_600_000, 1)
        summary = {
            "trades": n,
            "winrate": round(wins / n * 100, 1) if n else 0.0,
            "final_balance": round(bal, 2),
            "total_pnl_pct": round((bal - PAPER_START_BAL) / PAPER_START_BAL * 100, 2),
            "max_profit": round(max(rets), 3) if rets else 0.0,
            "max_loss":   round(min(rets), 3) if rets else 0.0,
            "avg_pct":    round(sum(rets) / n, 3) if n else 0.0,
            "fee_entry": fee_entry, "fee_exit": fee_exit,
            "sl": SL_DIST, "tp": TP_DIST, "tr": TR_DIST,
            "source": "ticks_reais_gravados",
            "ticks_loaded": len(ticks),
            "recorded_hours": span_h,          # how much real time was recorded
            "candles_skipped_no_ticks": skipped,
        }
        _backtest.update({"running": False, "progress": "concluído",
                          "result": {"summary": summary, "trades": trades[::-1]}})
        logger.info(f"[BT-TICKS] {n} trades, bal={bal:.2f}, "
                    f"pnl={summary['total_pnl_pct']}%, {span_h}h gravadas")
    except Exception as e:
        logger.error(f"[BT-TICKS] {type(e).__name__}: {e}")
        _backtest.update({"running": False, "progress": "erro",
                          "error": f"{type(e).__name__}: {e}"})

def run_backtest(n_candles=300, fee_entry=0.0, fee_exit=0.0, slot=0, params=None):
    """Backtest synthetic (1m). `slot` (0..N_BT_SLOTS-1) permite execuções
    SIMULTÂNEAS e `params` (sl_pts/tp_pts/tr_pts/mintick_sl/mintick_tp/
    gain_limit) faz ESTA execução usar as próprias variáveis, sem tocar no
    config global — requisito do otimizador paralelo (2026-07-02)."""
    bt = _backtests[slot]
    bt.update({"running": True, "result": None, "error": None,
               "progress": "preparando…"})
    global _bt_fetch_err
    _bt_fetch_err = None
    p    = params or {}
    msl  = float(p.get("mintick_sl", config["mintick_sl"]))
    mtp  = float(p.get("mintick_tp", config["mintick_tp"]))
    sl_d = round(float(p.get("sl_pts", config["sl_pts"])) * msl, 8)
    tp_d = round(float(p.get("tp_pts", config["tp_pts"])) * mtp, 8)
    tr_d = round(float(p.get("tr_pts", config["tr_pts"])) * mtp, 8)
    gl   = int(p.get("gain_limit", config.get("gain_limit", 900)))
    try:
        try:
            bt_ex.load_markets()
        except Exception as e:
            logger.warning(f"[BT] load_markets: {type(e).__name__}: {e}")
        warm = 200                                   # warmup bars so EC/EMA/period converge like TV
        c30  = _fetch_30m_cached(n_candles + warm, bt)   # cache: baixa 1x, reutiliza
        if not c30 or len(c30) < 70:
            raise RuntimeError(_few_candles_msg(len(c30) if c30 else 0))
        c30 = c30[-(n_candles + warm):]
        # only the LAST n_candles are traded; the earlier ones are warmup-only
        start_i = max(60, len(c30) - 1 - n_candles)
        # 1m only for the TRADED window (cache: baixa/monta buckets 1x)
        c1m, buckets = _fetch_1m_cached(c30[start_i][0], c30[-1][0] + 1_800_000, bt)

        signals = _signals_cached(c30, gl, bt)       # cache por (gain_limit, janela)

        bt["progress"] = "simulando trades (replay 1m)…"
        bal = PAPER_START_BAL; trades = []; wins = 0
        for i in range(start_i, len(c30) - 1):
            sig = signals[i]
            if not sig: continue
            entry = float(c30[i][4])                 # candle i close = candle i+1 open
            nxt   = c30[i + 1]
            subs  = buckets.get((nxt[0] // 1_800_000) * 1_800_000) or [nxt]
            ex_px, reason, mfe, mae = _simulate_trade(sig, entry, subs, float(nxt[4]),
                                                      sl_d, tp_d, tr_d)
            qty = (bal / entry) if entry else 0.0
            raw = (ex_px - entry) * qty if sig == "LONG" else (entry - ex_px) * qty
            fee = (entry * qty * fee_entry / 100) + (ex_px * qty * fee_exit / 100)
            raw -= fee                                   # net of entry + exit fees
            bal += raw
            pct = (raw / (entry * qty) * 100) if (entry * qty) else 0.0
            if pct > 0: wins += 1
            trades.append({
                "candle": datetime.fromtimestamp(nxt[0] / 1000, tz=timezone.utc)
                          .strftime("%Y-%m-%d %H:%M UTC"),
                "side": sig, "entry": round(entry, 2), "exit": round(ex_px, 2),
                "reason": reason, "pnl_pct": round(pct, 3),
                # MFE/MAE shifted by the entry fee so they read EXACTLY like the
                # live dashboard's _unrealized_pct (which subtracts the entry fee
                # from every reading). With fees=0 this is a no-op.
                "mfe": round(mfe - fee_entry, 3), "mae": round(mae - fee_entry, 3),
                "balance": round(bal, 2),
            })
        n = len(trades)
        rets = [t["pnl_pct"] for t in trades]
        summary = {
            "trades": n,
            "winrate": round(wins / n * 100, 1) if n else 0.0,
            "final_balance": round(bal, 2),
            "total_pnl_pct": round((bal - PAPER_START_BAL) / PAPER_START_BAL * 100, 2),
            "max_profit": round(max(rets), 3) if rets else 0.0,   # biggest single-trade win %
            "max_loss":   round(min(rets), 3) if rets else 0.0,   # biggest single-trade loss %
            "avg_pct":    round(sum(rets) / n, 3) if n else 0.0,  # average % per trade
            "fee_entry": fee_entry, "fee_exit": fee_exit,
            "candles": n_candles, "slot": slot,
            "sl": sl_d, "tp": tp_d, "tr": tr_d,
            "mintick_sl": msl, "mintick_tp": mtp, "gain_limit": gl,
            "sl_pts": float(p.get("sl_pts", config["sl_pts"])),
            "tp_pts": float(p.get("tp_pts", config["tp_pts"])),
            "tr_pts": float(p.get("tr_pts", config["tr_pts"])),
            "intrabar_minutes_loaded": len(c1m),
            "intrabar_jag": INTRABAR_JAG,   # 0 = old straight-line; >0 = realistic wiggle
        }
        bt.update({"running": False, "progress": "concluído",
                   "result": {"summary": summary, "trades": trades[::-1]}})
        logger.info(f"[BT] slot {slot} done: {n} trades, bal={bal:.2f}, "
                    f"pnl={summary['total_pnl_pct']}%")
    except Exception as e:
        logger.error(f"[BT] slot {slot}: {type(e).__name__}: {e}")
        bt.update({"running": False, "progress": "erro",
                   "error": f"{type(e).__name__}: {e}"})

# ── Emulador do TradingView (fonte "tv") ─────────────────────────────────────
# Reproduz o backtest do TradingView COMO O USUÁRIO O OBSERVOU rodando ao vivo
# via webhook (Pine v3, calc_on_every_tick=false, sem bar magnifier):
#   • TODA trade vive DENTRO do seu candle: entra na ABERTURA do candle seguinte
#     ao sinal, o trailing/SL agem ao longo do candle e, se NENHUM dos dois
#     bater, fecha no CLOSE do candle (CANDLE_END). Confirmado pelo usuário
#     (2026-07-01, com certeza absoluta): no TV dele toda trade fecha no mesmo
#     candle — NÃO reintroduzir posição multi-candle aqui.
#   • caminho intra-candle do broker emulator: o extremo mais PRÓXIMO da
#     abertura é visitado primeiro (O→H→L→C se o high está mais perto do open,
#     senão O→L→H→C) — regra documentada do TV, sem pullbacks inventados;
#   • sizing do Pine: qty = risk·saldo/(fixedSL·mintick), cap em MAX_LOTS,
#     saldo = capital inicial + lucro realizado (compounding igual ao TV).
def _tv_walk_bar(pos, o, h, l, c, sl_d, tp_d, tr_d):
    """Anda um candle de 30m pelos pivôs do broker emulator, armando/arrastando
    o trailing ao longo do candle. Atualiza peak/trail_stop/mfe/mae em `pos` e
    retorna (saiu, preço, motivo). Se retornar False, quem chama fecha a trade
    no close do candle (CANDLE_END)."""
    is_long = pos["side"] == "long"
    entry   = pos["entry_price"]
    fixed   = entry - sl_d if is_long else entry + sl_d

    def _stop():
        return pos["trail_stop"] if pos["trail_active"] else fixed

    def _fav(px):                    # movimento a favor: ratchet do trailing
        if is_long:
            if px > pos["peak"]:
                pos["peak"] = px
            if pos["peak"] - entry >= tp_d:
                nt = round(pos["peak"] - tr_d, 8)
                if not pos["trail_active"] or nt > pos["trail_stop"]:
                    pos["trail_active"], pos["trail_stop"] = True, nt
        else:
            if px < pos["peak"]:
                pos["peak"] = px
            if entry - pos["peak"] >= tp_d:
                nt = round(pos["peak"] + tr_d, 8)
                if not pos["trail_active"] or nt < pos["trail_stop"]:
                    pos["trail_active"], pos["trail_stop"] = True, nt

    def _exc(px):                    # excursão máx. favorável/adversa (%)
        d   = (px - entry) if is_long else (entry - px)
        pct = d / entry * 100 if entry else 0.0
        if pct > pos["mfe"]: pos["mfe"] = pct
        if pct < pos["mae"]: pos["mae"] = pct

    def _rsn():
        return "TRAIL_TP" if pos["trail_active"] else "SL"

    # gap na abertura: abriu além do stop → preenche na própria abertura
    _exc(o)
    s = _stop()
    if (is_long and o <= s) or (not is_long and o >= s):
        return True, o, _rsn()
    _fav(o)

    pts = [h, l, c] if (h - o) < (o - l) else [l, h, c]
    cur = o
    for px in pts:
        contra = (px < cur) if is_long else (px > cur)
        if contra:
            s = _stop()
            if (is_long and px <= s) or (not is_long and px >= s):
                _exc(s)
                return True, s, _rsn()
            _exc(px)
        else:
            _fav(px)
            _exc(px)
        cur = px
    return False, 0.0, ""

def run_backtest_tv(n_candles=300, fee_entry=0.0, fee_exit=0.0):
    """Backtest "tv": emula o TradingView candle a candle (30m O/H/L/C, sem
    sub-candles): entrada no OPEN do candle seguinte ao sinal, trailing/SL pelo
    caminho do broker emulator e, se nada bater, CANDLE_END no close — toda
    trade fecha no próprio candle, como o usuário observou no TV. Sizing
    risk/(SL·mintick) igual ao Pine — é o modo pra comparar com o painel do TV."""
    _backtest.update({"running": True, "result": None, "error": None,
                      "progress": "buscando candles de 30m…"})
    global _bt_fetch_err
    _bt_fetch_err = None
    try:
        try:
            bt_ex.load_markets()
        except Exception as e:
            logger.warning(f"[BT-TV] load_markets: {type(e).__name__}: {e}")
        warm = 200                                  # warmup pro EC/EMA convergir como no TV
        c30  = _fetch_30m_candles(n_candles + warm)
        if not c30 or len(c30) < 70:
            raise RuntimeError(_few_candles_msg(len(c30) if c30 else 0))
        c30 = c30[-(n_candles + warm):]
        _backtest["progress"] = "calculando sinais (EC×EMA por candle)…"
        signals = strat.calculate_series([r[4] for r in c30],
                                         int(config.get("gain_limit", 900)))

        _backtest["progress"] = "simulando (emulador do Strategy Tester)…"
        bal    = PAPER_START_BAL
        pos    = None
        pend   = None
        trades = []
        wins   = 0

        def _fmt(ts):
            return datetime.fromtimestamp(ts / 1000, tz=timezone.utc) \
                           .strftime("%Y-%m-%d %H:%M UTC")

        def _close(px, reason, ts):
            nonlocal bal, pos, wins
            entry, qty = pos["entry_price"], pos["qty"]
            raw = (px - entry) * qty if pos["side"] == "long" else (entry - px) * qty
            raw -= (entry * qty * fee_entry / 100) + (px * qty * fee_exit / 100)
            bal += raw
            notional = entry * qty
            pct = (raw / notional * 100) if notional else 0.0
            if pct > 0:
                wins += 1
            trades.append({
                "candle":  _fmt(ts),
                "side":    "LONG" if pos["side"] == "long" else "SHORT",
                "entry":   round(entry, 2), "exit": round(px, 2),
                "reason":  reason, "pnl_pct": round(pct, 3),
                "pnl_usd": round(raw, 2),
                "mfe": round(pos["mfe"], 3), "mae": round(pos["mae"], 3),
                "balance": round(bal, 2),
            })
            pos = None

        start_i = max(61, len(c30) - n_candles)
        for i in range(start_i, len(c30)):
            ts = c30[i][0]
            o, h, l, c = (float(c30[i][1]), float(c30[i][2]),
                          float(c30[i][3]), float(c30[i][4]))

            # 1) a ordem do candle anterior executa na ABERTURA deste candle
            if pend and bal > 0:
                side = "long" if pend == "LONG" else "short"
                qty  = (RISK * bal / SL_DIST) if SL_DIST else 0.0
                pos  = {"side": side, "entry_price": o, "qty": min(qty, MAX_LOTS),
                        "peak": o, "trail_active": False, "trail_stop": 0.0,
                        "mfe": 0.0, "mae": 0.0}
            pend = None

            # 2) trailing/SL agem ao longo do candle; se NENHUM bater, a trade
            #    fecha no CLOSE do candle (CANDLE_END) — toda trade vive dentro
            #    do próprio candle, como no TV do usuário.
            if pos:
                ex, px, ex_rsn = _tv_walk_bar(pos, o, h, l, c,
                                              SL_DIST, TP_DIST, TR_DIST)
                if ex:
                    _close(px, ex_rsn, ts)
                else:
                    _close(c, "CANDLE_END", ts)

            # 3) sinal no fechamento vira ordem pro próximo candle
            if signals[i]:
                pend = signals[i]

        n    = len(trades)
        rets = [t["pnl_pct"] for t in trades]
        usds = [t["pnl_usd"] for t in trades]
        summary = {
            "trades": n,
            "winrate": round(wins / n * 100, 1) if n else 0.0,
            "final_balance": round(bal, 2),
            "total_pnl_pct": round((bal - PAPER_START_BAL) / PAPER_START_BAL * 100, 2),
            "max_profit": round(max(rets), 3) if rets else 0.0,
            "max_loss":   round(min(rets), 3) if rets else 0.0,
            "avg_pct":    round(sum(rets) / n, 3) if n else 0.0,
            "avg_usd":    round(sum(usds) / n, 2) if n else 0.0,  # = "Avg P&L" do TV
            "fee_entry": fee_entry, "fee_exit": fee_exit,
            "candles": n_candles,
            "sl": SL_DIST, "tp": TP_DIST, "tr": TR_DIST,
            "mintick_sl": MINTICK_SL, "mintick_tp": MINTICK_TP,
            "gain_limit": int(config.get("gain_limit", 900)),
            "bars30m": len(c30) - start_i,
            "source": "tv_emulator",
        }
        _backtest.update({"running": False, "progress": "concluído (emulador TV)",
                          "result": {"summary": summary, "trades": trades[::-1]}})
        logger.info(f"[BT-TV] {n} trades, bal={bal:.2f}, "
                    f"pnl={summary['total_pnl_pct']}%")
    except Exception as e:
        logger.error(f"[BT-TV] {type(e).__name__}: {e}")
        _backtest.update({"running": False, "progress": "erro",
                          "error": f"{type(e).__name__}: {e}"})

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# FLASK
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
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
header{background:#161b22;border-bottom:1px solid #30363d;padding:16px 28px;
       display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:8px}
header h1{font-size:1.15rem;color:#58a6ff}
#clock{font-size:.78rem;color:#8b949e}
.container{max-width:1120px;margin:24px auto;padding:0 18px}
.top-row{display:flex;gap:14px;align-items:flex-start;flex-wrap:wrap;margin-bottom:20px}
.controls{display:flex;gap:10px}
button{padding:9px 24px;border:none;border-radius:8px;font-size:.9rem;font-weight:600;cursor:pointer}
#btn-start{background:#238636;color:#fff}#btn-start:hover:not(:disabled){background:#2ea043}
#btn-stop{background:#da3633;color:#fff}#btn-stop:hover:not(:disabled){background:#f85149}
button:disabled{opacity:.35;cursor:not-allowed}
.fee-box{background:#161b22;border:1px solid #30363d;border-radius:10px;padding:14px 18px;
         display:flex;gap:16px;align-items:center;flex-wrap:wrap}
.fee-box label{font-size:.72rem;color:#8b949e;text-transform:uppercase;letter-spacing:.05em}
.fee-box input{width:72px;background:#0d1117;border:1px solid #30363d;border-radius:6px;
               color:#e6edf3;font-size:.88rem;padding:5px 8px;text-align:center}
.fee-box input:focus{outline:none;border-color:#58a6ff}
#btn-fee{padding:7px 16px;font-size:.82rem;background:#1f6feb;color:#fff;border:none;
         border-radius:6px;cursor:pointer}#btn-fee:hover{background:#388bfd}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(142px,1fr));gap:11px;margin-bottom:22px}
.card{background:#161b22;border:1px solid #30363d;border-radius:10px;padding:14px 16px}
.lbl{font-size:.67rem;color:#8b949e;text-transform:uppercase;letter-spacing:.06em;margin-bottom:4px}
.val{font-size:1.05rem;font-weight:600}
.long{color:#3fb950}.short{color:#f85149}
.running{color:#3fb950}.stopped{color:#8b949e}
.err{color:#e3b341;font-size:.74rem;word-break:break-all}
.pos-num{color:#3fb950}.neg-num{color:#f85149}
.st{font-size:.76rem;color:#8b949e;text-transform:uppercase;letter-spacing:.06em;margin-bottom:8px}
table{width:100%;border-collapse:collapse;background:#161b22;border:1px solid #30363d;
      border-radius:10px;overflow:hidden}
th{background:#21262d;color:#8b949e;font-size:.72rem;text-transform:uppercase;
   padding:10px 12px;text-align:left}
td{padding:9px 12px;font-size:.83rem;border-top:1px solid #21262d}
tr:hover td{background:#1c2128}
.badge{display:inline-block;padding:2px 8px;border-radius:20px;font-size:.72rem;font-weight:700}
.badge.long{background:#0d4429;color:#3fb950}
.badge.short{background:#3d1010;color:#f85149}
.badge.close{background:#1e2533;color:#8b949e}
#toast{position:fixed;bottom:18px;right:18px;padding:10px 16px;border-radius:8px;
       font-size:.84rem;display:none;z-index:99;color:#fff}
footer{text-align:center;color:#484f58;font-size:.71rem;margin:28px 0 14px}
.head-right{display:flex;align-items:center;gap:22px}
#liveprice{text-align:right;line-height:1.1}
#liveprice .lbl2{font-size:.6rem;color:#8b949e;text-transform:uppercase;letter-spacing:.05em;display:block;margin-bottom:2px}
#lp-val{font-size:1.4rem;font-weight:700;color:#3fb950}
#lp-val.stale{color:#e3b341}
.live-dot{display:inline-block;width:8px;height:8px;border-radius:50%;background:#3fb950;
          margin-right:7px;vertical-align:middle;box-shadow:0 0 6px #3fb950}
.live-dot.stale{background:#e3b341;box-shadow:none}
.tabs{display:flex;gap:8px;margin-bottom:16px;border-bottom:1px solid #30363d}
.tabbtn{background:none;border:none;color:#8b949e;padding:10px 18px;font-size:.92rem;
        font-weight:600;cursor:pointer;border-bottom:2px solid transparent;border-radius:0}
.tabbtn.active{color:#58a6ff;border-bottom-color:#58a6ff}
.cfg-box{background:#161b22;border:1px solid #30363d;border-radius:10px;padding:14px 18px;
         display:flex;gap:16px;align-items:flex-end;flex-wrap:wrap;margin-bottom:18px}
.cfg-box label{font-size:.7rem;color:#8b949e;text-transform:uppercase;letter-spacing:.04em}
.cfg-box input{width:90px;background:#0d1117;border:1px solid #30363d;border-radius:6px;
               color:#e6edf3;font-size:.88rem;padding:6px 8px;text-align:center}
.cfg-box input:focus{outline:none;border-color:#58a6ff}
#btn-cfg,#btn-bt{padding:7px 16px;font-size:.82rem;background:#1f6feb;color:#fff;border:none;
                 border-radius:6px;cursor:pointer}#btn-cfg:hover,#btn-bt:hover{background:#388bfd}
.cfg-hint{font-size:.72rem;color:#8b949e}
.bt-ctrl{display:flex;gap:12px;align-items:flex-end;flex-wrap:wrap}
.bt-ctrl input{width:90px;background:#0d1117;border:1px solid #30363d;border-radius:6px;
               color:#e6edf3;font-size:.88rem;padding:6px 8px;text-align:center}
.badge.tp{background:#0d4429;color:#3fb950}.badge.slr{background:#3d1010;color:#f85149}
.badge.ce{background:#1e2533;color:#8b949e}
</style>
</head>
<body>
<header>
  <h1>⚡ AZLEMA Bot — Paper Trading</h1>
  <div class="head-right">
    <div id="liveprice">
      <span class="lbl2">ETH/USDT ao vivo</span>
      <span class="live-dot" id="lp-dot"></span><span id="lp-val">—</span>
    </div>
    <span id="clock"></span>
  </div>
</header>
<div class="container">
  <div class="tabs">
    <button class="tabbtn active" id="tb-paper" onclick="showTab('paper')">📈 Paper Trading</button>
    <button class="tabbtn" id="tb-backtest" onclick="showTab('backtest')">🧪 Backtesting</button>
  </div>

  <div class="cfg-box">
    <div><label>Stop Loss (pontos)</label><br><input id="cfg-sl" type="number" step="1" min="1"></div>
    <div><label>TP ativação (pontos)</label><br><input id="cfg-tp" type="number" step="1" min="1"></div>
    <div><label>Trailing (pontos)</label><br><input id="cfg-tr" type="number" step="1" min="1"></div>
    <div><label>Valor do ponto SL ($)</label><br><input id="cfg-mt-sl" type="number" step="0.001" min="0.0001"></div>
    <div><label>Valor do ponto TP/Trail ($)</label><br><input id="cfg-mt-tp" type="number" step="0.001" min="0.0001"></div>
    <div><label>Gain Limit</label><br><input id="cfg-gl" type="number" step="1" min="1"></div>
    <button id="btn-cfg" onclick="saveConfig()">Salvar Config</button>
    <span class="cfg-hint" id="cfg-eq">vale pro bot ao vivo E pro backtest</span>
  </div>

  <div id="tab-paper">
  <div class="top-row">
    <div class="controls">
      <button id="btn-start" onclick="ctrl('start')">▶ Start</button>
      <button id="btn-stop"  onclick="ctrl('stop')" disabled>⏹ Stop</button>
    </div>
    <div class="fee-box">
      <div>
        <label>Fee Entrada %</label><br>
        <input id="fee-entry" type="number" step="0.01" min="0" value="0">
      </div>
      <div>
        <label>Fee Saída %</label><br>
        <input id="fee-exit" type="number" step="0.01" min="0" value="0">
      </div>
      <button id="btn-fee" onclick="saveFees()">Salvar Fees</button>
    </div>
  </div>

  <div class="cards">
    <div class="card"><div class="lbl">Status</div><div class="val" id="c-status">—</div></div>
    <div class="card"><div class="lbl">Sinal</div><div class="val" id="c-signal">—</div></div>
    <div class="card"><div class="lbl">Posição</div><div class="val" id="c-pos">—</div></div>
    <div class="card"><div class="lbl">Saldo (USDT)</div><div class="val" id="c-bal">—</div></div>
    <div class="card"><div class="lbl">PnL Total %</div><div class="val" id="c-pnl">—</div></div>
    <div class="card"><div class="lbl">Não Realizado %</div><div class="val" id="c-unr">—</div></div>
    <div class="card"><div class="lbl">Lucro Máx % (aberta)</div><div class="val" id="c-mfe">—</div></div>
    <div class="card"><div class="lbl">Perda Máx % (aberta)</div><div class="val" id="c-mae">—</div></div>
    <div class="card"><div class="lbl">Winrate</div><div class="val" id="c-win">—</div></div>
    <div class="card"><div class="lbl">Trail Stop</div><div class="val" id="c-trail">—</div></div>
    <div class="card"><div class="lbl">EMA</div><div class="val" id="c-ema">—</div></div>
    <div class="card"><div class="lbl">EC</div><div class="val" id="c-ec">—</div></div>
    <div class="card"><div class="lbl">Período</div><div class="val" id="c-period">—</div></div>
    <div class="card"><div class="lbl">Último Candle</div>
      <div class="val" style="font-size:.8rem" id="c-candle">—</div></div>
    <div class="card"><div class="lbl">Próximo Candle (entrada)</div>
      <div class="val" style="font-size:.85rem" id="c-next">—</div></div>
  </div>

  <div class="st">Histórico de Trades</div>
  <table>
    <thead>
      <tr><th>Hora</th><th>Candle</th><th>Lado</th>
          <th>Preço</th><th>Qtd</th><th>PnL %</th>
          <th>Lucro Máx %</th><th>Perda Máx %</th><th>Saldo</th><th>ID</th></tr>
    </thead>
    <tbody id="tbody">
      <tr><td colspan="10" style="text-align:center;color:#484f58;padding:20px">
        Nenhuma trade ainda</td></tr>
    </tbody>
  </table>
  </div><!-- /tab-paper -->

  <div id="tab-backtest" style="display:none">
    <div class="top-row">
      <div class="bt-ctrl">
        <div><label>Fonte</label><br>
          <select id="bt-source">
            <option value="synthetic">Candles 1m (qualquer período)</option>
            <option value="tv">Emulador TradingView (Strategy Tester)</option>
            <option value="ticks">Ticks reais gravados (1:1 com paper)</option>
          </select></div>
        <span class="cfg-hint">"Emulador TradingView" = candle 30m puro (O/H/L/C): entra na abertura do candle seguinte, trailing/SL agem no candle e fecha no fim do candle se nada bater — é o modo pra comparar com o TV. "Ticks reais" grava sozinho enquanto o bot roda (Start) e só cobre esse período; é o mais fiel ao paper trading (veja /ticks-info).</span>
        <div><label>Candles de 30m</label><br>
          <input id="bt-candles" type="number" step="10" min="50" max="35040" value="300"></div>
        <span class="cfg-hint">300 ≈ 6 dias · 1440 ≈ 1 mês · 17520 ≈ 1 ano (longo demora minutos) — só p/ fonte "Candles 1m"</span>
        <div><label>Fee Entrada %</label><br>
          <input id="bt-fee-entry" type="number" step="0.01" min="0" value="0"></div>
        <div><label>Fee Saída %</label><br>
          <input id="bt-fee-exit" type="number" step="0.01" min="0" value="0"></div>
        <button id="btn-bt" onclick="runBacktest()">▶ Rodar Backtest</button>
        <span id="bt-progress" class="cfg-hint"></span>
      </div>
    </div>
    <div class="cards">
      <div class="card"><div class="lbl">Trades</div><div class="val" id="bt-trades">—</div></div>
      <div class="card"><div class="lbl">Winrate</div><div class="val" id="bt-win">—</div></div>
      <div class="card"><div class="lbl">PnL Total %</div><div class="val" id="bt-pnl">—</div></div>
      <div class="card"><div class="lbl">Lucro Médio %/trade</div><div class="val" id="bt-avg">—</div></div>
      <div class="card"><div class="lbl">Maior Lucro %</div><div class="val" id="bt-max">—</div></div>
      <div class="card"><div class="lbl">Maior Perda %</div><div class="val" id="bt-min">—</div></div>
      <div class="card"><div class="lbl">Saldo Final</div><div class="val" id="bt-bal">—</div></div>
      <div class="card"><div class="lbl">Candles 1m</div><div class="val" id="bt-1m">—</div></div>
    </div>
    <div class="st">Trades do Backtest — replay intra-candle (1m), regras iguais às do bot</div>
    <table>
      <thead><tr><th>Candle</th><th>Lado</th><th>Entrada</th><th>Saída</th>
        <th>Motivo</th><th>PnL %</th><th>Lucro Máx %</th><th>Perda Máx %</th><th>Saldo</th></tr></thead>
      <tbody id="bt-tbody">
        <tr><td colspan="9" style="text-align:center;color:#484f58;padding:20px">
          Rode um backtest para ver as trades</td></tr>
      </tbody>
    </table>
  </div><!-- /tab-backtest -->
</div>
<div id="toast"></div>
<footer>AZLEMA · Paper Trading Interno · 30 m · ETH/USDT · Preços Phemex Live</footer>
<script>
function p(n){return String(n).padStart(2,'0')}
function tick(){const d=new Date();document.getElementById('clock').textContent=
  `${d.getUTCFullYear()}-${p(d.getUTCMonth()+1)}-${p(d.getUTCDate())} `+
  `${p(d.getUTCHours())}:${p(d.getUTCMinutes())}:${p(d.getUTCSeconds())} UTC`;
  // next 30m candle boundary (UTC) + countdown → when the next entry happens
  const nx=new Date(d);
  if(d.getUTCMinutes()<30){nx.setUTCMinutes(30,0,0);}
  else{nx.setUTCHours(d.getUTCHours()+1,0,0,0);}
  const secs=Math.max(0,Math.round((nx-d)/1000)),mm=Math.floor(secs/60),ss=secs%60;
  const el=document.getElementById('c-next');
  if(el)el.textContent=`${p(nx.getUTCHours())}:${p(nx.getUTCMinutes())} UTC (em ${mm}:${p(ss)})`;
}
setInterval(tick,1000);tick();

async function pollPrice(){
  try{
    const r=await fetch('/price');const d=await r.json();
    const v=document.getElementById('lp-val'),dot=document.getElementById('lp-dot');
    if(d.price>0){
      v.textContent='$'+Number(d.price).toLocaleString('en-US',
        {minimumFractionDigits:2,maximumFractionDigits:2});
      const stale=(d.age_s==null||d.age_s>6);
      v.classList.toggle('stale',stale);dot.classList.toggle('stale',stale);
    }
  }catch(e){}
}
setInterval(pollPrice,1000);pollPrice();

// ── Tabs ──
function showTab(t){
  document.getElementById('tab-paper').style.display    = (t==='paper')?'':'none';
  document.getElementById('tab-backtest').style.display = (t==='backtest')?'':'none';
  document.getElementById('tb-paper').classList.toggle('active', t==='paper');
  document.getElementById('tb-backtest').classList.toggle('active', t==='backtest');
}

// ── Config (SL / TP / trailing — vale pro bot e pro backtest) ──
function loadConfig(){
  fetch('/config').then(r=>r.json()).then(d=>{
    document.getElementById('cfg-sl').value=d.sl_pts;
    document.getElementById('cfg-tp').value=d.tp_pts;
    document.getElementById('cfg-tr').value=d.tr_pts;
    document.getElementById('cfg-mt-sl').value=d.mintick_sl;
    document.getElementById('cfg-mt-tp').value=d.mintick_tp;
    document.getElementById('cfg-gl').value=d.gain_limit;
    document.getElementById('cfg-eq').textContent=
      `1pt SL = $${d.mintick_sl} · 1pt TP/Trail = $${d.mintick_tp} → `+
      `SL $${d.sl_dist} · TP $${d.tp_dist} · Trail $${d.tr_dist}  (vale pro bot e backtest)`;
  });
}
async function saveConfig(){
  const body={sl_pts:parseFloat(document.getElementById('cfg-sl').value),
              tp_pts:parseFloat(document.getElementById('cfg-tp').value),
              tr_pts:parseFloat(document.getElementById('cfg-tr').value),
              mintick_sl:parseFloat(document.getElementById('cfg-mt-sl').value),
              mintick_tp:parseFloat(document.getElementById('cfg-mt-tp').value),
              gain_limit:parseInt(document.getElementById('cfg-gl').value)};
  const r=await fetch('/config',{method:'POST',headers:{'Content-Type':'application/json'},
                                 body:JSON.stringify(body)});
  const d=await r.json();toast(d.message,d.ok);loadConfig();
}

// ── Backtest ──
let btTimer=null;
function btBadge(r){
  if(r==='TRAIL_TP')return '<span class="badge tp">TRAIL_TP</span>';
  if(r==='SL')return '<span class="badge slr">SL</span>';
  return '<span class="badge ce">'+r+'</span>';
}
async function runBacktest(){
  const n=parseInt(document.getElementById('bt-candles').value)||300;
  const fe=parseFloat(document.getElementById('bt-fee-entry').value)||0;
  const fx=parseFloat(document.getElementById('bt-fee-exit').value)||0;
  const src=document.getElementById('bt-source').value;
  document.getElementById('btn-bt').disabled=true;
  const r=await fetch('/backtest',{method:'POST',headers:{'Content-Type':'application/json'},
                                   body:JSON.stringify({candles:n,fee_entry:fe,fee_exit:fx,source:src})});
  const d=await r.json();toast(d.message,d.ok);
  if(!d.ok){document.getElementById('btn-bt').disabled=false;return;}
  if(btTimer)clearInterval(btTimer);
  btTimer=setInterval(pollBacktest,1500);pollBacktest();
}
async function pollBacktest(){
  try{
    const r=await fetch('/backtest-result');const d=await r.json();
    const pr=document.getElementById('bt-progress');
    if(d.running){pr.textContent='⏳ '+(d.progress||'rodando…');return;}
    if(btTimer){clearInterval(btTimer);btTimer=null;}
    document.getElementById('btn-bt').disabled=false;
    if(d.error){pr.textContent='❌ '+d.error;return;}
    if(d.result){pr.textContent='✅ '+(d.progress||'concluído');renderBacktest(d.result);}
  }catch(e){}
}
function renderBacktest(res){
  const s=res.summary;
  document.getElementById('bt-trades').textContent=s.trades;
  document.getElementById('bt-win').textContent=s.winrate+'%';
  const pl=document.getElementById('bt-pnl');
  pl.textContent=s.total_pnl_pct+'%';pl.className='val '+(s.total_pnl_pct>=0?'pos-num':'neg-num');
  const avg=document.getElementById('bt-avg');
  avg.textContent=s.avg_pct+'%'+(s.avg_usd!==undefined?' ($'+s.avg_usd+')':'');
  avg.className='val '+(s.avg_pct>=0?'pos-num':'neg-num');
  document.getElementById('bt-max').textContent='+'+s.max_profit+'%';
  document.getElementById('bt-max').className='val pos-num';
  document.getElementById('bt-min').textContent=s.max_loss+'%';
  document.getElementById('bt-min').className='val neg-num';
  document.getElementById('bt-bal').textContent='$'+s.final_balance;
  document.getElementById('bt-1m').textContent=
    s.intrabar_minutes_loaded!==undefined?s.intrabar_minutes_loaded:
    (s.bars30m!==undefined?s.bars30m+' ×30m':
    (s.ticks_loaded!==undefined?s.ticks_loaded+' ticks':'—'));
  const tb=document.getElementById('bt-tbody');
  tb.innerHTML=res.trades.length?res.trades.map(r=>`<tr>
    <td style="font-size:.73rem">${r.candle}</td>
    <td>${badge(r.side)}</td>
    <td>$${r.entry.toFixed(2)}</td>
    <td>$${r.exit.toFixed(2)}</td>
    <td>${btBadge(r.reason)}</td>
    <td class="${r.pnl_pct>=0?'pos-num':'neg-num'}">${r.pnl_pct.toFixed(3)}%</td>
    <td class="pos-num">${r.mfe.toFixed(3)}%</td>
    <td class="neg-num">${r.mae.toFixed(3)}%</td>
    <td>$${r.balance.toFixed(2)}</td></tr>`).join('')
    :'<tr><td colspan="9" style="text-align:center;color:#484f58;padding:20px">Sem trades</td></tr>';
}
loadConfig();

function toast(msg,ok){const e=document.getElementById('toast');
  e.textContent=msg;e.style.background=ok?'#238636':'#da3633';
  e.style.display='block';setTimeout(()=>e.style.display='none',3000)}

function pnlCls(v){const n=parseFloat(v);return n>0?'val pos-num':n<0?'val neg-num':'val'}

function badge(s){
  if(s==='LONG')  return '<span class="badge long">LONG</span>';
  if(s==='SHORT') return '<span class="badge short">SHORT</span>';
  return `<span class="badge close">${s}</span>`}

async function ctrl(a){
  const r=await fetch('/'+a,{method:'POST'});
  const d=await r.json();toast(d.message,d.ok);fetchAll()}

async function saveFees(){
  const entry=parseFloat(document.getElementById('fee-entry').value)||0;
  const exit =parseFloat(document.getElementById('fee-exit').value)||0;
  const r=await fetch('/settings',{method:'POST',
    headers:{'Content-Type':'application/json'},
    body:JSON.stringify({fee_entry:entry,fee_exit:exit})});
  const d=await r.json();toast(d.message,d.ok)}

async function fetchAll(){
  const[sr,hr]=await Promise.all([fetch('/status'),fetch('/history')]);
  const s=await sr.json(),t=await hr.json();

  const sv=document.getElementById('c-status');sv.textContent=s.status;
  sv.className='val '+(s.status==='Running'?'running':
    (s.status.startsWith('Error')||s.status.startsWith('Waiting'))?'err':'stopped');
  const sg=document.getElementById('c-signal');sg.textContent=s.signal;
  sg.className='val '+(s.signal==='LONG'?'long':s.signal==='SHORT'?'short':'');
  const ps=document.getElementById('c-pos');ps.textContent=s.position;
  ps.className='val '+(s.position==='long'?'long':s.position==='short'?'short':'');
  document.getElementById('c-bal').textContent='$'+s.balance;
  const pnl=document.getElementById('c-pnl');
  pnl.textContent=s.total_pnl_pct+'%';pnl.className=pnlCls(s.total_pnl_pct);
  const unr=document.getElementById('c-unr');
  unr.textContent=s.unrealized+'%';unr.className=pnlCls(s.unrealized);
  const mfe=document.getElementById('c-mfe');
  mfe.textContent=s.mfe+'%';mfe.className=pnlCls(s.mfe);
  const mae=document.getElementById('c-mae');
  mae.textContent=s.mae+'%';mae.className=pnlCls(s.mae);
  document.getElementById('c-win').textContent=s.winrate;
  document.getElementById('c-trail').textContent=s.trail_stop;
  ['ema','ec','period'].forEach(k=>document.getElementById('c-'+k).textContent=s[k]);
  document.getElementById('c-candle').textContent=s.last_candle;
  document.getElementById('btn-start').disabled=s.running;
  document.getElementById('btn-stop').disabled=!s.running;

  const tb=document.getElementById('tbody');
  tb.innerHTML=t.length?t.map(r=>`<tr>
    <td>${r.time}</td>
    <td style="font-size:.73rem">${r.candle}</td>
    <td>${badge(r.side)}</td>
    <td>$${Number(r.price).toFixed(2)}</td>
    <td>${r.qty||'—'}</td>
    <td class="${r.pnl_pct&&r.pnl_pct!=='—'?(parseFloat(r.pnl_pct)>=0?'pos-num':'neg-num'):''}">
      ${r.pnl_pct&&r.pnl_pct!=='—'?r.pnl_pct+'%':'—'}</td>
    <td class="${r.mfe&&r.mfe!=='—'?'pos-num':''}">${r.mfe&&r.mfe!=='—'?r.mfe+'%':'—'}</td>
    <td class="${r.mae&&r.mae!=='—'?'neg-num':''}">${r.mae&&r.mae!=='—'?r.mae+'%':'—'}</td>
    <td>$${r.balance}</td>
    <td style="font-size:.68rem;color:#8b949e">${r.order_id}</td></tr>`).join('')
    :'<tr><td colspan="10" style="text-align:center;color:#484f58;padding:20px">Nenhuma trade ainda</td></tr>'}
// Load saved fee values on page open
fetch('/settings').then(r=>r.json()).then(d=>{
  document.getElementById('fee-entry').value=d.fee_entry;
  document.getElementById('fee-exit').value=d.fee_exit;
  // Backtest uses the SAME fees as the live bot by default — pre-fill them here
  // so the numbers line up with paper trading unless you deliberately change them.
  document.getElementById('bt-fee-entry').value=d.fee_entry;
  document.getElementById('bt-fee-exit').value=d.fee_exit;
});
fetchAll();setInterval(fetchAll,10000);
</script>
</body>
</html>"""

@app.route("/")
def index(): return HTML

@app.route("/ping")
@app.route("/health")
def ping(): return jsonify({"ok": True, "time": datetime.utcnow().isoformat()})

@app.route("/status")
def status(): return jsonify({**state, "running": state["running"]})

@app.route("/price")
def price():
    """Fast live-price read for the UI (polled ~every second)."""
    age = round(time_mod.time() - _last_price_ts, 1) if _last_price_ts else None
    return jsonify({
        "price":  _last_good_price,
        "age_s":  age,                 # seconds since last update; small = live
        "ws":     _ws_active,          # websocket delivering ticks?
        "symbol": SYMBOL or "ETH/USDT:USDT",
    })

@app.route("/history")
def history(): return jsonify(trade_history)

@app.route("/settings", methods=["GET"])
def get_settings():
    return jsonify({"fee_entry": fees["entry"], "fee_exit": fees["exit"]})

@app.route("/settings", methods=["POST"])
def settings():
    data = request.get_json() or {}
    if "fee_entry" in data:
        fees["entry"] = float(data["fee_entry"])
    if "fee_exit" in data:
        fees["exit"] = float(data["fee_exit"])
    _save_fees()
    logger.info(f"[FEES] entry={fees['entry']}%  exit={fees['exit']}%")
    return jsonify({"ok": True, "message": f"Fees: entrada {fees['entry']}% / saída {fees['exit']}%"})

@app.route("/config", methods=["GET"])
def get_config():
    # POINT counts + os dois valores de ponto (editable) + $ distances (display)
    return jsonify({"sl_pts": config["sl_pts"], "tp_pts": config["tp_pts"],
                    "tr_pts": config["tr_pts"],
                    "mintick_sl": config.get("mintick_sl", MINTICK_DEFAULT),
                    "mintick_tp": config.get("mintick_tp", MINTICK_DEFAULT),
                    "gain_limit": config.get("gain_limit", 900),
                    "sl_dist": SL_DIST, "tp_dist": TP_DIST, "tr_dist": TR_DIST})

@app.route("/config", methods=["POST"])
def set_config():
    """Edit the exit distances in POINTS + the strategy Gain Limit."""
    data = request.get_json() or {}
    for k in ("sl_pts", "tp_pts", "tr_pts"):
        if k in data:
            try:
                v = float(data[k])
                if v > 0: config[k] = v
            except Exception:
                return jsonify({"ok": False, "message": f"valor inválido em {k}"})
    for k in ("mintick_sl", "mintick_tp"):
        if k in data:
            try:
                mt = float(data[k])
                if mt > 0: config[k] = mt
            except Exception:
                return jsonify({"ok": False, "message": f"{k} inválido"})
    if "gain_limit" in data:
        try:
            gl = int(data["gain_limit"])
            if gl >= 1: config["gain_limit"] = gl
        except Exception:
            return jsonify({"ok": False, "message": "Gain Limit inválido"})
    _save_config()
    _apply_config()
    logger.info(f"[CONFIG] SL={config['sl_pts']}pt×{MINTICK_SL}=${SL_DIST}  "
                f"TP={config['tp_pts']}pt×{MINTICK_TP}=${TP_DIST}  "
                f"trail={config['tr_pts']}pt×{MINTICK_TP}=${TR_DIST}")
    return jsonify({"ok": True,
                    "message": f"Config: SL {config['sl_pts']}pt × {MINTICK_SL} = ${SL_DIST} · "
                               f"TP {config['tp_pts']}pt × {MINTICK_TP} = ${TP_DIST} · "
                               f"Trail {config['tr_pts']}pt × {MINTICK_TP} = ${TR_DIST} · "
                               f"GainLimit={config.get('gain_limit', 900)}"})

@app.route("/backtest", methods=["POST"])
def backtest():
    data = request.get_json() or {}
    # Default the fees to the SAME ones the live paper bot uses (fees.json / the
    # Settings tab). The two used to be decoupled — running a backtest with 0 fees
    # while the live bot charged a fee silently produced different numbers. Now an
    # omitted/blank field inherits the live fee; the backtest form can still
    # override it explicitly.
    n = 300; fe = float(fees["entry"]); fx = float(fees["exit"])
    # No 1000-candle cap — long backtests (e.g. 1 year = 17 520 candles) allowed.
    # 35 040 ≈ 2 years is a sanity ceiling so a typo can't hang the server forever.
    try: n  = max(50, min(35040, int(data.get("candles", 300))))
    except Exception: pass
    try: fe = max(0.0, float(data.get("fee_entry", fees["entry"])))
    except Exception: pass
    try: fx = max(0.0, float(data.get("fee_exit", fees["exit"])))
    except Exception: pass
    # source: "synthetic" (1m candles + realistic path — works for any length,
    # any period), "tv" (TradingView Strategy Tester emulator — multi-candle
    # positions, no CANDLE_END, Pine sizing; the mode to compare with the TV
    # panel) or "ticks" (replay the REAL recorded 1s stream — 1:1 with paper
    # trading, but only for the period RECORD_TICKS captured).
    source = str(data.get("source", "synthetic")).lower()
    # slot (0..N_BT_SLOTS-1): permite backtests synthetic SIMULTÂNEOS, cada um
    # com variáveis próprias passadas inline (sl_pts/tp_pts/tr_pts/mintick_sl/
    # mintick_tp/gain_limit) — sem tocar no config global. tv/ticks: slot 0.
    try:
        slot = max(0, min(N_BT_SLOTS - 1, int(data.get("slot", 0))))
    except Exception:
        slot = 0
    pkeys  = ("sl_pts", "tp_pts", "tr_pts", "mintick_sl", "mintick_tp", "gain_limit")
    params = {k: data[k] for k in pkeys if k in data} or None
    if source == "tv":
        if _backtests[0]["running"]:
            return jsonify({"ok": False, "message": "Backtest já está rodando (slot 0)"})
        threading.Thread(target=run_backtest_tv, args=(n, fe, fx), daemon=True).start()
        return jsonify({"ok": True,
                        "message": f"Backtest emulador TradingView ({n} candles) iniciado"})
    if source == "ticks":
        if _backtests[0]["running"]:
            return jsonify({"ok": False, "message": "Backtest já está rodando (slot 0)"})
        if not _load_recorded_ticks():
            return jsonify({"ok": False, "message":
                "Nenhum tick gravado ainda. A gravação é automática enquanto a "
                "strategy está iniciada (Start) — deixe o bot rodando um tempo e "
                "tente de novo. Acompanhe o quanto já foi gravado em /ticks-info."})
        threading.Thread(target=run_backtest_ticks, args=(fe, fx), daemon=True).start()
        return jsonify({"ok": True, "message": "Backtest por ticks reais gravados iniciado"})
    if _backtests[slot]["running"]:
        return jsonify({"ok": False, "message": f"Backtest já está rodando (slot {slot})"})
    threading.Thread(target=run_backtest, args=(n, fe, fx, slot, params),
                     daemon=True).start()
    return jsonify({"ok": True, "slot": slot,
                    "message": f"Backtest de {n} candles iniciado (slot {slot})"})

@app.route("/backtest-result")
def backtest_result():
    try:
        slot = max(0, min(N_BT_SLOTS - 1, int(request.args.get("slot", 0))))
    except Exception:
        slot = 0
    return jsonify(_backtests[slot])

@app.route("/backtest-slots")
def backtest_slots():
    """Visão rápida dos 5 slots paralelos (pro otimizador escolher um livre)."""
    return jsonify([{"slot": i, "running": b["running"], "progress": b["progress"],
                     "has_result": b["result"] is not None, "error": b["error"]}
                    for i, b in enumerate(_backtests)])

@app.route("/ticks-info")
def ticks_info():
    """How much real 1s data has been recorded so far (for the ticks backtest)."""
    ticks = _load_recorded_ticks()
    if not ticks:
        return jsonify({"recording": RECORD_TICKS, "ticks": 0, "hours": 0.0,
                        "note": ("gravação ativa — clique Start e deixe o bot "
                                 "rodando pra acumular ticks" if RECORD_TICKS
                                 else "gravação desligada (RECORD_TICKS=0)")})
    span_h = round((ticks[-1][0] - ticks[0][0]) / 3_600_000, 2)
    return jsonify({"recording": RECORD_TICKS, "ticks": len(ticks),
                    "hours": span_h,
                    "from": datetime.fromtimestamp(ticks[0][0]/1000, tz=timezone.utc)
                            .strftime("%Y-%m-%d %H:%M UTC"),
                    "to":   datetime.fromtimestamp(ticks[-1][0]/1000, tz=timezone.utc)
                            .strftime("%Y-%m-%d %H:%M UTC")})

@app.route("/debug-ticker")
def debug_ticker():
    """
    Diagnoses the live-price feed AND exit engine. If a position is stuck open,
    open this: it shows whether prices are arriving (price_age_s should be small),
    whether the websocket is delivering ticks, and where the SL/trailing sits vs
    the current price — so you can see exactly why it has or hasn't fired.
    """
    t0 = time_mod.time()
    price = get_live_price()
    elapsed_ms = round((time_mod.time() - t0) * 1000, 1)
    age = round(time_mod.time() - _last_price_ts, 1) if _last_price_ts else None
    return jsonify({
        "live_price":        price,
        "last_good_price":   _last_good_price,
        "price_age_s":       age,            # seconds since any price update; small = healthy
        "ws_available":      _HAS_WS,        # ccxt.pro imported ok
        "ws_active":         _ws_active,     # websocket currently delivering ticks
        "ws_tick_count":     _ws_tick_count, # total ticks seen (should keep rising)
        "realtime_exits":    REALTIME_EXITS, # MUST be true or exits only at candle close
        "consecutive_fails": _ticker_fail_count,
        "fetch_time_ms":     elapsed_ms,
        "symbol":            SYMBOL,
        # ── current position / why it may not have exited ──
        "position":          paper["side"],
        "entry_price":       paper["entry_price"],
        "peak":              paper["peak"],
        "trail_active":      paper["trail_active"],
        "trail_stop":        paper["trail_stop"],
        "sl_dist":           SL_DIST,
        "tp_activation_dist": TP_DIST,
        "trail_offset_dist": TR_DIST,
        "mintick_sl":        MINTICK_SL,   # valor de 1 ponto do SL
        "mintick_tp":        MINTICK_TP,   # valor de 1 ponto do TP/trailing
    })

@app.route("/debug-raw")
def debug_raw():
    """
    Surfaces the RAW reason the live price is (not) arriving. Runs a fresh
    load_markets + fetch_ticker WITHOUT swallowing the error, so the exact cause —
    a Phemex geo-block (403/451), rate-limit (DDoSProtection), timeout, or a
    symbol/endpoint change — shows up directly here in the browser instead of only
    in the Render logs. Open /debug-raw and read `fetch_ticker`.
    """
    info = {
        "symbol":             SYMBOL,
        "symbol_used":        SYMBOL or "ETH/USDT:USDT",
        "last_good_price":    _last_good_price,
        "price_age_s":        round(time_mod.time() - _last_price_ts, 1) if _last_price_ts else None,
        "ticker_fail_count":  _ticker_fail_count,
        "ws_available":       _HAS_WS,
        "ws_active":          _ws_active,
        "record_ticks":       RECORD_TICKS,
    }
    # 1) Can we even load the market list from Phemex?
    try:
        mkts = ticker_ex.load_markets()
        info["load_markets"]  = "ok"
        info["markets_count"] = len(mkts)
        info["eth_usdt_swap_present"] = ("ETH/USDT:USDT" in mkts)
    except Exception as e:
        info["load_markets"] = f"FAIL: {type(e).__name__}: {str(e)[:300]}"
    # 2) The actual price call — the exact error, unswallowed.
    try:
        t = ticker_ex.fetch_ticker(SYMBOL or "ETH/USDT:USDT")
        info["fetch_ticker"] = "ok"
        info["last"] = t.get("last")
        info["bid"]  = t.get("bid")
        info["ask"]  = t.get("ask")
    except Exception as e:
        info["fetch_ticker"] = f"FAIL: {type(e).__name__}: {str(e)[:400]}"
    return jsonify(info)

@app.route("/diagnostico")
def diagnostico():
    """UM link que testa TUDO que o bot precisa da Phemex e mostra o erro real
    de cada passo. Abrir quando "não puxa candles" ou "não tem trades":
    cada item vem com ok:true ou o erro exato (geo-block 403/451, rate-limit,
    timeout, DNS…). bt_ex/ticker_ex são usados aqui pra não concorrer com a
    sessão do loop ao vivo (live_ex)."""
    sym = SYMBOL or "ETH/USDT:USDT"
    out = {
        "symbol":            sym,
        "strategy_rodando":  state["running"],
        "status_atual":      state["status"],
        "ws_ativo":          _ws_active,
        "preco_atual":       _last_good_price,
        "idade_preco_s":     round(time_mod.time() - _last_price_ts, 1) if _last_price_ts else None,
        "erro_candles_bot":  _ohlcv_last_err,     # último erro do loop ao vivo
        "erro_fetch_backtest": _bt_fetch_err,     # último erro do backtest
    }
    def step(name, fn):
        try:
            out[name] = {"ok": True, "info": fn()}
        except Exception as e:
            out[name] = {"ok": False, "erro": f"{type(e).__name__}: {str(e)[:300]}"}
    # Passos com timeout CURTO e HTTP cru (sem ccxt): o gunicorn mata o worker
    # em ~30 s — o diagnóstico inteiro tem que caber nisso MESMO com tudo
    # falhando (6+8+8 = 22 s no pior caso), senão o próprio diagnóstico derruba
    # o bot. E o IP/país de saída prova em qual região o serviço está de fato.
    step("1_ip_saida", lambda: {k: req.get("https://ipwho.is/", timeout=6).json().get(k)
                                for k in ("ip", "country", "city")})
    step("2_phemex_products", lambda: (lambda r:
         f"HTTP {r.status_code} em {r.elapsed.total_seconds():.1f}s")(
         req.get("https://api.phemex.com/public/products", timeout=8)))
    step("3_phemex_ticker", lambda: (lambda r:
         f"HTTP {r.status_code} em {r.elapsed.total_seconds():.1f}s")(
         req.get("https://api.phemex.com/md/v2/ticker/24hr?symbol=ETHUSDT",
                 timeout=8)))
    ok_net    = out["1_ip_saida"]["ok"]
    ok_phemex = (out["2_phemex_products"]["ok"]
                 and "HTTP 200" in str(out["2_phemex_products"].get("info", "")))
    if ok_phemex:
        out["conclusao"] = ("Phemex RESPONDE deste servidor. Se o bot segue sem preço/"
                            "candles, reinicie o serviço na Render e clique Start; se o "
                            "status mostrar 'Erro candles 30m', me mande o texto.")
    elif ok_net:
        out["conclusao"] = ("Internet OK mas a Phemex NÃO responde deste servidor (IP/"
                            "região bloqueada). Olhe 1_ip_saida: se country NÃO for "
                            "Singapore, a região não mudou de verdade — na Render é "
                            "preciso CRIAR UM SERVIÇO NOVO na região desejada (não dá "
                            "pra mudar um existente). Se já for Singapore, a Phemex está "
                            "bloqueando os IPs desse datacenter — teste outra hospedagem.")
    else:
        out["conclusao"] = ("Sem internet de saída no servidor — problema na hospedagem, "
                            "não no bot.")
    return jsonify(out)

@app.route("/start", methods=["POST"])
def start():
    global _strategy_thread
    # Only refuse if the strategy thread is actually ALIVE. If `running` got stuck
    # True but the thread died (e.g. an exception outside the loop), the old check
    # said "Já em execução" forever and nothing could restart it → "não inicia
    # trade nenhuma". Now a dead thread is restarted.
    alive = _strategy_thread is not None and _strategy_thread.is_alive()
    if state["running"] and alive:
        return jsonify({"ok": False, "message": "Já em execução"})
    state["running"] = True
    state["status"]  = "Iniciando…"
    _strategy_thread = threading.Thread(target=strategy_loop, daemon=True)
    _strategy_thread.start()
    return jsonify({"ok": True, "message": "Strategy iniciada"})

@app.route("/stop", methods=["POST"])
def stop():
    if not state["running"]:
        return jsonify({"ok": False, "message": "Já parada"})
    state["running"] = False
    return jsonify({"ok": True, "message": "Strategy parada"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT)
