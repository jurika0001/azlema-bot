import os
import json
import time
import logging
import threading
from datetime import datetime

from flask import Flask, jsonify, render_template

import keepalive
from mexc_client import MEXCPaperClient, TIMEFRAME_SECONDS
from strategy import calculate_signal

# ── logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ── Flask ─────────────────────────────────────────────────────────────────────
app = Flask(__name__)

# ── persistent state ─────────────────────────────────────────────────────────
TRADES_FILE = os.environ.get("TRADES_FILE", "trades.json")

trade_history: list[dict] = []

bot_status = {
    "running":     False,
    "position":    "FLAT",   # FLAT | LONG | SHORT
    "last_signal": None,
    "last_update": None,
    "ema":         None,
    "ec":          None,
    "last_error":  None,
    "symbol":      os.environ.get("SYMBOL", "ETH_USDT"),
    "timeframe":   os.environ.get("TIMEFRAME", "Min15"),
}

client = MEXCPaperClient()


# ── helpers ───────────────────────────────────────────────────────────────────

def _load_trades() -> None:
    global trade_history
    try:
        if os.path.exists(TRADES_FILE):
            with open(TRADES_FILE, "r") as f:
                trade_history = json.load(f)
            logger.info(f"Loaded {len(trade_history)} trades from {TRADES_FILE}")
    except Exception as e:
        logger.warning(f"Could not load trades file: {e}")
        trade_history = []


def _save_trades() -> None:
    try:
        with open(TRADES_FILE, "w") as f:
            json.dump(trade_history, f, indent=2)
    except Exception as e:
        logger.error(f"Could not save trades: {e}")


def _close_open_trade(exit_price: float, exit_time: str) -> None:
    """Mark the last open trade as closed and calculate PnL."""
    for trade in reversed(trade_history):
        if trade.get("status") == "OPEN":
            trade["status"]     = "CLOSED"
            trade["exit_price"] = exit_price
            trade["exit_time"]  = exit_time
            entry = trade["entry_price"]
            if trade["type"] == "LONG":
                trade["pnl_pct"] = round((exit_price - entry) / entry * 100, 3)
            else:
                trade["pnl_pct"] = round((entry - exit_price) / entry * 100, 3)
            break


# ── trading loop ─────────────────────────────────────────────────────────────

def _trading_loop() -> None:
    """
    Runs in a background daemon thread.
    Waits until the close of the next candle, then evaluates the strategy
    and executes the appropriate order on MEXC paper futures.
    """
    global bot_status, trade_history

    logger.info("Trading loop started — setting leverage …")
    try:
        resp = client.set_leverage()
        logger.info(f"set_leverage: {resp}")
    except Exception as e:
        logger.warning(f"set_leverage failed: {e}")

    tf_secs = TIMEFRAME_SECONDS.get(client.timeframe, 900)
    logger.info(f"Timeframe: {client.timeframe} ({tf_secs}s per candle)")

    bot_status["running"] = True

    while True:
        try:
            # ── wait for next candle close ────────────────────────────────
            now        = time.time()
            next_close = (now // tf_secs + 1) * tf_secs
            wait       = next_close - now
            logger.info(f"Next candle in {wait:.0f}s …")
            time.sleep(wait + 1)  # +1 s margin so close is confirmed

            # ── fetch data & calculate strategy ───────────────────────────
            closes = client.get_klines(limit=200)
            if len(closes) < 20:
                logger.warning("Not enough kline data — skipping candle")
                continue

            signal, ema, ec, err = calculate_signal(closes)
            price                = client.get_ticker()
            ts                   = datetime.utcnow().isoformat() + "Z"

            bot_status.update({
                "last_signal": signal,
                "last_update": ts,
                "ema":         round(ema,   6),
                "ec":          round(ec,    6),
                "last_error":  round(err,   6),
            })

            logger.info(f"Signal={signal}  price={price}  EC={ec:.4f}  EMA={ema:.4f}")

            current_pos = bot_status["position"]

            # ── LONG ──────────────────────────────────────────────────────
            if signal == "LONG" and current_pos != "LONG":
                if current_pos == "SHORT":
                    res = client.place_order(2)          # close short
                    logger.info(f"close SHORT → {res}")
                    _close_open_trade(price, ts)

                res = client.place_order(1)              # open long
                logger.info(f"open LONG  → {res}")

                trade_history.append({
                    "id":          len(trade_history) + 1,
                    "open_time":   ts,
                    "type":        "LONG",
                    "entry_price": price,
                    "exit_price":  None,
                    "exit_time":   None,
                    "pnl_pct":     None,
                    "status":      "OPEN",
                    "ema":         round(ema, 6),
                    "ec":          round(ec,  6),
                })
                bot_status["position"] = "LONG"
                _save_trades()

            # ── SHORT ─────────────────────────────────────────────────────
            elif signal == "SHORT" and current_pos != "SHORT":
                if current_pos == "LONG":
                    res = client.place_order(4)          # close long
                    logger.info(f"close LONG  → {res}")
                    _close_open_trade(price, ts)

                res = client.place_order(3)              # open short
                logger.info(f"open SHORT → {res}")

                trade_history.append({
                    "id":          len(trade_history) + 1,
                    "open_time":   ts,
                    "type":        "SHORT",
                    "entry_price": price,
                    "exit_price":  None,
                    "exit_time":   None,
                    "pnl_pct":     None,
                    "status":      "OPEN",
                    "ema":         round(ema, 6),
                    "ec":          round(ec,  6),
                })
                bot_status["position"] = "SHORT"
                _save_trades()

            else:
                logger.info(f"No action — already {current_pos}, signal={signal}")

        except Exception as exc:
            logger.exception(f"Trading loop error: {exc}")
            time.sleep(60)


# ── Flask routes ─────────────────────────────────────────────────────────────

@app.route("/")
def dashboard():
    return render_template(
        "index.html",
        bot=bot_status,
        trades=list(reversed(trade_history)),
        ka=keepalive.status,
    )


@app.route("/api/status")
def api_status():
    return jsonify(bot_status)


@app.route("/api/trades")
def api_trades():
    return jsonify(list(reversed(trade_history)))


@app.route("/api/keepalive")
def api_keepalive():
    return jsonify(keepalive.status)


# ── Internal keep-alive endpoints ─────────────────────────────────────────────

@app.route("/ka/1")
def ka1():
    keepalive.status["signal_1"]["last_ping"] = datetime.utcnow().isoformat() + "Z"
    return "ok-1", 200


@app.route("/ka/2")
def ka2():
    keepalive.status["signal_2"]["last_ping"] = datetime.utcnow().isoformat() + "Z"
    return "ok-2", 200


@app.route("/ka/3")
def ka3():
    keepalive.status["signal_3"]["last_ping"] = datetime.utcnow().isoformat() + "Z"
    return "ok-3", 200


# ── External keep-alive endpoint (UptimeRobot) ────────────────────────────────

@app.route("/uptimerobot")
def uptimerobot():
    keepalive.record_external()
    return jsonify({
        "status": "alive",
        "time":   datetime.utcnow().isoformat() + "Z",
        "bot":    bot_status["running"],
    })


# ── Generic health / ping ─────────────────────────────────────────────────────

@app.route("/health")
@app.route("/ping")
@app.route("/alive")
def health():
    return jsonify({"status": "ok", "time": datetime.utcnow().isoformat() + "Z"})


# ── Startup ───────────────────────────────────────────────────────────────────

def _start_background_threads() -> None:
    _load_trades()

    # 3 internal keep-alive threads
    keepalive.start()

    # Main trading loop thread
    t = threading.Thread(target=_trading_loop, daemon=True, name="trading-loop")
    t.start()
    logger.info("All background threads started")


_start_background_threads()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
