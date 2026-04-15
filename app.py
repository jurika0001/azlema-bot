import os
import time
import threading
import logging
import requests
from flask import Flask, render_template, jsonify
from bot import TradingBot, trade_history, bot_status

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

app  = Flask(__name__)
_bot = TradingBot()


# ── Routes ─────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html",
                           status=bot_status,
                           trades=list(reversed(trade_history)))

@app.route("/ping")
def ping():
    """External keepalive endpoint — point UptimeRobot here."""
    return jsonify({"ok": True, "ts": time.time()}), 200

@app.route("/health")
def health():
    return jsonify(bot_status), 200

@app.route("/api/trades")
def api_trades():
    return jsonify(list(reversed(trade_history))), 200


# ── Internal keepalive workers ─────────────────────────────────────────────────
def _keepalive(interval: int, sid: int):
    """
    Pings own /ping endpoint every `interval` seconds.
    Waits 20 s on startup to let the server come up first.
    """
    time.sleep(20)
    port = int(os.environ.get("PORT", 5000))
    url  = f"http://127.0.0.1:{port}/ping"
    while True:
        try:
            r = requests.get(url, timeout=5)
            logger.debug(f"[KA-{sid}] {interval}s → HTTP {r.status_code}")
        except Exception as e:
            logger.warning(f"[KA-{sid}] {interval}s → {e}")
        time.sleep(interval)


# ── Background startup ─────────────────────────────────────────────────────────
_started = False

def _start_all():
    """Called once when gunicorn imports the module (workers=1, no preload)."""
    global _started
    if _started:
        return
    _started = True

    # Signal 1 → every 8 s
    # Signal 2 → every 15 s
    # Signal 3 → every 23 s
    for sid, iv in enumerate([8, 15, 23], start=1):
        t = threading.Thread(target=_keepalive, args=(iv, sid),
                             daemon=True, name=f"ka-{sid}")
        t.start()
        logger.info(f"[KA-{sid}] started — interval {iv}s")

    # Trading bot
    bt = threading.Thread(target=_bot.run, daemon=True, name="trading-bot")
    bt.start()
    logger.info("[Bot] thread started")


_start_all()   # runs at import time inside the gunicorn worker process


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False,
            use_reloader=False, threaded=True)
