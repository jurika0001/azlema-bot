import os
import time
import logging
import threading
from datetime import datetime, timezone

import ccxt
import numpy as np

import strategy

logger = logging.getLogger(__name__)

# ── Shared state (imported by app.py) ─────────────────────────────────────────
trade_history: list = []
bot_status: dict = {
    "running":    False,
    "updated_at": "—",
    "signal":     "NONE",
    "position":   "NONE",
    "ec":         None,
    "ema":        None,
    "balance":    None,
    "symbol":     "ETH/USDT:USDT",
    "error":      None,
}
_lock = threading.Lock()


class TradingBot:
    # ── Strategy / risk config ────────────────────────────────────────────────
    SYMBOL     = "ETH/USDT:USDT"
    TIMEFRAME  = "45m"
    CANDLES    = 200
    LEVERAGE   = 1
    RISK       = 0.01          # 1 % risk per trade
    PERIOD     = 20
    GAIN_LIMIT = 50
    THRESHOLD  = 0.0
    ADAPTIVE   = "Cos IFM"
    SL_POINTS  = 2000          # stop-loss ticks
    POLL       = 60            # seconds between main-loop iterations

    def __init__(self):
        self.ex      = self._connect()
        self._pos    = "NONE"   # LONG | SHORT | NONE
        self._qty    = 0.0
        self._entry  = 0.0

    # ── Exchange helpers ───────────────────────────────────────────────────────
    def _connect(self) -> ccxt.phemex:
        ex = ccxt.phemex({
            "apiKey":  os.environ.get("PHEMEX_API_KEY",    ""),
            "secret":  os.environ.get("PHEMEX_API_SECRET", ""),
            "options": {"defaultType": "swap"},
            "enableRateLimit": True,
        })
        ex.set_sandbox_mode(True)   # ← paper / testnet mode
        ex.load_markets()
        return ex

    def _set_leverage(self):
        try:
            self.ex.set_leverage(self.LEVERAGE, self.SYMBOL)
            logger.info(f"[Bot] leverage = {self.LEVERAGE}x")
        except Exception as e:
            logger.warning(f"[Bot] set_leverage: {e}")

    def _balance(self) -> float:
        try:
            b = self.ex.fetch_balance()
            return float(b.get("USDT", {}).get("free") or 1000)
        except Exception:
            return 1000.0

    def _ohlcv(self) -> np.ndarray:
        raw = self.ex.fetch_ohlcv(self.SYMBOL, self.TIMEFRAME, limit=self.CANDLES)
        return np.array([c[4] for c in raw], dtype=float)

    def _qty_calc(self, price: float, balance: float) -> float:
        sl_usdt = self.SL_POINTS * 0.01          # 2000 pts × $0.01 = $20
        qty     = (self.RISK * balance) / sl_usdt
        qty     = max(qty, 0.001)
        try:
            qty = float(self.ex.amount_to_precision(self.SYMBOL, qty))
        except Exception:
            qty = round(qty, 3)
        return qty

    # ── Order helpers ──────────────────────────────────────────────────────────
    def _close(self, reason: str = "signal"):
        if self._pos == "NONE":
            return
        side = "sell" if self._pos == "LONG" else "buy"
        try:
            o     = self.ex.create_market_order(
                        self.SYMBOL, side, self._qty,
                        params={"reduceOnly": True})
            price = float(o.get("average") or o.get("price") or self._entry)
            pnl   = (price - self._entry) * self._qty * (1 if self._pos == "LONG" else -1)
            self._record("CLOSE", self._pos, price, self._qty, pnl, reason)
            logger.info(f"[Bot] CLOSE {self._pos} @ {price:.4f}  pnl={pnl:.4f}")
        except Exception as e:
            logger.error(f"[Bot] _close: {e}")
            bot_status["error"] = str(e)
        finally:
            self._pos   = "NONE"
            self._qty   = 0.0
            self._entry = 0.0

    def _open(self, direction: str, qty: float, price: float):
        side = "buy" if direction == "LONG" else "sell"
        try:
            o           = self.ex.create_market_order(self.SYMBOL, side, qty)
            fill_price  = float(o.get("average") or o.get("price") or price)
            self._pos   = direction
            self._qty   = qty
            self._entry = fill_price
            self._record("OPEN", direction, fill_price, qty, 0.0, "signal")
            logger.info(f"[Bot] OPEN {direction} @ {fill_price:.4f}  qty={qty}")
        except Exception as e:
            logger.error(f"[Bot] _open: {e}")
            bot_status["error"] = str(e)

    def _record(self, action, direction, price, qty, pnl, reason):
        with _lock:
            trade_history.append({
                "ts":        datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
                "action":    action,
                "direction": direction,
                "price":     round(price, 4),
                "qty":       round(qty,   4),
                "pnl":       round(pnl,   4),
                "reason":    reason,
            })

    # ── Main loop ──────────────────────────────────────────────────────────────
    def run(self):
        bot_status["running"] = True
        self._set_leverage()
        logger.info("[Bot] trading loop started")

        while True:
            try:
                closes  = self._ohlcv()
                balance = self._balance()

                sig, ec_v, ema_v, _ = strategy.signal(
                    closes,
                    period     = self.PERIOD,
                    gain_limit = self.GAIN_LIMIT,
                    threshold  = self.THRESHOLD,
                    adaptive   = self.ADAPTIVE,
                )
                price = float(closes[-1])
                qty   = self._qty_calc(price, balance)

                bot_status.update({
                    "updated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
                    "signal":     sig,
                    "position":   self._pos,
                    "ec":         round(ec_v,  4),
                    "ema":        round(ema_v, 4),
                    "balance":    round(balance, 2),
                    "error":      None,
                })

                if sig == "LONG"  and self._pos != "LONG":
                    self._close("reverse")
                    self._open("LONG",  qty, price)

                elif sig == "SHORT" and self._pos != "SHORT":
                    self._close("reverse")
                    self._open("SHORT", qty, price)

            except Exception as e:
                logger.error(f"[Bot] loop error: {e}")
                bot_status["error"] = str(e)

            time.sleep(self.POLL)
