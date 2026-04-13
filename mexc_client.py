import hmac
import hashlib
import time
import json
import requests
import os
import logging

logger = logging.getLogger(__name__)

MEXC_BASE_URL = os.environ.get("MEXC_BASE_URL", "https://contract.mexc.com")

TIMEFRAME_SECONDS = {
    "Min1":  60,
    "Min5":  300,
    "Min15": 900,
    "Min30": 1800,
    "Min60": 3600,
    "Hour4": 14400,
    "Hour8": 28800,
    "Day1":  86400,
}


class MEXCPaperClient:
    def __init__(self):
        self.api_key    = os.environ.get("MEXC_API_KEY", "")
        self.api_secret = os.environ.get("MEXC_API_SECRET", "")
        self.symbol     = os.environ.get("SYMBOL", "ETH_USDT")
        self.leverage   = int(os.environ.get("LEVERAGE", "1"))
        self.timeframe  = os.environ.get("TIMEFRAME", "Min15")
        self.vol        = float(os.environ.get("CONTRACT_SIZE", "1"))
        self.base_url   = MEXC_BASE_URL

    # ------------------------------------------------------------------
    # Auth
    # ------------------------------------------------------------------
    def _headers(self, body_str: str = "") -> dict:
        ts = str(int(time.time() * 1000))
        raw = self.api_key + ts + body_str
        sig = hmac.new(
            self.api_secret.encode(),
            raw.encode(),
            hashlib.sha256
        ).hexdigest()
        return {
            "ApiKey":       self.api_key,
            "Request-Time": ts,
            "Signature":    sig,
            "Content-Type": "application/json",
        }

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------
    def get_klines(self, limit: int = 200) -> list[float]:
        tf_secs = TIMEFRAME_SECONDS.get(self.timeframe, 900)
        end     = int(time.time())
        start   = end - limit * tf_secs
        url     = f"{self.base_url}/api/v1/contract/kline/{self.symbol}"
        params  = {"interval": self.timeframe, "start": start, "end": end}
        try:
            r    = requests.get(url, params=params, timeout=10)
            data = r.json()
            if data.get("success") and data.get("data"):
                closes = [float(c) for c in data["data"].get("close", [])]
                return closes[-limit:]
        except Exception as e:
            logger.error(f"get_klines error: {e}")
        return []

    def get_ticker(self) -> float:
        url    = f"{self.base_url}/api/v1/contract/ticker"
        params = {"symbol": self.symbol}
        try:
            r    = requests.get(url, params=params, timeout=10)
            data = r.json()
            if data.get("success"):
                return float(data["data"].get("lastPrice", 0))
        except Exception as e:
            logger.error(f"get_ticker error: {e}")
        return 0.0

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------
    def _post(self, path: str, body: dict) -> dict:
        body_str = json.dumps(body, separators=(",", ":"))
        headers  = self._headers(body_str)
        url      = f"{self.base_url}{path}"
        try:
            r = requests.post(url, data=body_str, headers=headers, timeout=10)
            return r.json()
        except Exception as e:
            logger.error(f"POST {path} error: {e}")
            return {"success": False, "message": str(e)}

    def _get_private(self, path: str, params: dict = None) -> dict:
        params    = params or {}
        qs        = "&".join(f"{k}={v}" for k, v in sorted(params.items()))
        headers   = self._headers(qs)
        url       = f"{self.base_url}{path}"
        try:
            r = requests.get(url, params=params, headers=headers, timeout=10)
            return r.json()
        except Exception as e:
            logger.error(f"GET {path} error: {e}")
            return {"success": False, "message": str(e)}

    def set_leverage(self) -> dict:
        return self._post("/api/v1/private/position/change_leverage", {
            "symbol":   self.symbol,
            "leverage": self.leverage,
            "openType": 2,          # 2 = cross margin
        })

    def get_positions(self) -> list:
        data = self._get_private("/api/v1/private/position/open_positions",
                                 {"symbol": self.symbol})
        if data.get("success"):
            return data.get("data", [])
        return []

    def place_order(self, side: int, vol: float = None) -> dict:
        """
        side: 1=open_long  2=close_short  3=open_short  4=close_long
        type: 5 = market
        """
        return self._post("/api/v1/private/order/submit", {
            "symbol":   self.symbol,
            "price":    0,
            "vol":      vol or self.vol,
            "leverage": self.leverage,
            "side":     side,
            "type":     5,
            "openType": 2,
        })

    def close_all(self) -> list:
        results = []
        for pos in self.get_positions():
            hold = float(pos.get("holdVol", 0))
            if hold > 0:
                ptype = pos.get("positionType")   # 1=long 2=short
                side  = 4 if ptype == 1 else 2    # 4=close_long 2=close_short
                results.append(self.place_order(side, hold))
        return results

    def get_account(self) -> dict:
        return self._get_private("/api/v1/private/account/assets")
