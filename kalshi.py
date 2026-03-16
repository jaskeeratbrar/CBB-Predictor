"""
Kalshi API client with RSA key authentication for the CBB trading bot.
"""

import base64
import logging
import time
from pathlib import Path
from typing import Optional

import requests
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, utils

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://api.elections.kalshi.com/trade-api/v2"

# Kalshi series/event prefixes commonly used for college basketball markets.
_CBB_SEARCH_TERMS = [
    "CBB",
    "NCAAM",
    "NCAAB",
    "MARCHMAD",
    "MMAD",
    "CBBGAME",
]


class KalshiClient:
    """Thin Kalshi REST client with RSA-PSS request signing."""

    def __init__(
        self,
        api_key_id: str,
        private_key_path: str,
        base_url: str = None,
    ) -> None:
        self.api_key_id = api_key_id
        self.base_url = (base_url or DEFAULT_BASE_URL).rstrip("/")
        self._private_key = self._load_private_key(private_key_path)
        self._last_request_time: float = 0.0

        self._session = requests.Session()
        self._session.headers.update({
            "User-Agent": "CBB-Predictor-Bot/1.0",
            "Content-Type": "application/json",
            "Accept": "application/json",
        })

    # ------------------------------------------------------------------
    # Authentication helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_private_key(path: str):
        """Load an RSA private key in PEM format from *path*."""
        resolved = Path(path).expanduser().resolve()
        pem_bytes = resolved.read_bytes()
        return serialization.load_pem_private_key(pem_bytes, password=None)

    def _sign_request(self, method: str, path: str, timestamp_ms: int) -> str:
        """Return a base64-encoded RSA-PSS signature for the request.

        The message to sign is: ``str(timestamp_ms) + method + path``
        where *method* is upper-case (``GET``, ``POST``, ...) and *path*
        starts with ``/trade-api/v2/...``.
        """
        message = f"{timestamp_ms}{method}{path}".encode("utf-8")
        signature = self._private_key.sign(
            message,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.MAX_LENGTH,
            ),
            hashes.SHA256(),
        )
        return base64.b64encode(signature).decode("utf-8")

    # ------------------------------------------------------------------
    # Core HTTP transport
    # ------------------------------------------------------------------

    def _request(
        self,
        method: str,
        path: str,
        params: dict = None,
        json_body: dict = None,
    ) -> dict:
        """Make an authenticated request to the Kalshi API.

        Raises ``requests.HTTPError`` on non-2xx responses.
        """
        # Simple rate-limit: wait if less than 100ms since last request.
        elapsed = time.time() - self._last_request_time
        if elapsed < 0.1:
            time.sleep(0.1 - elapsed)

        url = f"{self.base_url}{path}"
        method_upper = method.upper()

        # Build the full path portion used for signing (includes the
        # /trade-api/v2 prefix that is already part of self.base_url).
        # Kalshi expects the path exactly as it appears after the host.
        full_path = f"/trade-api/v2{path}"

        timestamp_ms = int(time.time() * 1000)
        signature = self._sign_request(method_upper, full_path, timestamp_ms)

        headers = {
            "KALSHI-ACCESS-KEY": self.api_key_id,
            "KALSHI-ACCESS-SIGNATURE": signature,
            "KALSHI-ACCESS-TIMESTAMP": str(timestamp_ms),
        }

        logger.debug(
            "API %s %s params=%s body=%s",
            method_upper,
            path,
            params,
            json_body,
        )

        self._last_request_time = time.time()
        response = self._session.request(
            method_upper,
            url,
            headers=headers,
            params=params,
            json=json_body,
        )
        response.raise_for_status()
        return response.json()

    # ------------------------------------------------------------------
    # Market endpoints
    # ------------------------------------------------------------------

    def get_events(
        self,
        series_ticker: str = None,
        limit: int = 200,
        cursor: str = None,
    ) -> Optional[dict]:
        """GET /events with optional filters."""
        try:
            params: dict = {"limit": limit}
            if series_ticker:
                params["series_ticker"] = series_ticker
            if cursor:
                params["cursor"] = cursor
            return self._request("GET", "/events", params=params)
        except Exception:
            logger.exception("Failed to fetch events")
            return None

    def get_markets(
        self,
        event_ticker: str = None,
        status: str = "active",
        limit: int = 200,
        cursor: str = None,
    ) -> Optional[dict]:
        """GET /markets with optional filters."""
        try:
            params: dict = {"limit": limit, "status": status}
            if event_ticker:
                params["event_ticker"] = event_ticker
            if cursor:
                params["cursor"] = cursor
            return self._request("GET", "/markets", params=params)
        except Exception:
            logger.exception("Failed to fetch markets")
            return None

    def get_market(self, ticker: str) -> Optional[dict]:
        """GET /markets/{ticker} for a single market."""
        try:
            return self._request("GET", f"/markets/{ticker}")
        except Exception:
            logger.exception("Failed to fetch market %s", ticker)
            return None

    def find_cbb_markets(self, date: str = None) -> list[dict]:
        """Search for college basketball markets on Kalshi.

        Iterates over common CBB-related series prefixes, collects their
        events and underlying markets, and returns a de-duplicated list of
        market dicts containing: ``ticker``, ``event_ticker``, ``title``,
        ``yes_price``, ``no_price``, and ``volume``.

        If *date* is provided (``YYYY-MM-DD``), only markets whose
        ``close_time`` falls on that date are included.
        """
        found: dict[str, dict] = {}

        for term in _CBB_SEARCH_TERMS:
            try:
                events_resp = self.get_events(series_ticker=term)
                if not events_resp or "events" not in events_resp:
                    continue

                for event in events_resp["events"]:
                    event_ticker = event.get("event_ticker", "")
                    markets_resp = self.get_markets(event_ticker=event_ticker)
                    if not markets_resp or "markets" not in markets_resp:
                        continue

                    for mkt in markets_resp["markets"]:
                        ticker = mkt.get("ticker", "")
                        if ticker in found:
                            continue

                        # Optional date filter on close_time
                        if date and not mkt.get("close_time", "").startswith(date):
                            continue

                        found[ticker] = {
                            "ticker": ticker,
                            "event_ticker": mkt.get("event_ticker", event_ticker),
                            "title": mkt.get("title", ""),
                            "yes_price": mkt.get("yes_ask", 0),
                            "no_price": mkt.get("no_ask", 0),
                            "volume": mkt.get("volume", 0),
                        }
            except Exception:
                logger.exception("Error searching CBB markets with term %s", term)

        results = list(found.values())
        logger.debug("Found %d CBB markets", len(results))
        return results

    # ------------------------------------------------------------------
    # Portfolio endpoints
    # ------------------------------------------------------------------

    def get_balance(self) -> Optional[int]:
        """GET /portfolio/balance -> balance in cents."""
        try:
            resp = self._request("GET", "/portfolio/balance")
            return resp.get("balance", 0)
        except Exception:
            logger.exception("Failed to fetch balance")
            return None

    def get_positions(self, event_ticker: str = None) -> Optional[list[dict]]:
        """GET /portfolio/positions, optionally filtered by event."""
        try:
            params: dict = {}
            if event_ticker:
                params["event_ticker"] = event_ticker
            resp = self._request("GET", "/portfolio/positions", params=params)
            return resp.get("market_positions", [])
        except Exception:
            logger.exception("Failed to fetch positions")
            return None

    def get_fills(
        self,
        ticker: str = None,
        limit: int = 100,
    ) -> Optional[list[dict]]:
        """GET /portfolio/fills."""
        try:
            params: dict = {"limit": limit}
            if ticker:
                params["ticker"] = ticker
            resp = self._request("GET", "/portfolio/fills", params=params)
            return resp.get("fills", [])
        except Exception:
            logger.exception("Failed to fetch fills")
            return None

    # ------------------------------------------------------------------
    # Order endpoints
    # ------------------------------------------------------------------

    def place_order(
        self,
        ticker: str,
        side: str,
        count: int,
        yes_price: int,
        order_type: str = "limit",
    ) -> Optional[dict]:
        """POST /portfolio/orders to place a new order.

        Parameters
        ----------
        ticker : str
            The market ticker (e.g. ``"CBB-DUKE-UNC-24MAR16"``).
        side : str
            ``"yes"`` or ``"no"``.
        count : int
            Number of contracts.
        yes_price : int
            Price in cents (1-99).
        order_type : str
            ``"limit"`` (default) or ``"market"``.

        Returns
        -------
        dict or None
            Order response dict containing ``order_id``, or ``None`` on error.
        """
        try:
            body = {
                "ticker": ticker,
                "action": "buy",
                "side": side,
                "count": count,
                "type": order_type,
                "yes_price": yes_price,
            }
            resp = self._request("POST", "/portfolio/orders", json_body=body)
            logger.info(
                "Placed %s order: %s %d@%dc -> order_id=%s",
                order_type,
                side,
                count,
                yes_price,
                resp.get("order", {}).get("order_id"),
            )
            return resp.get("order")
        except Exception:
            logger.exception(
                "Failed to place order: %s %s %d@%dc",
                ticker,
                side,
                count,
                yes_price,
            )
            return None

    def cancel_order(self, order_id: str) -> Optional[dict]:
        """DELETE /portfolio/orders/{order_id}."""
        try:
            resp = self._request("DELETE", f"/portfolio/orders/{order_id}")
            logger.info("Cancelled order %s", order_id)
            return resp
        except Exception:
            logger.exception("Failed to cancel order %s", order_id)
            return None

    def get_orders(self, status: str = "resting") -> Optional[list[dict]]:
        """GET /portfolio/orders filtered by status."""
        try:
            params: dict = {"status": status}
            resp = self._request("GET", "/portfolio/orders", params=params)
            return resp.get("orders", [])
        except Exception:
            logger.exception("Failed to fetch orders")
            return None

    # ------------------------------------------------------------------
    # Exchange
    # ------------------------------------------------------------------

    def get_exchange_status(self) -> Optional[dict]:
        """GET /exchange/status."""
        try:
            return self._request("GET", "/exchange/status")
        except Exception:
            logger.exception("Failed to fetch exchange status")
            return None
