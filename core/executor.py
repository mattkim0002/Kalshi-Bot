"""
Trade Executor for Kalshi.com.

Handles order placement via the Kalshi REST API.
Supports both dry-run (paper trading) and live execution.
"""
import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import aiohttp

import config

logger = logging.getLogger(__name__)


@dataclass
class OrderResult:
    """Result of an order execution."""
    success: bool
    order_id: Optional[str]
    fill_price: float
    shares: float
    amount_usd: float
    is_simulation: bool
    error: Optional[str] = None
    timestamp: float = 0.0

    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = time.time()


class TradeExecutor:
    """
    Executes trades on Kalshi.com via their REST API.

    In DRY_RUN mode: simulates fills at current market price.
    In live mode: places limit orders near the ask price.

    token_id = market ticker (same for YES and NO, side determined by direction).
    """

    def __init__(self):
        self.dry_run = config.DRY_RUN
        self._session: Optional[aiohttp.ClientSession] = None
        self._headers: dict = {}
        self._private_key = None  # set by _init_auth if credentials are valid

        if not self.dry_run:
            self._init_auth()

    def _init_auth(self):
        """Set up RSA-PSS auth headers for Kalshi API."""
        if not config.KALSHI_API_KEY_ID:
            logger.error("KALSHI_API_KEY_ID not set. Falling back to dry run.")
            self.dry_run = True
            return
        try:
            import base64
            from cryptography.hazmat.primitives import hashes, serialization
            from cryptography.hazmat.primitives.asymmetric import padding

            key_path = Path(config.KALSHI_PRIVATE_KEY_PATH)
            if not key_path.exists():
                logger.error(f"Private key not found at {key_path}. Falling back to dry run.")
                self.dry_run = True
                return

            self._private_key_pem = key_path.read_bytes()
            self._private_key = serialization.load_pem_private_key(
                self._private_key_pem, password=None
            )
            logger.info("Kalshi live trading client initialized successfully")
        except ImportError:
            logger.error("cryptography package not installed. Run: pip install cryptography")
            self.dry_run = True
        except Exception as e:
            logger.error(f"Failed to load Kalshi private key: {e}. Falling back to dry run.")
            self.dry_run = True

    def _sign_request(self, method: str, path: str) -> dict:
        """Generate RSA-PSS signed headers for Kalshi API."""
        from cryptography.hazmat.primitives import hashes
        from cryptography.hazmat.primitives.asymmetric import padding
        import base64

        timestamp_ms = str(int(time.time() * 1000))
        # Strip query params from path for signing
        sign_path = path.split("?")[0]
        message = (timestamp_ms + method.upper() + sign_path).encode()

        signature = self._private_key.sign(
            message,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.DIGEST_LENGTH,
            ),
            hashes.SHA256(),
        )

        return {
            "KALSHI-ACCESS-KEY": config.KALSHI_API_KEY_ID,
            "KALSHI-ACCESS-TIMESTAMP": timestamp_ms,
            "KALSHI-ACCESS-SIGNATURE": base64.b64encode(signature).decode(),
            "Content-Type": "application/json",
        }

    async def _ensure_session(self):
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()

    async def execute_buy(
        self,
        token_id: str,
        amount_usd: float,
        current_price: float,
        direction: str,
        question: str = "",
    ) -> OrderResult:
        """
        Execute a buy order.

        Args:
            token_id: Kalshi market ticker
            amount_usd: Dollar amount to spend
            current_price: Current market price 0-1 (for simulation / sizing)
            direction: "YES" or "NO"
            question: Market question (for logging)
        """
        # Convert dollars to number of contracts
        # Each contract costs current_price dollars and pays $1 if correct
        price_cents = int(round(current_price * 100))
        contracts = max(1, int(amount_usd / current_price))
        shares = float(contracts)

        if self.dry_run:
            return self._simulate_buy(token_id, amount_usd, current_price, shares, direction, question)
        else:
            return await self._live_buy(token_id, amount_usd, current_price, contracts, price_cents, direction, question)

    async def execute_sell(
        self,
        token_id: str,
        shares: float,
        current_price: float,
        direction: str,
        question: str = "",
    ) -> OrderResult:
        """Execute a sell order (exit a position)."""
        amount_usd = shares * current_price

        if self.dry_run:
            return self._simulate_sell(token_id, amount_usd, current_price, shares, direction, question)
        else:
            price_cents = int(round(current_price * 100))
            contracts = max(1, int(shares))
            return await self._live_sell(token_id, amount_usd, current_price, contracts, price_cents, direction, question)

    # ── Simulation ───────────────────────────────────

    def _simulate_buy(self, token_id, amount, price, shares, direction, question) -> OrderResult:
        logger.info(
            f"[DRY RUN] BUY {direction} | {question[:50]}... | "
            f"{shares:.0f} contracts @ ${price:.2f} = ${amount:.2f}"
        )
        return OrderResult(
            success=True,
            order_id=f"sim_{int(time.time())}",
            fill_price=price,
            shares=shares,
            amount_usd=amount,
            is_simulation=True,
        )

    def _simulate_sell(self, token_id, amount, price, shares, direction, question) -> OrderResult:
        logger.info(
            f"[DRY RUN] SELL {direction} | {question[:50]}... | "
            f"{shares:.0f} contracts @ ${price:.2f} = ${amount:.2f}"
        )
        return OrderResult(
            success=True,
            order_id=f"sim_{int(time.time())}",
            fill_price=price,
            shares=shares,
            amount_usd=amount,
            is_simulation=True,
        )

    # ── Live Execution ───────────────────────────────

    async def _live_buy(self, token_id, amount, price, contracts, price_cents, direction, question) -> OrderResult:
        """Place a limit buy order on Kalshi."""
        await self._ensure_session()
        try:
            path = "/trade-api/v2/portfolio/orders"
            url = f"https://api.elections.kalshi.com{path}"
            headers = self._sign_request("POST", path)

            payload = {
                "ticker": token_id,
                "side": direction.lower(),  # "yes" or "no"
                "action": "buy",
                "count": contracts,
                "type": "limit",
                f"{direction.lower()}_price": price_cents,
                "client_order_id": str(uuid.uuid4()),
            }

            async with self._session.post(
                url, json=payload, headers=headers,
                timeout=aiohttp.ClientTimeout(total=30),
            ) as resp:
                data = await resp.json()
                if resp.status not in (200, 201):
                    error = data.get("error", str(data))
                    logger.error(f"[LIVE] Buy error {resp.status}: {error}")
                    self._log_failed_trade("BUY", token_id, amount, direction, question, error)
                    return OrderResult(success=False, order_id=None, fill_price=0, shares=0, amount_usd=0, is_simulation=False, error=error)

                order = data.get("order", {})
                order_id = order.get("order_id", "unknown")
                logger.info(f"[LIVE] BUY {direction} | {question[:50]}... | {contracts} contracts @ ${price:.2f} | Order: {order_id}")
                return OrderResult(success=True, order_id=order_id, fill_price=price, shares=float(contracts), amount_usd=amount, is_simulation=False)

        except Exception as e:
            logger.error(f"[LIVE] Buy execution error: {e}")
            self._log_failed_trade("BUY", token_id, amount, direction, question, str(e))
            return OrderResult(success=False, order_id=None, fill_price=0, shares=0, amount_usd=0, is_simulation=False, error=str(e))

    async def _live_sell(self, token_id, amount, price, contracts, price_cents, direction, question) -> OrderResult:
        """Sell/close a position on Kalshi."""
        await self._ensure_session()
        try:
            path = "/trade-api/v2/portfolio/orders"
            url = f"https://api.elections.kalshi.com{path}"
            headers = self._sign_request("POST", path)

            payload = {
                "ticker": token_id,
                "side": direction.lower(),
                "action": "sell",
                "count": contracts,
                "type": "limit",
                f"{direction.lower()}_price": price_cents,
                "client_order_id": str(uuid.uuid4()),
                "reduce_only": True,
            }

            async with self._session.post(
                url, json=payload, headers=headers,
                timeout=aiohttp.ClientTimeout(total=30),
            ) as resp:
                data = await resp.json()
                if resp.status not in (200, 201):
                    error = data.get("error", str(data))
                    logger.error(f"[LIVE] Sell error {resp.status}: {error}")
                    self._log_failed_trade("SELL", token_id, amount, direction, question, error)
                    return OrderResult(success=False, order_id=None, fill_price=0, shares=0, amount_usd=0, is_simulation=False, error=error)

                order = data.get("order", {})
                order_id = order.get("order_id", "unknown")
                logger.info(f"[LIVE] SELL {direction} | {question[:50]}... | {contracts} contracts @ ${price:.2f} | Order: {order_id}")
                return OrderResult(success=True, order_id=order_id, fill_price=price, shares=float(contracts), amount_usd=amount, is_simulation=False)

        except Exception as e:
            logger.error(f"[LIVE] Sell execution error: {e}")
            self._log_failed_trade("SELL", token_id, amount, direction, question, str(e))
            return OrderResult(success=False, order_id=None, fill_price=0, shares=0, amount_usd=0, is_simulation=False, error=str(e))

    async def get_balance(self) -> Optional[float]:
        """Fetch account balance from Kalshi in dollars."""
        if self.dry_run or self._private_key is None:
            return None
        await self._ensure_session()
        try:
            path = "/trade-api/v2/portfolio/balance"
            url = f"https://api.elections.kalshi.com{path}"
            headers = self._sign_request("GET", path)

            async with self._session.get(url, headers=headers, timeout=aiohttp.ClientTimeout(total=15)) as resp:
                if resp.status != 200:
                    return None
                data = await resp.json()
                # Balance is in cents
                balance_cents = data.get("balance", 0)
                return float(balance_cents) / 100
        except Exception as e:
            logger.warning(f"Could not fetch Kalshi balance: {e}")
            return None

    def _log_failed_trade(self, action, ticker, amount, direction, question, error):
        """Append a failed trade to failed_trades.json."""
        entry = {
            "timestamp": time.time(),
            "action": action,
            "ticker": ticker,
            "amount_usd": amount,
            "direction": direction,
            "question": question,
            "error": error,
        }
        path = Path("failed_trades.json")
        try:
            existing = json.loads(path.read_text()) if path.exists() else []
            existing.append(entry)
            path.write_text(json.dumps(existing[-50:], indent=2))
        except Exception:
            pass
