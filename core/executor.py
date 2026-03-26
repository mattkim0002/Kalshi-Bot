"""
Trade Executor for Polymarket US.

Handles order placement via the polymarket-us SDK.
Supports both dry-run (paper trading) and live execution.
"""
import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Optional

import config

logger = logging.getLogger(__name__)

_sdk_available = False
try:
    from polymarket_us import PolymarketUS
    _sdk_available = True
except ImportError:
    logger.info(
        "polymarket-us not installed. Live trading disabled. "
        "Install with: pip install polymarket-us"
    )


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
    Executes trades on Polymarket US.

    In DRY_RUN mode: simulates fills at current market price.
    In live mode: places FOK market orders via the polymarket-us SDK.

    token_id parameters are market slugs on Polymarket US.
    direction ("YES"/"NO") maps to intents (BUY_LONG/BUY_SHORT, SELL_LONG/SELL_SHORT).
    """

    def __init__(self):
        self.dry_run = config.DRY_RUN
        self.client: Optional[object] = None

        if not self.dry_run:
            self._init_live_client()

    def _init_live_client(self):
        """Initialize the Polymarket US client for live trading."""
        if not _sdk_available:
            logger.error(
                "Cannot enable live trading: polymarket-us not installed. "
                "Run: pip install polymarket-us"
            )
            self.dry_run = True
            return

        if not config.POLYMARKET_KEY_ID or not config.POLYMARKET_SECRET_KEY:
            logger.error(
                "POLYMARKET_KEY_ID or POLYMARKET_SECRET_KEY not set. "
                "Falling back to dry run."
            )
            self.dry_run = True
            return

        try:
            self.client = PolymarketUS(
                key_id=config.POLYMARKET_KEY_ID,
                secret_key=config.POLYMARKET_SECRET_KEY,
            )
            logger.info("Live trading client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize live client: {e}. Falling back to dry run.")
            self.dry_run = True

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
            token_id: Market slug on Polymarket US
            amount_usd: Dollar amount to spend
            current_price: Current market price (for simulation / share calculation)
            direction: "YES" or "NO"
            question: Market question (for logging)
        """
        shares = amount_usd / current_price if current_price > 0 else 0

        if self.dry_run:
            return self._simulate_buy(token_id, amount_usd, current_price, shares, direction, question)
        else:
            return await self._live_buy(token_id, amount_usd, current_price, shares, direction, question)

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
            return await self._live_sell(token_id, shares, current_price, direction, question)

    # ── Simulation ───────────────────────────────────

    def _simulate_buy(
        self, token_id, amount, price, shares, direction, question,
    ) -> OrderResult:
        logger.info(
            f"[DRY RUN] BUY {direction} | {question[:50]}... | "
            f"{shares:.1f} shares @ ${price:.4f} = ${amount:.2f}"
        )
        return OrderResult(
            success=True,
            order_id=f"sim_{int(time.time())}",
            fill_price=price,
            shares=shares,
            amount_usd=amount,
            is_simulation=True,
        )

    def _simulate_sell(
        self, token_id, amount, price, shares, direction, question,
    ) -> OrderResult:
        logger.info(
            f"[DRY RUN] SELL {direction} | {question[:50]}... | "
            f"{shares:.1f} shares @ ${price:.4f} = ${amount:.2f}"
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

    async def _live_buy(
        self, token_id, amount, price, shares, direction, question,
    ) -> OrderResult:
        """Place a market buy order on Polymarket US."""
        try:
            intent = "BUY_LONG" if direction == "YES" else "BUY_SHORT"
            result = await asyncio.to_thread(
                self.client.orders.create,
                {
                    "marketSlug": token_id,
                    "intent": intent,
                    "type": "ORDER_TYPE_MARKET",
                    "cashOrderQty": amount,
                    "tif": "FOK",
                    "synchronousExecution": True,
                },
            )

            order_id = result.get("id", "unknown") if isinstance(result, dict) else "unknown"
            logger.info(
                f"[LIVE] BUY {direction} | {question[:50]}... | "
                f"${amount:.2f} | Order: {order_id}"
            )
            return OrderResult(
                success=True,
                order_id=order_id,
                fill_price=price,
                shares=shares,
                amount_usd=amount,
                is_simulation=False,
            )

        except Exception as e:
            logger.error(f"[LIVE] Buy execution error: {e}")
            return OrderResult(
                success=False,
                order_id=None,
                fill_price=0,
                shares=0,
                amount_usd=0,
                is_simulation=False,
                error=str(e),
            )

    async def _live_sell(
        self, token_id, shares, price, direction, question,
    ) -> OrderResult:
        """Close a position on Polymarket US."""
        try:
            intent = "SELL_LONG" if direction == "YES" else "SELL_SHORT"
            result = await asyncio.to_thread(
                self.client.orders.close_position,
                {
                    "marketSlug": token_id,
                    "intent": intent,
                    "synchronousExecution": True,
                },
            )

            amount = shares * price
            order_id = result.get("id", "unknown") if isinstance(result, dict) else "unknown"
            logger.info(
                f"[LIVE] SELL {direction} | {question[:50]}... | "
                f"${amount:.2f} | Order: {order_id}"
            )
            return OrderResult(
                success=True,
                order_id=order_id,
                fill_price=price,
                shares=shares,
                amount_usd=amount,
                is_simulation=False,
            )

        except Exception as e:
            logger.error(f"[LIVE] Sell execution error: {e}")
            return OrderResult(
                success=False,
                order_id=None,
                fill_price=0,
                shares=0,
                amount_usd=0,
                is_simulation=False,
                error=str(e),
            )
