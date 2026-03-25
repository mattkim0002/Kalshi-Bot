"""
Trade Executor.

Handles the actual placement of orders on Polymarket via the CLOB API.
Supports both dry-run (paper trading) and live execution.
"""
import logging
import time
from typing import Optional
from dataclasses import dataclass

import config

logger = logging.getLogger(__name__)

# Polymarket SDK imports (only needed for live trading)
_clob_available = False
try:
    from py_clob_client.client import ClobClient
    from py_clob_client.clob_types import MarketOrderArgs, OrderType
    from py_clob_client.constants import POLYGON
    _clob_available = True
except ImportError:
    logger.info(
        "py-clob-client not installed. Live trading disabled. "
        "Install with: pip install py-clob-client"
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
    Executes trades on Polymarket.
    
    In DRY_RUN mode: simulates fills at current market price.
    In live mode: places FOK (Fill-or-Kill) market orders via the CLOB API.
    """

    def __init__(self):
        self.dry_run = config.DRY_RUN
        self.client: Optional[object] = None
        
        if not self.dry_run:
            self._init_live_client()

    def _init_live_client(self):
        """Initialize the Polymarket CLOB client for live trading."""
        if not _clob_available:
            logger.error(
                "Cannot enable live trading: py-clob-client not installed. "
                "Run: pip install py-clob-client"
            )
            self.dry_run = True
            return
        
        if not config.POLYMARKET_PRIVATE_KEY:
            logger.error("POLYMARKET_PRIVATE_KEY not set. Falling back to dry run.")
            self.dry_run = True
            return

        try:
            self.client = ClobClient(
                config.CLOB_API_BASE,
                key=config.POLYMARKET_PRIVATE_KEY,
                chain_id=config.CHAIN_ID,
                signature_type=config.SIGNATURE_TYPE,
                funder=config.POLYMARKET_FUNDER_ADDRESS,
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
            token_id: The Polymarket token ID to buy
            amount_usd: Dollar amount to spend
            current_price: Current market price (for simulation / logging)
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
        """
        Execute a sell order (exit a position).
        """
        amount_usd = shares * current_price

        if self.dry_run:
            return self._simulate_sell(token_id, amount_usd, current_price, shares, direction, question)
        else:
            return await self._live_sell(token_id, shares, current_price, direction, question)

    # ── Simulation ───────────────────────────────────

    def _simulate_buy(
        self, token_id: str, amount: float, price: float,
        shares: float, direction: str, question: str,
    ) -> OrderResult:
        """Simulate a buy fill at current price."""
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
        self, token_id: str, amount: float, price: float,
        shares: float, direction: str, question: str,
    ) -> OrderResult:
        """Simulate a sell fill at current price."""
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
        self, token_id: str, amount: float, price: float,
        shares: float, direction: str, question: str,
    ) -> OrderResult:
        """Place a real Fill-or-Kill market buy order."""
        try:
            order_args = MarketOrderArgs(
                token_id=token_id,
                amount=amount,
                side="BUY",
                order_type=OrderType.FOK,
            )
            
            signed_order = self.client.create_market_order(order_args)
            result = self.client.post_order(signed_order, OrderType.FOK)
            
            if result and not result.get("errorMsg"):
                order_id = result.get("orderID", "unknown")
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
            else:
                error = result.get("errorMsg", "Unknown error") if result else "No response"
                logger.error(f"[LIVE] Order failed: {error}")
                return OrderResult(
                    success=False,
                    order_id=None,
                    fill_price=0,
                    shares=0,
                    amount_usd=0,
                    is_simulation=False,
                    error=error,
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
        self, token_id: str, shares: float, price: float,
        direction: str, question: str,
    ) -> OrderResult:
        """Place a real sell order to close a position."""
        try:
            amount = shares * price
            order_args = MarketOrderArgs(
                token_id=token_id,
                amount=amount,
                side="SELL",
                order_type=OrderType.FOK,
            )
            
            signed_order = self.client.create_market_order(order_args)
            result = self.client.post_order(signed_order, OrderType.FOK)
            
            if result and not result.get("errorMsg"):
                order_id = result.get("orderID", "unknown")
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
            else:
                error = result.get("errorMsg", "Unknown error") if result else "No response"
                logger.error(f"[LIVE] Sell failed: {error}")
                return OrderResult(
                    success=False,
                    order_id=None,
                    fill_price=0,
                    shares=0,
                    amount_usd=0,
                    is_simulation=False,
                    error=error,
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
