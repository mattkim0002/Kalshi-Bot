"""
Trade Executor for Polymarket.com.

Handles order placement via the py-clob-client SDK.
Supports both dry-run (paper trading) and live execution.
"""
import asyncio
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import config

logger = logging.getLogger(__name__)

_sdk_available = False
try:
    from py_clob_client.client import ClobClient
    from py_clob_client.clob_types import MarketOrderArgs, OrderType
    from py_clob_client.order_builder.constants import BUY, SELL
    _sdk_available = True
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
    Executes trades on Polymarket.com using py-clob-client.

    In DRY_RUN mode: simulates fills at current market price.
    In live mode: places FOK market orders via the CLOB API.

    token_id parameters are the YES/NO token IDs from the CLOB.
    direction ("YES"/"NO") determines which token to buy/sell.
    """

    def __init__(self):
        self.dry_run = config.DRY_RUN
        self.client: Optional[object] = None

        if not self.dry_run:
            self._init_live_client()

    def _init_live_client(self):
        """Initialize the Polymarket CLOB client for live trading."""
        if not _sdk_available:
            logger.error(
                "Cannot enable live trading: py-clob-client not installed. "
                "Run: pip install py-clob-client"
            )
            self.dry_run = True
            return

        if not config.POLYMARKET_PRIVATE_KEY:
            logger.error(
                "POLYMARKET_PRIVATE_KEY not set. Falling back to dry run."
            )
            self.dry_run = True
            return

        try:
            self.client = ClobClient(
                config.CLOB_HOST,
                key=config.POLYMARKET_PRIVATE_KEY,
                chain_id=config.CHAIN_ID,
                signature_type=config.POLYMARKET_SIGNATURE_TYPE,
                funder=config.POLYMARKET_FUNDER_ADDRESS or None,
            )
            # Derive and set API credentials
            api_creds = self.client.create_or_derive_api_creds()
            self.client.set_api_creds(api_creds)
            logger.info("Live trading client initialized successfully (Polymarket.com)")
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
            token_id: YES token_id for YES trades, NO token_id for NO trades
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
        """Place a market buy order on Polymarket.com CLOB."""
        try:
            order_args = MarketOrderArgs(
                token_id=token_id,
                amount=amount,
                side=BUY,
            )
            signed_order = await asyncio.to_thread(
                self.client.create_market_order, order_args
            )
            result = await asyncio.to_thread(
                self.client.post_order, signed_order, OrderType.FOK
            )

            order_id = result.get("orderID", result.get("id", "unknown")) if isinstance(result, dict) else "unknown"
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
            self._log_failed_trade("BUY", token_id, amount, direction, question, str(e))
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
        """Close a position on Polymarket.com CLOB."""
        try:
            amount = shares * price
            order_args = MarketOrderArgs(
                token_id=token_id,
                amount=amount,
                side=SELL,
            )
            signed_order = await asyncio.to_thread(
                self.client.create_market_order, order_args
            )
            result = await asyncio.to_thread(
                self.client.post_order, signed_order, OrderType.FOK
            )

            order_id = result.get("orderID", result.get("id", "unknown")) if isinstance(result, dict) else "unknown"
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
            self._log_failed_trade("SELL", token_id, 0, direction, question, str(e))
            return OrderResult(
                success=False,
                order_id=None,
                fill_price=0,
                shares=0,
                amount_usd=0,
                is_simulation=False,
                error=str(e),
            )

    def _log_failed_trade(self, action, token_id, amount, direction, question, error):
        """Append a failed trade attempt to failed_trades.json."""
        entry = {
            "timestamp": time.time(),
            "action": action,
            "token_id": token_id,
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
