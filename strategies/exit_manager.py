"""
Exit Strategy Manager.

Determines when to close positions based on:
- Take-profit: edge has collapsed (market moved to your estimate)
- Stop-loss: position has lost too much
- Time-based: position has been held too long without resolution
- Re-evaluation: Claude re-estimates probability and edge has flipped
"""
import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Optional

import config
from core.portfolio import PortfolioManager, Position
from core.estimator import ProbabilityEstimator
from core.scanner import Market

logger = logging.getLogger(__name__)


@dataclass
class ExitSignal:
    """A signal to exit a position."""
    market_id: str
    reason: str
    exit_type: str          # "take_profit", "stop_loss", "time_exit", "edge_flip"
    urgency: str            # "immediate", "normal"
    current_price: float
    estimated_new_prob: Optional[float] = None


class ExitManager:
    """
    Monitors open positions and generates exit signals.
    
    This runs as a background loop, periodically checking all
    positions against exit criteria.
    """

    def __init__(
        self,
        portfolio: PortfolioManager,
        estimator: ProbabilityEstimator,
    ):
        self.portfolio = portfolio
        self.estimator = estimator

    async def check_all_positions(self) -> list[ExitSignal]:
        """
        Check all open positions for exit conditions.
        Returns a list of exit signals.
        """
        signals = []
        
        for market_id, position in list(self.portfolio.positions.items()):
            signal = await self.check_position(position)
            if signal:
                signals.append(signal)
        
        if signals:
            logger.info(f"Generated {len(signals)} exit signals")
        
        return signals

    async def check_position(self, pos: Position) -> Optional[ExitSignal]:
        """
        Check a single position against all exit criteria.
        Returns an ExitSignal if the position should be closed, None otherwise.
        
        Priority order:
        1. Stop-loss (immediate)
        2. Take-profit (normal)
        3. Time-based re-evaluation (normal)
        """
        # 1. STOP-LOSS: Position has lost too much
        if pos.cost_basis > 0:
            loss_pct = -pos.unrealized_pnl / pos.cost_basis
            if loss_pct >= config.STOP_LOSS_PCT:
                return ExitSignal(
                    market_id=pos.market_id,
                    reason=(
                        f"Stop-loss triggered: position down {loss_pct:.0%} "
                        f"(limit: {config.STOP_LOSS_PCT:.0%})"
                    ),
                    exit_type="stop_loss",
                    urgency="immediate",
                    current_price=pos.current_price,
                )

        # 2. TAKE-PROFIT: Edge has collapsed (price moved to our estimate)
        remaining_edge = abs(pos.estimated_prob - pos.current_price)
        if remaining_edge < config.TAKE_PROFIT_EDGE:
            return ExitSignal(
                market_id=pos.market_id,
                reason=(
                    f"Take-profit: remaining edge {remaining_edge:.2%} "
                    f"below threshold {config.TAKE_PROFIT_EDGE:.2%}. "
                    f"Market has priced in your view."
                ),
                exit_type="take_profit",
                urgency="normal",
                current_price=pos.current_price,
            )

        # 3. TIME-BASED: Held too long — re-evaluate with fresh Claude analysis
        if pos.hold_hours >= config.TIME_EXIT_HOURS:
            return await self._reevaluate_position(pos)

        return None

    async def _reevaluate_position(self, pos: Position) -> Optional[ExitSignal]:
        """
        Re-evaluate a position by asking Claude for an updated probability.
        If the edge has flipped or disappeared, signal an exit.
        """
        logger.info(f"Re-evaluating position: {pos.question[:60]}...")
        
        # Create a market object for the estimator
        market = Market(
            condition_id=pos.market_id,
            question=pos.question,
            slug="",
            yes_price=pos.current_price,
            no_price=1 - pos.current_price,
            yes_token_id=pos.token_id,
            no_token_id="",
            volume=0,
            liquidity=0,
        )
        
        try:
            updated = await self.estimator.estimate(market)
            
            if updated.estimated_prob is None:
                logger.warning("Re-evaluation failed, keeping position")
                return None
            
            new_prob = updated.estimated_prob
            new_edge = new_prob - pos.current_price
            
            # Check if edge has flipped
            original_direction_positive = (
                (pos.direction == "YES" and pos.estimated_prob > pos.entry_price) or
                (pos.direction == "NO" and pos.estimated_prob < pos.entry_price)
            )
            
            new_direction_positive = (
                (pos.direction == "YES" and new_prob > pos.current_price) or
                (pos.direction == "NO" and new_prob < pos.current_price)
            )
            
            if not new_direction_positive:
                return ExitSignal(
                    market_id=pos.market_id,
                    reason=(
                        f"Edge flipped on re-evaluation. "
                        f"Old estimate: {pos.estimated_prob:.2f}, "
                        f"new estimate: {new_prob:.2f}, "
                        f"current price: {pos.current_price:.2f}"
                    ),
                    exit_type="edge_flip",
                    urgency="normal",
                    current_price=pos.current_price,
                    estimated_new_prob=new_prob,
                )
            
            # Edge still exists but may have shrunk
            remaining_edge = abs(new_edge)
            if remaining_edge < config.TAKE_PROFIT_EDGE:
                return ExitSignal(
                    market_id=pos.market_id,
                    reason=(
                        f"Edge diminished on re-evaluation. "
                        f"New estimate: {new_prob:.2f}, "
                        f"remaining edge: {remaining_edge:.2%}"
                    ),
                    exit_type="take_profit",
                    urgency="normal",
                    current_price=pos.current_price,
                    estimated_new_prob=new_prob,
                )
            
            # Edge still valid — update estimate and keep holding
            pos.estimated_prob = new_prob
            pos.entry_time = time.time()  # Reset hold timer
            logger.info(
                f"Re-evaluation confirms edge. Updated estimate: {new_prob:.2f}, "
                f"edge: {remaining_edge:.2%}. Continuing to hold."
            )
            return None
            
        except Exception as e:
            logger.error(f"Error during re-evaluation: {e}")
            return None
