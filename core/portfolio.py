"""
Portfolio Manager.

Tracks open positions, P&L, exposure, and enforces risk limits.
Persists state to disk for crash recovery.
Syncs bankroll from live Polymarket US account balance.
"""
import asyncio
import json
import time
import logging
from dataclasses import dataclass, field, asdict
from typing import Optional
from pathlib import Path

import config

logger = logging.getLogger(__name__)


@dataclass
class Position:
    """An open position in a Polymarket market."""
    market_id: str
    question: str
    direction: str          # "YES" or "NO"
    entry_price: float
    current_price: float
    shares: float
    cost_basis: float       # Total USD paid
    current_value: float
    unrealized_pnl: float
    entry_time: float       # Unix timestamp
    estimated_prob: float
    confidence: str
    token_id: str
    url: str = ""

    @property
    def pnl_pct(self) -> float:
        if self.cost_basis == 0:
            return 0.0
        return self.unrealized_pnl / self.cost_basis

    @property
    def hold_hours(self) -> float:
        return (time.time() - self.entry_time) / 3600


@dataclass
class TradeRecord:
    """Record of a completed trade."""
    market_id: str
    question: str
    direction: str
    action: str             # "BUY" or "SELL"
    price: float
    shares: float
    amount_usd: float
    timestamp: float
    reason: str
    estimated_prob: float
    market_price_at_trade: float


class PortfolioManager:
    """
    Manages all open positions, enforces risk limits,
    tracks P&L, and persists state.
    """

    def __init__(self):
        self.positions: dict[str, Position] = {}  # market_id -> Position
        self.trade_history: list[TradeRecord] = []
        self.daily_pnl: float = 0.0
        self.daily_pnl_reset_time: float = time.time()
        self.bankroll: float = config.BANKROLL  # updated from live API
        self._load_state()

    async def sync_balance(self, client) -> float:
        """
        Fetch live buying power from Polymarket US and update bankroll.
        Falls back to config.BANKROLL if unavailable (dry run / error).
        """
        if config.DRY_RUN or client is None:
            return self.bankroll
        try:
            data = await asyncio.to_thread(client.account.balances)
            if isinstance(data, dict) and "balances" in data:
                entry = data["balances"][0] if data["balances"] else {}
                balance = entry.get("buyingPower")
            elif isinstance(data, dict):
                balance = data.get("buyingPower")
            else:
                balance = getattr(data, "buyingPower", None)
            if balance is not None:
                self.bankroll = float(balance)
                logger.info(f"Account balance synced: ${self.bankroll:.2f}")
            else:
                logger.warning(f"Could not find balance in response: {data}")
        except Exception as e:
            logger.warning(f"Could not sync balance from API: {e}. Using ${self.bankroll:.2f}")
        return self.bankroll

    # ── Position Management ──────────────────────────

    def open_position(
        self,
        market_id: str,
        question: str,
        direction: str,
        price: float,
        shares: float,
        estimated_prob: float,
        confidence: str,
        token_id: str,
        url: str = "",
    ) -> Position:
        """Record a new position."""
        cost = price * shares
        
        pos = Position(
            market_id=market_id,
            question=question,
            direction=direction,
            entry_price=price,
            current_price=price,
            shares=shares,
            cost_basis=cost,
            current_value=cost,
            unrealized_pnl=0.0,
            entry_time=time.time(),
            estimated_prob=estimated_prob,
            confidence=confidence,
            token_id=token_id,
            url=url,
        )
        
        self.positions[market_id] = pos
        
        self.trade_history.append(TradeRecord(
            market_id=market_id,
            question=question,
            direction=direction,
            action="BUY",
            price=price,
            shares=shares,
            amount_usd=cost,
            timestamp=time.time(),
            reason="New position opened",
            estimated_prob=estimated_prob,
            market_price_at_trade=price,
        ))
        
        self._save_state()
        logger.info(
            f"OPENED {direction} {market_id[:16]}... | "
            f"{shares:.1f} shares @ ${price:.4f} = ${cost:.2f}"
        )
        return pos

    def close_position(self, market_id: str, exit_price: float, reason: str) -> Optional[TradeRecord]:
        """Close a position and record the trade."""
        pos = self.positions.get(market_id)
        if not pos:
            logger.warning(f"No position found for {market_id}")
            return None

        exit_value = exit_price * pos.shares
        realized_pnl = exit_value - pos.cost_basis
        
        trade = TradeRecord(
            market_id=market_id,
            question=pos.question,
            direction=pos.direction,
            action="SELL",
            price=exit_price,
            shares=pos.shares,
            amount_usd=exit_value,
            timestamp=time.time(),
            reason=reason,
            estimated_prob=pos.estimated_prob,
            market_price_at_trade=exit_price,
        )
        
        self.trade_history.append(trade)
        self.daily_pnl += realized_pnl
        
        del self.positions[market_id]
        self._save_state()
        
        logger.info(
            f"CLOSED {pos.direction} {market_id[:16]}... | "
            f"PnL: ${realized_pnl:+.2f} ({realized_pnl/pos.cost_basis:+.1%}) | "
            f"Reason: {reason}"
        )
        return trade

    def update_price(self, market_id: str, new_price: float):
        """Update the current price and unrealized P&L for a position."""
        pos = self.positions.get(market_id)
        if not pos:
            return
        
        pos.current_price = new_price
        pos.current_value = new_price * pos.shares
        pos.unrealized_pnl = pos.current_value - pos.cost_basis

    # ── Risk Checks ──────────────────────────────────

    def can_open_position(self, amount_usd: float) -> tuple[bool, str]:
        """Check if opening a new position is allowed by risk limits."""
        # Check position count
        if len(self.positions) >= config.MAX_POSITIONS:
            return False, f"Max positions ({config.MAX_POSITIONS}) reached"

        # Check daily loss limit
        self._maybe_reset_daily_pnl()
        if self.daily_pnl <= -config.MAX_DAILY_LOSS:
            return False, f"Daily loss limit (${config.MAX_DAILY_LOSS}) reached"

        # Check total exposure
        current_exposure = self.total_exposure
        new_exposure = current_exposure + amount_usd
        max_exposure_usd = self.bankroll * config.MAX_TOTAL_EXPOSURE
        
        if new_exposure > max_exposure_usd:
            return False, (
                f"Total exposure ${new_exposure:.2f} would exceed "
                f"limit ${max_exposure_usd:.2f}"
            )

        return True, "OK"

    @property
    def total_exposure(self) -> float:
        """Total USD currently deployed across all positions."""
        return sum(p.cost_basis for p in self.positions.values())

    @property
    def total_unrealized_pnl(self) -> float:
        """Total unrealized P&L across all positions."""
        return sum(p.unrealized_pnl for p in self.positions.values())

    @property
    def total_value(self) -> float:
        """Total current value of all positions."""
        return sum(p.current_value for p in self.positions.values())

    @property
    def available_capital(self) -> float:
        """How much capital is available to deploy."""
        return self.bankroll - self.total_exposure

    def _maybe_reset_daily_pnl(self):
        """Reset daily P&L counter if a new day has started."""
        if time.time() - self.daily_pnl_reset_time > 86400:
            self.daily_pnl = 0.0
            self.daily_pnl_reset_time = time.time()

    # ── Reporting ────────────────────────────────────

    def summary(self) -> dict:
        """Generate a portfolio summary."""
        return {
            "open_positions": len(self.positions),
            "total_exposure": round(self.total_exposure, 2),
            "available_capital": round(self.available_capital, 2),
            "total_unrealized_pnl": round(self.total_unrealized_pnl, 2),
            "daily_realized_pnl": round(self.daily_pnl, 2),
            "total_trades": len(self.trade_history),
            "positions": {
                mid: {
                    "question": p.question[:80],
                    "direction": p.direction,
                    "entry": p.entry_price,
                    "current": p.current_price,
                    "pnl": round(p.unrealized_pnl, 2),
                    "pnl_pct": f"{p.pnl_pct:+.1%}",
                    "hold_hours": round(p.hold_hours, 1),
                }
                for mid, p in self.positions.items()
            },
        }

    def print_summary(self):
        """Print a formatted portfolio summary to the console."""
        s = self.summary()
        print("\n" + "=" * 70)
        print("PORTFOLIO SUMMARY")
        print("=" * 70)
        print(f"  Open positions:      {s['open_positions']}")
        print(f"  Total exposure:      ${s['total_exposure']:,.2f}")
        print(f"  Available capital:   ${s['available_capital']:,.2f}")
        print(f"  Unrealized P&L:      ${s['total_unrealized_pnl']:+,.2f}")
        print(f"  Daily realized P&L:  ${s['daily_realized_pnl']:+,.2f}")
        print(f"  Total trades:        {s['total_trades']}")
        
        if self.positions:
            print("\n  POSITIONS:")
            print(f"  {'Question':<50} {'Dir':>4} {'Entry':>6} {'Now':>6} {'P&L':>8}")
            print("  " + "-" * 78)
            for mid, info in s["positions"].items():
                q = info["question"][:48]
                print(
                    f"  {q:<50} {info['direction']:>4} "
                    f"{info['entry']:>6.2f} {info['current']:>6.2f} "
                    f"${info['pnl']:>+7.2f}"
                )
        print("=" * 70 + "\n")

    # ── Persistence ──────────────────────────────────

    def _save_state(self):
        """Save positions and trade history to disk."""
        try:
            # Positions
            pos_data = {mid: asdict(p) for mid, p in self.positions.items()}
            Path(config.POSITIONS_FILE).write_text(
                json.dumps(pos_data, indent=2)
            )
            
            # Trade history
            trades_data = [asdict(t) for t in self.trade_history[-1000:]]  # Keep last 1000
            Path(config.TRADES_FILE).write_text(
                json.dumps(trades_data, indent=2)
            )
        except Exception as e:
            logger.error(f"Error saving state: {e}")

    def _load_state(self):
        """Load positions and trade history from disk."""
        try:
            pos_path = Path(config.POSITIONS_FILE)
            if pos_path.exists():
                data = json.loads(pos_path.read_text())
                for mid, pdata in data.items():
                    self.positions[mid] = Position(**pdata)
                logger.info(f"Loaded {len(self.positions)} positions from disk")
            
            trades_path = Path(config.TRADES_FILE)
            if trades_path.exists():
                data = json.loads(trades_path.read_text())
                self.trade_history = [TradeRecord(**t) for t in data]
                logger.info(f"Loaded {len(self.trade_history)} trade records from disk")
                
        except Exception as e:
            logger.warning(f"Could not load saved state: {e}")
