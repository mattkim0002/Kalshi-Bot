"""
LMSR (Logarithmic Market Scoring Rule) Engine.

Implements Robin Hanson's cost function for prediction market pricing,
Kelly criterion position sizing, and price impact modeling.

Reference: C(q) = b * ln(Σ e^(qi/b))
"""
import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class TradeAnalysis:
    """Result of analyzing a potential trade."""
    market_price: float
    estimated_prob: float
    edge: float
    direction: str           # "YES" or "NO"
    kelly_fraction: float
    position_size_usd: float
    shares: float
    avg_fill_price: float
    price_after_trade: float
    price_impact: float
    impact_vs_edge: float    # What % of edge is eaten by impact
    ev_per_dollar: float
    should_trade: bool
    skip_reason: Optional[str] = None


class LMSREngine:
    """
    Core pricing and sizing engine based on Hanson's LMSR.
    
    This is the mathematical foundation. It:
    - Computes market-implied probabilities from share quantities
    - Calculates trade costs via the cost function
    - Sizes positions using Kelly criterion with fractional scaling
    - Models price impact before committing capital
    """

    def __init__(self, b: float = 100.0):
        self.b = b

    # ── Core LMSR Functions ──────────────────────────

    def cost(self, q: np.ndarray) -> float:
        """
        LMSR cost function: C(q) = b * ln(Σ e^(qi/b))
        
        This is the running "tab" of the market. The difference in cost
        before and after a trade is what the trader pays.
        """
        q = np.asarray(q, dtype=np.float64)
        # Numerical stability: subtract max to prevent overflow
        q_scaled = q / self.b
        max_q = np.max(q_scaled)
        return self.b * (max_q + np.log(np.sum(np.exp(q_scaled - max_q))))

    def price(self, q: np.ndarray, outcome: int) -> float:
        """
        Instantaneous price (probability) for an outcome.
        
        p_k(q) = e^(qk/b) / Σ e^(qi/b)
        
        This is the softmax function — same math as neural network output layers.
        Prices always sum to 1 and live in (0, 1).
        """
        q = np.asarray(q, dtype=np.float64)
        q_scaled = q / self.b
        max_q = np.max(q_scaled)
        exp_q = np.exp(q_scaled - max_q)
        return float(exp_q[outcome] / np.sum(exp_q))

    def prices(self, q: np.ndarray) -> np.ndarray:
        """All outcome prices (probabilities) at once."""
        q = np.asarray(q, dtype=np.float64)
        q_scaled = q / self.b
        max_q = np.max(q_scaled)
        exp_q = np.exp(q_scaled - max_q)
        return exp_q / np.sum(exp_q)

    def trade_cost(self, q_before: np.ndarray, q_after: np.ndarray) -> float:
        """
        Cost of a trade = C(q_after) - C(q_before).
        
        Positive = trader pays. Negative = trader receives.
        """
        return self.cost(q_after) - self.cost(q_before)

    def max_loss(self, n_outcomes: int) -> float:
        """
        Maximum possible loss for the market maker.
        L_max = b * ln(n)
        """
        return self.b * np.log(n_outcomes)

    # ── Price Impact Simulation ──────────────────────

    def simulate_impact(
        self,
        q: np.ndarray,
        shares: float,
        outcome: int,
        steps: int = 50
    ) -> dict:
        """
        Simulate buying `shares` of `outcome` in small increments.
        Returns the average fill price and final market price.
        
        This models the convexity of the cost function — larger trades
        move prices more, degrading your average fill.
        """
        q = np.asarray(q, dtype=np.float64).copy()
        start_price = self.price(q, outcome)
        total_cost = 0.0
        shares_per_step = shares / steps

        for _ in range(steps):
            q_after = q.copy()
            q_after[outcome] += shares_per_step
            step_cost = self.trade_cost(q, q_after)
            total_cost += step_cost
            q = q_after

        avg_fill = total_cost / shares if shares > 0 else start_price
        final_price = self.price(q, outcome)

        return {
            "start_price": start_price,
            "avg_fill": avg_fill,
            "final_price": final_price,
            "total_cost": total_cost,
            "price_impact": abs(final_price - start_price),
            "slippage": abs(avg_fill - start_price),
        }

    # ── Kelly Criterion ──────────────────────────────

    @staticmethod
    def kelly_fraction(true_prob: float, market_price: float) -> float:
        """
        Kelly criterion for binary prediction market.
        
        f* = (p * b - q) / b
        
        where:
          p = true probability of winning
          q = 1 - p
          b = payout ratio = (1/market_price) - 1
        
        Returns the optimal fraction of bankroll to bet.
        Negative means no edge (don't trade).
        """
        if market_price <= 0 or market_price >= 1:
            return 0.0

        payout_ratio = (1.0 / market_price) - 1.0
        if payout_ratio <= 0:
            return 0.0

        f = (true_prob * payout_ratio - (1 - true_prob)) / payout_ratio
        return max(f, 0.0)  # Don't return negative (no edge)

    @staticmethod
    def kelly_from_edge(true_prob: float, market_price: float) -> dict:
        """
        Full Kelly analysis for both YES and NO sides.
        Returns the better direction and sizing.
        """
        # YES side
        yes_payout = (1.0 / market_price) - 1.0 if market_price > 0 else 0
        yes_kelly = 0.0
        if yes_payout > 0:
            yes_kelly = (true_prob * yes_payout - (1 - true_prob)) / yes_payout
            yes_kelly = max(yes_kelly, 0.0)

        # NO side
        no_price = 1.0 - market_price
        no_payout = (1.0 / no_price) - 1.0 if no_price > 0 else 0
        no_kelly = 0.0
        if no_payout > 0:
            no_kelly = ((1 - true_prob) * no_payout - true_prob) / no_payout
            no_kelly = max(no_kelly, 0.0)

        if yes_kelly >= no_kelly:
            return {
                "direction": "YES",
                "kelly": yes_kelly,
                "trade_price": market_price,
                "edge": true_prob - market_price,
            }
        else:
            return {
                "direction": "NO",
                "kelly": no_kelly,
                "trade_price": no_price,
                "edge": (1 - true_prob) - no_price,
            }

    # ── Full Trade Analysis ──────────────────────────

    def analyze_trade(
        self,
        market_price: float,
        estimated_prob: float,
        bankroll: float,
        kelly_multiplier: float = 0.25,
        min_edge: float = 0.05,
        max_position_pct: float = 0.06,
        impact_threshold: float = 0.50,
    ) -> TradeAnalysis:
        """
        Complete trade analysis combining LMSR pricing, Kelly sizing,
        and price impact modeling.
        
        This is the decision engine: given a market price, your probability
        estimate, and your bankroll, it tells you whether to trade,
        which direction, and exactly how much.
        """
        edge = estimated_prob - market_price
        kelly_info = self.kelly_from_edge(estimated_prob, market_price)

        direction = kelly_info["direction"]
        raw_kelly = kelly_info["kelly"]
        trade_price = kelly_info["trade_price"]
        actual_edge = kelly_info["edge"]

        # Check minimum edge
        if abs(actual_edge) < min_edge:
            return TradeAnalysis(
                market_price=market_price,
                estimated_prob=estimated_prob,
                edge=edge,
                direction=direction,
                kelly_fraction=raw_kelly,
                position_size_usd=0,
                shares=0,
                avg_fill_price=trade_price,
                price_after_trade=market_price,
                price_impact=0,
                impact_vs_edge=0,
                ev_per_dollar=0,
                should_trade=False,
                skip_reason=f"Edge {abs(actual_edge):.2%} below minimum {min_edge:.2%}",
            )

        # Kelly position sizing with fractional scaling and cap
        sized_kelly = raw_kelly * kelly_multiplier
        capped_kelly = min(sized_kelly, max_position_pct)
        position_usd = bankroll * capped_kelly

        # Calculate shares
        if trade_price > 0:
            shares = position_usd / trade_price
        else:
            shares = 0

        # Estimate price impact (using simplified model since we may not
        # have actual LMSR q values — Polymarket uses an order book)
        # Impact is approximated as proportional to position size
        estimated_impact = (position_usd / bankroll) * abs(actual_edge) * 0.3
        impact_ratio = estimated_impact / abs(actual_edge) if actual_edge != 0 else 0

        # EV per dollar
        if direction == "YES":
            ev_per_dollar = estimated_prob * (1 - trade_price) - (1 - estimated_prob) * trade_price
        else:
            ev_per_dollar = (1 - estimated_prob) * (1 - (1 - market_price)) - estimated_prob * (1 - market_price)

        # Should we trade?
        should_trade = True
        skip_reason = None

        if impact_ratio > impact_threshold:
            should_trade = False
            skip_reason = f"Price impact ({impact_ratio:.0%}) exceeds threshold ({impact_threshold:.0%})"

        if position_usd < 1.0:
            should_trade = False
            skip_reason = "Position size below $1 minimum"

        return TradeAnalysis(
            market_price=market_price,
            estimated_prob=estimated_prob,
            edge=edge,
            direction=direction,
            kelly_fraction=raw_kelly,
            position_size_usd=round(position_usd, 2),
            shares=round(shares, 2),
            avg_fill_price=round(trade_price, 4),
            price_after_trade=round(trade_price + estimated_impact, 4),
            price_impact=round(estimated_impact, 4),
            impact_vs_edge=round(impact_ratio, 4),
            ev_per_dollar=round(ev_per_dollar, 4),
            should_trade=should_trade,
            skip_reason=skip_reason,
        )
