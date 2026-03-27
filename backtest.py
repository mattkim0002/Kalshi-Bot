"""
Polymarket.com Backtester

Fetches resolved markets, simulates our edge-based strategy,
and reports win rate, ROI, drawdown, and Kelly performance.

Usage:
    python3 backtest.py                    # Run full backtest
    python3 backtest.py --min-edge 0.05    # Only trades with 5%+ edge
    python3 backtest.py --bankroll 100     # Start with $100
    python3 backtest.py --limit 500        # Analyze 500 resolved markets
"""

import argparse
import asyncio
import json
import sys
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import aiohttp


GAMMA_API = "https://gamma-api.polymarket.com"


# ── Data Structures ──────────────────────────────────────

@dataclass
class ResolvedMarket:
    question: str
    slug: str
    yes_price_at_sample: float      # Market price we "observe"
    resolved_yes: bool              # Did YES win?
    volume: float
    end_date: str = ""
    category: str = ""


@dataclass
class Trade:
    question: str
    direction: str          # YES or NO
    entry_price: float
    position_size: float    # USD
    shares: float
    resolved_yes: bool
    pnl: float
    roi_pct: float
    edge_used: float


@dataclass
class BacktestResult:
    trades: list[Trade] = field(default_factory=list)
    bankroll_history: list[float] = field(default_factory=list)

    @property
    def total_trades(self): return len(self.trades)

    @property
    def winning_trades(self): return [t for t in self.trades if t.pnl > 0]

    @property
    def losing_trades(self): return [t for t in self.trades if t.pnl <= 0]

    @property
    def win_rate(self):
        if not self.trades: return 0
        return len(self.winning_trades) / len(self.trades)

    @property
    def total_pnl(self): return sum(t.pnl for t in self.trades)

    @property
    def avg_edge(self):
        if not self.trades: return 0
        return sum(t.edge_used for t in self.trades) / len(self.trades)

    @property
    def profit_factor(self):
        gross_profit = sum(t.pnl for t in self.winning_trades)
        gross_loss = abs(sum(t.pnl for t in self.losing_trades))
        return gross_profit / gross_loss if gross_loss > 0 else float("inf")

    @property
    def max_drawdown(self):
        if not self.bankroll_history: return 0
        peak = self.bankroll_history[0]
        max_dd = 0
        for b in self.bankroll_history:
            if b > peak:
                peak = b
            dd = (peak - b) / peak
            max_dd = max(max_dd, dd)
        return max_dd

    @property
    def roi(self):
        if not self.bankroll_history: return 0
        return (self.bankroll_history[-1] - self.bankroll_history[0]) / self.bankroll_history[0]


# ── Market Fetcher ────────────────────────────────────────

async def fetch_resolved_markets(limit: int = 300) -> list[ResolvedMarket]:
    """Fetch resolved markets from Gamma API with their price history."""
    print(f"Fetching resolved markets from Polymarket.com...")
    markets = []
    offset = 0
    batch_size = 100

    async with aiohttp.ClientSession() as session:
        while len(markets) < limit:
            params = {
                "closed": "true",
                "limit": batch_size,
                "offset": offset,
                "order": "volume",
                "ascending": "false",
            }
            try:
                async with session.get(f"{GAMMA_API}/markets", params=params, timeout=aiohttp.ClientTimeout(total=30)) as resp:
                    if resp.status != 200:
                        print(f"API error: {resp.status}")
                        break
                    items = await resp.json()
            except Exception as e:
                print(f"Error fetching markets: {e}")
                break

            if not items:
                break

            for data in items:
                market = _parse_resolved(data)
                if market:
                    markets.append(market)

            offset += batch_size
            if len(items) < batch_size:
                break

            print(f"  Fetched {len(markets)} resolved markets...", end="\r")

    print(f"  Loaded {len(markets)} resolved markets.         ")
    return markets[:limit]


def _parse_resolved(data: dict) -> Optional[ResolvedMarket]:
    """Parse a resolved market entry."""
    try:
        if not data.get("closed"):
            return None

        tokens = data.get("tokens", [])
        if len(tokens) < 2:
            return None

        yes_token = next((t for t in tokens if t.get("outcome", "").upper() == "YES"), tokens[0])
        no_token = next((t for t in tokens if t.get("outcome", "").upper() == "NO"), tokens[1])

        # resolvedPrice: 1.0 = YES won, 0.0 = NO won
        resolved_price = float(data.get("resolutionPrice", yes_token.get("winner", 0)) or 0)
        resolved_yes = resolved_price > 0.5

        # Use the price the market was at before resolution (last traded price)
        # We sample at what would have been the "entry" price
        yes_price = float(yes_token.get("price", 0) or 0)

        # Skip markets that were already near resolution when we'd sample them
        # (too close to 0 or 1 — no edge opportunity)
        if yes_price <= 0.05 or yes_price >= 0.95:
            return None

        volume = float(data.get("volume", 0) or 0)
        if volume < 1000:  # Skip tiny markets
            return None

        return ResolvedMarket(
            question=data.get("question", ""),
            slug=data.get("slug", ""),
            yes_price_at_sample=yes_price,
            resolved_yes=resolved_yes,
            volume=volume,
            end_date=data.get("endDate", ""),
            category=data.get("category", ""),
        )
    except (ValueError, KeyError, TypeError):
        return None


# ── Strategy Simulator ────────────────────────────────────

def simulate_strategy(
    markets: list[ResolvedMarket],
    bankroll: float,
    min_edge: float,
    kelly_fraction: float,
    max_position_pct: float,
    assumed_accuracy: float,  # How often is our edge estimate correct?
) -> BacktestResult:
    """
    Simulate our edge-based Kelly strategy on resolved markets.

    Key assumption: We model our AI estimator as having a certain accuracy.
    If accuracy=0.6, we get the direction right 60% of the time when we
    detect edge. This lets us test how sensitive our strategy is to model quality.
    """
    import random
    random.seed(42)  # Reproducible results

    result = BacktestResult()
    current_bankroll = bankroll
    result.bankroll_history.append(current_bankroll)

    for market in markets:
        yes_price = market.yes_price_at_sample

        # Simulate our AI estimating a probability
        # With assumed_accuracy, we get the right direction
        # Our estimate has some spread around the true probability
        true_prob = 1.0 if market.resolved_yes else 0.0

        # Model our estimator: correct direction with probability = assumed_accuracy
        if random.random() < assumed_accuracy:
            # We get it right — our estimate is closer to true_prob
            our_estimate = true_prob * 0.7 + yes_price * 0.3
        else:
            # We get it wrong — our estimate is on the wrong side
            our_estimate = (1 - true_prob) * 0.7 + yes_price * 0.3

        # Add some noise
        our_estimate = max(0.01, min(0.99, our_estimate + random.gauss(0, 0.05)))

        # Calculate edge
        edge = our_estimate - yes_price

        # Skip if not enough edge
        if abs(edge) < min_edge:
            continue

        # Determine direction
        if edge > 0:
            direction = "YES"
            entry_price = yes_price
            win_condition = market.resolved_yes
        else:
            direction = "NO"
            entry_price = 1 - yes_price
            win_condition = not market.resolved_yes
            edge = abs(edge)

        # Kelly position sizing
        # Kelly fraction = edge / (1 - entry_price) for binary bets
        kelly_bet = edge / (1 - entry_price) if (1 - entry_price) > 0 else 0
        kelly_bet = kelly_bet * kelly_fraction  # quarter-Kelly
        kelly_bet = min(kelly_bet, max_position_pct)  # cap

        position_size = current_bankroll * kelly_bet
        if position_size < 1.0:
            continue  # Too small

        shares = position_size / entry_price

        # Calculate P&L
        if win_condition:
            pnl = shares * (1 - entry_price)  # Win: shares * profit per share
        else:
            pnl = -position_size  # Loss: lose entire position

        roi_pct = pnl / position_size

        trade = Trade(
            question=market.question[:60],
            direction=direction,
            entry_price=entry_price,
            position_size=position_size,
            shares=shares,
            resolved_yes=market.resolved_yes,
            pnl=pnl,
            roi_pct=roi_pct,
            edge_used=edge,
        )
        result.trades.append(trade)

        current_bankroll += pnl
        result.bankroll_history.append(current_bankroll)

        if current_bankroll <= 0:
            print("  ⚠️  Bankroll depleted!")
            break

    return result


# ── Report ────────────────────────────────────────────────

def print_report(result: BacktestResult, bankroll: float, params: dict):
    """Print a formatted backtest report."""
    print()
    print("=" * 70)
    print("  POLYMARKET BACKTEST RESULTS")
    print("=" * 70)
    print(f"  Strategy Parameters:")
    print(f"    Min Edge:          {params['min_edge']:.0%}")
    print(f"    Kelly Fraction:    {params['kelly_fraction']:.0%} (Quarter-Kelly)")
    print(f"    Max Position:      {params['max_position_pct']:.0%} of bankroll")
    print(f"    AI Accuracy:       {params['assumed_accuracy']:.0%} (model correctness)")
    print(f"    Starting Bankroll: ${bankroll:.2f}")
    print()
    print(f"  Performance:")
    print(f"    Total Trades:      {result.total_trades}")
    print(f"    Win Rate:          {result.win_rate:.1%}")
    print(f"    Profit Factor:     {result.profit_factor:.2f}x")
    print(f"    Total P&L:         ${result.total_pnl:+.2f}")
    print(f"    ROI:               {result.roi:+.1%}")
    print(f"    Avg Edge/Trade:    {result.avg_edge:.1%}")
    print()
    if result.bankroll_history:
        print(f"    Starting Capital:  ${result.bankroll_history[0]:.2f}")
        print(f"    Final Capital:     ${result.bankroll_history[-1]:.2f}")
        print(f"    Max Drawdown:      {result.max_drawdown:.1%}")
    print()

    # Sensitivity analysis
    print("  Sensitivity Analysis (AI accuracy vs ROI):")
    print(f"  {'Accuracy':>10} | {'Win Rate':>9} | {'ROI':>9} | {'Drawdown':>9}")
    print("  " + "-" * 46)

    # We'll run quick mini-sims at different accuracy levels
    # (using the same markets already loaded)
    print()

    # Top 5 winning trades
    sorted_trades = sorted(result.trades, key=lambda t: t.pnl, reverse=True)
    print("  Top 5 Trades:")
    print(f"  {'Question':<45} {'Dir':>4} {'Edge':>6} {'P&L':>8}")
    print("  " + "-" * 67)
    for t in sorted_trades[:5]:
        print(f"  {t.question:<45} {t.direction:>4} {t.edge_used:>5.1%} ${t.pnl:>7.2f}")

    print()
    print("  Bottom 5 Trades:")
    print(f"  {'Question':<45} {'Dir':>4} {'Edge':>6} {'P&L':>8}")
    print("  " + "-" * 67)
    for t in sorted_trades[-5:]:
        print(f"  {t.question:<45} {t.direction:>4} {t.edge_used:>5.1%} ${t.pnl:>7.2f}")

    print()
    print("=" * 70)

    # Verdict
    if result.roi > 0.10 and result.win_rate > 0.55:
        verdict = "STRONG ✓ — Strategy profitable at this accuracy level"
    elif result.roi > 0:
        verdict = "MARGINAL — Profitable but thin margins"
    else:
        verdict = "UNPROFITABLE ✗ — Needs higher accuracy or edge threshold"

    print(f"  Verdict: {verdict}")
    print("=" * 70)


async def run_sensitivity(markets, bankroll, min_edge, kelly_fraction, max_pos):
    """Run backtest at multiple accuracy levels to show sensitivity."""
    print()
    print("  Sensitivity Analysis (how good does our AI need to be?):")
    print(f"  {'AI Accuracy':>12} | {'Trades':>7} | {'Win Rate':>9} | {'ROI':>8} | {'Drawdown':>9}")
    print("  " + "-" * 58)

    for accuracy in [0.50, 0.55, 0.60, 0.65, 0.70, 0.75]:
        r = simulate_strategy(markets, bankroll, min_edge, kelly_fraction, max_pos, accuracy)
        print(f"  {accuracy:>12.0%} | {r.total_trades:>7} | {r.win_rate:>8.1%} | {r.roi:>7.1%}% | {r.max_drawdown:>8.1%}")


# ── Main ──────────────────────────────────────────────────

async def main():
    parser = argparse.ArgumentParser(description="Polymarket Strategy Backtester")
    parser.add_argument("--bankroll", type=float, default=100.0, help="Starting bankroll in USD")
    parser.add_argument("--min-edge", type=float, default=0.03, help="Minimum edge to trade (default: 0.03)")
    parser.add_argument("--kelly", type=float, default=0.25, help="Kelly fraction (default: 0.25)")
    parser.add_argument("--max-position", type=float, default=0.15, help="Max position % (default: 0.15)")
    parser.add_argument("--accuracy", type=float, default=0.62, help="AI model accuracy (default: 0.62)")
    parser.add_argument("--limit", type=int, default=300, help="Number of resolved markets to analyze")
    args = parser.parse_args()

    # Fetch resolved markets
    markets = await fetch_resolved_markets(limit=args.limit)

    if not markets:
        print("No resolved markets found. Check your internet connection.")
        sys.exit(1)

    print(f"Running backtest on {len(markets)} resolved markets...")

    # Main simulation
    params = {
        "min_edge": args.min_edge,
        "kelly_fraction": args.kelly,
        "max_position_pct": args.max_position,
        "assumed_accuracy": args.accuracy,
    }
    result = simulate_strategy(
        markets,
        bankroll=args.bankroll,
        **params,
    )

    print_report(result, args.bankroll, params)

    # Sensitivity analysis
    await run_sensitivity(markets, args.bankroll, args.min_edge, args.kelly, args.max_position)

    # Save results
    output = {
        "params": params,
        "bankroll": args.bankroll,
        "markets_analyzed": len(markets),
        "total_trades": result.total_trades,
        "win_rate": result.win_rate,
        "roi": result.roi,
        "total_pnl": result.total_pnl,
        "max_drawdown": result.max_drawdown,
        "profit_factor": result.profit_factor,
        "bankroll_history": result.bankroll_history,
    }
    with open("backtest_results.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Results saved to backtest_results.json")


if __name__ == "__main__":
    asyncio.run(main())
