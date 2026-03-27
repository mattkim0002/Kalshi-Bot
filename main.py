"""
Polymarket AI Trading System — Main Orchestrator.

Ties together all components into a continuous trading loop:
1. Scan markets → 2. Estimate probabilities → 3. Find edge →
4. Size positions → 5. Execute trades → 6. Monitor exits → Repeat

Usage:
    python main.py              # Run with settings from config.py
    python main.py --dry-run    # Force paper trading mode
    python main.py --once       # Run one scan cycle then exit
"""
import asyncio
import argparse
import logging
import signal
import sys
import time
from pathlib import Path

import config
from core.lmsr import LMSREngine
from core.scanner import MarketScanner, Market
from core.estimator import ProbabilityEstimator
from core.portfolio import PortfolioManager
from core.executor import TradeExecutor
from strategies.exit_manager import ExitManager

# ─────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(config.LOG_FILE),
    ],
)
logger = logging.getLogger("main")


class TradingSystem:
    """
    The main orchestrator. Runs the scan → analyze → trade → monitor loop.
    """

    def __init__(self, dry_run: bool = None):
        if dry_run is not None:
            config.DRY_RUN = dry_run
        
        self.lmsr = LMSREngine()
        self.executor = TradeExecutor()
        # Pass executor's signing method so scanner makes authenticated requests
        self.scanner = MarketScanner(sign_request=self.executor._sign_request if not self.executor.dry_run else None)
        self.estimator = ProbabilityEstimator()
        self.portfolio = PortfolioManager()
        self.exit_manager = ExitManager(self.portfolio, self.estimator)
        
        self._running = True
        self._cycle_count = 0

    async def run(self, once: bool = False):
        """Main trading loop."""
        self._validate_config()

        try:
            # Sync live balance before printing banner so it shows real balance
            await self.portfolio.sync_balance(self.executor)
            self._print_banner()

            while self._running:
                self._cycle_count += 1
                logger.info(f"\n{'='*60}\nCYCLE {self._cycle_count} STARTING\n{'='*60}")

                await self._run_cycle()

                # Re-sync balance after each cycle to pick up deposits/withdrawals
                await self.portfolio.sync_balance(self.executor)
                
                self.portfolio.print_summary()
                
                if once:
                    logger.info("Single-cycle mode. Exiting.")
                    break
                
                logger.info(
                    f"Cycle complete. Next scan in "
                    f"{config.SCAN_INTERVAL_SECONDS // 60} minutes."
                )
                await asyncio.sleep(config.SCAN_INTERVAL_SECONDS)

        except KeyboardInterrupt:
            logger.info("Shutdown requested by user")
        finally:
            await self._shutdown()

    async def _run_cycle(self):
        """Execute one full scan → analyze → trade → monitor cycle."""
        
        # ── Step 1: Check exits on existing positions ────────
        if self.portfolio.positions:
            logger.info("Step 1: Checking exit conditions on open positions...")
            await self._check_exits()
        
        # ── Step 2: Scan for new opportunities ───────────────
        logger.info("Step 2: Scanning markets...")
        markets = await self.scanner.fetch_markets()
        
        if not markets:
            logger.warning("No markets found. Skipping cycle.")
            return
        
        logger.info(f"Found {len(markets)} markets meeting volume/liquidity criteria")
        
        # Filter out markets we already have positions in
        open_ids = set(self.portfolio.positions.keys())
        # Also filter out questions we already hold (1 position per market question)
        open_questions = {p.question for p in self.portfolio.positions.values()}
        new_markets = [
            m for m in markets
            if m.condition_id not in open_ids and m.question not in open_questions
        ]
        
        # ── Step 3: Estimate probabilities ───────────────────
        # Only estimate a batch at a time to manage API costs/rate limits
        batch_size = min(config.ESTIMATOR_BATCH_SIZE, len(new_markets))
        batch = new_markets[:batch_size]

        logger.info(f"Step 3: Estimating probabilities for top {batch_size} markets...")
        estimated = await self.estimator.estimate_batch(batch)
        
        # ── Step 4: Rank by edge and filter ──────────────────
        ranked = self.scanner.rank_by_edge(estimated)
        tradeable = self.scanner.filter_tradeable(ranked)
        
        # Also filter by confidence
        tradeable = [
            m for m in tradeable
            if self.estimator.meets_confidence_threshold(m)
        ]
        
        logger.info(
            f"Step 4: {len(tradeable)} markets with sufficient edge "
            f"and confidence"
        )
        
        if tradeable:
            self._print_opportunities(tradeable[:10])
        
        # ── Step 5: Execute trades ───────────────────────────
        logger.info("Step 5: Executing trades...")
        for market in tradeable:
            await self._maybe_trade(market)

    async def _maybe_trade(self, market: Market):
        """
        Analyze a market opportunity and execute if all criteria are met.
        """
        # Drawdown protection (Chan): pause if 30% down, halve sizing if 20% down
        dd_multiplier = self.portfolio.drawdown_kelly_multiplier
        if dd_multiplier == 0.0:
            logger.warning("Trading paused — drawdown limit reached")
            return

        # Signal health check (Simons): reduce sizing if recent win rate < 45%
        signal_multiplier = self.portfolio.signal_health_multiplier

        # Correlation check (Simons): skip if new bet is too correlated with existing positions
        direction = "YES" if (market.estimated_prob or 0) > market.yes_price else "NO"
        correlation = self.portfolio.correlation_risk(market.question, direction)
        if correlation > 0.6:
            logger.info(f"SKIP: {market.question[:50]}... — Too correlated with existing positions ({correlation:.0%})")
            return

        # Long-shot bias correction (Taleb): underdog markets (5-20%) are
        # systematically underpriced by crowds.
        if 0.05 <= market.yes_price <= 0.20:
            market.estimated_prob = min(0.99, (market.estimated_prob or 0) * 1.08)

        # Combined Kelly multiplier
        combined_multiplier = config.KELLY_FRACTION * dd_multiplier * signal_multiplier

        # Run full trade analysis
        analysis = self.lmsr.analyze_trade(
            market_price=market.yes_price,
            estimated_prob=market.estimated_prob,
            bankroll=self.portfolio.available_capital,
            kelly_multiplier=combined_multiplier,
            min_edge=config.MIN_EDGE,
            max_position_pct=config.MAX_POSITION_PCT,
            impact_threshold=config.IMPACT_THRESHOLD,
        )
        
        if not analysis.should_trade:
            logger.info(
                f"SKIP: {market.question[:50]}... — {analysis.skip_reason}"
            )
            return

        # Minimum position size guard (Kalshi has ~$1 minimum and tiny bets waste fees)
        if analysis.position_size_usd < config.MIN_POSITION_USD:
            logger.info(
                f"SKIP: {market.question[:50]}... — "
                f"Position ${analysis.position_size_usd:.2f} below minimum ${config.MIN_POSITION_USD:.2f}"
            )
            return

        # Risk check
        can_trade, reason = self.portfolio.can_open_position(analysis.position_size_usd)
        if not can_trade:
            logger.info(f"BLOCKED: {reason}")
            return
        
        # Determine token
        if analysis.direction == "YES":
            token_id = market.yes_token_id
            trade_price = market.yes_price
        else:
            token_id = market.no_token_id
            trade_price = market.no_price
        
        # Execute
        result = await self.executor.execute_buy(
            token_id=token_id,
            amount_usd=analysis.position_size_usd,
            current_price=trade_price,
            direction=analysis.direction,
            question=market.question,
        )
        
        if result.success:
            # Record in portfolio
            self.portfolio.open_position(
                market_id=market.condition_id,
                question=market.question,
                direction=analysis.direction,
                price=result.fill_price,
                shares=result.shares,
                estimated_prob=market.estimated_prob,
                confidence=market.confidence or "Medium",
                token_id=token_id,
                url=market.url,
            )
            
            sim_tag = " [SIMULATED]" if result.is_simulation else ""
            logger.info(
                f"✓ TRADE{sim_tag}: {analysis.direction} "
                f"{market.question[:50]}... | "
                f"${analysis.position_size_usd:.2f} @ {trade_price:.4f} | "
                f"Edge: {abs(market.edge):.2%} | "
                f"EV/$ {analysis.ev_per_dollar:.4f}"
            )
        else:
            logger.error(f"✗ TRADE FAILED: {result.error}")

    async def _check_exits(self):
        """Check all positions for exit signals and execute exits."""
        signals = await self.exit_manager.check_all_positions()
        
        for signal in signals:
            pos = self.portfolio.positions.get(signal.market_id)
            if not pos:
                continue
            
            logger.info(
                f"EXIT SIGNAL [{signal.exit_type}]: {pos.question[:50]}... — "
                f"{signal.reason}"
            )
            
            # Execute the exit
            result = await self.executor.execute_sell(
                token_id=pos.token_id,
                shares=pos.shares,
                current_price=signal.current_price,
                direction=pos.direction,
                question=pos.question,
            )
            
            if result.success:
                self.portfolio.close_position(
                    market_id=signal.market_id,
                    exit_price=result.fill_price,
                    reason=f"{signal.exit_type}: {signal.reason}",
                )
            else:
                logger.error(f"Failed to exit position: {result.error}")

    # ── Display ──────────────────────────────────────

    def _print_banner(self):
        mode = "PAPER TRADING (DRY RUN)" if config.DRY_RUN else "⚠️  LIVE TRADING ⚠️"
        print(f"""
╔══════════════════════════════════════════════════════════════╗
║                      BOT TRADER                             ║
║                                                              ║
║  Mode:           {mode:<40} ║
║  Bankroll:       ${self.portfolio.bankroll:>10,.2f}                              ║
║  Kelly fraction: {config.KELLY_FRACTION:.0%} ({"Quarter" if config.KELLY_FRACTION == 0.25 else "Half" if config.KELLY_FRACTION == 0.5 else "Custom"}-Kelly){' ' * 28}║
║  Min edge:       {config.MIN_EDGE:.0%}{' ' * 42}║
║  Max position:   {config.MAX_POSITION_PCT:.0%} of bankroll{' ' * 30}║
║  Scan interval:  {config.SCAN_INTERVAL_SECONDS // 60} minutes{' ' * 34}║
║  Claude model:   {config.CLAUDE_MODEL[:38]:<38} ║
╚══════════════════════════════════════════════════════════════╝
""")

    def _print_opportunities(self, markets: list[Market]):
        """Print top opportunities found."""
        print("\n  TOP OPPORTUNITIES:")
        print(f"  {'Question':<45} {'Market':>6} {'Est.':>6} {'Edge':>7} {'Conf':>6}")
        print("  " + "-" * 75)
        for m in markets:
            q = m.question[:43]
            edge = m.edge or 0
            direction = "YES↑" if edge > 0 else "NO↑"
            print(
                f"  {q:<45} {m.yes_price:>6.2f} "
                f"{m.estimated_prob:>6.2f} {edge:>+6.2%} "
                f"{m.confidence or '?':>6}"
            )
        print()

    def _validate_config(self):
        """Validate configuration before starting."""
        if not config.ANTHROPIC_API_KEY:
            logger.error("ANTHROPIC_API_KEY not set. Cannot estimate probabilities.")
            sys.exit(1)
        
        if not config.DRY_RUN:
            if not config.KALSHI_API_KEY_ID:
                logger.error(
                    "Live trading enabled but KALSHI_API_KEY_ID not set. "
                    "Set DRY_RUN=True or add your Kalshi API key to .env"
                )
                sys.exit(1)
            
            print("\n⚠️  WARNING: LIVE TRADING MODE ENABLED")
            print("Real money will be used. Press Ctrl+C within 10 seconds to abort.\n")
            try:
                time.sleep(10)
            except KeyboardInterrupt:
                print("\nAborted.")
                sys.exit(0)

    async def _shutdown(self):
        """Clean up resources."""
        logger.info("Shutting down...")
        await self.scanner.close()
        await self.estimator.close()
        self.portfolio.print_summary()
        logger.info("Shutdown complete.")


def main():
    parser = argparse.ArgumentParser(description="Polymarket AI Trading System")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Force paper trading mode (overrides config)",
    )
    parser.add_argument(
        "--once", action="store_true",
        help="Run one scan cycle then exit",
    )
    args = parser.parse_args()

    dry_run = True if args.dry_run else None
    system = TradingSystem(dry_run=dry_run)

    # Handle graceful shutdown
    loop = asyncio.new_event_loop()
    
    def signal_handler(sig, frame):
        system._running = False

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        loop.run_until_complete(system.run(once=args.once))
    finally:
        loop.close()


if __name__ == "__main__":
    main()
