# Polymarket-Hack

AI-powered Polymarket prediction market trading system using Claude AI, LMSR pricing, and Kelly criterion.

## What This Is

A complete autonomous trading bot that:
1. **Scans** 500+ active Polymarket markets every 10 minutes
2. **Estimates** true probabilities using Claude AI with web search
3. **Finds edge** where Claude's estimate differs from the market price
4. **Sizes positions** using the Kelly Criterion (quarter-Kelly)
5. **Executes trades** via Polymarket's CLOB API
6. **Manages exits** with stop-loss, take-profit, and dynamic re-evaluation

## Background

Based on the viral experiment (~March 10, 2026) where a Claude-powered agent turned $1,000 into ~$14,200 in 48 hours on Polymarket, executing 5,200+ trades with a 1,322% return.

See [RESEARCH.md](RESEARCH.md) for the complete mathematical analysis, formulas, and strategy breakdown.

## Architecture

```
┌─────────────┐    ┌──────────────┐    ┌─────────────┐    ┌──────────────┐
│   Scanner   │───▶│  Estimator   │───▶│  Analyzer   │───▶│  Executor    │
│  (Gamma API)│    │  (Claude AI) │    │ (LMSR+Kelly)│    │  (CLOB API)  │
└─────────────┘    └──────────────┘    └─────────────┘    └──────────────┘
                          ▲
                   ┌──────┘
                   │ Exit Manager
                   │ (TP/SL/Reeval)
                   └──────────────┘
```

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Set up credentials
cp .env.example .env
# Edit .env with your Polymarket + Anthropic API keys

# Paper trade (recommended first)
python main.py --dry-run --once

# Continuous paper trading
python main.py --dry-run

# Live trading (after validating)
# Set DRY_RUN=False in config.py
python main.py
```

## Project Structure

```
├── main.py                    # Main orchestrator and trading loop
├── config.py                  # All configuration and risk parameters
├── RESEARCH.md                # Complete math, formulas, and analysis
├── requirements.txt           # Python dependencies
├── .env.example               # Template for API credentials
├── core/
│   ├── lmsr.py               # LMSR pricing engine + Kelly criterion
│   ├── scanner.py            # Multi-market scanner (Gamma API)
│   ├── estimator.py          # Claude AI probability estimator
│   ├── portfolio.py          # Position tracking, P&L, risk limits
│   └── executor.py           # Trade execution (paper + live)
└── strategies/
    └── exit_manager.py        # Exit logic: TP, SL, time, re-evaluation
```

## Key Formulas

| Formula | Purpose |
|---------|---------|
| `C(q) = b × ln(Σ e^(qi/b))` | LMSR cost function — how the market prices trades |
| `p_k = e^(qk/b) / Σ e^(qi/b)` | Market price (softmax) — same math as neural networks |
| `f* = (p × b - q) / b` | Kelly criterion — optimal bet sizing |
| `EV = p_true × (1-price) - (1-p_true) × price` | Expected value per dollar |

## Risk Controls

| Parameter | Default | Purpose |
|-----------|---------|---------|
| Kelly Fraction | 25% | Quarter-Kelly conservative sizing |
| Min Edge | 5% | Skip markets with less than 5% edge |
| Max Position | 6% | Never risk >6% of bankroll per market |
| Max Daily Loss | $50 | Stop trading after $50 daily loss |
| Max Exposure | 50% | Never have >50% of bankroll deployed |
| Stop-Loss | 30% | Exit if position drops 30% |
| Take-Profit Edge | 2% | Exit when remaining edge < 2% |

## ⚠️ Warnings

- **Start with DRY_RUN=True** — paper trade for at least 1-2 weeks first
- **Only use money you can afford to lose entirely**
- **The viral $14K result is unverified** — no auditable trade records exist
- **87% of Polymarket wallets lose money**
- **Check your jurisdiction** for prediction market legality
- **This is a tool, not financial advice**
