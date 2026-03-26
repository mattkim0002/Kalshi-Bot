"""
Configuration for the Polymarket AI Trading System.
Fill in your credentials and adjust parameters to your risk tolerance.
"""
import os
from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────────
# API CREDENTIALS
# ─────────────────────────────────────────────
# Polymarket US credentials — get from polymarket.us/developer
POLYMARKET_KEY_ID = os.getenv("POLYMARKET_KEY_ID", "")
POLYMARKET_SECRET_KEY = os.getenv("POLYMARKET_SECRET_KEY", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

# ─────────────────────────────────────────────
# TRADING MODE
# ─────────────────────────────────────────────
DRY_RUN = False  # True = paper trading, False = live with real USDC.

# ─────────────────────────────────────────────
# RISK MANAGEMENT
# ─────────────────────────────────────────────
BANKROLL = 30.0             # Total capital allocated (in USDC)
KELLY_FRACTION = 0.25       # Quarter-Kelly (0.25 recommended, max 0.5)
MIN_EDGE = 0.05             # Minimum edge to trade (5%). Below this, skip.
MAX_POSITION_PCT = 0.10     # Max 10% of bankroll per market (~$3 max)
MAX_DAILY_LOSS = 10.0       # Stop trading if daily losses exceed $10
MAX_TOTAL_EXPOSURE = 0.50   # Max 50% of bankroll deployed at any time
MAX_POSITIONS = 5           # Max simultaneous open positions
IMPACT_THRESHOLD = 0.50     # Warn if slippage eats >50% of edge

# ─────────────────────────────────────────────
# MARKET SCANNING
# ─────────────────────────────────────────────
SCAN_INTERVAL_SECONDS = 600     # Scan markets every 10 minutes
MIN_VOLUME = 5000               # Only consider markets with >$5K volume
MIN_LIQUIDITY = 1000            # Only consider markets with >$1K liquidity
MAX_MARKETS_TO_SCAN = 500       # Fetch up to 500 active markets per scan

# ─────────────────────────────────────────────
# CLAUDE AI ESTIMATOR
# ─────────────────────────────────────────────
CLAUDE_MODEL = "claude-sonnet-4-6"
WEB_SEARCH_MAX = 3              # Max web searches per market analysis
MIN_CONFIDENCE = "Medium"       # Minimum confidence to consider: Low, Medium, High
CONFIDENCE_MAP = {"Low": 1, "Medium": 2, "High": 3}

# ─────────────────────────────────────────────
# EXIT STRATEGY
# ─────────────────────────────────────────────
TAKE_PROFIT_EDGE = 0.02        # Exit if remaining edge drops below 2%
STOP_LOSS_PCT = 0.30           # Exit if position loses 30% of entry value
TIME_EXIT_HOURS = 48           # Re-evaluate all positions every 48 hours
REEVAL_INTERVAL_SECONDS = 3600 # Re-evaluate positions every hour

# ─────────────────────────────────────────────
# LOGGING & PERSISTENCE
# ─────────────────────────────────────────────
LOG_FILE = "trading.log"
TRADES_FILE = "trades.json"
POSITIONS_FILE = "positions.json"
PNL_FILE = "pnl.json"

# ─────────────────────────────────────────────
# POLYMARKET US API
# ─────────────────────────────────────────────
POLYMARKET_API_BASE = "https://api.polymarket.us/v1"
POLYMARKET_GATEWAY_BASE = "https://gateway.polymarket.us/v1"
