"""
Configuration for the Kalshi AI Trading System.
Fill in your credentials and adjust parameters to your risk tolerance.
"""
import os
from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────────
# API CREDENTIALS
# ─────────────────────────────────────────────
# Kalshi credentials — get from kalshi.com → Settings → API
# Download your private key as a .pem file
KALSHI_API_KEY_ID = os.getenv("KALSHI_API_KEY_ID", "")
KALSHI_PRIVATE_KEY_PATH = os.getenv("KALSHI_PRIVATE_KEY_PATH", "kalshi_private_key.pem")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

# ─────────────────────────────────────────────
# TRADING MODE
# ─────────────────────────────────────────────
DRY_RUN = False  # True = paper trading, False = live with real money.

# ─────────────────────────────────────────────
# RISK MANAGEMENT
# ─────────────────────────────────────────────
BANKROLL = 10.0             # Fallback for dry-run only — live mode always syncs from API
KELLY_FRACTION = 0.25       # Quarter-Kelly (0.25 recommended, max 0.5)
MIN_EDGE = 0.06             # Minimum edge to trade (6% — higher bar for small account)
MAX_POSITION_PCT = 0.15     # Max 15% of bankroll per market (~$1.50 on $10)
MIN_POSITION_USD = 1.00     # Never place an order smaller than $1 (below Kalshi minimums)
MAX_DAILY_LOSS = 2.00       # Stop trading if daily losses exceed $2 (20% of bankroll)
MAX_TOTAL_EXPOSURE = 0.40   # Max 40% of bankroll deployed at any time ($4 max out)
MAX_POSITIONS = 4           # Max 4 simultaneous positions on a $10 account
MAX_POSITIONS_PER_MARKET = 1  # Max 1 position per unique market question
IMPACT_THRESHOLD = 0.50     # Warn if slippage eats >50% of edge

# ─────────────────────────────────────────────
# MARKET SCANNING
# ─────────────────────────────────────────────
SCAN_INTERVAL_SECONDS = 600     # Scan markets every 10 minutes
MAX_DAYS_TO_EXPIRY = 730        # Only trade markets resolving within 2 years
MIN_VOLUME = 0                  # No volume floor — let Claude filter by confidence instead
MIN_LIQUIDITY = 0               # No open interest requirement (Kalshi markets often show 0)
MAX_MARKETS_TO_SCAN = 100       # Fetch up to 100 markets per scan (1 API call, no pagination)

# ─────────────────────────────────────────────
# MARKET FILTERS
# ─────────────────────────────────────────────
# Categories and keywords to exclude (sports = unpredictable, gambling-like)
EXCLUDED_CATEGORIES = {
    "sports", "sport", "basketball", "football", "soccer", "baseball",
    "hockey", "tennis", "golf", "mma", "ufc", "boxing", "nfl", "nba",
    "mlb", "nhl", "fifa", "olympics", "racing", "esports", "cricket",
}
EXCLUDED_KEYWORDS = [
    "nfl", "nba", "mlb", "nhl", "ufc", "mma", "super bowl", "world series",
    "stanley cup", "championship game", "playoffs", "march madness",
    "world cup", "premier league", "champions league", "formula 1", "f1",
    "grand prix", "wimbledon", "us open tennis", "masters golf",
    "who will win the game", "cover the spread", "total points",
    "rushing yards", "passing yards", "home run", "touchdowns",
]

# ─────────────────────────────────────────────
# CLAUDE AI ESTIMATOR
# ─────────────────────────────────────────────
CLAUDE_MODEL = "claude-sonnet-4-6"
WEB_SEARCH_MAX = 1              # 1 web search per market (fast + news-aware)
MIN_CONFIDENCE = "Medium"       # Minimum confidence: Low, Medium, High
CONFIDENCE_MAP = {"Low": 1, "Medium": 2, "High": 3}
ESTIMATOR_BATCH_SIZE = 5        # Markets to analyze per cycle (top 5 by volume)
ESTIMATOR_STAGGER_SECS = 5      # Seconds between Claude calls

# ─────────────────────────────────────────────
# EXIT STRATEGY
# ─────────────────────────────────────────────
TAKE_PROFIT_EDGE = 0.02        # Exit if remaining edge drops below 2%
STOP_LOSS_PCT = 0.20           # Exit if position loses 20% of entry value (tighter on small account)
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
# KALSHI API
# ─────────────────────────────────────────────
KALSHI_API_BASE = "https://api.elections.kalshi.com/trade-api/v2"
