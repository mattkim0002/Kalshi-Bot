"""
Multi-Market Scanner for Polymarket US.

Fetches active markets via the polymarket-us SDK,
filters by volume/liquidity, and ranks by estimated edge.
"""
import asyncio
import json
import logging
from dataclasses import dataclass
from typing import Optional

import config

logger = logging.getLogger(__name__)

_sdk_available = False
try:
    from polymarket_us import PolymarketUS
    _sdk_available = True
except ImportError:
    logger.warning(
        "polymarket-us not installed. Run: pip install polymarket-us"
    )


@dataclass
class Market:
    """Represents a single Polymarket US market."""
    condition_id: str       # slug, used as internal ID
    question: str
    slug: str
    yes_price: float        # implied YES probability (lastTradePrice or mid)
    no_price: float
    yes_token_id: str       # repurposed: holds slug for order placement
    no_token_id: str        # repurposed: holds slug for order placement
    volume: float
    liquidity: float
    end_date: Optional[str] = None
    category: str = ""
    description: str = ""
    url: str = ""
    # Filled by the estimator
    estimated_prob: Optional[float] = None
    confidence: Optional[str] = None
    reasoning: Optional[str] = None
    edge: Optional[float] = None


class MarketScanner:
    """
    Scans Polymarket US for active markets and identifies candidates.

    Flow:
    1. Fetch active markets via polymarket-us SDK
    2. Filter by minimum volume and liquidity
    3. Return sorted by volume (most liquid first)
    """

    def __init__(self):
        self._client: Optional[object] = None
        if _sdk_available:
            try:
                self._client = PolymarketUS(
                    key_id=config.POLYMARKET_KEY_ID,
                    secret_key=config.POLYMARKET_SECRET_KEY,
                )
            except Exception as e:
                logger.error(f"Failed to initialize Polymarket US client: {e}")

    async def close(self):
        pass  # SDK handles cleanup internally

    async def fetch_markets(
        self,
        limit: int = None,
        min_volume: float = None,
        min_liquidity: float = None,
    ) -> list[Market]:
        """
        Fetch active markets from Polymarket US.
        Returns markets sorted by volume (highest first).
        """
        if not self._client:
            logger.error("Polymarket US SDK not available. Cannot fetch markets.")
            return []

        limit = limit or config.MAX_MARKETS_TO_SCAN
        min_volume = min_volume or config.MIN_VOLUME
        min_liquidity = min_liquidity or config.MIN_LIQUIDITY

        markets = []
        offset = 0
        batch_size = 100

        while len(markets) < limit:
            try:
                params = {
                    "active": True,
                    "closed": False,
                    "limit": batch_size,
                    "offset": offset,
                }

                raw = await asyncio.to_thread(self._client.markets.list, params)

                # SDK may return a dict with 'markets' key or a list directly
                items = raw.get("markets", raw) if isinstance(raw, dict) else raw
                if not items:
                    break

                for m in items:
                    data = m if isinstance(m, dict) else vars(m)
                    market = self._parse_market(data)
                    if market is None:
                        continue
                    markets.append(market)

                offset += batch_size
                if len(items) < batch_size:
                    break

            except Exception as e:
                logger.error(f"Error fetching markets: {e}")
                break

        logger.info(f"Fetched {len(markets)} markets meeting criteria")
        return markets[:limit]

    def _parse_market(self, data: dict) -> Optional[Market]:
        """Parse a market from polymarket-us SDK response."""
        try:
            slug = data.get("slug", "")
            if not slug:
                return None

            # Skip closed/resolved markets
            if data.get("closed", False):
                return None

            # outcomePrices is a JSON string e.g. '["0.62","0.38"]'
            raw_prices = data.get("outcomePrices", "[]")
            prices = json.loads(raw_prices) if isinstance(raw_prices, str) else raw_prices

            if len(prices) < 2:
                return None

            yes_price = float(prices[0])

            # Skip unpriced or fully resolved markets
            if yes_price <= 0 or yes_price >= 1:
                return None

            return Market(
                condition_id=slug,
                question=data.get("question", ""),
                slug=slug,
                yes_price=yes_price,
                no_price=float(prices[1]),
                yes_token_id=slug,   # slug used for order placement
                no_token_id=slug,    # slug used for order placement
                volume=0,            # not provided by this endpoint
                liquidity=0,
                end_date=data.get("endDate"),
                category=data.get("category", ""),
                description=data.get("description", ""),
                url=f"https://polymarket.us/event/{slug}",
            )
        except (ValueError, KeyError, TypeError, json.JSONDecodeError) as e:
            logger.debug(f"Skipping malformed market: {e}")
            return None

    async def fetch_single_market(self, slug: str) -> Optional[Market]:
        """Fetch a single market by slug (used for position re-evaluation)."""
        if not self._client:
            return None
        try:
            raw = await asyncio.to_thread(self._client.markets.retrieve_by_slug, slug)
            data = raw if isinstance(raw, dict) else vars(raw)
            return self._parse_market(data)
        except Exception as e:
            logger.error(f"Error fetching market {slug}: {e}")
            return None

    def rank_by_edge(self, markets: list[Market]) -> list[Market]:
        """Rank markets by absolute edge (requires estimated_prob to be set)."""
        estimated = [m for m in markets if m.estimated_prob is not None]
        for m in estimated:
            m.edge = m.estimated_prob - m.yes_price
        return sorted(estimated, key=lambda m: abs(m.edge or 0), reverse=True)

    def filter_tradeable(
        self,
        markets: list[Market],
        min_edge: float = None,
    ) -> list[Market]:
        """Filter to only markets with sufficient edge."""
        min_edge = min_edge or config.MIN_EDGE
        return [
            m for m in markets
            if m.edge is not None and abs(m.edge) >= min_edge
        ]
