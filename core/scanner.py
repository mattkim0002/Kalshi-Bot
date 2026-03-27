"""
Multi-Market Scanner for Kalshi.com.

Fetches active markets via the Kalshi REST API,
filters by volume/liquidity, and ranks by estimated edge.
"""
import asyncio
import logging
from dataclasses import dataclass
from typing import Optional

import aiohttp

import config

logger = logging.getLogger(__name__)


@dataclass
class Market:
    """Represents a single Kalshi market."""
    condition_id: str       # ticker used as internal ID
    question: str
    slug: str               # ticker
    yes_price: float        # YES probability (0-1)
    no_price: float         # NO probability (0-1)
    yes_token_id: str       # ticker (used for order placement)
    no_token_id: str        # ticker (used for order placement)
    volume: float           # 24h volume in USD
    liquidity: float        # open interest in USD
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
    Scans Kalshi for active markets using their REST API.

    Flow:
    1. Fetch active markets from Kalshi API
    2. Filter by minimum volume and open interest
    3. Return sorted by volume (most liquid first)
    """

    def __init__(self):
        self._session: Optional[aiohttp.ClientSession] = None

    async def _ensure_session(self):
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()

    async def fetch_markets(
        self,
        limit: int = None,
        min_volume: float = None,
        min_liquidity: float = None,
    ) -> list[Market]:
        """
        Fetch active markets from Kalshi API.
        Returns markets sorted by volume (highest first).
        """
        await self._ensure_session()

        limit = limit or config.MAX_MARKETS_TO_SCAN
        min_volume = min_volume or config.MIN_VOLUME
        min_liquidity = min_liquidity or config.MIN_LIQUIDITY

        markets = []
        cursor = None
        retries = 0
        max_retries = 4
        pages_fetched = 0
        max_pages = 3  # Never fetch more than 3 pages (600 raw markets max)

        while pages_fetched < max_pages:
            try:
                params = {
                    "status": "open",
                    "limit": 200,
                }
                if cursor:
                    params["cursor"] = cursor

                async with self._session.get(
                    f"{config.KALSHI_API_BASE}/markets",
                    params=params,
                    timeout=aiohttp.ClientTimeout(total=30),
                ) as resp:
                    if resp.status == 429:
                        retries += 1
                        if retries > max_retries:
                            logger.error("Rate limit retries exhausted. Using markets fetched so far.")
                            break
                        wait = 2 ** retries  # 2, 4, 8, 16 seconds
                        logger.warning(f"Rate limited by Kalshi. Waiting {wait}s before retry {retries}/{max_retries}...")
                        await asyncio.sleep(wait)
                        continue
                    if resp.status != 200:
                        logger.error(f"Kalshi API error {resp.status}: {await resp.text()}")
                        break
                    retries = 0  # reset on success
                    data = await resp.json()

                pages_fetched += 1
                items = data.get("markets", [])
                cursor = data.get("cursor")

                for item in items:
                    market = self._parse_market(item, min_volume, min_liquidity)
                    if market is not None:
                        markets.append(market)

                if not cursor or not items:
                    break

                # Polite pause between pages to avoid triggering rate limits
                await asyncio.sleep(0.5)

            except Exception as e:
                logger.error(f"Error fetching markets: {e}")
                break

        # Sort by volume descending
        markets.sort(key=lambda m: m.volume, reverse=True)
        logger.info(f"Fetched {len(markets)} markets meeting criteria")
        return markets[:limit]

    def _parse_market(self, data: dict, min_volume: float, min_liquidity: float) -> Optional[Market]:
        """Parse a Kalshi market entry."""
        try:
            if data.get("status") != "active":
                return None

            ticker = data.get("ticker", "")
            if not ticker:
                return None

            # Prices come as cents (1-99), convert to 0-1
            yes_ask = data.get("yes_ask", 0) or 0
            yes_bid = data.get("yes_bid", 0) or 0
            last_price = data.get("last_price", 0) or 0

            # Use mid-price or last trade price
            if yes_ask > 0 and yes_bid > 0:
                yes_price = (yes_ask + yes_bid) / 2 / 100
            elif last_price > 0:
                yes_price = last_price / 100
            else:
                return None

            # Skip near-resolved markets
            if yes_price <= 0.03 or yes_price >= 0.97:
                return None

            no_price = 1 - yes_price

            # Volume in cents → dollars
            volume = float(data.get("volume_24h", 0) or 0) / 100
            liquidity = float(data.get("open_interest", 0) or 0) / 100

            if volume < min_volume or liquidity < min_liquidity:
                return None

            title = data.get("title", "") or data.get("subtitle", "") or ticker
            event_ticker = data.get("event_ticker", "")
            category = data.get("category", "") or ""

            # ── Sports filter ────────────────────────────────────
            if category.lower().strip() in config.EXCLUDED_CATEGORIES:
                return None
            title_lower = title.lower()
            if any(kw in title_lower for kw in config.EXCLUDED_KEYWORDS):
                return None

            return Market(
                condition_id=ticker,
                question=title,
                slug=ticker,
                yes_price=yes_price,
                no_price=no_price,
                yes_token_id=ticker,
                no_token_id=ticker,
                volume=volume,
                liquidity=liquidity,
                end_date=data.get("expiration_time", ""),
                category=category,
                description=data.get("subtitle", ""),
                url=f"https://kalshi.com/markets/{event_ticker}/{ticker}",
            )
        except (ValueError, KeyError, TypeError) as e:
            logger.debug(f"Skipping malformed market: {e}")
            return None

    async def fetch_single_market(self, ticker: str) -> Optional[Market]:
        """Fetch a single market by ticker."""
        await self._ensure_session()
        try:
            async with self._session.get(
                f"{config.KALSHI_API_BASE}/markets/{ticker}",
                timeout=aiohttp.ClientTimeout(total=15),
            ) as resp:
                if resp.status != 200:
                    return None
                data = await resp.json()
                market_data = data.get("market", data)
                return self._parse_market(market_data, 0, 0)
        except Exception as e:
            logger.error(f"Error fetching market {ticker}: {e}")
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
