"""
Multi-Market Scanner for Polymarket.com.

Fetches active markets via the Gamma API (metadata + prices),
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
    """Represents a single Polymarket.com market."""
    condition_id: str       # condition_id used as internal ID
    question: str
    slug: str
    yes_price: float        # implied YES probability
    no_price: float
    yes_token_id: str       # CLOB token ID for YES outcome
    no_token_id: str        # CLOB token ID for NO outcome
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
    Scans Polymarket.com for active markets using the Gamma API.

    Flow:
    1. Fetch active markets from gamma-api.polymarket.com
    2. Filter by minimum volume and liquidity
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
        Fetch active markets from Polymarket.com Gamma API.
        Returns markets sorted by volume (highest first).
        """
        await self._ensure_session()

        limit = limit or config.MAX_MARKETS_TO_SCAN
        min_volume = min_volume or config.MIN_VOLUME
        min_liquidity = min_liquidity or config.MIN_LIQUIDITY

        markets = []
        offset = 0
        batch_size = 100

        while len(markets) < limit:
            try:
                params = {
                    "active": "true",
                    "closed": "false",
                    "limit": batch_size,
                    "offset": offset,
                    "order": "volume24hr",
                    "ascending": "false",
                }
                async with self._session.get(
                    f"{config.GAMMA_API_BASE}/markets",
                    params=params,
                    timeout=aiohttp.ClientTimeout(total=30),
                ) as resp:
                    if resp.status != 200:
                        logger.error(f"Gamma API error {resp.status}")
                        break
                    items = await resp.json()

                if not items:
                    break

                for data in items:
                    market = self._parse_market(data, min_volume, min_liquidity)
                    if market is not None:
                        markets.append(market)

                offset += batch_size
                if len(items) < batch_size:
                    break

            except Exception as e:
                logger.error(f"Error fetching markets: {e}")
                break

        logger.info(f"Fetched {len(markets)} markets meeting criteria")
        return markets[:limit]

    def _parse_market(self, data: dict, min_volume: float, min_liquidity: float) -> Optional[Market]:
        """Parse a market from Gamma API response."""
        try:
            # Skip closed/resolved markets
            if data.get("closed") or not data.get("active"):
                return None

            # Must have token data
            tokens = data.get("tokens", [])
            if len(tokens) < 2:
                return None

            # Find YES and NO tokens
            yes_token = next((t for t in tokens if t.get("outcome", "").upper() == "YES"), tokens[0])
            no_token = next((t for t in tokens if t.get("outcome", "").upper() == "NO"), tokens[1])

            yes_price = float(yes_token.get("price", 0) or 0)
            no_price = float(no_token.get("price", 0) or 0)

            # Skip unpriced or near-resolved markets
            if yes_price <= 0.02 or yes_price >= 0.98:
                return None

            volume = float(data.get("volume", 0) or data.get("volume24hr", 0) or 0)
            liquidity = float(data.get("liquidity", 0) or 0)

            if volume < min_volume or liquidity < min_liquidity:
                return None

            slug = data.get("slug", "")
            condition_id = data.get("conditionId", slug)

            return Market(
                condition_id=condition_id,
                question=data.get("question", ""),
                slug=slug,
                yes_price=yes_price,
                no_price=no_price,
                yes_token_id=yes_token.get("token_id", ""),
                no_token_id=no_token.get("token_id", ""),
                volume=volume,
                liquidity=liquidity,
                end_date=data.get("endDate"),
                category=data.get("category", ""),
                description=data.get("description", ""),
                url=f"https://polymarket.com/event/{slug}",
            )
        except (ValueError, KeyError, TypeError) as e:
            logger.debug(f"Skipping malformed market: {e}")
            return None

    async def fetch_single_market(self, condition_id: str) -> Optional[Market]:
        """Fetch a single market by condition ID."""
        await self._ensure_session()
        try:
            async with self._session.get(
                f"{config.GAMMA_API_BASE}/markets",
                params={"conditionId": condition_id},
                timeout=aiohttp.ClientTimeout(total=15),
            ) as resp:
                if resp.status != 200:
                    return None
                items = await resp.json()
                if items:
                    return self._parse_market(items[0], 0, 0)
        except Exception as e:
            logger.error(f"Error fetching market {condition_id}: {e}")
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
