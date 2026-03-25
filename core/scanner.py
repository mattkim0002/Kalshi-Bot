"""
Multi-Market Scanner.

Fetches all active markets from Polymarket's Gamma API,
filters by volume/liquidity, and ranks by estimated edge.
"""
import asyncio
import aiohttp
import logging
from dataclasses import dataclass, field
from typing import Optional

import config

logger = logging.getLogger(__name__)


@dataclass
class Market:
    """Represents a single Polymarket market."""
    condition_id: str
    question: str
    slug: str
    yes_price: float
    no_price: float
    yes_token_id: str
    no_token_id: str
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
    Scans Polymarket for active markets and identifies candidates.
    
    Flow:
    1. Fetch all active markets from Gamma API
    2. Filter by minimum volume and liquidity
    3. Return sorted by volume (most liquid first)
    """

    def __init__(self):
        self.base_url = config.GAMMA_API_BASE
        self.session: Optional[aiohttp.ClientSession] = None

    async def _ensure_session(self):
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession()

    async def close(self):
        if self.session and not self.session.closed:
            await self.session.close()

    async def fetch_markets(
        self,
        limit: int = None,
        min_volume: float = None,
        min_liquidity: float = None,
    ) -> list[Market]:
        """
        Fetch active markets from the Gamma API.
        
        Returns markets sorted by volume (highest first).
        """
        await self._ensure_session()
        
        limit = limit or config.MAX_MARKETS_TO_SCAN
        min_volume = min_volume or config.MIN_VOLUME
        min_liquidity = min_liquidity or config.MIN_LIQUIDITY
        
        markets = []
        offset = 0
        batch_size = 100  # Gamma API page size

        while len(markets) < limit:
            try:
                params = {
                    "active": "true",
                    "closed": "false",
                    "limit": batch_size,
                    "offset": offset,
                    "order": "volume",
                    "ascending": "false",
                }
                
                async with self.session.get(
                    f"{self.base_url}/markets",
                    params=params,
                    timeout=aiohttp.ClientTimeout(total=30),
                ) as resp:
                    if resp.status != 200:
                        logger.error(f"Gamma API returned {resp.status}")
                        break
                    
                    data = await resp.json()
                    
                    if not data:
                        break
                    
                    for m in data:
                        market = self._parse_market(m)
                        if market is None:
                            continue
                        if market.volume < min_volume:
                            continue
                        if market.liquidity < min_liquidity:
                            continue
                        markets.append(market)
                    
                    offset += batch_size
                    
                    if len(data) < batch_size:
                        break

            except asyncio.TimeoutError:
                logger.warning(f"Timeout fetching markets at offset {offset}")
                break
            except Exception as e:
                logger.error(f"Error fetching markets: {e}")
                break

        logger.info(f"Fetched {len(markets)} markets meeting criteria")
        return markets[:limit]

    def _parse_market(self, data: dict) -> Optional[Market]:
        """Parse a market from Gamma API response."""
        try:
            # Extract token IDs and prices
            tokens = data.get("clobTokenIds", "")
            if isinstance(tokens, str):
                tokens = [t.strip() for t in tokens.split(",") if t.strip()]
            
            prices = data.get("outcomePrices", "")
            if isinstance(prices, str):
                prices = [p.strip() for p in prices.split(",") if p.strip()]
            
            if len(tokens) < 2 or len(prices) < 2:
                return None

            yes_price = float(prices[0])
            no_price = float(prices[1])
            
            # Sanity check
            if yes_price <= 0 or yes_price >= 1:
                return None

            slug = data.get("slug", "") or data.get("marketSlug", "")
            
            return Market(
                condition_id=str(data.get("conditionId", "")),
                question=data.get("question", ""),
                slug=slug,
                yes_price=yes_price,
                no_price=no_price,
                yes_token_id=str(tokens[0]),
                no_token_id=str(tokens[1]),
                volume=float(data.get("volume", 0) or 0),
                liquidity=float(data.get("liquidity", 0) or 0),
                end_date=data.get("endDate"),
                category=data.get("groupItemTitle", "") or data.get("category", ""),
                description=data.get("description", ""),
                url=f"https://polymarket.com/event/{slug}" if slug else "",
            )
        except (ValueError, KeyError, IndexError) as e:
            logger.debug(f"Skipping malformed market: {e}")
            return None

    async def fetch_single_market(self, slug: str) -> Optional[Market]:
        """Fetch a single market by its URL slug."""
        await self._ensure_session()
        
        try:
            # Try events endpoint first
            async with self.session.get(
                f"{self.base_url}/events",
                params={"slug": slug},
                timeout=aiohttp.ClientTimeout(total=15),
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    if data:
                        event = data[0] if isinstance(data, list) else data
                        markets = event.get("markets", [])
                        if markets:
                            return self._parse_market(markets[0])

            # Fallback to markets endpoint
            async with self.session.get(
                f"{self.base_url}/markets",
                params={"slug": slug},
                timeout=aiohttp.ClientTimeout(total=15),
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    if data:
                        m = data[0] if isinstance(data, list) else data
                        return self._parse_market(m)

        except Exception as e:
            logger.error(f"Error fetching market {slug}: {e}")

        return None

    def rank_by_edge(self, markets: list[Market]) -> list[Market]:
        """
        Rank markets by absolute edge (requires estimated_prob to be set).
        Markets without estimates are excluded.
        """
        estimated = [m for m in markets if m.estimated_prob is not None]
        for m in estimated:
            m.edge = m.estimated_prob - m.yes_price
        
        return sorted(estimated, key=lambda m: abs(m.edge or 0), reverse=True)

    def filter_tradeable(
        self,
        markets: list[Market],
        min_edge: float = None,
    ) -> list[Market]:
        """
        Filter to only markets with sufficient edge.
        """
        min_edge = min_edge or config.MIN_EDGE
        return [
            m for m in markets
            if m.edge is not None and abs(m.edge) >= min_edge
        ]
