"""
Multi-Market Scanner for Kalshi.com.

Fetches active markets via the Kalshi REST API,
filters by volume/liquidity, and ranks by estimated edge.
"""
import asyncio
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
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
    1. Fetch active markets from Kalshi API (authenticated)
    2. Filter by minimum volume and open interest
    3. Return sorted by volume (most liquid first)
    """

    def __init__(self, sign_request=None):
        self._session: Optional[aiohttp.ClientSession] = None
        self._sign_request = sign_request  # executor's signing method

    async def _ensure_session(self):
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()

    # Categories — Financials and Crypto first for same-day markets
    TARGET_CATEGORIES = [
        "Financials", "Crypto", "Economics",
        "Politics", "World", "Science", "Climate and weather",
    ]

    async def fetch_markets(
        self,
        limit: int = None,
        min_volume: float = None,
        min_liquidity: float = None,
    ) -> list[Market]:
        """
        Fetch active markets from Kalshi via the events API.
        Filters by category so we only get politics/economics/crypto markets.
        """
        await self._ensure_session()

        limit = limit or config.MAX_MARKETS_TO_SCAN
        min_volume = min_volume or config.MIN_VOLUME
        min_liquidity = min_liquidity or config.MIN_LIQUIDITY

        markets = []

        for category in self.TARGET_CATEGORIES:
            try:
                path = "/trade-api/v2/events"
                headers = self._sign_request("GET", path) if self._sign_request else {}
                params = {"status": "open", "category": category, "limit": 100, "with_nested_markets": "true"}

                async with self._session.get(
                    f"{config.KALSHI_API_BASE}/events",
                    params=params,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=30),
                ) as resp:
                    if resp.status == 429:
                        await asyncio.sleep(4)
                        continue
                    if resp.status != 200:
                        logger.warning(f"Events API error {resp.status} for category {category}: {await resp.text()}")
                        continue
                    data = await resp.json()

                events = data.get("events", [])
                logger.info(f"Category '{category}': {len(events)} events")

                for event in events:
                    for mkt in event.get("markets", []):
                        market = self._parse_market(mkt, min_volume, min_liquidity)
                        if market is not None:
                            markets.append(market)

                await asyncio.sleep(0.3)  # polite pause between categories

            except Exception as e:
                logger.error(f"Error fetching category {category}: {e}")
                continue

        # Deduplicate by question text (same event can appear multiple times)
        seen = set()
        unique = []
        for m in markets:
            key = m.question.strip().lower()
            if key not in seen:
                seen.add(key)
                unique.append(m)
        markets = unique

        # Sort by soonest expiry first so bot targets near-term markets
        def expiry_key(m):
            try:
                return datetime.fromisoformat(m.end_date.replace("Z", "+00:00"))
            except Exception:
                return datetime.max.replace(tzinfo=timezone.utc)
        markets.sort(key=expiry_key)
        logger.info(f"Fetched {len(markets)} unique markets. Rejections: {self._debug_counts}")
        self._debug_counts = {}
        return markets[:limit]

    # Track filter rejection reasons across all markets in a scan
    _debug_counts: dict = {}

    def _parse_market(self, data: dict, min_volume: float, min_liquidity: float) -> Optional[Market]:
        """Parse a Kalshi market entry."""
        try:
            if data.get("status") != "active":
                self._debug_counts["not_active"] = self._debug_counts.get("not_active", 0) + 1
                return None

            ticker = data.get("ticker", "")
            if not ticker:
                return None

            # Prices returned as dollar strings e.g. "0.6500" (already 0-1 scale)
            yes_ask = float(data.get("yes_ask_dollars", 0) or 0)
            yes_bid = float(data.get("yes_bid_dollars", 0) or 0)
            last_price = float(data.get("last_price_dollars", 0) or 0)

            # Use mid-price or last trade price
            if yes_ask > 0 and yes_bid > 0:
                yes_price = (yes_ask + yes_bid) / 2
            elif last_price > 0:
                yes_price = last_price
            else:
                self._debug_counts["no_price"] = self._debug_counts.get("no_price", 0) + 1
                return None

            # Skip near-resolved markets
            if yes_price <= 0.03 or yes_price >= 0.97:
                self._debug_counts["near_resolved"] = self._debug_counts.get("near_resolved", 0) + 1
                return None


            no_price = 1 - yes_price

            # Volume and liquidity already in dollars (fp = fixed point string)
            volume = float(data.get("volume_24h_fp", 0) or 0)
            liquidity = float(data.get("open_interest_fp", 0) or 0)

            if volume < min_volume or liquidity < min_liquidity:
                self._debug_counts["low_volume"] = self._debug_counts.get("low_volume", 0) + 1
                return None

            title = data.get("title", "") or data.get("subtitle", "") or ticker
            event_ticker = data.get("event_ticker", "")
            category = data.get("category", "") or ""

            # ── Sports filter ────────────────────────────────────
            if category.lower().strip() in config.EXCLUDED_CATEGORIES:
                self._debug_counts["sports_category"] = self._debug_counts.get("sports_category", 0) + 1
                return None
            title_lower = title.lower()
            ticker_lower = ticker.lower()
            if any(kw in title_lower or kw in ticker_lower for kw in config.EXCLUDED_KEYWORDS):
                self._debug_counts["sports_keyword"] = self._debug_counts.get("sports_keyword", 0) + 1
                return None
            # Kalshi Exchange (KX prefix) is almost entirely sports.
            # Whitelist only KX crypto/finance tickers we want; block all other KX.
            KX_ALLOWED = ("kxbtc", "kxeth", "kxsol", "kxspy", "kxqqq", "kxgold", "kxoil")
            if ticker_lower.startswith("kx") and not any(ticker_lower.startswith(a) for a in KX_ALLOWED):
                self._debug_counts["kx_sports"] = self._debug_counts.get("kx_sports", 0) + 1
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
