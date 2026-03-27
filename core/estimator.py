"""
Claude AI Probability Estimator.

Sends Polymarket questions to Claude with web search enabled,
receives structured probability estimates and confidence levels.
"""
import asyncio
import aiohttp
import json
import logging
import time
from typing import Optional

import config
from core.scanner import Market

logger = logging.getLogger(__name__)


# Tool definition forces Claude to return structured output
ANALYSIS_TOOL = {
    "name": "submit_prediction",
    "description": "Submit a structured prediction for a prediction market question.",
    "input_schema": {
        "type": "object",
        "properties": {
            "probability": {
                "type": "number",
                "description": (
                    "Your estimated probability that YES is the correct outcome, "
                    "between 0.0 and 1.0. Be precise — e.g., 0.72, not just 0.7."
                ),
            },
            "confidence": {
                "type": "string",
                "enum": ["Low", "Medium", "High"],
                "description": (
                    "How confident you are in this estimate. "
                    "Low = speculative/limited info. "
                    "Medium = reasonable basis but some uncertainty. "
                    "High = strong evidence supports this estimate."
                ),
            },
            "reasoning": {
                "type": "string",
                "description": (
                    "2-4 sentence explanation of your reasoning, including "
                    "key evidence and any significant uncertainty."
                ),
            },
            "key_factors": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Top 3 factors that most influence this prediction.",
            },
        },
        "required": ["probability", "confidence", "reasoning", "key_factors"],
    },
}


SYSTEM_PROMPT = """You are a quantitative prediction market analyst. Your job is to estimate
the true probability of events as accurately as possible.

CRITICAL RULES:
1. Do ONE web search to find the latest relevant news before estimating.
2. Consider base rates — how often does this type of event actually happen?
3. Be calibrated: if you say 70%, events like this should happen ~70% of the time.
4. Don't anchor too heavily on the current market price. Think independently.
5. Account for time remaining — events far in the future have more uncertainty.
6. If information is extremely limited, express that through LOW confidence.
7. Be concise — short reasoning only.

You MUST call the submit_prediction tool with your analysis."""


class BtcContext:
    """Fetches live Bitcoin price and trend from CoinGecko (no API key needed)."""

    COINGECKO_URL = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart"
    _cache: dict = {}
    _cache_ttl = 300  # refresh every 5 minutes

    @classmethod
    async def get(cls, session: aiohttp.ClientSession) -> str:
        """Return a short BTC market context string for Claude's prompt."""
        now = time.time()
        if cls._cache and now - cls._cache.get("ts", 0) < cls._cache_ttl:
            return cls._cache["text"]

        try:
            params = {"vs_currency": "usd", "days": "7", "interval": "daily"}
            async with session.get(
                cls.COINGECKO_URL,
                params=params,
                timeout=aiohttp.ClientTimeout(total=10),
            ) as resp:
                if resp.status != 200:
                    return ""
                data = await resp.json()

            prices = [p[1] for p in data.get("prices", [])]
            if len(prices) < 2:
                return ""

            current = prices[-1]
            week_ago = prices[0]
            pct_7d = (current - week_ago) / week_ago * 100
            direction = "up" if pct_7d > 2 else "down" if pct_7d < -2 else "flat"

            text = (
                f"MACRO CONTEXT — Bitcoin is currently ${current:,.0f} "
                f"({pct_7d:+.1f}% over 7 days, trending {direction}). "
                f"Factor this into your analysis where relevant."
            )
            cls._cache = {"ts": now, "text": text}
            logger.info(f"BTC context: {text}")
            return text

        except Exception as e:
            logger.warning(f"Could not fetch BTC context: {e}")
            return ""


class ProbabilityEstimator:
    """
    Uses Claude AI with web search to estimate true probabilities
    for Polymarket questions.
    """

    def __init__(self):
        self.api_key = config.ANTHROPIC_API_KEY
        self.model = config.CLAUDE_MODEL
        self.max_searches = config.WEB_SEARCH_MAX
        self.min_confidence = config.MIN_CONFIDENCE
        self.session: Optional[aiohttp.ClientSession] = None
        self._semaphore = asyncio.Semaphore(config.ESTIMATOR_BATCH_SIZE)  # concurrent Claude calls

    async def _ensure_session(self):
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession()

    async def close(self):
        if self.session and not self.session.closed:
            await self.session.close()

    async def estimate(self, market: Market) -> Market:
        """
        Estimate the true probability for a single market.
        Updates the market object in place and returns it.
        """
        async with self._semaphore:
            try:
                result = await self._call_claude(market)
                if result:
                    market.estimated_prob = result["probability"]
                    market.confidence = result["confidence"]
                    market.reasoning = result["reasoning"]
                    market.edge = market.estimated_prob - market.yes_price
                    
                    logger.info(
                        f"Estimated {market.question[:60]}... → "
                        f"P={market.estimated_prob:.2f} "
                        f"(market={market.yes_price:.2f}, "
                        f"edge={market.edge:+.2f}, "
                        f"conf={market.confidence})"
                    )
                else:
                    logger.warning(f"No estimate for: {market.question[:60]}...")
                    
            except Exception as e:
                logger.error(f"Error estimating {market.question[:60]}...: {e}")

        return market

    async def estimate_batch(
        self,
        markets: list[Market],
        max_concurrent: int = 5,
    ) -> list[Market]:
        """
        Estimate probabilities for multiple markets concurrently.
        All markets are fired in parallel (bounded by semaphore) so total
        time is ~1 Claude call instead of N × stagger_secs.
        """
        logger.info(f"Estimating probabilities for {len(markets)} markets in parallel...")

        tasks = [self.estimate(market) for market in markets]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter out exceptions
        valid = [r for r in results if isinstance(r, Market)]
        successful = [r for r in valid if r.estimated_prob is not None]
        logger.info(f"Successfully estimated {len(successful)}/{len(markets)} markets")

        return valid

    async def _call_claude(self, market: Market) -> Optional[dict]:
        """
        Call Claude API with the market question and web search tool.
        Returns structured prediction or None.
        """
        await self._ensure_session()

        btc_context = await BtcContext.get(self.session)
        user_message = self._build_prompt(market, btc_context)

        # Build tools list
        tools = [ANALYSIS_TOOL]
        if self.max_searches > 0:
            tools.append({
                "type": "web_search_20250305",
                "name": "web_search",
                "max_uses": self.max_searches,
            })

        payload = {
            "model": self.model,
            "max_tokens": 1024,
            "system": SYSTEM_PROMPT,
            "tools": tools,
            "messages": [{"role": "user", "content": user_message}],
        }

        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }

        for attempt in range(3):
            try:
                async with self.session.post(
                    "https://api.anthropic.com/v1/messages",
                    json=payload,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=60),
                ) as resp:
                    if resp.status == 429:
                        wait = 15 * (attempt + 1)
                        logger.warning(f"Rate limited. Waiting {wait}s before retry...")
                        await asyncio.sleep(wait)
                        continue
                    if resp.status != 200:
                        error = await resp.text()
                        logger.error(f"Claude API error {resp.status}: {error[:200]}")
                        return None
                    data = await resp.json()
                    return self._parse_response(data)
            except asyncio.TimeoutError:
                logger.warning(f"Claude API timeout for: {market.question[:60]}...")
                return None
            except Exception as e:
                logger.error(f"Claude API call failed: {e}")
                return None
        return None

    def _build_prompt(self, market: Market, btc_context: str = "") -> str:
        """Build the analysis prompt for Claude."""
        parts = [
            f"Analyze this prediction market question and estimate the TRUE probability.\n",
            f"QUESTION: {market.question}\n",
        ]
        
        if market.description:
            parts.append(f"DESCRIPTION: {market.description[:500]}\n")
        
        parts.extend([
            f"CURRENT MARKET PRICE: YES = {market.yes_price:.2f} ({market.yes_price:.0%}), "
            f"NO = {market.no_price:.2f} ({market.no_price:.0%})",
            f"MARKET VOLUME: ${market.volume:,.0f}",
            f"CATEGORY: {market.category}" if market.category else "",
            f"END DATE: {market.end_date}" if market.end_date else "",
            btc_context if btc_context else "",
            "",
            "Search for the latest news on this topic, then call submit_prediction "
            "with your probability estimate, confidence level, and brief reasoning.",
        ])
        
        return "\n".join(p for p in parts if p)

    def _parse_response(self, data: dict) -> Optional[dict]:
        """Extract the structured prediction from Claude's response."""
        for block in data.get("content", []):
            if block.get("type") == "tool_use" and block.get("name") == "submit_prediction":
                inp = block.get("input", {})
                
                prob = inp.get("probability")
                if prob is None or not (0.0 <= prob <= 1.0):
                    logger.warning(f"Invalid probability: {prob}")
                    return None
                
                return {
                    "probability": float(prob),
                    "confidence": inp.get("confidence", "Low"),
                    "reasoning": inp.get("reasoning", ""),
                    "key_factors": inp.get("key_factors", []),
                }
        
        logger.warning("No submit_prediction tool call found in response")
        return None

    def meets_confidence_threshold(self, market: Market) -> bool:
        """Check if market's confidence meets minimum threshold."""
        if market.confidence is None:
            return False
        return (
            config.CONFIDENCE_MAP.get(market.confidence, 0)
            >= config.CONFIDENCE_MAP.get(self.min_confidence, 2)
        )
