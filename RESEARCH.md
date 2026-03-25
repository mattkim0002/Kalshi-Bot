# Polymarket AI Trading — Complete Research & Analysis

## Table of Contents
1. [The Viral Experiment: Claude $1K → $14K in 48 Hours](#the-viral-experiment)
2. [How Polymarket Works](#how-polymarket-works)
3. [LMSR — The Math Behind Market Pricing](#lmsr)
4. [Kelly Criterion — Optimal Position Sizing](#kelly-criterion)
5. [Expected Value — Finding Edge](#expected-value)
6. [Price Impact & Slippage](#price-impact)
7. [Cognitive Biases That Destroy Traders](#cognitive-biases)
8. [How the Claude AI Agent Actually Worked](#how-claude-agent-worked)
9. [The Complete Decision System](#complete-system)
10. [Risks & Caveats](#risks)

---

## The Viral Experiment

Around March 10, 2026, someone ran an experiment where two autonomous AI agents were each given **$1,000 and 48 hours** to trade on Polymarket — a decentralized prediction market — with zero human intervention.

**Results:**
| Agent | Starting Capital | Ending Capital | Return | Trades |
|-------|-----------------|----------------|--------|--------|
| **Claude (Anthropic)** | $1,000 | ~$14,200 | **+1,322%** | 5,200+ |
| **OpenClaw** | $1,000 | $0 | **-100%** | <200 |

The post on X garnered over 1.2 million views. The Claude agent executed over 5,200 trades (~1 every 33 seconds) and even covered its own API costs from profits.

**Why Claude won:** Conservative pattern recognition, disciplined risk management, small diversified bets across many uncorrelated markets.

**Why OpenClaw lost:** Aggressive momentum chasing, concentrated bets, no stop-losses, 94% drawdown before total liquidation.

**Important caveat:** No independently auditable trade records were published. Reddit discussions flagged skepticism about potential fabrication. The stat that "87% of Polymarket wallets lose money" is a grounding reality check.

---

## How Polymarket Works

Polymarket is a prediction market where you trade on the outcomes of real-world events. 

- Every share is priced between **$0.00 and $1.00**
- The price represents the market's belief in the probability of an outcome
- If "Yes" shares trade at $0.35, the market thinks there's a 35% chance
- If the event occurs, your $0.35 share pays out **$1.00** (profit: $0.65)
- If it doesn't, you lose your $0.35

**Key mechanics:**
- Uses USDC.e (stablecoin) on the Polygon blockchain
- Peer-to-peer order book (CLOB) — you trade against other users, not a house
- YES + NO always sums to $1.00
- Shares are ERC1155 tokens settled via smart contracts

**The edge opportunity:** If you believe the TRUE probability of an event is higher than what the market price reflects, you have mathematical edge. This is what AI agents exploit.

---

## LMSR

### The Logarithmic Market Scoring Rule

Invented by computer scientist Robin Hanson in 2002. Instead of matching buyers and sellers, LMSR maintains a **cost function** — you trade against pure math.

### The Cost Function

```
C(q) = b × ln(Σ e^(qi / b))
```

Where:
- **C(q)** = The total "cost state" of the market (a running tab)
- **q** = Vector of outstanding shares (qi = shares for outcome i)
- **b** = Liquidity parameter (controls market depth)
- **n** = Number of possible outcomes
- **e** = Euler's number (~2.718)

### Prices as Probabilities (Softmax)

The instantaneous price for outcome k is the partial derivative of the cost function:

```
p_k(q) = e^(qk/b) / Σ e^(qi/b)
```

This is the **softmax function** — the exact same math used in the final layer of neural networks. Polymarket prices beliefs the same way GPT outputs token probabilities.

Properties:
- Prices always satisfy p_k ∈ (0, 1)
- All prices sum to 1: Σ p_k = 1
- Prices behave exactly like probabilities

### Trade Cost

If you buy Δ shares of outcome k, you pay:

```
TradeCost = C(q + Δ·e_k) - C(q)
```

This is elegant: the market doesn't need a counterparty. The formula IS the market maker.

### Worked Example

Market with 3 outcomes. q = [10, 20, 23], b = 10.

**Price of outcome 1:**
```
p_1 = e^(10/10) / (e^(10/10) + e^(20/10) + e^(23/10)) = 0.1354
```
That's a 13.5% implied probability.

**Cost to buy 7 shares of outcome 1** (q₁: 10 → 17):
```
Cost before: 10 × ln(e¹ + e² + e²·³) = 29.998
Cost after:  10 × ln(e¹·⁷ + e² + e²·³) = 31.284
You pay: 31.284 - 29.998 = $1.29
```

### The Liquidity Parameter (b)

- **Small b** (e.g., 50): Prices move fast. One whale swings the market 20%.
- **Large b** (e.g., 100,000): Prices barely move. Requires serious capital.

**Maximum market maker loss:** `L_max = b × ln(n)`

For a binary market with b = 100,000: ~$69,315. That's the subsidy the platform provides for truth discovery.

### Python Implementation

```python
import numpy as np

def lmsr_cost(q, b):
    q = np.asarray(q, dtype=np.float64)
    q_scaled = q / b
    max_q = np.max(q_scaled)
    return b * (max_q + np.log(np.sum(np.exp(q_scaled - max_q))))

def lmsr_price(q, b, outcome):
    q = np.asarray(q, dtype=np.float64)
    q_scaled = q / b
    max_q = np.max(q_scaled)
    exp_q = np.exp(q_scaled - max_q)
    return float(exp_q[outcome] / np.sum(exp_q))
```

---

## Kelly Criterion

### The Formula

The Kelly criterion determines the optimal fraction of your bankroll to bet to maximize long-term growth while minimizing ruin risk.

For a prediction market with price **p** (market price) and belief **q** (your estimated probability):

**Standard form:**
```
f* = (p_win × b - q_lose) / b
```

Where:
- f* = Optimal fraction of bankroll to bet
- p_win = Your estimated probability of winning
- q_lose = 1 - p_win
- b = Payout ratio = (1/market_price) - 1

**Prediction market form (from the Kelly/LMSR paper):**
```
f = q - p(1-q)/(1-p)
```

Where p = market price, q = your belief.

### Worked Example

Market price p = $0.40 (market says 40%). Your estimate q = 0.65 (you think 65%).

```
Payout ratio b = (1/0.40) - 1 = 1.5
f* = (0.65 × 1.5 - 0.35) / 1.5 = 0.417
```

Full Kelly says bet 41.7% of bankroll. **But full Kelly is way too aggressive.**

### Fractional Kelly

50 years of real-world trading has shown: use **quarter-Kelly to half-Kelly** (0.25x - 0.5x).

```
f_actual = f* × 0.25 = 0.417 × 0.25 = 0.104
```

So on a $1,000 bankroll: bet ~$104 on this trade.

### Edge Condition

When f* ≤ 0, there is **no edge**. Do not trade.

```python
def kelly_fraction(true_prob, market_price):
    if market_price <= 0 or market_price >= 1:
        return 0.0
    payout_ratio = (1.0 / market_price) - 1.0
    f = (true_prob * payout_ratio - (1 - true_prob)) / payout_ratio
    return max(f, 0.0)
```

---

## Expected Value

### The Core Formula

```
EV = Σ(probability × payoff)
```

### Applied to Polymarket

Market at $0.40. Your estimated true probability: 60%.

```
EV = 0.60 × (1 - 0.40) - 0.40 × (1 - 0.60)
   = 0.60 × 0.60 - 0.40 × 0.40
   = 0.36 - 0.16
   = +$0.20 per dollar risked
```

20 cents of edge per dollar. In traditional finance, people build careers on 2-cent edges.

### The Compounding Effect

With an average edge of ~2% per trade and 5,200 trades:
```
$1,000 × (1.005)^500 ≈ $12,180
```

Even tiny per-trade returns compound explosively across thousands of bets on uncorrelated events.

---

## Price Impact

### Why It Matters

Your own buying moves the price against you. Buy 1,000 shares of YES and the price isn't $0.40 anymore — it's $0.55, $0.60, climbing with every share.

The LMSR cost function is **convex**: larger trades move prices more. This is called **slippage**.

### Measuring Impact

```python
def simulate_buy_impact(initial_q, shares, b, outcome=0, steps=20):
    q = list(initial_q)
    total_cost = 0
    shares_per_step = shares / steps
    
    for _ in range(steps):
        q_after = q.copy()
        q_after[outcome] += shares_per_step
        step_cost = lmsr_cost(q_after, b) - lmsr_cost(q, b)
        total_cost += step_cost
        q = q_after
    
    avg_fill = total_cost / shares
    return avg_fill, lmsr_price(q, b, outcome)
```

### Why 5,200 Small Trades

This is why the Claude agent made 5,200 small trades instead of a few big ones — to avoid eating its own edge through slippage. Small bets across many markets keep the average fill close to the quoted price.

**Rule of thumb:** If slippage eats more than 50% of your edge, reduce position size.

---

## Cognitive Biases

The 5 mental traps that destroy Polymarket traders (from extensive market analysis):

### 1. Base Rate Neglect
A 99% accurate test on a 1/1000 disease gives only a ~9% true positive rate, not 99%. On Polymarket: markets at 85¢ resolve NO roughly 15% of the time. That's every 6-7 contracts.

**Fix:** Always ask: what's the base rate?

### 2. Sunk Cost Fallacy
You bought YES at $0.70. Price drops to $0.40. New info says NO is right. Most traders hold. The market doesn't know your entry price. The only question: would you buy YES at $0.40 right now? If no, sell.

**Fix:** Evaluate positions as if you held cash, not shares.

### 3. Survivorship Bias
The $500 → $50K screenshot gets 200K likes. The 500 people who lost everything deleted their accounts. **87% of Polymarket wallets lose money.**

**Fix:** Look for the denominator. How many tried? What fraction succeeded?

### 4. Bad Bayesian Updating
Bayes' theorem:
```
P(belief | evidence) = P(evidence | belief) × P(belief) / P(evidence)
```
Start with a prior. Update proportionally to evidence. Don't overreact to one poll. Don't ignore ten.

**Fix:** The best traders update fastest — not the ones who are right from the start.

### 5. Anti-Kelly Position Sizing
Most people either YOLO everything or bet tiny amounts. Both are wrong. Kelly criterion gives the mathematically optimal size. Full Kelly is too aggressive; quarter-Kelly is the sweet spot.

**Fix:** Calculate, don't feel.

---

## How Claude Agent Worked

### The Architecture

```
Loop every 10 minutes:
1. SCAN: Fetch 500-1000 active markets via Polymarket Gamma API
2. ESTIMATE: Send each question to Claude with web search enabled
3. COMPARE: Claude's probability vs. market price = edge
4. FILTER: Only trade if edge > threshold (e.g., 5-8%)
5. SIZE: Kelly criterion with fractional scaling (quarter-Kelly, capped at 6%)
6. EXECUTE: Place Fill-or-Kill order via CLOB API
7. MONITOR: Re-evaluate positions, enforce stop-loss/take-profit
```

### Claude's Role as Probability Estimator

Claude receives each market question with:
- The question text and description
- Current market price
- Access to web search (up to 3 searches per question)

Claude returns:
- **Probability estimate** (0.0 to 1.0)
- **Confidence level** (Low / Medium / High)
- **Reasoning** (2-4 sentences)
- **Key factors** (top 3 influences)

### Why Claude Had an Edge

1. **Web search access:** Could pull real-time data (news, weather, sports, official statements)
2. **Reasoning at scale:** Could analyze hundreds of markets that humans skip
3. **No emotional bias:** No loss aversion, no sunk cost attachment
4. **Speed:** One analysis every ~10 seconds vs. minutes/hours for a human
5. **Calibration:** Language models can be surprisingly well-calibrated on probability estimates

### Key Parameters Used

| Parameter | Value | Purpose |
|-----------|-------|---------|
| Mispricing threshold | >8% | Only trade meaningful edge |
| Kelly fraction | 0.25 | Quarter-Kelly — conservative |
| Position cap | 6% of bankroll | No single trade too large |
| Scan frequency | Every 10 min | Catch new opportunities |
| Web searches | 3 per question | Balance cost vs. accuracy |
| Confidence minimum | Medium | Skip low-confidence estimates |

---

## Complete System

These aren't random tricks. They form a complete decision-making pipeline:

| Component | Question It Answers |
|-----------|-------------------|
| **LMSR** | How does the market mechanically set prices? |
| **Expected Value** | Is this trade worth making? |
| **Base Rates** | What's the ground truth frequency? |
| **Bayesian Updating** | How do I revise my beliefs with new data? |
| **Survivorship Bias** | What am I NOT seeing in the data? |
| **Sunk Cost Fallacy** | What should I ignore when cutting losses? |
| **Kelly Criterion** | How much should I risk? |
| **Price Impact** | How does my own trading degrade my edge? |

The top Polymarket traders all think this way — not because they read the same books, but because the market beat it into them through losses.

---

## Risks

### Unverified Claims
The $14K result has no auditable trade records. Dashboards can be faked. Treat it as directionally interesting, not as proof.

### Survivorship Bias in the Experiment Itself
We hear about the agent that made 1,322%. We don't hear about the hundreds of similar experiments that lost money.

### Market Saturation
14 of the top 20 Polymarket traders are already bots. As more AI agents enter, mispricings get smaller and edge disappears.

### Scale Limitations
$1,000 can find mispriced edges that $1M would instantly erase through market impact. This doesn't scale linearly.

### Claude Is Not Infallible
Its probability estimates can be wrong. The Kelly sizing and stop-losses are your safety net — they limit damage when the model is wrong.

### API Costs
5,200 trades in 48 hours = thousands of Claude API calls with web search. That's real spend on Anthropic credits.

### Legal
Prediction market trading may have regulatory implications depending on your jurisdiction. Polymarket is not regulated by the CFTC.

### The Uncomfortable Truth
> "You don't feel irrational when you're being irrational. You feel certain. The certainty IS the bug."

---

## Sources

- Robin Hanson, "Logarithmic Market Scoring Rules for Modular Combinatorial Information Aggregation" (2002)
- Bernhard K. Meister, "Application of the Kelly Criterion to Prediction Markets" (arXiv:2412.14144, 2024)
- Polymarket Documentation: https://docs.polymarket.com
- Gensyn LMSR Implementation: https://blog.gensyn.ai/lmsr-logarithmic-market-scoring-rule/
- Cultivate Labs LMSR Guide: https://www.cultivatelabs.com/crowdsourced-forecasting-guide/how-does-logarithmic-market-scoring-rule-lmsr-work
- Phemex Analysis: https://phemex.com/news/article/ai-trading-agent-claude-achieves-1322-return-on-polymarket-65634
- Robot Traders Tutorial: https://robottraders.io/blog/polymarket-ai-bot-claude-python
