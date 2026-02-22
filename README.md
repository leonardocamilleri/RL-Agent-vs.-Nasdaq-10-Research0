# Time-Varying Reinforcement Learning Agent for Dynamic Factor Investment in Equity Markets
### Research Project – Nasdaq-100 Stock Selection

This project investigates whether deep reinforcement learning can improve risk-adjusted performance through dynamic stock selection within the Nasdaq-100 universe.

Rather than predicting returns directly, the agent learns an allocation policy that adapts to changing market regimes using multi-horizon factor inputs and risk-aware reward shaping.

---

## Research Motivation

Traditional factor models assume relatively stable relationships between signals and returns. Financial markets, however, are non-stationary.

This research explores:

- Can an RL agent adapt dynamically to regime changes?
- Does policy-based allocation improve Sharpe ratio vs passive exposure?
- How sensitive is performance to reward design and feature construction?

---

## Methodology

### Feature Engineering
- Multi-horizon momentum (1d, 5d, 21d, 63d, 252d)
- Rolling volatility measures
- Trend indicators
- Standardized cross-sectional factor inputs

### Environment Design
A custom Gym environment models stock selection:

- Observation: per-stock feature tensor
- Action: top-k allocation selection
- Reward: portfolio return with turnover penalty
- Optional volatility-adjusted objective

### Learning Algorithm
- Proximal Policy Optimization (PPO)
- Stable Baselines3 implementation
- VecNormalize for observation scaling
- Rolling-window training for out-of-sample robustness

---

## Evaluation Framework

Performance is evaluated using:

- CAGR
- Sharpe Ratio
- Sortino Ratio
- Maximum Drawdown
- Turnover metrics
- Benchmark comparison vs QQQ

Out-of-sample testing is emphasized to reduce overfitting risk.

---

## Key Insights

- Reward shaping materially impacts turnover and risk profile.
- Multi-horizon momentum features improve stability.
- Policy learning adapts better during volatility clustering than static allocation rules.

---

## Limitations & Future Work

- Transaction costs and slippage modeling can be refined.
- Regime conditioning may improve robustness.
- Further cross-validation across market cycles is ongoing.

---

## Repository Scope

This repository contains the research implementation used for experimentation and backtesting.

Live execution and brokerage infrastructure are intentionally excluded.
