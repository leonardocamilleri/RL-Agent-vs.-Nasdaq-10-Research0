# Reinforcement Learning for Dynamic Equity Allocation
### Research Project – Nasdaq-100 Stock Selection

This project explores the use of deep reinforcement learning to dynamically select and allocate capital across stocks in the Nasdaq-100 universe.

The objective is to investigate whether an RL agent can improve risk-adjusted returns relative to a passive benchmark by adapting to changing market regimes.

---

## Methodology

The system is built around:

- Multi-horizon momentum and volatility features
- Custom Gym-based stock selection environment
- Proximal Policy Optimization (PPO)
- Risk-adjusted reward formulation
- Out-of-sample evaluation

The agent learns to allocate capital across a subset of stocks while incorporating turnover constraints and volatility considerations.

---

## Key Components

- Feature engineering pipeline
- Custom reinforcement learning environment
- PPO training framework (Stable Baselines3)
- Backtesting and performance evaluation tools

---

## Research Questions

- Can reinforcement learning adapt to structural market shifts?
- Does dynamic stock selection improve Sharpe ratio vs passive exposure?
- How sensitive is performance to feature design and reward structure?

---

## Status

This repository contains the research implementation used for experimentation and backtesting.

Execution and live trading infrastructure are intentionally excluded.
