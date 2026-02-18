"""
Gymnasium environment for Nasdaq-100 stock-selection via RL.

Action: continuous scores per stock  → top-K equal-weight portfolio.
Reward: daily portfolio return  – cost – turnover penalty – drawdown penalty.
Observation: per-stock features (pre-scaled) + portfolio state.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces


class NasdaqStockSelectionEnv(gym.Env):
    """
    RL environment for daily top-K stock selection.

    Parameters
    ----------
    features       : np.ndarray   (n_steps, n_tickers * n_features)  pre-scaled
    forward_returns: np.ndarray   (n_steps, n_tickers)
    bench_returns  : np.ndarray   (n_steps,)
    dates          : list         length n_steps
    tickers        : list[str]    ordered ticker list
    K              : int          number of stocks to hold
    alpha          : float        EWMA smoothing  (0 = no smoothing)
    cost_rate      : float        proportional transaction cost
    cash_buffer    : float        fraction of portfolio always in cash
    max_weight     : float        max weight per name
    max_trade_frac : float        max total turnover per step (fraction of NAV)
    lambda_turnover: float        turnover penalty coefficient
    lambda_dd      : float        drawdown-increase penalty coefficient
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        features: np.ndarray,
        forward_returns: np.ndarray,
        bench_returns: np.ndarray,
        dates: list,
        tickers: list[str],
        K: int = 20,
        alpha: float = 0.10,
        cost_rate: float = 0.001,
        cash_buffer: float = 0.02,
        max_weight: float = 0.10,
        max_trade_frac: float = 0.20,
        lambda_turnover: float = 0.01,
        lambda_dd: float = 0.001,
    ):
        super().__init__()

        self.features = np.asarray(features, dtype=np.float32)
        self.forward_returns = np.asarray(forward_returns, dtype=np.float64)
        self.bench_returns = np.asarray(bench_returns, dtype=np.float64)
        self.dates = list(dates)
        self.tickers = list(tickers)
        self.n_stocks = len(tickers)
        self.n_steps = len(self.features)

        self.K = K
        self.alpha = alpha
        self.cost_rate = cost_rate
        self.cash_buffer = cash_buffer
        self.max_weight = max_weight
        self.max_trade_frac = max_trade_frac
        self.lambda_turnover = lambda_turnover
        self.lambda_dd = lambda_dd

        # Observation: per-stock features + prev_weights + cash_weight + drawdown
        feat_dim = self.features.shape[1]
        self.obs_dim = feat_dim + self.n_stocks + 2  # +cash +dd

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=-10.0, high=10.0, shape=(self.n_stocks,), dtype=np.float32,
        )

        # Runtime state (set in reset)
        self.current_step: int = 0
        self.weights = np.zeros(self.n_stocks, dtype=np.float64)
        self.cash_weight: float = 1.0
        self.value: float = 1.0
        self.peak_value: float = 1.0
        self.drawdown: float = 0.0

    # ------------------------------------------------------------------
    # Portfolio construction
    # ------------------------------------------------------------------
    def _scores_to_target_weights(self, scores: np.ndarray) -> np.ndarray:
        """Convert raw scores → top-K equal-weight target (before smoothing)."""
        idx = np.argsort(scores)[::-1][:self.K]
        w = np.zeros(self.n_stocks, dtype=np.float64)
        w[idx] = 1.0 / self.K
        # Clip to max_weight and renormalise
        w = np.minimum(w, self.max_weight)
        s = w.sum()
        if s > 0:
            w /= s
        else:
            w = np.ones(self.n_stocks, dtype=np.float64) / self.n_stocks
        # Scale down for cash buffer
        w *= (1.0 - self.cash_buffer)
        return w

    def _apply_smoothing(self, target_w: np.ndarray) -> np.ndarray:
        """EWMA smooth toward target weights."""
        return (1.0 - self.alpha) * self.weights + self.alpha * target_w

    def _apply_trade_limit(self, new_w: np.ndarray) -> np.ndarray:
        """Cap total turnover per step at max_trade_frac of NAV."""
        delta = new_w - self.weights
        total_trade = np.sum(np.abs(delta))
        if total_trade > self.max_trade_frac:
            scale = self.max_trade_frac / total_trade
            new_w = self.weights + delta * scale
            # Ensure non-negative (long-only)
            new_w = np.maximum(new_w, 0.0)
        return new_w

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------
    def _build_obs(self) -> np.ndarray:
        feat = self.features[min(self.current_step, self.n_steps - 1)]
        port_state = np.concatenate([
            self.weights.astype(np.float32),
            np.array([self.cash_weight, self.drawdown], dtype=np.float32),
        ])
        return np.concatenate([feat, port_state])

    # ------------------------------------------------------------------
    # Gym API
    # ------------------------------------------------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.value = 1.0
        self.peak_value = 1.0
        self.drawdown = 0.0

        # Start with equal weight across all stocks
        investable = 1.0 - self.cash_buffer
        self.weights = np.full(self.n_stocks, investable / self.n_stocks, dtype=np.float64)
        self.cash_weight = self.cash_buffer

        return self._build_obs(), {}

    def step(self, action):
        scores = np.asarray(action, dtype=np.float64).flatten()

        # 1) Target weights from scores
        target_w = self._scores_to_target_weights(scores)

        # 2) EWMA smoothing
        new_w = self._apply_smoothing(target_w)

        # 3) Trade-size limit
        new_w = self._apply_trade_limit(new_w)

        # Ensure long-only, sum ≤ 1
        new_w = np.maximum(new_w, 0.0)
        total = new_w.sum()
        if total > (1.0 - self.cash_buffer):
            new_w *= (1.0 - self.cash_buffer) / total
        self.cash_weight = 1.0 - new_w.sum()

        # 4) Turnover
        turnover = 0.5 * float(np.sum(np.abs(new_w - self.weights)))
        cost = self.cost_rate * 2.0 * turnover  # turnover = half-turn; cost on full

        # 5) Portfolio return
        ret_vec = self.forward_returns[self.current_step]
        port_ret_gross = float(np.dot(new_w, ret_vec))  # cash earns 0
        port_ret_net = port_ret_gross - cost
        bench_ret = float(self.bench_returns[self.current_step])

        # 6) Update value / drawdown
        self.value *= (1.0 + port_ret_net)
        self.peak_value = max(self.peak_value, self.value)
        prev_dd = self.drawdown
        self.drawdown = 1.0 - self.value / self.peak_value
        dd_increment = max(0.0, self.drawdown - prev_dd)

        # 7) Reward
        reward = (
            port_ret_net
            - self.lambda_turnover * (2.0 * turnover)
            - self.lambda_dd * dd_increment
        )

        # 8) Drift weights by asset returns (for next step's prev_weights)
        asset_growth = new_w * (1.0 + ret_vec)
        total_growth = asset_growth.sum() + self.cash_weight
        if total_growth > 0:
            self.weights = asset_growth / total_growth
            self.cash_weight = self.cash_weight / total_growth
        else:
            self.weights = np.full(self.n_stocks, (1.0 - self.cash_buffer) / self.n_stocks, dtype=np.float64)
            self.cash_weight = self.cash_buffer

        self.current_step += 1
        terminated = self.current_step >= self.n_steps
        truncated = False
        obs = self._build_obs()

        info = {
            "date": self.dates[min(self.current_step - 1, self.n_steps - 1)],
            "value": self.value,
            "weights": new_w.copy(),
            "turnover": 2.0 * turnover,
            "cost": cost,
            "port_ret_gross": port_ret_gross,
            "port_ret_net": port_ret_net,
            "bench_ret": bench_ret,
            "drawdown": self.drawdown,
            "cash_weight": self.cash_weight,
            "n_held": int(np.sum(new_w > 1e-6)),
        }

        return obs, float(reward), terminated, truncated, info
