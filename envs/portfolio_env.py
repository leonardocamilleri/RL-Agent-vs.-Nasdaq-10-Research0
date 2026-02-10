import numpy as np
import gymnasium as gym
from gymnasium import spaces


class PortfolioEnv(gym.Env):

    metadata = {"render_modes": []}

    def __init__(self, features, forward_returns, dates, qqq_index=0,
                 cost_rate=0.001, alpha=0.10):
        super().__init__()
        self.features = np.array(features, dtype=np.float32)
        self.forward_returns = np.array(forward_returns, dtype=np.float64)
        self.dates = list(dates)
        self.qqq_index = qqq_index
        self.cost_rate = cost_rate
        self.alpha = alpha
        self.n_assets = self.forward_returns.shape[1]
        self.n_steps = len(self.features)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.features.shape[1],), dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0,
            shape=(self.n_assets,), dtype=np.float32,
        )

    def _softmax(self, x):
        e = np.exp(x - np.max(x))
        return e / e.sum()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.weights = np.ones(self.n_assets, dtype=np.float64) / self.n_assets
        self.value = 1.0
        return self.features[0].copy(), {}

    def step(self, action):
        raw = np.asarray(action, dtype=np.float64).flatten()
        target_weights = self._softmax(raw)
        new_weights = (1.0 - self.alpha) * self.weights + self.alpha * target_weights

        turnover = float(np.sum(np.abs(new_weights - self.weights)))
        cost = self.cost_rate * turnover

        ret_next = self.forward_returns[self.current_step]
        bench_ret = float(ret_next[self.qqq_index])
        port_ret_gross = float(np.dot(new_weights, ret_next))
        port_ret_net = max(port_ret_gross - cost, -0.999)

        self.value *= 1.0 + port_ret_net

        reward = float(np.log1p(port_ret_net) - np.log1p(bench_ret))

        asset_growth = new_weights * (1.0 + ret_next)
        total_growth = asset_growth.sum()
        if total_growth > 0:
            self.weights = asset_growth / total_growth
        else:
            self.weights = np.ones(self.n_assets, dtype=np.float64) / self.n_assets

        info = {
            "date": self.dates[self.current_step],
            "value": self.value,
            "turnover": turnover,
            "cost": cost,
            "port_ret_gross": port_ret_gross,
            "port_ret_net": port_ret_net,
            "bench_ret": bench_ret,
            "weights": new_weights.copy(),
        }

        self.current_step += 1
        terminated = self.current_step >= self.n_steps
        truncated = False
        obs = self.features[min(self.current_step, self.n_steps - 1)].copy()
        return obs, reward, terminated, truncated, info
