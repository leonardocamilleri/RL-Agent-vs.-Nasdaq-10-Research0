import numpy as np


def _run_strategy(forward_returns, tickers, cost_rate, target_fn):
    n_periods = len(forward_returns)
    n_assets = len(tickers)
    qqq_idx = tickers.index("QQQ")
    eq_w = np.ones(n_assets) / n_assets
    weights = eq_w.copy()
    value = 1.0
    values = [1.0]
    net_returns_list = []
    turnovers_list = []
    excess_returns_list = []
    for i in range(n_periods):
        new_w = target_fn(i, weights, tickers)
        turnover = np.sum(np.abs(new_w - weights))
        cost = cost_rate * turnover
        gross_ret = np.dot(new_w, forward_returns[i])
        net_ret = gross_ret - cost
        bench_ret = forward_returns[i][qqq_idx]
        value *= 1.0 + net_ret
        values.append(value)
        net_returns_list.append(net_ret)
        turnovers_list.append(turnover)
        excess_returns_list.append(net_ret - bench_ret)
        asset_growth = new_w * (1.0 + forward_returns[i])
        total = asset_growth.sum()
        if total > 0:
            weights = asset_growth / total
        else:
            weights = eq_w.copy()
    return (
        np.array(values),
        np.array(net_returns_list),
        np.array(turnovers_list),
        np.array(excess_returns_list),
    )


def buy_and_hold_qqq(forward_returns, tickers, cost_rate=0.001):
    qqq_idx = tickers.index("QQQ")
    n_assets = len(tickers)
    target = np.zeros(n_assets)
    target[qqq_idx] = 1.0

    def target_fn(i, weights, _tickers):
        if i == 0:
            return target.copy()
        return weights.copy()

    return _run_strategy(forward_returns, tickers, cost_rate, target_fn)


def equal_weight(forward_returns, tickers, cost_rate=0.001):
    n_assets = len(tickers)
    target = np.ones(n_assets) / n_assets

    def target_fn(i, weights, _tickers):
        return target.copy()

    return _run_strategy(forward_returns, tickers, cost_rate, target_fn)


def fixed_qqq_plus_sectors_equal(forward_returns, tickers, cost_rate=0.001):
    n_assets = len(tickers)
    qqq_idx = tickers.index("QQQ")
    n_sectors = n_assets - 1
    target = np.zeros(n_assets)
    target[qqq_idx] = 0.5
    for j in range(n_assets):
        if j != qqq_idx:
            target[j] = 0.5 / n_sectors

    def target_fn(i, weights, _tickers):
        return target.copy()

    return _run_strategy(forward_returns, tickers, cost_rate, target_fn)
