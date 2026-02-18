import numpy as np


def compute_cagr(values):
    n_months = len(values) - 1
    if n_months <= 0 or values[0] <= 0:
        return 0.0
    return (values[-1] / values[0]) ** (12.0 / n_months) - 1.0


def compute_annual_volatility(monthly_returns):
    if len(monthly_returns) < 2:
        return 0.0
    return float(np.std(monthly_returns, ddof=1) * np.sqrt(12))


def compute_sharpe(monthly_returns):
    if len(monthly_returns) < 2:
        return 0.0
    vol = np.std(monthly_returns, ddof=1)
    if vol < 1e-12:
        return 0.0
    return float(np.mean(monthly_returns) / vol * np.sqrt(12))


def compute_sortino(monthly_returns):
    if len(monthly_returns) < 2:
        return 0.0
    downside = np.minimum(monthly_returns, 0.0)
    downside_dev = np.sqrt(np.mean(downside ** 2))
    if downside_dev < 1e-12:
        return 0.0
    return float(np.mean(monthly_returns) / downside_dev * np.sqrt(12))


def compute_max_drawdown(values):
    values = np.asarray(values, dtype=np.float64)
    cummax = np.maximum.accumulate(values)
    drawdown = (values - cummax) / cummax
    return float(drawdown.min())


def compute_avg_turnover(turnovers):
    if len(turnovers) == 0:
        return 0.0
    return float(np.mean(turnovers))


def compute_outperformance(agent_returns, benchmark_returns):
    excess = np.asarray(agent_returns) - np.asarray(benchmark_returns)
    return float(np.mean(excess))


def compute_excess_cagr(strategy_values, benchmark_values):
    s_cagr = compute_cagr(strategy_values)
    b_cagr = compute_cagr(benchmark_values)
    return s_cagr - b_cagr


def compute_excess_sharpe(strategy_returns, benchmark_returns):
    excess = np.asarray(strategy_returns) - np.asarray(benchmark_returns)
    return compute_sharpe(excess)


def compute_all_metrics(values, monthly_returns, turnovers):
    values = np.asarray(values, dtype=np.float64)
    monthly_returns = np.asarray(monthly_returns, dtype=np.float64)
    turnovers = np.asarray(turnovers, dtype=np.float64)
    return {
        "CAGR": compute_cagr(values),
        "Ann. Vol": compute_annual_volatility(monthly_returns),
        "Sharpe": compute_sharpe(monthly_returns),
        "Sortino": compute_sortino(monthly_returns),
        "Max DD": compute_max_drawdown(values),
        "Avg Turnover": compute_avg_turnover(turnovers),
    }


# ── Daily-frequency helpers (stock-selection pipeline) ──────────────

def compute_cagr_daily(values):
    """CAGR from a daily equity curve (252 trading days/year)."""
    values = np.asarray(values, dtype=np.float64)
    n_days = len(values) - 1
    if n_days <= 0 or values[0] <= 0:
        return 0.0
    return float((values[-1] / values[0]) ** (252.0 / n_days) - 1.0)


def compute_annual_volatility_daily(daily_returns):
    """Annualised volatility from daily returns."""
    if len(daily_returns) < 2:
        return 0.0
    return float(np.std(daily_returns, ddof=1) * np.sqrt(252))


def compute_sharpe_daily(daily_returns):
    """Annualised Sharpe ratio from daily returns (rf=0)."""
    if len(daily_returns) < 2:
        return 0.0
    vol = np.std(daily_returns, ddof=1)
    if vol < 1e-12:
        return 0.0
    return float(np.mean(daily_returns) / vol * np.sqrt(252))


def compute_sortino_daily(daily_returns):
    """Annualised Sortino ratio from daily returns (rf=0)."""
    if len(daily_returns) < 2:
        return 0.0
    downside = np.minimum(daily_returns, 0.0)
    downside_dev = np.sqrt(np.mean(downside ** 2))
    if downside_dev < 1e-12:
        return 0.0
    return float(np.mean(daily_returns) / downside_dev * np.sqrt(252))


def compute_all_metrics_daily(values, daily_returns, turnovers):
    """Full metrics dict using daily frequency."""
    values = np.asarray(values, dtype=np.float64)
    daily_returns = np.asarray(daily_returns, dtype=np.float64)
    turnovers = np.asarray(turnovers, dtype=np.float64)
    return {
        "CAGR": compute_cagr_daily(values),
        "Ann. Vol": compute_annual_volatility_daily(daily_returns),
        "Sharpe": compute_sharpe_daily(daily_returns),
        "Sortino": compute_sortino_daily(daily_returns),
        "Max DD": compute_max_drawdown(values),
        "Avg Turnover": compute_avg_turnover(turnovers),
    }
