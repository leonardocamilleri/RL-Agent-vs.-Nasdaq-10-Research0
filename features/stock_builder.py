"""
Feature builder for individual stock selection pipeline.

Produces per-stock daily features (causal, no lookahead).
Output: 3-D array  (n_dates, n_tickers, n_features)  ready for the env.
"""

import numpy as np
import pandas as pd

FEATURE_NAMES = [
    "r_1",
    "r_5",
    "r_21",
    "r_63",
    "r_252",
    "vol_21",
    "trend_50",
    "trend_200",
    "dd_252",
    "vol_zscore_21",
]
N_FEATURES_PER_STOCK = len(FEATURE_NAMES)

# Minimum history (trading days) required before the first usable date.
MIN_WARMUP = 252


def compute_stock_features(daily_prices: pd.DataFrame,
                           daily_volumes: pd.DataFrame | None,
                           tickers: list[str]) -> dict[str, pd.DataFrame]:
    """
    Compute per-stock feature DataFrames.

    Args:
        daily_prices:  DataFrame (date × tickers), adjusted close prices.
        daily_volumes: DataFrame (date × tickers), trading volume. Optional.
        tickers:       Ordered list of tickers to compute features for.

    Returns:
        dict  {ticker: DataFrame (date × features)}
    """
    daily_returns = daily_prices[tickers].pct_change()
    daily_returns.iloc[0] = 0.0

    all_features: dict[str, pd.DataFrame] = {}

    for ticker in tickers:
        p = daily_prices[ticker]
        r = daily_returns[ticker]

        df = pd.DataFrame(index=daily_prices.index)
        df["r_1"] = r
        df["r_5"] = p / p.shift(5) - 1
        df["r_21"] = p / p.shift(21) - 1
        df["r_63"] = p / p.shift(63) - 1
        df["r_252"] = p / p.shift(252) - 1
        df["vol_21"] = r.rolling(21, min_periods=15).std() * np.sqrt(252)
        sma50 = p.rolling(50, min_periods=40).mean()
        sma200 = p.rolling(200, min_periods=150).mean()
        df["trend_50"] = p / sma50 - 1
        df["trend_200"] = p / sma200 - 1
        df["dd_252"] = p / p.rolling(252, min_periods=200).max() - 1

        # Volume z-score (dollar volume)
        if (daily_volumes is not None
                and ticker in daily_volumes.columns
                and daily_volumes[ticker].notna().sum() > 30):
            v = daily_volumes[ticker].clip(lower=0)
            dv = np.log1p(v * p)
            m21 = dv.rolling(21, min_periods=15).mean()
            s21 = dv.rolling(21, min_periods=15).std()
            df["vol_zscore_21"] = (dv - m21) / (s21 + 1e-8)
        else:
            # Fallback: absolute-return z-score as liquidity proxy
            abs_r = r.abs()
            m21 = abs_r.rolling(21, min_periods=15).mean()
            s21 = abs_r.rolling(21, min_periods=15).std()
            df["vol_zscore_21"] = (abs_r - m21) / (s21 + 1e-8)

        all_features[ticker] = df[FEATURE_NAMES]

    return all_features


def build_stock_dataset(daily_prices: pd.DataFrame,
                        daily_volumes: pd.DataFrame | None,
                        tickers: list[str],
                        benchmark_ticker: str = "QQQ"):
    """
    End-to-end dataset builder for the stock-selection environment.

    Returns
    -------
    dict with keys:
        features   : np.ndarray  (n_steps, n_tickers, n_features)  RAW
        forward_ret: np.ndarray  (n_steps, n_tickers)
        bench_ret  : np.ndarray  (n_steps,)
        dates      : list[pd.Timestamp]        length n_steps
        tickers    : list[str]                 ordered ticker list
        feature_names: list[str]
    """
    all_feat = compute_stock_features(daily_prices, daily_volumes, tickers)

    # Find first date where ALL tickers have valid (non-NaN) features
    valid_per_ticker = {}
    for ticker in tickers:
        valid_per_ticker[ticker] = all_feat[ticker].notna().all(axis=1)

    combined_valid = pd.DataFrame(valid_per_ticker).all(axis=1)
    if not combined_valid.any():
        raise ValueError("No date with valid features for all tickers.")

    first_valid = combined_valid.idxmax()
    valid_dates = daily_prices.index[daily_prices.index >= first_valid]
    valid_dates = valid_dates[combined_valid.reindex(valid_dates, fill_value=False)]
    valid_dates = sorted(valid_dates)

    n_dates = len(valid_dates)
    n_tickers = len(tickers)

    # Build 3-D feature array  (n_dates, n_tickers, n_features)
    feat_3d = np.zeros((n_dates, n_tickers, N_FEATURES_PER_STOCK), dtype=np.float32)
    for j, ticker in enumerate(tickers):
        feat_3d[:, j, :] = all_feat[ticker].loc[valid_dates, FEATURE_NAMES].values

    feat_3d = np.nan_to_num(feat_3d, nan=0.0)

    # Forward returns: close-to-close from date[i] to date[i+1]
    prices_arr = daily_prices.loc[valid_dates, tickers].values.astype(np.float64)
    forward_ret = prices_arr[1:] / prices_arr[:-1] - 1.0
    forward_ret = np.nan_to_num(forward_ret, nan=0.0)

    # Benchmark forward returns
    bench_prices = daily_prices.loc[valid_dates, benchmark_ticker].values.astype(np.float64)
    bench_ret = bench_prices[1:] / bench_prices[:-1] - 1.0
    bench_ret = np.nan_to_num(bench_ret, nan=0.0)

    # Trim features to match forward returns (drop last row)
    feat_3d = feat_3d[:-1]
    step_dates = valid_dates[:-1]

    return {
        "features": feat_3d,
        "forward_ret": forward_ret,
        "bench_ret": bench_ret,
        "dates": step_dates,
        "tickers": tickers,
        "feature_names": FEATURE_NAMES,
    }


def obs_for_date(feature_row_flat: np.ndarray,
                 prev_weights: np.ndarray,
                 cash_weight: float,
                 drawdown: float) -> np.ndarray:
    """
    Build a single observation vector from pre-scaled feature row + portfolio state.
    """
    portfolio_state = np.concatenate([
        prev_weights,
        np.array([cash_weight, drawdown], dtype=np.float32),
    ])
    return np.concatenate([feature_row_flat, portfolio_state]).astype(np.float32)
