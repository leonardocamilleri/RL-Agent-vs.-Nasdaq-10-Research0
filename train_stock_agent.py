#!/usr/bin/env python3
"""
Train a PPO stock-selection agent on Nasdaq-100 constituents.

Usage:
    python train_stock_agent.py                     # full pipeline
    python train_stock_agent.py --refresh           # re-download prices
    python train_stock_agent.py --timesteps 500000  # custom budget
    python train_stock_agent.py --topk 10           # hold 10 stocks
"""

import argparse
import json
import os
import pickle
import random
import sys
import time

import numpy as np
import pandas as pd
import torch
import yfinance as yf
from sklearn.preprocessing import StandardScaler
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from features.stock_builder import (
    build_stock_dataset,
    N_FEATURES_PER_STOCK,
    FEATURE_NAMES,
)
from envs.nasdaq_stock_env import NasdaqStockSelectionEnv
from evaluation.metrics import (
    compute_all_metrics_daily,
    compute_cagr_daily,
    compute_sharpe_daily,
    compute_max_drawdown,
)

# ═══════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════
SEED = 42
COST_RATE = 0.001
ALPHA = 0.10
K_DEFAULT = 20
CASH_BUFFER = 0.02
MAX_WEIGHT = 0.10
MAX_TRADE_FRAC = 0.20
LAMBDA_TURNOVER = 0.01
LAMBDA_DD = 0.001

# PPO hyper-parameters
N_ENVS = 1
N_STEPS = 2048
BATCH_SIZE = 256
GAMMA = 0.99
GAE_LAMBDA = 0.95
N_EPOCHS = 5
ENT_COEF = 0.01
LEARNING_RATE = 3e-4
POLICY_KWARGS = dict(net_arch=[128, 128])

TOTAL_TIMESTEPS = 2_000_000

# Date splits
TRAIN_END = "2018-12-31"
VAL_START = "2019-01-01"
VAL_END = "2021-12-31"
TEST_START = "2022-01-01"

# Paths
UNIVERSE_CSV = "data/universe/nasdaq100.csv"
PRICE_CACHE = "data/raw/stock_prices.csv"
VOLUME_CACHE = "data/raw/stock_volumes.csv"
SCALER_PATH = "data/processed/stock_scaler.pkl"
MODEL_PATH = "agents/ppo_stock_final"
VECNORM_PATH = "agents/vecnormalize_stock.pkl"
TICKERS_PATH = "agents/stock_tickers.json"
REPORT_PATH = "reports/stock_agent_metrics.json"


# ═══════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_universe() -> list[str]:
    df = pd.read_csv(UNIVERSE_CSV)
    return sorted(df["ticker"].dropna().astype(str).tolist())


def download_prices(tickers: list[str], refresh: bool = False):
    """Download (or load cached) daily OHLCV via yfinance."""
    if not refresh and os.path.exists(PRICE_CACHE):
        print("  Loading cached prices …")
        prices = pd.read_csv(PRICE_CACHE, index_col=0, parse_dates=True)
        volumes = None
        if os.path.exists(VOLUME_CACHE):
            volumes = pd.read_csv(VOLUME_CACHE, index_col=0, parse_dates=True)
        return prices, volumes

    # Download from yfinance — include QQQ benchmark
    all_tickers = list(set(tickers + ["QQQ"]))
    print(f"  Downloading {len(all_tickers)} tickers from yfinance …")
    data = yf.download(all_tickers, start="2008-01-01", auto_adjust=True, threads=True)

    if isinstance(data.columns, pd.MultiIndex):
        prices = data["Close"].copy()
        volumes = data["Volume"].copy()
    else:
        prices = data.copy()
        volumes = None

    if prices.index.tz is not None:
        prices.index = prices.index.tz_localize(None)
    if volumes is not None and volumes.index.tz is not None:
        volumes.index = volumes.index.tz_localize(None)

    prices = prices.ffill()
    os.makedirs(os.path.dirname(PRICE_CACHE), exist_ok=True)
    prices.to_csv(PRICE_CACHE)
    if volumes is not None:
        volumes.to_csv(VOLUME_CACHE)
    return prices, volumes


def filter_tickers(prices: pd.DataFrame, tickers: list[str],
                   min_start: str = "2009-06-01") -> list[str]:
    """Keep only tickers with data from before min_start."""
    valid = []
    min_dt = pd.Timestamp(min_start)
    for t in tickers:
        if t not in prices.columns:
            print(f"    {t}: not in downloaded data — skipped")
            continue
        first_valid = prices[t].first_valid_index()
        if first_valid is None or first_valid > min_dt:
            print(f"    {t}: first data {first_valid} > {min_start} — skipped")
            continue
        # Check enough non-NaN values
        if prices[t].notna().sum() < 500:
            print(f"    {t}: fewer than 500 data points — skipped")
            continue
        valid.append(t)
    return sorted(valid)


def split_by_date(dataset: dict, start: str | None, end: str | None):
    """Slice dataset dict by date range."""
    dates = np.array(dataset["dates"])
    mask = np.ones(len(dates), dtype=bool)
    if start is not None:
        mask &= dates >= pd.Timestamp(start)
    if end is not None:
        mask &= dates <= pd.Timestamp(end)
    idx = np.where(mask)[0]
    if len(idx) == 0:
        return None
    return {
        "features": dataset["features"][idx],
        "forward_ret": dataset["forward_ret"][idx],
        "bench_ret": dataset["bench_ret"][idx],
        "dates": [dataset["dates"][i] for i in idx],
        "tickers": dataset["tickers"],
    }


# ═══════════════════════════════════════════════════════════════════
# Environment factory
# ═══════════════════════════════════════════════════════════════════
def make_env_fn(features_flat, forward_ret, bench_ret, dates, tickers,
                K, alpha, cost_rate, seed=0):
    """Create a single env inside a closure (for DummyVecEnv)."""
    f = features_flat.copy()
    r = forward_ret.copy()
    b = bench_ret.copy()
    d = list(dates)
    tk = list(tickers)
    s = int(seed)

    def _init():
        env = NasdaqStockSelectionEnv(
            features=f, forward_returns=r, bench_returns=b,
            dates=d, tickers=tk, K=K, alpha=alpha, cost_rate=cost_rate,
            cash_buffer=CASH_BUFFER, max_weight=MAX_WEIGHT,
            max_trade_frac=MAX_TRADE_FRAC,
            lambda_turnover=LAMBDA_TURNOVER, lambda_dd=LAMBDA_DD,
        )
        env.reset(seed=s)
        return env

    return _init


def make_train_env(features_flat, forward_ret, bench_ret, dates, tickers,
                   K, alpha, cost_rate, n_envs=N_ENVS):
    env_fns = [
        make_env_fn(features_flat, forward_ret, bench_ret, dates, tickers,
                    K, alpha, cost_rate, seed=SEED + i)
        for i in range(n_envs)
    ]
    venv = DummyVecEnv(env_fns)
    venv = VecNormalize(venv, norm_obs=True, norm_reward=False, clip_obs=10.0)
    return venv


# ═══════════════════════════════════════════════════════════════════
# Evaluation
# ═══════════════════════════════════════════════════════════════════
def evaluate_agent(model, vecnorm_path, features_flat, forward_ret,
                   bench_ret, dates, tickers, K, alpha, cost_rate):
    """Run deterministic evaluation; return metrics-friendly arrays."""
    venv = DummyVecEnv([
        make_env_fn(features_flat, forward_ret, bench_ret, dates, tickers,
                    K, alpha, cost_rate, seed=SEED)
    ])
    venv = VecNormalize.load(vecnorm_path, venv)
    venv.training = False
    venv.norm_reward = False

    obs = venv.reset()
    infos_list = []
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, dones, infos = venv.step(action)
        infos_list.append(infos[0])
        done = dones[0]
    venv.close()

    values = np.array([1.0] + [x["value"] for x in infos_list])
    net_rets = np.array([x["port_ret_net"] for x in infos_list])
    bench_rets_out = np.array([x["bench_ret"] for x in infos_list])
    turnovers = np.array([x["turnover"] for x in infos_list])
    n_held = np.array([x["n_held"] for x in infos_list])
    cash_wts = np.array([x["cash_weight"] for x in infos_list])
    dts = [x["date"] for x in infos_list]

    return {
        "values": values,
        "net_returns": net_rets,
        "bench_returns": bench_rets_out,
        "turnovers": turnovers,
        "n_held": n_held,
        "cash_weights": cash_wts,
        "dates": dts,
    }


def momentum_baseline(forward_ret, bench_ret, daily_prices_slice,
                       tickers, K, cost_rate, rebalance_every=21):
    """
    Simple 252-day momentum top-K baseline (equal-weight, rebalance monthly).
    daily_prices_slice must be aligned to forward_ret dates.
    """
    n_steps, n_stocks = forward_ret.shape
    w = np.ones(n_stocks, dtype=np.float64) / n_stocks * 0.98
    value = 1.0
    values = [1.0]
    net_rets_list = []
    turnovers_list = []

    for t in range(n_steps):
        if t % rebalance_every == 0 and daily_prices_slice is not None:
            p = daily_prices_slice[tickers].iloc[t].values
            if t >= 252:
                p_past = daily_prices_slice[tickers].iloc[t - 252].values
                mom = p / (p_past + 1e-12) - 1.0
            else:
                mom = np.zeros(n_stocks)
            mom = np.nan_to_num(mom, nan=-999.0)
            idx = np.argsort(mom)[::-1][:K]
            new_w = np.zeros(n_stocks, dtype=np.float64)
            new_w[idx] = 0.98 / K
        else:
            new_w = w.copy()

        turnover = 0.5 * np.sum(np.abs(new_w - w))
        cost = cost_rate * 2.0 * turnover
        r = float(np.dot(new_w, forward_ret[t])) - cost
        value *= (1.0 + r)
        values.append(value)
        net_rets_list.append(r)
        turnovers_list.append(2.0 * turnover)

        # Drift weights
        growth = new_w * (1.0 + forward_ret[t])
        g_total = growth.sum() + (1.0 - new_w.sum())
        if g_total > 0:
            w = growth / g_total
        else:
            w = np.ones(n_stocks, dtype=np.float64) / n_stocks * 0.98

    return np.array(values), np.array(net_rets_list), np.array(turnovers_list)


def qqq_buyhold(bench_ret):
    """QQQ buy-and-hold daily equity curve."""
    values = [1.0]
    v = 1.0
    for r in bench_ret:
        v *= (1.0 + r)
        values.append(v)
    return np.array(values), bench_ret.copy()


def print_metrics_table(results: dict):
    """Print a compact comparison table."""
    print("\n" + "=" * 100)
    header = (f"{'Strategy':<22} {'CAGR':>8} {'Vol':>8} {'Sharpe':>8} "
              f"{'Sortino':>8} {'MaxDD':>8} {'Turnover':>10}")
    print(header)
    print("-" * 100)
    for name, m in results.items():
        print(
            f"{name:<22} {m['CAGR']:>7.2%} {m['Ann. Vol']:>7.2%} "
            f"{m['Sharpe']:>8.3f} {m['Sortino']:>8.3f} "
            f"{m['Max DD']:>7.2%} {m['Avg Turnover']:>10.4f}"
        )
    print("=" * 100)


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="Train Nasdaq-100 stock-selection PPO")
    parser.add_argument("--refresh", action="store_true", help="Re-download price data")
    parser.add_argument("--timesteps", type=int, default=TOTAL_TIMESTEPS)
    parser.add_argument("--topk", type=int, default=K_DEFAULT)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    set_seed(args.seed)
    K = args.topk

    # Directories
    for d in ["data/raw", "data/processed", "agents", "reports",
              "evaluation/plots", "data/universe"]:
        os.makedirs(d, exist_ok=True)

    # ----------------------------------------------------------
    # 1. Load universe
    # ----------------------------------------------------------
    universe = load_universe()
    print(f"Universe: {len(universe)} tickers from {UNIVERSE_CSV}")

    # ----------------------------------------------------------
    # 2. Download / load prices
    # ----------------------------------------------------------
    print("Loading price data …")
    prices, volumes = download_prices(universe, refresh=args.refresh)
    print(f"  Price matrix: {prices.shape[0]} days × {prices.shape[1]} tickers")
    print(f"  Range: {prices.index[0].date()} – {prices.index[-1].date()}")

    # ----------------------------------------------------------
    # 3. Filter to tickers with sufficient history
    # ----------------------------------------------------------
    print("Filtering tickers …")
    valid_tickers = filter_tickers(prices, universe, min_start="2009-06-01")
    print(f"  {len(valid_tickers)} tickers with enough history")
    if len(valid_tickers) < 10:
        print("ERROR: fewer than 10 valid tickers. Cannot proceed.")
        sys.exit(1)

    # Ensure QQQ is in prices (for benchmark) but NOT in universe
    if "QQQ" not in prices.columns:
        print("ERROR: QQQ benchmark not in price data.")
        sys.exit(1)

    # ----------------------------------------------------------
    # 4. Build features
    # ----------------------------------------------------------
    print("Building features …")
    t0 = time.time()
    dataset = build_stock_dataset(prices, volumes, valid_tickers, benchmark_ticker="QQQ")
    print(f"  Built in {time.time() - t0:.1f}s")
    n_steps = len(dataset["dates"])
    n_tickers = len(dataset["tickers"])
    print(f"  Steps: {n_steps}   Tickers: {n_tickers}   "
          f"Features/stock: {N_FEATURES_PER_STOCK}")
    print(f"  Date range: {dataset['dates'][0].date()} – {dataset['dates'][-1].date()}")
    feat_3d = dataset["features"]
    print(f"  NaN check: {np.isnan(feat_3d).sum()}")

    # ----------------------------------------------------------
    # 5. Split data
    # ----------------------------------------------------------
    train_ds = split_by_date(dataset, None, TRAIN_END)
    val_ds = split_by_date(dataset, VAL_START, VAL_END)
    test_ds = split_by_date(dataset, TEST_START, None)

    for label, ds in [("Train", train_ds), ("Val", val_ds), ("Test", test_ds)]:
        if ds is None:
            print(f"  {label}: 0 days (empty split)")
        else:
            print(f"  {label}: {len(ds['dates'])} days  "
                  f"({ds['dates'][0].date()} → {ds['dates'][-1].date()})")

    if train_ds is None or len(train_ds["dates"]) < 100:
        print("ERROR: training set too small.")
        sys.exit(1)

    # ----------------------------------------------------------
    # 6. Scale features
    # ----------------------------------------------------------
    print("Fitting StandardScaler on train …")
    train_feat_2d = train_ds["features"].reshape(len(train_ds["dates"]), -1)
    scaler = StandardScaler()
    scaler.fit(train_feat_2d)
    with open(SCALER_PATH, "wb") as f:
        pickle.dump(scaler, f)

    def scale_split(ds):
        """Scale features in-place and return flattened 2-D array."""
        n = len(ds["dates"])
        flat = ds["features"].reshape(n, -1)
        return scaler.transform(flat).astype(np.float32)

    train_feat = scale_split(train_ds)
    val_feat = scale_split(val_ds) if val_ds else None
    test_feat = scale_split(test_ds) if test_ds else None

    obs_dim = train_feat.shape[1] + n_tickers + 2
    print(f"  Observation dim: {obs_dim}  (feat={train_feat.shape[1]} + "
          f"weights={n_tickers} + cash + dd)")

    # Save tickers used
    with open(TICKERS_PATH, "w") as f:
        json.dump(valid_tickers, f, indent=2)

    # ----------------------------------------------------------
    # 7. Train PPO
    # ----------------------------------------------------------
    print(f"\nTraining PPO  (timesteps={args.timesteps:,}, K={K}, "
          f"n_envs={N_ENVS}, seed={args.seed}) …")

    venv = make_train_env(
        train_feat, train_ds["forward_ret"], train_ds["bench_ret"],
        train_ds["dates"], valid_tickers, K, ALPHA, COST_RATE,
    )

    model = PPO(
        "MlpPolicy", venv,
        learning_rate=LEARNING_RATE,
        n_steps=N_STEPS,
        batch_size=BATCH_SIZE,
        gamma=GAMMA,
        gae_lambda=GAE_LAMBDA,
        n_epochs=N_EPOCHS,
        ent_coef=ENT_COEF,
        seed=args.seed,
        verbose=1,
        policy_kwargs=POLICY_KWARGS,
    )

    t0 = time.time()
    model.learn(total_timesteps=args.timesteps)
    elapsed = time.time() - t0
    print(f"  Training finished in {elapsed / 60:.1f} min")

    model.save(MODEL_PATH)
    venv.save(VECNORM_PATH)
    venv.close()
    print(f"  Saved {MODEL_PATH}.zip + {VECNORM_PATH}")

    # ----------------------------------------------------------
    # 8. Evaluate
    # ----------------------------------------------------------
    report = {}

    for split_name, feat, ds in [
        ("Val", val_feat, val_ds),
        ("Test", test_feat, test_ds),
    ]:
        if ds is None:
            print(f"\n{split_name} set is empty — skipping evaluation.")
            continue

        print(f"\nEvaluating on {split_name} set …")
        ev = evaluate_agent(
            model, VECNORM_PATH, feat,
            ds["forward_ret"], ds["bench_ret"],
            ds["dates"], valid_tickers, K, ALPHA, COST_RATE,
        )

        # RL metrics
        rl_m = compute_all_metrics_daily(ev["values"], ev["net_returns"], ev["turnovers"])
        rl_m["Avg Held"] = float(np.mean(ev["n_held"]))
        rl_m["Avg Cash"] = float(np.mean(ev["cash_weights"]))

        # QQQ Buy & Hold
        bh_vals, bh_rets = qqq_buyhold(ds["bench_ret"])
        bh_m = compute_all_metrics_daily(bh_vals, bh_rets, np.zeros(len(bh_rets)))

        # Momentum baseline
        # Need aligned daily_prices for momentum lookback
        dates_for_mom = pd.DatetimeIndex(ds["dates"])
        prices_aligned = prices.loc[prices.index.isin(
            pd.DatetimeIndex(dataset["dates"])  # use all valid dates for alignment
        )].copy()
        # Reindex to ds dates range
        mom_prices = prices.loc[
            (prices.index >= dates_for_mom[0] - pd.Timedelta(days=400)) &
            (prices.index <= dates_for_mom[-1] + pd.Timedelta(days=5))
        ].copy()
        # Map ds dates into mom_prices index
        date_positions = [mom_prices.index.get_indexer([d], method="ffill")[0]
                          for d in dates_for_mom]
        mom_prices_slice = mom_prices.iloc[
            max(0, min(date_positions) - 252): max(date_positions) + 2
        ]

        mom_v, mom_r, mom_t = momentum_baseline(
            ds["forward_ret"], ds["bench_ret"],
            mom_prices_slice, valid_tickers, K, COST_RATE,
        )
        mom_m = compute_all_metrics_daily(mom_v, mom_r, mom_t)

        split_results = {
            f"PPO Agent (K={K})": rl_m,
            "QQQ Buy&Hold": bh_m,
            f"Momentum Top-{K}": mom_m,
        }

        print_metrics_table(split_results)

        # Extra RL diagnostics
        print(f"  Avg stocks held: {rl_m['Avg Held']:.1f}")
        print(f"  Avg cash weight: {rl_m['Avg Cash']:.4f}")

        report[split_name] = {k: {kk: round(vv, 6) for kk, vv in v.items()}
                               for k, v in split_results.items()}

    # ----------------------------------------------------------
    # 9. Save report
    # ----------------------------------------------------------
    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
    report["config"] = {
        "K": K, "alpha": ALPHA, "cost_rate": COST_RATE,
        "timesteps": args.timesteps, "seed": args.seed,
        "n_tickers": n_tickers, "obs_dim": obs_dim,
        "train_days": len(train_ds["dates"]) if train_ds else 0,
        "val_days": len(val_ds["dates"]) if val_ds else 0,
        "test_days": len(test_ds["dates"]) if test_ds else 0,
    }
    with open(REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport saved to {REPORT_PATH}")
    print("Done.")


if __name__ == "__main__":
    main()
