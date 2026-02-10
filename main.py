import os
import random
import pickle

import numpy as np
import pandas as pd
import torch
import yfinance as yf
from sklearn.preprocessing import StandardScaler
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from features.builder import get_month_end_dates, compute_features
from envs.portfolio_env import PortfolioEnv
from benchmarks.baselines import (
    buy_and_hold_qqq,
    equal_weight,
    fixed_qqq_plus_sectors_equal,
)
from evaluation.metrics import (
    compute_all_metrics,
    compute_sharpe,
    compute_excess_cagr,
    compute_excess_sharpe,
)
from evaluation.plots import plot_equity_curves, plot_drawdowns

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

TICKERS = ["QQQ", "XLK", "XLF", "XLE", "XLV", "XLY", "XLP"]
QQQ_INDEX = TICKERS.index("QQQ")
COST_RATE = 0.001
ALPHA = 0.10
N_ENVS = 4
N_STEPS = 1024
BATCH_SIZE = 256
GAMMA = 0.99
GAE_LAMBDA = 0.95
N_EPOCHS = 5
HP_TIMESTEPS = 200_000
FINAL_TIMESTEPS = 1_000_000
POLICY_KWARGS = dict(net_arch=[256, 256])


def make_dirs():
    for d in [
        "data/raw",
        "data/processed",
        "features",
        "agents",
        "evaluation/plots",
    ]:
        os.makedirs(d, exist_ok=True)


def download_prices():
    data = yf.download(TICKERS, start="2010-01-01", auto_adjust=True)
    if isinstance(data.columns, pd.MultiIndex):
        prices = data["Close"][TICKERS].copy()
    else:
        prices = data[TICKERS].copy()
    if prices.index.tz is not None:
        prices.index = prices.index.tz_localize(None)
    prices = prices.ffill().dropna(how="any")
    prices.to_csv("data/raw/daily_prices.csv")
    return prices


def build_dataset(prices):
    daily_returns = prices.pct_change()
    daily_returns.iloc[0] = 0.0

    month_end_dates = get_month_end_dates(prices)
    features_df = compute_features(prices, daily_returns, month_end_dates, TICKERS)

    monthly_prices = prices.loc[prices.index.isin(month_end_dates)]
    monthly_returns = monthly_prices.pct_change()
    fwd_returns = monthly_returns.shift(-1)
    fwd_returns = fwd_returns.reindex(features_df.index)

    valid = fwd_returns.notna().all(axis=1) & features_df.notna().all(axis=1)
    features_df = features_df[valid]
    fwd_returns = fwd_returns[valid]
    return features_df, fwd_returns


def split_data(features_df, fwd_returns):
    train_mask = features_df.index <= "2016-12-31"
    val_mask = (features_df.index >= "2017-01-01") & (
        features_df.index <= "2019-12-31"
    )
    test_mask = (features_df.index >= "2020-01-01") & (
        features_df.index <= "2024-12-31"
    )
    splits = {}
    for name, mask in [("train", train_mask), ("val", val_mask), ("test", test_mask)]:
        splits[name] = (features_df[mask], fwd_returns[mask])
    return splits


def make_env_fn(feat, fwd_ret, dates, qqq_index, cost_rate, alpha, seed=0):
    f = np.array(feat, dtype=np.float32)
    r = np.array(fwd_ret, dtype=np.float64)
    d = list(dates)
    s = int(seed)

    def _init():
        env = PortfolioEnv(f, r, d, qqq_index=qqq_index,
                           cost_rate=cost_rate, alpha=alpha)
        env.reset(seed=s)
        return env

    return _init


def make_train_env(feat, fwd_ret, dates, qqq_index, cost_rate, alpha, n_envs=N_ENVS):
    env_fns = [
        make_env_fn(feat, fwd_ret, dates, qqq_index, cost_rate, alpha, seed=SEED + i)
        for i in range(n_envs)
    ]
    venv = DummyVecEnv(env_fns)
    venv = VecNormalize(venv, norm_obs=True, norm_reward=False, clip_obs=10.0)
    return venv


def evaluate_agent_vecnorm(model, vecnorm_path, feat, fwd_ret, dates,
                           qqq_index, cost_rate, alpha):
    venv = DummyVecEnv([make_env_fn(feat, fwd_ret, dates, qqq_index,
                                     cost_rate, alpha, seed=SEED)])
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
    net_returns = np.array([x["port_ret_net"] for x in infos_list])
    bench_returns = np.array([x["bench_ret"] for x in infos_list])
    turnovers = np.array([x["turnover"] for x in infos_list])
    dates_out = [x["date"] for x in infos_list]
    return values, net_returns, bench_returns, turnovers, dates_out


def main():
    make_dirs()

    print("Downloading data...")
    prices = download_prices()
    print(
        f"  Prices: {prices.index[0].date()} to {prices.index[-1].date()}, "
        f"shape={prices.shape}"
    )

    print("Building dataset...")
    features_df, fwd_returns = build_dataset(prices)
    print(f"  Dataset: {features_df.shape[0]} months, {features_df.shape[1]} features")
    print(
        f"  Range: {features_df.index[0].date()} to {features_df.index[-1].date()}"
    )

    splits = split_data(features_df, fwd_returns)
    for name in ["train", "val", "test"]:
        f, r = splits[name]
        if len(f) > 0:
            print(f"  {name}: {len(f)} months  ({f.index[0].date()} -> {f.index[-1].date()})")
        else:
            print(f"  {name}: 0 months")

    scaler = StandardScaler()
    scaler.fit(splits["train"][0].values)

    features_df.to_csv("data/processed/features.csv")
    fwd_returns.to_csv("data/processed/forward_returns.csv")
    with open("data/processed/scaler.pkl", "wb") as fh:
        pickle.dump(scaler, fh)

    train_feat = scaler.transform(splits["train"][0].values)
    train_ret = splits["train"][1].values
    train_dates = splits["train"][0].index.tolist()

    val_feat = scaler.transform(splits["val"][0].values)
    val_ret = splits["val"][1].values
    val_dates = splits["val"][0].index.tolist()

    print("\nHyperparameter selection (validation excess Sharpe vs QQQ)...")
    candidate_lrs = [1e-4, 3e-4, 5e-4]
    best_lr = candidate_lrs[0]
    best_sharpe = -np.inf

    for lr in candidate_lrs:
        env = make_train_env(train_feat, train_ret, train_dates,
                             QQQ_INDEX, COST_RATE, ALPHA)
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=lr,
            n_steps=N_STEPS,
            batch_size=BATCH_SIZE,
            gamma=GAMMA,
            gae_lambda=GAE_LAMBDA,
            n_epochs=N_EPOCHS,
            seed=SEED,
            verbose=0,
            policy_kwargs=POLICY_KWARGS,
        )
        model.learn(total_timesteps=HP_TIMESTEPS)
        env.save("agents/_hp_vecnorm.pkl")
        env.close()

        _, vr, vb, _, _ = evaluate_agent_vecnorm(
            model, "agents/_hp_vecnorm.pkl",
            val_feat, val_ret, val_dates,
            QQQ_INDEX, COST_RATE, ALPHA,
        )
        excess = vr - vb
        sharpe = compute_sharpe(excess)
        print(f"  LR={lr:.0e}  Val ExcessSharpe={sharpe:.4f}")
        if sharpe > best_sharpe:
            best_sharpe = sharpe
            best_lr = lr

    print(f"  Selected LR={best_lr:.0e}  (ExcessSharpe={best_sharpe:.4f})")

    print("\nTraining final model on train+val...")
    combined_feat = np.vstack([train_feat, val_feat])
    combined_ret = np.vstack([train_ret, val_ret])
    combined_dates = train_dates + val_dates

    env = make_train_env(combined_feat, combined_ret, combined_dates,
                         QQQ_INDEX, COST_RATE, ALPHA)
    final_model = PPO(
        "MlpPolicy",
        env,
        learning_rate=best_lr,
        n_steps=N_STEPS,
        batch_size=BATCH_SIZE,
        gamma=GAMMA,
        gae_lambda=GAE_LAMBDA,
        n_epochs=N_EPOCHS,
        seed=SEED,
        verbose=1,
        policy_kwargs=POLICY_KWARGS,
    )
    final_model.learn(total_timesteps=FINAL_TIMESTEPS)
    final_model.save("agents/ppo_final")
    env.save("agents/vecnormalize.pkl")
    env.close()
    print("  Saved agents/ppo_final.zip + agents/vecnormalize.pkl")

    print("\nEvaluating on test set...")
    test_feat = scaler.transform(splits["test"][0].values)
    test_ret = splits["test"][1].values
    test_dates = splits["test"][0].index.tolist()

    rl_vals, rl_rets, rl_bench, rl_tvrs, rl_dts = evaluate_agent_vecnorm(
        final_model, "agents/vecnormalize.pkl",
        test_feat, test_ret, test_dates,
        QQQ_INDEX, COST_RATE, ALPHA,
    )

    bh_vals, bh_rets, bh_tvrs, bh_exc = buy_and_hold_qqq(test_ret, TICKERS, COST_RATE)
    ew_vals, ew_rets, ew_tvrs, ew_exc = equal_weight(test_ret, TICKERS, COST_RATE)
    fx_vals, fx_rets, fx_tvrs, fx_exc = fixed_qqq_plus_sectors_equal(
        test_ret, TICKERS, COST_RATE
    )

    rl_excess = rl_rets - rl_bench

    all_results = {
        "PPO Agent": (rl_vals, rl_rets, rl_tvrs, rl_excess),
        "Buy&Hold QQQ": (bh_vals, bh_rets, bh_tvrs, bh_exc),
        "Equal Weight": (ew_vals, ew_rets, ew_tvrs, ew_exc),
        "50/50 QQQ+Sec": (fx_vals, fx_rets, fx_tvrs, fx_exc),
    }

    print("\n" + "=" * 120)
    print(
        f"{'Strategy':<16} {'CAGR':>8} {'Vol':>8} {'Sharpe':>8} "
        f"{'Sortino':>8} {'MaxDD':>8} {'Turnover':>10} "
        f"{'ExcCAGR':>9} {'ExcSharpe':>10}"
    )
    print("-" * 120)
    for name, (v, r, t, exc) in all_results.items():
        m = compute_all_metrics(v, r, t)
        exc_cagr = compute_excess_cagr(v, bh_vals)
        exc_sharpe = compute_excess_sharpe(r, bh_rets)
        print(
            f"{name:<16} {m['CAGR']:>7.2%} {m['Ann. Vol']:>7.2%} "
            f"{m['Sharpe']:>8.3f} {m['Sortino']:>8.3f} "
            f"{m['Max DD']:>7.2%} {m['Avg Turnover']:>10.4f} "
            f"{exc_cagr:>8.2%} {exc_sharpe:>10.3f}"
        )
    print("=" * 120)

    initial_date = pd.Timestamp(test_dates[0]) - pd.DateOffset(months=1)
    plot_dates = [initial_date] + [pd.Timestamp(d) for d in test_dates]

    plot_results = {}
    for name, (v, r, t, exc) in all_results.items():
        plot_results[name] = {"values": list(v), "dates": plot_dates}

    plot_equity_curves(plot_results, "evaluation/plots")
    plot_drawdowns(plot_results, "evaluation/plots")
    print("\nPlots saved to evaluation/plots/")


if __name__ == "__main__":
    main()
