#!/usr/bin/env python3
"""
Live paper-trading execution for the Nasdaq-100 stock-selection RL agent.

Uses IBKR TWS/Gateway via ib_insync.  Default: DRY RUN (prints orders only).

Usage:
    python live_paper_execute_stocks.py                          # dry run
    python live_paper_execute_stocks.py --live                   # real orders
    python live_paper_execute_stocks.py --live --run-at 16:45    # schedule
    python live_paper_execute_stocks.py --topk 15                # hold 15 names
"""

import argparse
import datetime as dt
import json
import logging
import os
import pickle
import sys
import time

import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from features.stock_builder import (
    build_stock_dataset,
    N_FEATURES_PER_STOCK,
)
from envs.nasdaq_stock_env import NasdaqStockSelectionEnv

# ═══════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════
ALPHA = 0.10
COST_RATE = 0.001
K_DEFAULT = 20
CASH_BUFFER = 0.02
MAX_WEIGHT = 0.10
MAX_TRADE_PCT_NAV = 0.20
MAX_SINGLE_WEIGHT = 0.50

MODEL_PATH = "agents/ppo_stock_final.zip"
VECNORM_PATH = "agents/vecnormalize_stock.pkl"
SCALER_PATH = "data/processed/stock_scaler.pkl"
TICKERS_PATH = "agents/stock_tickers.json"

TWS_HOST = "127.0.0.1"
TWS_PORT = 7497
TWS_CLIENT_ID = 20
HISTORY_DURATION = "2 Y"

LOG_DIR = "logs"
LOG_FILE = "live_trading_stocks.log"


# ═══════════════════════════════════════════════════════════════════
# Logging
# ═══════════════════════════════════════════════════════════════════
def setup_logging():
    os.makedirs(LOG_DIR, exist_ok=True)
    log_path = os.path.join(LOG_DIR, LOG_FILE)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_path, mode="a"),
        ],
    )
    return logging.getLogger("stock_live")


# ═══════════════════════════════════════════════════════════════════
# IBKR helpers
# ═══════════════════════════════════════════════════════════════════
def connect_ibkr(host, port, client_id):
    from ib_insync import IB
    ib = IB()
    ib.connect(host, port, clientId=client_id, timeout=30)
    return ib


def fetch_daily_bars(ib, ticker, duration=HISTORY_DURATION):
    """Fetch daily OHLCV bars from IBKR for a single US stock."""
    from ib_insync import Stock
    contract = Stock(ticker, "SMART", "USD")
    ib.qualifyContracts(contract)
    bars = ib.reqHistoricalData(
        contract,
        endDateTime="",
        durationStr=duration,
        barSizeSetting="1 day",
        whatToShow="TRADES",
        useRTH=True,
        formatDate=1,
    )
    if not bars:
        return None
    df = pd.DataFrame([{
        "date": b.date, "open": b.open, "high": b.high,
        "low": b.low, "close": b.close, "volume": b.volume,
    } for b in bars])
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    return df


def get_nav_base_currency(ib):
    """Read NetLiquidation from IBKR account summary."""
    ib.reqAccountSummary()
    ib.sleep(3)
    summaries = ib.accountSummary()
    nl_entries = [s for s in summaries if s.tag == "NetLiquidation"]
    for entry in nl_entries:
        logging.getLogger("stock_live").info(
            f"  NetLiq entry: {entry.value} {entry.currency}"
        )
    if not nl_entries:
        raise RuntimeError("No NetLiquidation entries from IBKR")
    entry = nl_entries[0]
    return float(entry.value), entry.currency


def get_fx_rate_hist(ib, base_ccy, quote_ccy, asof_date=None):
    """Fetch historical FX rate (e.g. EURUSD) via IDEALPRO daily bars."""
    from ib_insync import Forex
    pair = f"{base_ccy}{quote_ccy}"
    contract = Forex(pair)
    try:
        ib.qualifyContracts(contract)
    except Exception:
        pass
    end_dt = "" if asof_date is None else asof_date.strftime("%Y%m%d %H:%M:%S")
    bars = ib.reqHistoricalData(
        contract, endDateTime=end_dt,
        durationStr="30 D", barSizeSetting="1 day",
        whatToShow="MIDPOINT", useRTH=False, formatDate=1,
    )
    if bars:
        return float(bars[-1].close)
    # Try inverse
    inv_pair = f"{quote_ccy}{base_ccy}"
    contract_inv = Forex(inv_pair)
    try:
        ib.qualifyContracts(contract_inv)
    except Exception:
        pass
    bars_inv = ib.reqHistoricalData(
        contract_inv, endDateTime=end_dt,
        durationStr="30 D", barSizeSetting="1 day",
        whatToShow="MIDPOINT", useRTH=False, formatDate=1,
    )
    if bars_inv:
        return 1.0 / float(bars_inv[-1].close)
    logging.getLogger("stock_live").warning(f"Could not fetch FX rate for {pair}")
    return None


def get_nav_usd(ib):
    """Get account NAV in USD, converting if needed."""
    log = logging.getLogger("stock_live")
    nav_val, ccy = get_nav_base_currency(ib)
    log.info(f"  NAV base: {nav_val:,.2f} {ccy}")
    if ccy == "USD":
        return nav_val
    fx = get_fx_rate_hist(ib, ccy, "USD")
    if fx is not None:
        nav_usd = nav_val * fx
        log.info(f"  FX {ccy}USD = {fx:.6f}  →  NAV USD = {nav_usd:,.2f}")
        return nav_usd
    log.warning(f"  FX unavailable; using raw NAV {nav_val:,.2f} {ccy}")
    return nav_val


def get_current_positions(ib, tickers):
    """Return dict {ticker: qty} for specified tickers."""
    positions = {}
    for pos in ib.positions():
        sym = pos.contract.symbol
        if sym in tickers:
            positions[sym] = int(pos.position)
    return positions


def get_last_price(ib, ticker, fallback_price=None, prefer_live=False):
    """Get price for sizing — prefer hist close, optionally try live."""
    if prefer_live:
        from ib_insync import Stock
        contract = Stock(ticker, "SMART", "USD")
        ib.qualifyContracts(contract)
        md = ib.reqMktData(contract, "", False, False)
        ib.sleep(2)
        price = None
        if md.last and md.last > 0:
            price = float(md.last)
        elif md.close and md.close > 0:
            price = float(md.close)
        ib.cancelMktData(contract)
        if price:
            return price
    if fallback_price is not None:
        return fallback_price
    return None


# ═══════════════════════════════════════════════════════════════════
# Feature pipeline
# ═══════════════════════════════════════════════════════════════════
def build_live_features(ib, tickers, scaler):
    """
    Fetch IBKR daily bars, compute features, scale, and return
    (obs_flat, hist_prices_dict).
    """
    log = logging.getLogger("stock_live")
    all_closes = {}
    all_volumes = {}
    hist_prices = {}

    for i, ticker in enumerate(tickers):
        log.info(f"  [{i + 1}/{len(tickers)}] Fetching {ticker} …")
        bars_df = fetch_daily_bars(ib, ticker)
        if bars_df is None or bars_df.empty:
            log.warning(f"    {ticker}: no data — will fill with zeros")
            continue
        all_closes[ticker] = bars_df["close"]
        all_volumes[ticker] = bars_df["volume"]
        hist_prices[ticker] = float(bars_df["close"].iloc[-1])
        time.sleep(0.5)  # IBKR pacing

    # Also fetch QQQ for benchmark
    log.info("  Fetching QQQ (benchmark) …")
    qqq_bars = fetch_daily_bars(ib, "QQQ")
    if qqq_bars is not None and not qqq_bars.empty:
        all_closes["QQQ"] = qqq_bars["close"]
        all_volumes["QQQ"] = qqq_bars["volume"]

    if len(all_closes) < 5:
        raise RuntimeError(f"Only {len(all_closes)} tickers returned data")

    # Build aligned DataFrames
    prices_df = pd.DataFrame(all_closes).sort_index().ffill()
    volumes_df = pd.DataFrame(all_volumes).sort_index().ffill()

    # Fill missing tickers with NaN → will become 0 after nan_to_num
    for t in tickers:
        if t not in prices_df.columns:
            prices_df[t] = np.nan
            volumes_df[t] = np.nan

    # Build features
    dataset = build_stock_dataset(prices_df, volumes_df, tickers, benchmark_ticker="QQQ")
    # Take last row
    last_feat = dataset["features"][-1:]  # (1, n_tickers, n_feat)
    flat = last_feat.reshape(1, -1)
    scaled = scaler.transform(flat).astype(np.float32)

    return scaled[0], hist_prices


# ═══════════════════════════════════════════════════════════════════
# Model loading
# ═══════════════════════════════════════════════════════════════════
def load_model_and_normalizer(tickers, K):
    """Load PPO model and VecNormalize with a dummy env."""
    log = logging.getLogger("stock_live")
    n_stocks = len(tickers)
    feat_dim = n_stocks * N_FEATURES_PER_STOCK
    obs_dim = feat_dim + n_stocks + 2

    dummy_feat = np.zeros((10, feat_dim), dtype=np.float32)
    dummy_fwd = np.zeros((10, n_stocks), dtype=np.float64)
    dummy_bench = np.zeros(10, dtype=np.float64)
    dummy_dates = list(range(10))

    def _init():
        return NasdaqStockSelectionEnv(
            features=dummy_feat, forward_returns=dummy_fwd,
            bench_returns=dummy_bench, dates=dummy_dates,
            tickers=tickers, K=K,
        )

    venv = DummyVecEnv([_init])
    venv = VecNormalize.load(VECNORM_PATH, venv)
    venv.training = False
    venv.norm_reward = False

    model = PPO.load(MODEL_PATH, env=venv)
    log.info(f"  Model loaded: obs_dim={obs_dim}  action_dim={n_stocks}")
    return model, venv


# ═══════════════════════════════════════════════════════════════════
# Portfolio construction
# ═══════════════════════════════════════════════════════════════════
_prev_weights = None


def scores_to_weights(scores, K, n_stocks, prev_w=None):
    """Convert raw PPO scores → smoothed target weights."""
    idx = np.argsort(scores)[::-1][:K]
    w = np.zeros(n_stocks, dtype=np.float64)
    w[idx] = 1.0 / K
    # Max weight clip
    w = np.minimum(w, MAX_WEIGHT)
    s = w.sum()
    if s > 0:
        w /= s
    w *= (1.0 - CASH_BUFFER)
    # EWMA smoothing
    if prev_w is not None:
        w = (1.0 - ALPHA) * prev_w + ALPHA * w
    return w


def positions_to_weights(positions, tickers, prices, nav):
    """Convert current positions → weight vector."""
    w = np.zeros(len(tickers), dtype=np.float64)
    for i, t in enumerate(tickers):
        qty = positions.get(t, 0)
        p = prices.get(t, 0.0)
        if p and nav > 0:
            w[i] = qty * p / nav
    return w


def compute_orders(tickers, target_weights, current_positions, prices, nav):
    """Compute target shares and order deltas."""
    log = logging.getLogger("stock_live")
    orders = []
    for i, ticker in enumerate(tickers):
        price = prices.get(ticker)
        if price is None or price <= 0:
            continue
        target_dollar = target_weights[i] * nav
        target_shares = int(target_dollar / price)
        current_shares = current_positions.get(ticker, 0)
        delta = target_shares - current_shares

        if delta == 0:
            continue

        # Max trade size per rebalance
        trade_notional = abs(delta) * price
        max_notional = MAX_TRADE_PCT_NAV * nav
        if trade_notional > max_notional:
            delta = int(np.sign(delta) * max_notional / price)
            if delta == 0:
                continue

        orders.append({
            "ticker": ticker,
            "delta": delta,
            "price": price,
            "est_notional": abs(delta) * price,
            "target_shares": target_shares,
            "current_shares": current_shares,
        })
    return orders


def submit_orders(ib, orders, dry_run=True):
    """Place MarketOrders (or log them if dry_run)."""
    from ib_insync import Stock, MarketOrder
    log = logging.getLogger("stock_live")
    for o in orders:
        side = "BUY" if o["delta"] > 0 else "SELL"
        qty = abs(o["delta"])
        log.info(f"  {'[DRY]' if dry_run else ''} {side} {qty} {o['ticker']} "
                 f"@ ~${o['price']:.2f}  (est ${o['est_notional']:,.0f})")
        if not dry_run:
            contract = Stock(o["ticker"], "SMART", "USD")
            ib.qualifyContracts(contract)
            order = MarketOrder(side, qty)
            trade = ib.placeOrder(contract, order)
            ib.sleep(1)
            log.info(f"    → order placed: {trade.order.orderId}")


# ═══════════════════════════════════════════════════════════════════
# Rebalance cycle
# ═══════════════════════════════════════════════════════════════════
def rebalance_cycle(ib, model, venv, tickers, K, scaler,
                    dry_run=True, prefer_live=False):
    """Run one full rebalance cycle."""
    global _prev_weights
    log = logging.getLogger("stock_live")
    n_stocks = len(tickers)

    log.info("─" * 60)
    log.info(f"Rebalance cycle  {dt.datetime.now().isoformat()}")
    log.info(f"  Mode: {'DRY RUN' if dry_run else 'LIVE'}  "
             f"Price: {'prefer-live' if prefer_live else 'hist-only'}  K={K}")

    # 1) Build features + get hist prices
    log.info("Building features …")
    scaled_feat, hist_prices = build_live_features(ib, tickers, scaler)

    # 2) Portfolio state
    nav = get_nav_usd(ib)
    current_pos = get_current_positions(ib, tickers)
    cur_weights = positions_to_weights(current_pos, tickers, hist_prices, nav)

    if _prev_weights is None:
        _prev_weights = cur_weights.copy()

    cash_weight = max(0.0, 1.0 - cur_weights.sum())
    drawdown = 0.0  # Simplified — no peak tracking in live

    # Build full observation
    port_state = np.concatenate([
        _prev_weights.astype(np.float32),
        np.array([cash_weight, drawdown], dtype=np.float32),
    ])
    obs_raw = np.concatenate([scaled_feat, port_state]).astype(np.float32)
    obs = obs_raw.reshape(1, -1)

    # Normalise
    obs_norm = venv.normalize_obs(obs)

    # 3) Predict
    action, _ = model.predict(obs_norm, deterministic=True)
    scores = action.flatten()

    # 4) Construct weights
    target_w = scores_to_weights(scores, K, n_stocks, prev_w=_prev_weights)

    log.info(f"  NAV: ${nav:,.2f}")
    log.info(f"  Stocks held (current): {int(np.sum(cur_weights > 1e-4))}")
    log.info(f"  Target stocks: {int(np.sum(target_w > 1e-4))}")
    top_idx = np.argsort(target_w)[::-1][:10]
    log.info(f"  Top-10 target: "
             + ", ".join(f"{tickers[i]} {target_w[i]:.3f}" for i in top_idx))
    turnover = np.sum(np.abs(target_w - cur_weights))
    log.info(f"  Turnover estimate: {turnover:.4f}")

    # 5) Get prices for sizing
    sizing_prices = {}
    for t in tickers:
        fb = hist_prices.get(t)
        p = get_last_price(ib, t, fallback_price=fb, prefer_live=prefer_live)
        if p:
            sizing_prices[t] = p

    # 6) Compute orders
    orders = compute_orders(tickers, target_w, current_pos, sizing_prices, nav)
    log.info(f"  Orders to place: {len(orders)}")

    # 7) Submit
    if orders:
        submit_orders(ib, orders, dry_run=dry_run)
    else:
        log.info("  No trades needed.")

    _prev_weights = target_w.copy()
    log.info("Rebalance cycle complete.")


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="Live paper trading — Nasdaq-100 stocks")
    parser.add_argument("--live", action="store_true", help="Submit real orders")
    parser.add_argument("--prefer-live", action="store_true",
                        help="Try reqMktData for pricing (else hist-only)")
    parser.add_argument("--topk", type=int, default=K_DEFAULT, help="Top-K stocks")
    parser.add_argument("--host", default=TWS_HOST)
    parser.add_argument("--port", type=int, default=TWS_PORT)
    parser.add_argument("--client-id", type=int, default=TWS_CLIENT_ID)
    parser.add_argument("--run-at", default="16:45",
                        help="Earliest run time HH:MM (default 16:45)")
    parser.add_argument("--timezone", default="Europe/Madrid")
    parser.add_argument("--skip-schedule", action="store_true",
                        help="Ignore run-at guard")
    args = parser.parse_args()

    log = setup_logging()

    # Schedule guard
    if not args.skip_schedule:
        try:
            import pytz
            tz = pytz.timezone(args.timezone)
            now = dt.datetime.now(tz)
            hh, mm = map(int, args.run_at.split(":"))
            run_time = now.replace(hour=hh, minute=mm, second=0, microsecond=0)
            if now < run_time:
                log.info(f"Too early ({now.strftime('%H:%M')} < {args.run_at} "
                         f"{args.timezone}). Exiting.")
                sys.exit(0)
        except ImportError:
            log.warning("pytz not installed — skipping schedule guard")

    # Load tickers
    if not os.path.exists(TICKERS_PATH):
        log.error(f"Tickers file not found: {TICKERS_PATH}. Train first.")
        sys.exit(1)
    with open(TICKERS_PATH) as f:
        tickers = json.load(f)
    log.info(f"Universe: {len(tickers)} tickers from {TICKERS_PATH}")

    # Load scaler
    if not os.path.exists(SCALER_PATH):
        log.error(f"Scaler not found: {SCALER_PATH}. Train first.")
        sys.exit(1)
    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)

    # Load model + VecNormalize
    if not os.path.exists(MODEL_PATH):
        log.error(f"Model not found: {MODEL_PATH}. Train first.")
        sys.exit(1)
    model, venv = load_model_and_normalizer(tickers, args.topk)

    # Connect to IBKR
    log.info(f"Connecting to IBKR {args.host}:{args.port} (client {args.client_id}) …")
    ib = connect_ibkr(args.host, args.port, args.client_id)
    log.info("  Connected.")

    try:
        dry_run = not args.live
        rebalance_cycle(
            ib, model, venv, tickers, args.topk, scaler,
            dry_run=dry_run, prefer_live=args.prefer_live,
        )
    except KeyboardInterrupt:
        log.info("Interrupted.")
    except Exception as e:
        log.exception(f"Error during rebalance: {e}")
    finally:
        ib.disconnect()
        log.info("Disconnected from IBKR.")


if __name__ == "__main__":
    main()
