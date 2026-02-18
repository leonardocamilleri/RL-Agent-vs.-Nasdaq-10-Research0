# Live Paper Trading — Nasdaq-100 Stock Selection

## Overview

`live_paper_execute_stocks.py` runs the trained PPO stock-selection agent
against an IBKR TWS/Gateway paper-trading account.  It selects the top-K
stocks from ~100 Nasdaq-100 constituents and rebalances daily.

**This script is independent from the ETF pipeline** (`live_paper_execute.py`).

---

## Prerequisites

| Requirement | Notes |
|---|---|
| **IBKR TWS or Gateway** | Must be running and logged in |
| **API enabled** | TWS → File → Global Configuration → API → Settings: enable ActiveX/Socket, allow localhost |
| **Paper account** | Use port `7497` (paper) not `7496` (live) |
| **Trained model** | Run `python train_stock_agent.py` first |

### Demo Account Limitations

IBKR demo/paper accounts **cannot subscribe** to real-time market data for
individual stocks.  The script defaults to **historical daily bars** for both
feature computation and price sizing — no live quotes needed.

---

## Files Required (from training)

```
agents/ppo_stock_final.zip        # PPO model weights
agents/vecnormalize_stock.pkl     # VecNormalize running stats
agents/stock_tickers.json         # Ordered ticker list used in training
data/processed/stock_scaler.pkl   # StandardScaler fitted on training features
```

---

## Usage

### Dry Run (default — no orders placed)

```bash
python live_paper_execute_stocks.py
```

### Live Paper Orders

```bash
python live_paper_execute_stocks.py --live
```

### Custom Top-K

```bash
python live_paper_execute_stocks.py --live --topk 15
```

### Schedule Guard

The script checks if the current time (Europe/Madrid) is past `--run-at`:

```bash
python live_paper_execute_stocks.py --live --run-at 16:45 --timezone Europe/Madrid
```

If it's too early, the script logs a message and exits cleanly.
To skip the guard: `--skip-schedule`.

### All Options

```
--live              Submit real orders (default: dry run)
--prefer-live       Try reqMktData for pricing (else hist-only)
--topk N            Number of stocks to hold (default: 20)
--host HOST         TWS host (default: 127.0.0.1)
--port PORT         TWS port (default: 7497)
--client-id ID      IBKR client ID (default: 20)
--run-at HH:MM      Earliest run time (default: 16:45)
--timezone TZ        Timezone for schedule (default: Europe/Madrid)
--skip-schedule     Ignore run-at guard
```

---

## How It Works

1. **Connect** to IBKR TWS/Gateway
2. **Fetch** 2 years of daily historical bars for each ticker (via `reqHistoricalData`)
3. **Compute features** using `features/stock_builder.py` (same code as training)
4. **Scale** features with the saved `StandardScaler`
5. **Build observation** = scaled features + current portfolio weights + cash + drawdown
6. **Normalize** with saved `VecNormalize` stats
7. **Predict** scores with PPO (deterministic)
8. **Construct weights**: top-K equal-weight → max-weight clip → EWMA smoothing
9. **Compute orders**: target shares - current shares, respecting max trade size
10. **Place orders** (Market orders) or log them in dry-run mode

### Pricing

- **Default (hist-only)**: uses last close from historical daily bars
- **`--prefer-live`**: tries `reqMktData` first, falls back to historical close

### EUR Account Support

If the account base currency is EUR, the script fetches the EURUSD historical
FX rate from IDEALPRO and converts NAV to USD for position sizing.

---

## Logging

All activity is logged to both stdout and `logs/live_trading_stocks.log`.

---

## Daily Scheduling

Recommended: run once daily after US market close.  Use `cron` or Task Scheduler:

```bash
# Example crontab entry (run at 22:30 CET every weekday)
30 22 * * 1-5 cd /path/to/RL\ Agent && source venv/bin/activate && python live_paper_execute_stocks.py --live --run-at 22:00 --timezone Europe/Madrid
```

Ensure TWS is running and the laptop stays awake.

---

## Safety Constraints

| Constraint | Default |
|---|---|
| Cash buffer | 2% of NAV |
| Max weight per stock | 10% |
| Max trade per rebalance | 20% NAV |
| EWMA smoothing (alpha) | 0.10 |
