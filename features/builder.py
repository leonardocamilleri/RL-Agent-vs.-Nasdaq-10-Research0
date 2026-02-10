import numpy as np
import pandas as pd


def get_month_end_dates(daily_prices):
    dates = daily_prices.index.tolist()
    month_ends = []
    for i in range(len(dates) - 1):
        if dates[i].month != dates[i + 1].month or dates[i].year != dates[i + 1].year:
            month_ends.append(dates[i])
    month_ends.append(dates[-1])
    return month_ends


def compute_features(daily_prices, daily_returns, month_end_dates, tickers):
    price_idx = daily_prices.index
    records = []
    for dt in month_end_dates:
        loc = price_idx.get_loc(dt)
        if loc < 252:
            continue
        row = {}
        for ticker in tickers:
            p = daily_prices[ticker].values
            r = daily_returns[ticker].values
            row[f"{ticker}_ret_1m"] = p[loc] / p[loc - 21] - 1
            row[f"{ticker}_ret_3m"] = p[loc] / p[loc - 63] - 1
            row[f"{ticker}_ret_6m"] = p[loc] / p[loc - 126] - 1
            row[f"{ticker}_ret_12m"] = p[loc] / p[loc - 252] - 1
            row[f"{ticker}_vol_1m"] = (
                np.nanstd(r[loc - 20 : loc + 1], ddof=1) * np.sqrt(252)
            )
            row[f"{ticker}_vol_3m"] = (
                np.nanstd(r[loc - 62 : loc + 1], ddof=1) * np.sqrt(252)
            )
        ret_window = daily_returns.iloc[loc - 62 : loc + 1]
        corr_matrix = ret_window.corr().values
        n = corr_matrix.shape[0]
        mask = np.triu(np.ones((n, n), dtype=bool), k=1)
        row["avg_corr"] = float(np.nanmean(corr_matrix[mask]))
        qqq_r = daily_returns["QQQ"].values[loc - 62 : loc + 1]
        row["qqq_vol_3m"] = float(np.nanstd(qqq_r, ddof=1) * np.sqrt(252))
        qqq_p = daily_prices["QQQ"].values[loc - 251 : loc + 1]
        cummax = np.maximum.accumulate(qqq_p)
        dd = (qqq_p - cummax) / cummax
        row["qqq_max_dd_12m"] = float(dd.min())
        records.append({"date": dt, **row})
    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date")
    return df
