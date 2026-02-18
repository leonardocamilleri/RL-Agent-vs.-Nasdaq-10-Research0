#!/usr/bin/env python3
"""
Refresh data/universe/nasdaq100.csv from Wikipedia.

Run manually when index composition changes:
    python data/universe/refresh_nasdaq100.py

Requires: pandas, lxml (pip install lxml)
"""
import os
import pandas as pd

URL = "https://en.wikipedia.org/wiki/Nasdaq-100"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "nasdaq100.csv")


def main():
    try:
        tables = pd.read_html(URL)
        tickers = None
        for df in tables:
            for col in ("Ticker", "Symbol"):
                if col in df.columns:
                    tickers = sorted(df[col].dropna().astype(str).tolist())
                    break
            if tickers:
                break
        if not tickers:
            print("Could not find ticker column in Wikipedia tables.")
            return
        pd.DataFrame({"ticker": tickers}).to_csv(OUT, index=False)
        print(f"Updated {OUT} with {len(tickers)} tickers")
    except Exception as e:
        print(f"Failed: {e}")
        print("Update data/universe/nasdaq100.csv manually.")


if __name__ == "__main__":
    main()
