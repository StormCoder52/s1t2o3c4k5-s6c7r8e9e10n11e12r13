# main.py -- downloads the latest candles from TradingView and merges them into CSV files.
# Designed to run unattended on GitHub Actions (also fine to run by hand).
import json
import os
import sys
import time

import pandas as pd
from tvDatafeed import Interval, TvDatafeed

# tvdatafeed returns naive timestamps in the machine's LOCAL timezone.
# Force UTC so every file stays consistent (Replit, GitHub Actions, your own PC).
os.environ["TZ"] = "UTC"
if hasattr(time, "tzset"):
    time.tzset()

CONFIG_FILE = "stocks_list.json"

# Timeframes to collect. TradingView gives ~5000 bars per request:
#   5m -> about 67 trading days of history,  1m -> only about 13 days.
# 1m history can NOT be backfilled later. Uncomment the 1m line to keep collecting it
# (it grows the repo ~5x faster).
# Each entry: (name, TradingView interval, folder to store the CSVs)
TIMEFRAMES = [
    ("5m", Interval.in_5_minute, "indian_stocks_ohlcv_5m"),
    # ("1m", Interval.in_1_minute, "indian_stocks_ohlcv_csv"),
]

N_BARS = 5000
DELAY_SEC = 1.25       # politeness delay between stocks
RETRIES = 3            # attempts per stock (connections drop now and then)
RETRY_WAIT = 5         # seconds, multiplied by the attempt number
MAX_FAIL_RATE = 0.5    # exit with an error (red run on GitHub) if more than this fail


def load_stock_tickers():
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, "r") as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                return []
    return []


def standardize(df):
    """TradingView frame -> columns Datetime, Open, High, Low, Close, Volume."""
    df = df.reset_index()
    df = df.rename(columns={
        "open": "Open", "high": "High", "low": "Low", "close": "Close",
        "volume": "Volume", "datetime": "Datetime", "index": "Datetime",
    })
    df = df[["Datetime", "Open", "High", "Low", "Close", "Volume"]].copy()
    df["Datetime"] = pd.to_datetime(df["Datetime"]).dt.strftime("%Y-%m-%d %H:%M:%S")
    return df


def update_stock_csv(file_path, new_df):
    """Merge new bars into the stock's CSV. Returns (total_rows, newly_added_rows)."""
    new_df = standardize(new_df)
    n_before = 0
    if os.path.exists(file_path):
        existing = pd.read_csv(file_path)
        n_before = len(existing)
        combined = pd.concat([existing, new_df], ignore_index=True)
    else:
        combined = new_df

    combined = (
        combined.dropna(subset=["Datetime"])
        .drop_duplicates(subset=["Datetime"], keep="last")
        .sort_values("Datetime")
    )
    tmp = file_path + ".tmp"          # write to a temp file first so a crash
    combined.to_csv(tmp, index=False)  # can never leave a half-written CSV
    os.replace(tmp, file_path)
    return len(combined), len(combined) - n_before


def fetch_with_retry(tv, symbol, interval):
    for attempt in range(1, RETRIES + 1):
        try:
            df = tv.get_hist(symbol=symbol, exchange="NSE", interval=interval, n_bars=N_BARS)
            if df is not None and not df.empty:
                return df
        except Exception as e:
            print(f" (attempt {attempt} error: {e})", end="")
        if attempt < RETRIES:
            time.sleep(RETRY_WAIT * attempt)
    return None


def main():
    tickers = load_stock_tickers()
    if not tickers:
        print("No tickers found in stocks_list.json.")
        return 1

    print(f"Loaded {len(tickers)} stocks. Initializing TvDatafeed (guest mode)...\n")
    tv = TvDatafeed()  # pass username/password here if TradingView starts limiting guests

    attempts, failures = 0, 0
    for tf_name, interval, data_dir in TIMEFRAMES:
        os.makedirs(data_dir, exist_ok=True)
        print(f"=== {tf_name} candles -> {data_dir}/ ===")
        failed = []
        for i, ticker in enumerate(tickers, 1):
            symbol = ticker.replace(".NS", "").replace(".BO", "")
            print(f"[{i}/{len(tickers)}] {symbol}", end="", flush=True)
            attempts += 1

            df = fetch_with_retry(tv, symbol, interval)
            if df is None:
                print(" -> NO DATA")
                failed.append(ticker)
                failures += 1
            else:
                total, added = update_stock_csv(os.path.join(data_dir, f"{ticker}.csv"), df)
                first, last = pd.Timestamp(df.index[0]), pd.Timestamp(df.index[-1])
                print(f" -> +{added} new | {total} rows in file | fetched {len(df)} bars "
                      f"({first:%Y-%m-%d} to {last:%Y-%m-%d})")
            time.sleep(DELAY_SEC)

        print(f"\n{tf_name}: {len(tickers) - len(failed)}/{len(tickers)} stocks updated.")
        if failed:
            print("Failed: " + ", ".join(failed))
        print()

    if attempts and failures / attempts > MAX_FAIL_RATE:
        print(f"ERROR: {failures}/{attempts} downloads failed (TradingView limiting us?).")
        return 1
    print("Data collection complete!")
    return 0


if __name__ == "__main__":
    sys.exit(main())