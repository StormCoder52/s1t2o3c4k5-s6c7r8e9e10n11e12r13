from datetime import datetime
import json
import os
import time
import pandas as pd
from tvDatafeed import Interval, TvDatafeed

CONFIG_FILE = "stocks_list.json"
DATA_DIR = "indian_stocks_ohlcv_csv"


def ensure_directory():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)


def load_stock_tickers():
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, "r") as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                return []
    return []


def update_stock_csv(ticker_full, new_df):
    file_path = os.path.join(DATA_DIR, f"{ticker_full}.csv")

    # Reset index so datetime becomes a column
    new_df = new_df.reset_index()

    # tvdatafeed columns are typically lowercase; standardize them
    rename_map = {
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close",
        "volume": "Volume",
        "datetime": "Datetime",
        "index": "Datetime",
    }
    new_df = new_df.rename(columns=rename_map)

    # Filter and format columns
    new_df = new_df[["Datetime", "Open", "High", "Low", "Close", "Volume"]]
    new_df["Datetime"] = pd.to_datetime(new_df["Datetime"]).dt.strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    if os.path.exists(file_path):
        existing_df = pd.read_csv(file_path)
        combined_df = pd.concat([existing_df, new_df], ignore_index=True)
        combined_df.drop_duplicates(subset=["Datetime"], keep="last", inplace=True)
        combined_df.sort_values(by="Datetime", inplace=True)
    else:
        combined_df = new_df

    combined_df.to_csv(file_path, index=False)


def main():
    ensure_directory()
    tickers = load_stock_tickers()

    if not tickers:
        print("No tickers found to process.")
        return

    print(
        f"Loaded {len(tickers)} stocks from {CONFIG_FILE}. Initializing TvDatafeed...\n"
    )

    # Initialize TvDatafeed in guest (no-login) mode.
    # (Optional: Pass username="your_user", password="your_password" if you hit limits)
    tv = TvDatafeed()

    for index, ticker_full in enumerate(tickers, 1):
        # Strip exchange suffixes like '.NS' because TradingView expects just the symbol name (e.g., 'RELIANCE')
        symbol = ticker_full.replace(".NS", "").replace(".BO", "")
        print(
            f"[{index}/{len(tickers)}] Fetching 1m data for: {symbol}",
            end="",
            flush=True,
        )

        try:
            # Fetch up to 5000 historical 1-minute bars from NSE exchange
            df = tv.get_hist(
                symbol=symbol,
                exchange="NSE",
                interval=Interval.in_1_minute,
                n_bars=5000,
            )

            if df is None or df.empty:
                print(" -> No data found on TradingView.")
                time.sleep(1.25)
                continue

            update_stock_csv(ticker_full, df)
            print(" -> Updated CSV successfully.")

        except Exception as e:
            print(f" -> Error: {e}")

        # Maintain your delay to avoid rapid requests
        time.sleep(1.25)

    print("\nData collection via TradingView complete!")


if __name__ == "__main__":
    main()
