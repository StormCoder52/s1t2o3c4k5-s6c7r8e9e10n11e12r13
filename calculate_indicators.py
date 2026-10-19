# calculate_indicators.py
import os
import pandas as pd

# Absolute imports using your correct folder name: indicators_logic
from indicators_logic.momentum import add_rsi, add_tsi
from indicators_logic.trend import add_adx, add_macd, add_moving_averages
from indicators_logic.volatility import add_atr, add_bollinger_bands, add_vwap

RAW_DIR = "indian_stocks_ohlcv_csv"
OUT_DIR = "calculated_output"


def ensure_output_directory():
  """Creates the output folder if it doesn't already exist."""
  if not os.path.exists(OUT_DIR):
    os.makedirs(OUT_DIR)


def process_all_stocks():
  ensure_output_directory()

  if not os.path.exists(RAW_DIR):
    print(
        f"Raw data directory '{RAW_DIR}' not found. Please run main.py first."
    )
    return

  csv_files = [f for f in os.listdir(RAW_DIR) if f.endswith(".csv")]
  if not csv_files:
    print(f"No stock CSV files found in '{RAW_DIR}'.")
    return

  print(
      f"Found {len(csv_files)} stocks. Starting indicator calculation"
      " pipeline...\n"
  )

  for index, filename in enumerate(csv_files, 1):
    ticker = filename.replace(".csv", "")
    raw_path = os.path.join(RAW_DIR, filename)
    out_path = os.path.join(OUT_DIR, filename)

    print(
        f"[{index}/{len(csv_files)}] Processing indicators for: {ticker}",
        end="",
        flush=True,
    )

    try:
      # 1. Load raw 1-min data into memory
      df = pd.read_csv(raw_path, parse_dates=["Datetime"])

      # Clean volume zeros to protect indicator formulas
      df["Volume"] = df["Volume"].replace(0, pd.NA).ffill().bfill().fillna(1)

      # 2. Pass DataFrame through all indicator functions
      df = add_moving_averages(df)
      df = add_macd(df)
      df = add_adx(df)
      df = add_rsi(df)
      df = add_tsi(df)
      df = add_vwap(df)
      df = add_bollinger_bands(df)
      df = add_atr(df)

      # 3. Save the heavy, fully computed DataFrame into calculated_output/
      df.to_csv(out_path, index=False)
      print(" -> Calculated & Saved.")

    except Exception as e:
      print(f" -> Error: {e}")

  print(
      f"\nAll indicator calculations complete! Your heavy indicator files are"
      f" stored in: '{OUT_DIR}/'"
  )


if __name__ == "__main__":
  process_all_stocks()