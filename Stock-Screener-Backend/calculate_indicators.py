# calculate_indicators.py
import os
import sys

import pandas as pd

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from indicators_logic.common import BAR_MINUTES, prepare_frame
from indicators_logic.momentum import add_rsi, add_tsi
from indicators_logic.trend import add_adx, add_macd, add_moving_averages
from indicators_logic.volatility import add_atr, add_bollinger_bands, add_vwap

# BAR_MINUTES (1 or 5) is set in indicators_logic/common.py
FOLDERS = {
    1: ("indian_stocks_ohlcv_csv", "calculated_output"),
    5: ("indian_stocks_ohlcv_5m", "calculated_output_5m"),
}
RAW_DIR, OUT_DIR = FOLDERS[BAR_MINUTES]
REPORT_FILE = f"data_quality_report_{BAR_MINUTES}m.csv"  # kept outside OUT_DIR on purpose
SOURCE_TZ = "auto"  # "auto" | "UTC" | "IST"


def compute_indicators(df):
    df = add_moving_averages(df)
    df = add_macd(df)
    df = add_adx(df)
    df = add_rsi(df)
    df = add_tsi(df)
    df = add_vwap(df)
    df = add_bollinger_bands(df)
    df = add_atr(df)
    return df


def process_all_stocks():
    """Returns 0 on success, 1 if nothing could be processed."""
    os.makedirs(OUT_DIR, exist_ok=True)
    if not os.path.exists(RAW_DIR):
        print(f"'{RAW_DIR}' not found. Run main.py first.")
        return 1

    files = sorted(f for f in os.listdir(RAW_DIR) if f.endswith(".csv"))
    if not files:
        print(f"No CSV files in '{RAW_DIR}'.")
        return 1

    print(f"{BAR_MINUTES}-minute data: found {len(files)} stocks in '{RAW_DIR}'.\n")
    report = []

    for i, filename in enumerate(files, 1):
        ticker = filename[: -len(".csv")]
        print(f"[{i}/{len(files)}] {ticker}", end="", flush=True)
        try:
            raw = pd.read_csv(os.path.join(RAW_DIR, filename))
            df, info = prepare_frame(raw, SOURCE_TZ)
            if df.empty:
                print(" -> no valid in-session rows, skipped.")
                report.append({"ticker": ticker, "status": "empty", **info})
                continue

            df = compute_indicators(df)
            num_cols = df.select_dtypes("number").columns
            df[num_cols] = df[num_cols].round(6)
            df.to_csv(os.path.join(OUT_DIR, filename), index=False)

            warn = ""
            if info["session_share"] < 0.95:
                warn += " [WARN: mixed timezones?]"
            if info["coverage"] < 0.90:
                warn += " [WARN: missing bars]"
            print(
                f" -> ok | tz={info['tz_used']} rows={info['rows_out']}"
                f" days={info['days']} coverage={info['coverage']:.0%}{warn}"
            )
            report.append({"ticker": ticker, "status": "ok", **info})
        except Exception as e:
            print(f" -> ERROR: {e}")
            report.append({"ticker": ticker, "status": f"error: {e}"})

    pd.DataFrame(report).to_csv(REPORT_FILE, index=False)
    n_ok = sum(1 for r in report if r["status"] == "ok")
    print(f"\nDone: {n_ok}/{len(files)} stocks. Indicators in '{OUT_DIR}/', report in '{REPORT_FILE}'.")
    return 0 if n_ok else 1


if __name__ == "__main__":
    sys.exit(process_all_stocks())