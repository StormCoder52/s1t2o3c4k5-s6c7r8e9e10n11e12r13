# check_bar_times.py -- which time slots of the day are present in the 5-minute data?
# Run after `git pull`:  python check_bar_times.py   and send me the output.
import os
import sys

import pandas as pd

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from indicators_logic.common import prepare_frame

FOLDER = "indian_stocks_ohlcv_5m"

for ticker in ["AXISBANK.NS", "E2E.NS"]:
    df, info = prepare_frame(pd.read_csv(f"{FOLDER}/{ticker}.csv"))
    days = df["Datetime"].dt.normalize().nunique()
    expected = pd.date_range("2026-01-01 09:15", "2026-01-01 15:25", freq="5min").strftime("%H:%M")
    share = (df.groupby(df["Datetime"].dt.strftime("%H:%M")).size() / days).reindex(expected).fillna(0)
    present = share[share > 0]
    missing = share[share < 0.9]
    print(f"{ticker}: {len(df)} bars over {days} days | slots seen: {len(present)} of 75")
    print("  first slot:", present.index[0], "| last slot:", present.index[-1], "(IST)")
    print("  slots missing or present on <90% of days:",
          ", ".join(f"{t} ({v:.0%})" for t, v in missing.items()) or "none")
