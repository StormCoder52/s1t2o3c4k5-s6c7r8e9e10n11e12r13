# ai_engine/data.py
"""Features + labels -> one dataset for all stocks."""
import json
import os

import numpy as np
import pandas as pd

from ai_engine import config as C
from ai_engine.labels import label_stock
from indicators_logic.common import BARS_PER_DAY

STOCK_FEATURES = [f"Ret_{m}" for m in C.RET_MINUTES] + [
    "Dist_VWAP", "Dist_EMA9", "Dist_EMA26", "Dist_EMA50", "EMA_Spread",
    "MACD_Hist_N", "RSI_14", "ADX", "DI_Diff", "TSI", "BB_PctB", "BB_Width",
    "ATR_Pct", "Vol_Ratio_20",
    "Gap_Pct", "Day_Ret", "Dist_DayHigh", "Dist_DayLow", "Dist_ORH", "Dist_ORL",
    "Tod",
]
MARKET_FEATURES = ["Mkt_Ret_5", "Mkt_Ret_15", "Rel_Ret_15"]
FEATURES = STOCK_FEATURES + MARKET_FEATURES


def build_features(df):
    """Scale-free, causal features (row t uses only data up to t)."""
    day = df["Datetime"].dt.normalize()
    close = df["Close"]
    atr = df["ATR"].where(df["ATR"] > 0)
    f = pd.DataFrame(index=df.index)

    for m in C.RET_MINUTES:  # day-aware: no returns across the overnight gap
        f[f"Ret_{m}"] = np.log(close / close.groupby(day).shift(m // C.BAR_MINUTES))

    f["Dist_VWAP"] = (close - df["VWAP"]) / atr
    for p in (9, 26, 50):
        f[f"Dist_EMA{p}"] = (close - df[f"EMA_{p}"]) / atr
    f["EMA_Spread"] = (df["EMA_9"] - df["EMA_26"]) / atr
    f["MACD_Hist_N"] = df["MACD_Hist"] / atr
    f["RSI_14"] = df["RSI_14"]
    f["ADX"] = df["ADX"]
    f["DI_Diff"] = df["Plus_DI"] - df["Minus_DI"]
    f["TSI"] = df["TSI"]
    f["BB_PctB"] = df["BB_PctB"]
    f["BB_Width"] = df["BB_Width"]
    f["ATR_Pct"] = atr / close
    vol_mean = df["Volume"].rolling(20, min_periods=20).mean()
    f["Vol_Ratio_20"] = (df["Volume"] / vol_mean.where(vol_mean > 0)).clip(upper=10)

    day_open = df["Open"].groupby(day).transform("first")
    prev_close = day.map(close.groupby(day).last().shift())
    f["Gap_Pct"] = day_open / prev_close - 1
    f["Day_Ret"] = close / day_open - 1
    day_high = df["High"].groupby(day).cummax()
    day_low = df["Low"].groupby(day).cummin()
    f["Dist_DayHigh"] = (close - day_high) / atr
    f["Dist_DayLow"] = (close - day_low) / atr

    in_or = df["Minute_Of_Day"] < 15  # opening range 09:15-09:29
    or_high = df["High"].where(in_or).groupby(day).transform("max")
    or_low = df["Low"].where(in_or).groupby(day).transform("min")
    f["Dist_ORH"] = ((close - or_high) / atr).where(~in_or)
    f["Dist_ORL"] = ((close - or_low) / atr).where(~in_or)
    f["Tod"] = df["Minute_Of_Day"] / 374
    return f[STOCK_FEATURES]


def stock_quality(df):
    day = df["Datetime"].dt.normalize()
    per_day = df.groupby(day).size()
    interior = per_day.iloc[1:-1] if len(per_day) >= 3 else per_day
    return {
        "days": int(len(per_day)),
        "coverage": float(interior.median() / BARS_PER_DAY) if len(interior) else 0.0,
    }


def get_ticker_ids(tickers):
    """Append-only ticker -> integer id (used for the embedding later)."""
    ids = {}
    if os.path.exists(C.TICKER_ID_FILE):
        with open(C.TICKER_ID_FILE) as f:
            ids = json.load(f)
    for t in sorted(tickers):
        if t not in ids:
            ids[t] = len(ids)
    with open(C.TICKER_ID_FILE, "w") as f:
        json.dump(ids, f, indent=2)
    return ids


def build_dataset(verbose=True):
    files = sorted(f for f in os.listdir(C.CALC_DIR) if f.endswith(".csv"))
    parts, skipped = [], []
    for fn in files:
        ticker = fn[: -len(".csv")]
        df = pd.read_csv(os.path.join(C.CALC_DIR, fn), parse_dates=["Datetime"])
        q = stock_quality(df)
        if q["days"] < C.MIN_DAYS or q["coverage"] < C.MIN_COVERAGE:
            skipped.append((ticker, q))
            continue
        part = pd.concat(
            [df[["Datetime", "Minute_Of_Day"]], build_features(df), label_stock(df)],
            axis=1,
        )
        part.insert(1, "Ticker", ticker)
        parts.append(part)
        if verbose:
            print(f"  {ticker}: {len(part)} rows, coverage {q['coverage']:.0%}")

    if not parts:
        raise RuntimeError("No stock passed the quality gate.")
    ds = pd.concat(parts, ignore_index=True)

    ids = get_ticker_ids(ds["Ticker"].unique())
    ds["Ticker_ID"] = ds["Ticker"].map(ids)

    g = ds.groupby("Datetime")  # cross-stock context at the same minute
    ds["Mkt_Ret_5"] = g["Ret_5"].transform("mean")
    ds["Mkt_Ret_15"] = g["Ret_15"].transform("mean")
    ds["Rel_Ret_15"] = ds["Ret_15"] - ds["Mkt_Ret_15"]

    ds["Date"] = ds["Datetime"].dt.strftime("%Y-%m-%d")
    r_cols = [C.col_r(sd, sl, rr) for sd in C.SIDES for sl, rr in C.COMBOS]
    ds["Eligible"] = ds[r_cols].notna().any(axis=1)  # tradable under >=1 setup

    os.makedirs(C.DATA_DIR, exist_ok=True)
    ds.to_csv(C.DATASET_FILE, index=False)
    if verbose:
        for t, q in skipped:
            print(f"  SKIPPED {t}: days={q['days']} coverage={q['coverage']:.0%}")
        print(f"Saved {len(ds)} rows ({int(ds['Eligible'].sum())} eligible) -> {C.DATASET_FILE}")
    return ds


def split_by_day(ds, val_days=C.VAL_DAYS, test_days=C.TEST_DAYS):
    """Chronological split. Labels never cross a day, so there is no leakage."""
    dates = sorted(ds["Date"].unique())
    test = set(dates[-test_days:])
    val = set(dates[-(test_days + val_days) : -test_days])
    el = ds[ds["Eligible"]]
    return (
        el[~el["Date"].isin(test | val)],
        el[el["Date"].isin(val)],
        el[el["Date"].isin(test)],
    )
