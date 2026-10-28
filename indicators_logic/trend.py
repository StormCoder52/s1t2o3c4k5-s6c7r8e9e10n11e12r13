# indicators_logic/trend.py
import numpy as np
import pandas as pd

from indicators_logic.common import true_range


def add_moving_averages(df, periods=(9, 12, 26, 50, 100)):
    """EMA and SMA. Both are NaN until the full window exists."""
    for p in periods:
        df[f"EMA_{p}"] = df["Close"].ewm(span=p, adjust=False, min_periods=p).mean()
        df[f"SMA_{p}"] = df["Close"].rolling(window=p).mean()
    return df


def add_macd(df, fast=12, slow=26, signal=9):
    ema_fast = df["Close"].ewm(span=fast, adjust=False, min_periods=fast).mean()
    ema_slow = df["Close"].ewm(span=slow, adjust=False, min_periods=slow).mean()
    df["MACD_Line"] = ema_fast - ema_slow
    df["MACD_Signal"] = (
        df["MACD_Line"].ewm(span=signal, adjust=False, min_periods=signal).mean()
    )
    df["MACD_Hist"] = df["MACD_Line"] - df["MACD_Signal"]
    return df


def add_adx(df, period=14):
    """ADX with +DI / -DI using Wilder's method."""
    up = df["High"].diff()
    down = -df["Low"].diff()  # positive when the low moves down
    valid = up.notna() & down.notna()

    plus_dm = up.where((up > down) & (up > 0), 0.0).where(valid)
    minus_dm = down.where((down > up) & (down > 0), 0.0).where(valid)

    alpha = 1 / period
    atr = true_range(df).ewm(alpha=alpha, adjust=False, min_periods=period).mean()
    atr = atr.where(atr > 0)

    plus_di = 100 * plus_dm.ewm(alpha=alpha, adjust=False, min_periods=period).mean() / atr
    minus_di = 100 * minus_dm.ewm(alpha=alpha, adjust=False, min_periods=period).mean() / atr

    di_sum = (plus_di + minus_di).where(lambda x: x > 0)
    dx = 100 * (plus_di - minus_di).abs() / di_sum

    df["Plus_DI"] = plus_di
    df["Minus_DI"] = minus_di
    df["ADX"] = dx.ewm(alpha=alpha, adjust=False, min_periods=period).mean()
    return df