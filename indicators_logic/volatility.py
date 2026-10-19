# indicators/volatility.py
import numpy as np
import pandas as pd


def add_vwap(df):
    """Calculates Volume Weighted Average Price (VWAP)."""
    q = df["Volume"]
    p = (df["High"] + df["Low"] + df["Close"]) / 3
    # Safe division guarding against zero volume
    cum_pv = (p * q).cumsum()
    cum_vol = q.cumsum().replace(0, np.nan)
    df["VWAP"] = cum_pv / cum_vol
    df["VWAP"] = df["VWAP"].fillna(df["Close"])
    return df


def add_bollinger_bands(df, window=20, num_std=2):
    """Calculates Bollinger Bands (Upper, Middle, Lower)."""
    df["BB_Middle"] = df["Close"].rolling(window=window).mean()
    rolling_std = df["Close"].rolling(window=window).std()
    df["BB_Upper"] = df["BB_Middle"] + (rolling_std * num_std)
    df["BB_Lower"] = df["BB_Middle"] - (rolling_std * num_std)
    return df


def add_atr(df, period=14):
    """Calculates Average True Range (ATR)."""
    high = df["High"]
    low = df["Low"]
    close = df["Close"]

    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    df["ATR"] = tr.ewm(alpha=1 / period, adjust=False).mean()
    return df
