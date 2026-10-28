# indicators_logic/volatility.py
import pandas as pd

from indicators_logic.common import true_range


def add_vwap(df):
    """Intraday VWAP that RESETS at every session start.

    Needs a 'Datetime' column (IST, from prepare_frame). Bars before the
    first traded volume of the day fall back to the typical price.
    """
    day = df["Datetime"].dt.normalize()
    typical = (df["High"] + df["Low"] + df["Close"]) / 3
    cum_pv = (typical * df["Volume"]).groupby(day).cumsum()
    cum_vol = df["Volume"].groupby(day).cumsum()
    df["VWAP"] = (cum_pv / cum_vol.where(cum_vol > 0)).fillna(typical)
    return df


def add_bollinger_bands(df, window=20, num_std=2):
    """Bollinger Bands (population std, as TradingView) + %B and width."""
    mid = df["Close"].rolling(window=window).mean()
    std = df["Close"].rolling(window=window).std(ddof=0)
    df["BB_Middle"] = mid
    df["BB_Upper"] = mid + num_std * std
    df["BB_Lower"] = mid - num_std * std
    band = (df["BB_Upper"] - df["BB_Lower"]).where(lambda x: x > 0)
    df["BB_PctB"] = (df["Close"] - df["BB_Lower"]) / band
    df["BB_Width"] = band / mid
    return df


def add_atr(df, period=14):
    """Average True Range (Wilder)."""
    df["ATR"] = true_range(df).ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    return df