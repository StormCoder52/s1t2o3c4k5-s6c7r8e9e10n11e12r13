# indicators/trend.py
import numpy as np
import pandas as pd


def add_moving_averages(df):
  """Adds SMAs and EMAs for periods: 9, 12, 26, 50, 100."""
  periods = [9, 12, 26, 50, 100]
  for p in periods:
    df[f"EMA_{p}"] = df["Close"].ewm(span=p, adjust=False).mean()
    df[f"SMA_{p}"] = df["Close"].rolling(window=p).mean()
  return df


def add_macd(df, fast=12, slow=26, signal=9):
  """Calculates Moving Average Convergence Divergence (MACD)."""
  exp1 = df["Close"].ewm(span=fast, adjust=False).mean()
  exp2 = df["Close"].ewm(span=slow, adjust=False).mean()
  df["MACD_Line"] = exp1 - exp2
  df["MACD_Signal"] = df["MACD_Line"].ewm(span=signal, adjust=False).mean()
  df["MACD_Hist"] = df["MACD_Line"] - df["MACD_Signal"]
  return df


def add_adx(df, period=14):
  """Calculates Average Directional Index (ADX) with Directional Movements."""
  high = df["High"]
  low = df["Low"]
  close = df["Close"]

  # Directional Movement (+DM, -DM)
  plus_dm = high.diff()
  minus_dm = low.diff()

  plus_dm = np.where(
      (plus_dm > minus_dm) & (plus_dm > 0), plus_dm, 0.0
  )
  minus_dm = np.where(
      (minus_dm > plus_dm) & (minus_dm > 0), minus_dm, 0.0
  )

  # True Range (TR)
  tr1 = high - low
  tr2 = (high - close.shift()).abs()
  tr3 = (low - close.shift()).abs()
  tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

  # Wilder's Smoothing
  tr_smooth = tr.ewm(alpha=1 / period, adjust=False).mean()
  plus_di = (
      100
      * pd.Series(plus_dm, index=df.index)
      .ewm(alpha=1 / period, adjust=False)
      .mean()
      / tr_smooth
  )
  minus_di = (
      100
      * pd.Series(minus_dm, index=df.index)
      .ewm(alpha=1 / period, adjust=False)
      .mean()
      / tr_smooth
  )

  dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
  dx = dx.fillna(0)
  df["ADX"] = dx.ewm(alpha=1 / period, adjust=False).mean()
  return df