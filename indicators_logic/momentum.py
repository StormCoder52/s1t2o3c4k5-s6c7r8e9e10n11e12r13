# indicators/momentum.py
import numpy as np
import pandas as pd


def add_rsi(df, period=14):
  """Calculates Relative Strength Index (RSI) using Wilder's smoothing."""
  delta = df["Close"].diff()
  gain = delta.clip(lower=0)
  loss = -1 * delta.clip(upper=0)

  # Wilder's smoothing via EWM
  avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
  avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()

  rs = avg_gain / avg_loss.replace(0, np.nan)
  df[f"RSI_{period}"] = 100 - (100 / (1 + rs))
  df[f"RSI_{period}"] = df[f"RSI_{period}"].fillna(
      50
  )  # Fallback neutral value
  return df


def add_tsi(df, r=25, s=13):
  """Calculates True Strength Index (TSI) using double-smoothed momentum."""
  momentum = df["Close"].diff()
  abs_momentum = momentum.abs()

  # First smoothing (r period EMA), Second smoothing (s period EMA)
  smoothed_m = (
      momentum.ewm(span=r, adjust=False).mean().ewm(span=s, adjust=False).mean()
  )
  smoothed_abs_m = (
      abs_momentum.ewm(span=r, adjust=False)
      .mean()
      .ewm(span=s, adjust=False)
      .mean()
  )

  df["TSI"] = 100 * (smoothed_m / smoothed_abs_m.replace(0, np.nan))
  df["TSI"] = df["TSI"].fillna(0)
  return df