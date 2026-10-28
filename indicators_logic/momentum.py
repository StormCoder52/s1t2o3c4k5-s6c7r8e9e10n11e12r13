# indicators_logic/momentum.py
import pandas as pd


def add_rsi(df, period=14):
    """RSI with Wilder smoothing. NaN until `period` bars are available."""
    delta = df["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()

    total = avg_gain + avg_loss
    # 100*G/(G+L) == 100 - 100/(1+G/L) but is safe when L == 0 (gives 100)
    rsi = 100 * avg_gain / total.where(total > 0)
    rsi = rsi.mask(total == 0, 50.0)  # completely flat price -> neutral
    df[f"RSI_{period}"] = rsi
    return df


def add_tsi(df, r=25, s=13, signal=13):
    """True Strength Index (double-smoothed momentum) + signal line."""
    m = df["Close"].diff()

    def double_smooth(x):
        return (
            x.ewm(span=r, adjust=False, min_periods=r)
            .mean()
            .ewm(span=s, adjust=False, min_periods=s)
            .mean()
        )

    den = double_smooth(m.abs())
    df["TSI"] = 100 * double_smooth(m) / den.where(den > 0)
    df["TSI_Signal"] = df["TSI"].ewm(span=signal, adjust=False).mean()
    return df