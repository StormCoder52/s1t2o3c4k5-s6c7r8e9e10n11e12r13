# indicators_logic/common.py
"""Shared helpers: timezone fixing, session filtering, cleaning, true range."""
import pandas as pd

IST = "Asia/Kolkata"
SESSION_START = "09:15"
SESSION_END = "15:29"  # last 1-minute bar of the NSE session
MINUTES_AT_OPEN = 9 * 60 + 15
BAR_MINUTES = 5                    # candle size of the data being processed: 1 or 5
BARS_PER_DAY = 375 // BAR_MINUTES  # NSE session = 375 minutes
OHLCV = ["Open", "High", "Low", "Close", "Volume"]
REQUIRED = ["Datetime"] + OHLCV


def true_range(df):
    """Wilder's True Range."""
    prev_close = df["Close"].shift()
    return pd.concat(
        [
            df["High"] - df["Low"],
            (df["High"] - prev_close).abs(),
            (df["Low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)


def _session_share(ts):
    """Fraction of timestamps that fall inside 09:15-15:29."""
    hm = ts.dt.strftime("%H:%M")
    return float(((hm >= SESSION_START) & (hm <= SESSION_END)).mean())


def to_ist(ts, assume="auto"):
    """Return (naive IST timestamps, source tz used, in-session share).

    tvdatafeed returns naive timestamps in the *machine's* timezone
    (UTC on Replit/Colab/GitHub), so the CSVs usually hold UTC.
    'auto' tries both readings and keeps the one that lands inside the
    NSE session.
    """
    ts = pd.to_datetime(ts)
    if ts.dt.tz is not None:
        out = ts.dt.tz_convert(IST).dt.tz_localize(None)
        return out, "tz-aware", _session_share(out)

    as_ist = ts
    as_utc = ts.dt.tz_localize("UTC").dt.tz_convert(IST).dt.tz_localize(None)
    if assume == "IST":
        return as_ist, "IST", _session_share(as_ist)
    if assume == "UTC":
        return as_utc, "UTC", _session_share(as_utc)

    s_ist, s_utc = _session_share(as_ist), _session_share(as_utc)
    if s_utc >= s_ist:
        return as_utc, "UTC", s_utc
    return as_ist, "IST", s_ist


def prepare_frame(raw, assume="auto"):
    """Clean one stock's raw 1m frame. Returns (df, info).

    - fixes timezone to IST, sorts, removes duplicate timestamps
    - drops bars outside 09:15-15:29 and invalid prices
    - leaves zero volume untouched (real data; indicators handle it)
    - adds Minute_Of_Day, Is_First_Bar, Gap_Minutes for the AI stage
    """
    missing = [c for c in REQUIRED if c not in raw.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df = raw[REQUIRED].copy()
    for c in OHLCV:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["Datetime"], tz_used, share = to_ist(df["Datetime"], assume)
    rows_in = len(df)

    df = df.dropna(subset=["Datetime", "Open", "High", "Low", "Close"])
    df = df[df["Close"] > 0]
    df = df.sort_values("Datetime").drop_duplicates("Datetime", keep="last")
    hm = df["Datetime"].dt.strftime("%H:%M")
    df = df[(hm >= SESSION_START) & (hm <= SESSION_END)]
    df["Volume"] = df["Volume"].fillna(0)
    df = df.reset_index(drop=True)

    day = df["Datetime"].dt.normalize()
    df["Minute_Of_Day"] = (
        df["Datetime"].dt.hour * 60 + df["Datetime"].dt.minute - MINUTES_AT_OPEN
    )
    df["Is_First_Bar"] = (day != day.shift()).astype(int)
    gap = df["Datetime"].diff().dt.total_seconds() / 60
    df["Gap_Minutes"] = gap.where(df["Is_First_Bar"] == 0)

    per_day = df.groupby(day).size()
    interior = per_day.iloc[1:-1] if len(per_day) >= 3 else per_day
    coverage = float(interior.median() / BARS_PER_DAY) if len(interior) else 0.0
    days = int(day.nunique())
    info = {
        "tz_used": tz_used,
        "session_share": round(share, 3),
        "rows_in": rows_in,
        "rows_out": len(df),
        "days": days,
        "coverage": round(coverage, 3),  # median bars/375 on full interior days
        "zero_volume_share": round(float((df["Volume"] == 0).mean()), 3)
        if len(df)
        else 0.0,
    }
    return df, info