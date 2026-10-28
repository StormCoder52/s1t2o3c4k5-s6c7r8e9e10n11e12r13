# ai_engine/labels.py
"""Triple-barrier labels for long and short intraday trades.

For each signal bar t:  entry = OPEN of bar t+1 (no look-ahead),
stop/target = entry -/+ k*ATR_t. Walk forward until target, stop, horizon or
the forced exit bar. If one bar touches both, the stop is assumed first.
Net R = gross R - cost/risk.
"""
import numpy as np
import pandas as pd

from ai_engine import config as C


def _first_true(mask):
    """Per row: index of first True, or width if none."""
    return np.where(mask.any(axis=1), mask.argmax(axis=1), mask.shape[1])


def label_stock(df):
    n = len(df)
    sl = np.array([c[0] for c in C.COMBOS], dtype=float)
    rr = np.array([c[1] for c in C.COMBOS], dtype=float)
    K = len(sl)
    R = np.full((n, 2, K), np.nan)
    O = np.full((n, 2, K), np.nan)

    o, h, l, c = (df[x].to_numpy(float) for x in ["Open", "High", "Low", "Close"])
    atr = df["ATR"].to_numpy(float)
    mod = df["Minute_Of_Day"].to_numpy()
    day = df["Datetime"].dt.normalize().to_numpy()

    starts = np.flatnonzero(np.r_[True, day[1:] != day[:-1]])
    ends = np.r_[starts[1:], n]

    for s, e_end in zip(starts, ends):
        ok = np.flatnonzero(mod[s:e_end] <= C.FORCED_EXIT_MIN)
        if len(ok) == 0:
            continue
        last_idx = s + ok[-1]
        for t in range(s, e_end - 1):
            if not (C.FIRST_ENTRY_MIN <= mod[t] <= C.LAST_ENTRY_MIN):
                continue
            a = atr[t]
            if not np.isfinite(a) or a <= 0:
                continue
            e = t + 1
            if e > last_idx:
                continue
            end = min(e + C.HORIZON_BARS - 1, last_idx)
            hw, lw = h[e : end + 1], l[e : end + 1]
            w = len(hw)
            entry = o[e]
            risk = sl * a
            cost_r = C.COST_PCT * entry / risk
            for si, side in enumerate(C.SIDES):
                if side == "long":
                    stop, tgt = entry - risk, entry + rr * risk
                    hit_s = lw[None, :] <= stop[:, None]
                    hit_t = hw[None, :] >= tgt[:, None]
                    timeout_r = (c[end] - entry) / risk
                else:
                    stop, tgt = entry + risk, entry - rr * risk
                    hit_s = hw[None, :] >= stop[:, None]
                    hit_t = lw[None, :] <= tgt[:, None]
                    timeout_r = (entry - c[end]) / risk
                fs, ft = _first_true(hit_s), _first_true(hit_t)
                stop_first = (fs < w) & (fs <= ft)
                tgt_first = (ft < w) & (ft < fs)
                gross = np.where(stop_first, -1.0, np.where(tgt_first, rr, timeout_r))
                code = np.where(stop_first, -1, np.where(tgt_first, 1, 0))
                ok_cost = cost_r <= C.MAX_COST_R  # else costs dominate -> not tradable
                R[t, si, :] = np.where(ok_cost, gross - cost_r, np.nan)
                O[t, si, :] = np.where(ok_cost, code, np.nan)

    cols = {}
    for si, side in enumerate(C.SIDES):
        for ki, (s_, r_) in enumerate(C.COMBOS):
            cols[C.col_r(side, s_, r_)] = R[:, si, ki]
            cols[C.col_o(side, s_, r_)] = O[:, si, ki]
    return pd.DataFrame(cols, index=df.index)