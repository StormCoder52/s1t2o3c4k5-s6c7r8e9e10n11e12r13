# build_dataset.py  -- run after calculate_indicators.py
import os
import sys

import pandas as pd

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ai_engine import config as C
from ai_engine.data import build_dataset, split_by_day


def baseline_table(ds):
    """What happens if you enter EVERY tradable bar. The model must beat this."""
    el = ds[ds["Eligible"]]
    rows = []
    for side in C.SIDES:
        for sl, rr in C.COMBOS:
            r_all = el[C.col_r(side, sl, rr)]
            m = r_all.notna()
            r, o = r_all[m], el[C.col_o(side, sl, rr)][m]
            rows.append({
                "side": side, "sl_atr": sl, "rr": rr,
                "tradable_%": round(m.mean() * 100, 1),
                "target_%": round((o == 1).mean() * 100, 1),
                "stop_%": round((o == -1).mean() * 100, 1),
                "timeout_%": round((o == 0).mean() * 100, 1),
                "mean_net_R": round(r.mean(), 3),
                "median_net_R": round(r.median(), 3),
                "worst_R": round(r.min(), 2),
            })
    return pd.DataFrame(rows)


def atr_table(ds):
    """1-min ATR as % of price per stock. Shows which stocks are too quiet for the cost."""
    g = ds.groupby("Ticker")["ATR_Pct"]
    t = pd.DataFrame({
        "min_%": g.min() * 100,
        "p5_%": g.quantile(0.05) * 100,
        "median_%": g.median() * 100,
    }).round(4)
    min_atr = C.COST_PCT / (C.MAX_COST_R * min(C.SL_ATR))  # ATR needed for the tightest setup
    t["tradable_at_tightest_%"] = (
        ds.assign(ok=ds["ATR_Pct"] >= min_atr).groupby("Ticker")["ok"].mean() * 100
    ).round(1)
    return t.sort_values("median_%")


if __name__ == "__main__":
    print("Building dataset...")
    ds = build_dataset()
    print("\n1-min ATR per stock (% of price):")
    print(atr_table(ds).to_string())
    print("\nBaseline (enter on every tradable bar):")
    print(baseline_table(ds).to_string(index=False))
    tr, va, te = split_by_day(ds)
    print(f"\nSplit rows -> train {len(tr)} | val {len(va)} | test {len(te)}")