# train_lgbm.py  -- Stage B: baseline model that picks WHEN to enter and WHICH setup
#
# Run order:  calculate_indicators.py -> build_dataset.py -> train_lgbm.py
# Needs:      pip install lightgbm   (falls back to scikit-learn if missing)
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ai_engine import config as C
from ai_engine.data import FEATURES, split_by_day

LGB_ERROR = None
try:
    from lightgbm import LGBMRegressor, early_stopping

    HAVE_LGB = True
except (ImportError, OSError) as e:  # OSError = missing libgomp (e.g. on Replit)
    from sklearn.ensemble import HistGradientBoostingRegressor

    HAVE_LGB = False
    LGB_ERROR = e

TARGETS = [(s, sl, rr) for s in C.SIDES for sl, rr in C.COMBOS]  # 12 trade setups
R_COLS = [C.col_r(*t) for t in TARGETS]
MIN_TRADES_VAL = 25   # a validation threshold must give at least this many trades
N_BOOT = 2000
COOLDOWN_MIN = C.HORIZON_BARS  # one trade per stock per 60 min => trades never overlap
MODEL_DIR = os.path.join("ai_engine", "checkpoints", "stage_b")


# ---------------------------------------------------------------- models
def make_model():
    if HAVE_LGB:
        return LGBMRegressor(
            n_estimators=400, learning_rate=0.03, num_leaves=15,
            min_child_samples=200, subsample=0.7, subsample_freq=1,
            colsample_bytree=0.7, reg_lambda=5.0, random_state=42,
            n_jobs=-1, verbose=-1,
        )
    return HistGradientBoostingRegressor(
        max_iter=150, learning_rate=0.05, max_leaf_nodes=15,
        min_samples_leaf=200, l2_regularization=5.0,
        early_stopping=False, random_state=42,
    )


def train_models(tr, va, features):
    models = {}
    for t, col in zip(TARGETS, R_COLS):
        mt, mv = tr[col].notna(), va[col].notna()
        if mt.sum() < 500:
            models[t] = None
            continue
        m = make_model()
        if HAVE_LGB:
            m.fit(tr.loc[mt, features], tr.loc[mt, col],
                  eval_set=[(va.loc[mv, features], va.loc[mv, col])],
                  callbacks=[early_stopping(50, verbose=False)])
        else:
            m.fit(tr.loc[mt, features], tr.loc[mt, col])
        models[t] = m
    return models


def predict_best(models, df, features):
    """Per row: best predicted net R across the 12 setups, and the R it really got."""
    P = np.full((len(df), len(TARGETS)), -np.inf)
    for i, t in enumerate(TARGETS):
        if models[t] is None:
            continue
        pred = models[t].predict(df[features])
        ok = (df["ATR_Pct"] >= C.COST_PCT / (C.MAX_COST_R * t[1])).to_numpy()
        P[:, i] = np.where(ok, pred, -np.inf)  # same tradability rule as the labels
    best = P.argmax(axis=1)
    score = P.max(axis=1)
    real = df[R_COLS].to_numpy()[np.arange(len(df)), best]
    valid = np.isfinite(score) & np.isfinite(real)
    return score, real, best, valid


# ------------------------------------------------------------- selection
def apply_cooldown(df, mask):
    """Keep at most one trade per stock per COOLDOWN_MIN minutes (df sorted by time)."""
    minutes = df["Datetime"].values.astype("datetime64[m]").astype("int64")
    tick = df["Ticker"].to_numpy()
    keep = np.zeros(len(df), dtype=bool)
    last = {}
    for i in np.flatnonzero(mask):
        if tick[i] not in last or minutes[i] - last[tick[i]] >= COOLDOWN_MIN:
            keep[i] = True
            last[tick[i]] = minutes[i]
    return keep


def select(df, score, valid, thr):
    return apply_cooldown(df, valid & (score > thr))


def pick_threshold(va, score, real, valid):
    rows = []
    for q in [0.5, 0.7, 0.8, 0.9, 0.95, 0.98]:
        thr = np.quantile(score[valid], q)
        sel = select(va, score, valid, thr)
        n = int(sel.sum())
        rows.append({"quantile": q, "threshold_R": round(float(thr), 3), "trades": n,
                     "mean_R": round(float(real[sel].mean()), 3) if n else np.nan})
    tbl = pd.DataFrame(rows)
    ok = tbl[tbl["trades"] >= MIN_TRADES_VAL].dropna()
    if ok.empty:
        return float(tbl.loc[tbl["quantile"] == 0.9, "threshold_R"].iloc[0]), tbl, False
    return float(ok.sort_values("mean_R").iloc[-1]["threshold_R"]), tbl, True


# ------------------------------------------------------------ evaluation
def evaluate(te, score, real, best, valid, thr, seed=0):
    sel = select(te, score, valid, thr)
    n = int(sel.sum())
    pool = real[valid]
    out = {"trades": n, "all_bars_mean_R": float(pool.mean())}
    if n == 0:
        return out
    r = real[sel]
    rng = np.random.default_rng(seed)
    boots = np.array([rng.choice(pool, n, replace=False).mean() for _ in range(N_BOOT)])
    out.update(mean_R=float(r.mean()), win_pct=float((r > 0).mean() * 100),
               total_R=float(r.sum()), p_value=float((boots >= r.mean()).mean()))
    picked = te.loc[sel].assign(R=r, setup=[TARGETS[b] for b in best[sel]])
    out["per_day"] = picked.groupby("Date")["R"].agg(["count", "mean"]).round(3)
    out["setups"] = picked["setup"].value_counts().head(5)
    out["side_split"] = picked["setup"].map(lambda s: s[0]).value_counts()
    return out


def run(ds, features=FEATURES, verbose=True):
    ds = ds.sort_values("Datetime", kind="stable").reset_index(drop=True)
    tr, va, te = (x.reset_index(drop=True) for x in split_by_day(ds))
    if verbose:
        print(f"Rows -> train {len(tr)} | val {len(va)} | test {len(te)}  "
              f"[{'LightGBM' if HAVE_LGB else 'scikit-learn fallback'}]")
    models = train_models(tr, va, features)

    s_va, r_va, _, v_va = predict_best(models, va, features)
    thr, tbl, found = pick_threshold(va, s_va, r_va, v_va)
    s_te, r_te, b_te, v_te = predict_best(models, te, features)
    res = evaluate(te, s_te, r_te, b_te, v_te, thr)
    res["threshold"], res["val_table"], res["val_ok"] = thr, tbl, found

    if verbose:
        print("\nValidation sweep (used only to pick the entry threshold):")
        print(tbl.to_string(index=False))
        if not found:
            print("  ! no threshold gave enough validation trades; using the 90% quantile")
        print(f"\nChosen threshold: predicted net R > {thr:.3f}")
        print("\n=== TEST (days the model never saw) ===")
        print(f"Enter every tradable bar with the model's setup: mean R = {res['all_bars_mean_R']:.3f}")
        if res["trades"] == 0:
            print("Model took 0 trades on the test days.")
        else:
            print(f"Model trades: {res['trades']} | mean R = {res['mean_R']:.3f} | "
                  f"win% = {res['win_pct']:.1f} | total R = {res['total_R']:.1f}")
            print(f"Chance that random timing does at least this well (p) = {res['p_value']:.3f}")
            print("\nPer test day:\n" + res["per_day"].to_string())
            print("\nLong/short split:\n" + res["side_split"].to_string())
            print("\nMost used setups (side, stop ATR, reward:risk):\n" + res["setups"].to_string())
    return models, res


def save_models(models, res, features):
    os.makedirs(MODEL_DIR, exist_ok=True)
    if HAVE_LGB:
        for t, m in models.items():
            if m is not None:
                getattr(m, "booster_").save_model(os.path.join(MODEL_DIR, f"{C.col_r(*t)}.txt"))
    else:
        import joblib  # installed together with scikit-learn

        joblib.dump({C.col_r(*t): m for t, m in models.items()},
                    os.path.join(MODEL_DIR, "sklearn_models.joblib"))
    with open(os.path.join(MODEL_DIR, "meta.json"), "w") as f:
        json.dump({"threshold": res["threshold"], "features": features,
                   "cost_pct": C.COST_PCT, "max_cost_r": C.MAX_COST_R}, f, indent=2)


def importance(models, features):
    gain = np.zeros(len(features))
    for m in models.values():
        if m is not None:
            gain += getattr(m, "booster_").feature_importance("gain")
    s = pd.Series(gain / gain.sum() * 100, index=features).sort_values(ascending=False)
    return s.head(12).round(1)


if __name__ == "__main__":
    if not os.path.exists(C.DATASET_FILE):
        sys.exit(f"{C.DATASET_FILE} not found. Run build_dataset.py first.")
    ds = pd.read_csv(C.DATASET_FILE, parse_dates=["Datetime"])
    if not HAVE_LGB:
        print(f"LightGBM unavailable ({LGB_ERROR}).\n"
              "Using scikit-learn's histogram gradient boosting instead.\n")
    models, res = run(ds)
    if HAVE_LGB:
        print("\nTop features (% of total gain):\n" + importance(models, FEATURES).to_string())
    save_models(models, res, FEATURES)
    print(f"\nModels saved to {MODEL_DIR}/")