# ai_engine/config.py
"""Central settings for the intraday AI. Edit values here only."""
import os

# ---- paths ----
CALC_DIR = "calculated_output"
DATA_DIR = "ai_data"
DATASET_FILE = os.path.join(DATA_DIR, "dataset.csv.gz")
TICKER_ID_FILE = os.path.join("ai_engine", "ticker_ids.json")  # append-only

# ---- stock quality gate ----
MIN_COVERAGE = 0.90   # median bars per full day / 375
MIN_DAYS = 8

# ---- trading rules (minutes since 09:15 IST) ----
FIRST_ENTRY_MIN = 15    # 09:30
LAST_ENTRY_MIN = 330    # 14:45
FORCED_EXIT_MIN = 360   # 15:15, MIS square-off safety
HORIZON_BARS = 60       # max holding time

# ---- trade setups the AI chooses between ----
SIDES = ["long", "short"]
SL_ATR = [2.0, 3.0, 5.0]        # stop distance in 1-min ATRs (60-bar moves are ~7 ATR, so 1 ATR is just noise)
REWARD_RISK = [1.0, 2.0]        # target = RR * stop distance
COMBOS = [(sl, rr) for sl in SL_ATR for rr in REWARD_RISK]

# round-trip cost as a fraction of price (brokerage+STT+charges+slippage). ASSUMPTION: edit it.
COST_PCT = 0.0007
# A setup is only tradable if round-trip cost is at most this many R (cost / stop distance).
# Removes frozen/ultra-low-ATR bars where costs would dwarf any possible gain.
MAX_COST_R = 0.5

# ---- walk-forward split (by trading day) ----
VAL_DAYS = 3
TEST_DAYS = 3


def col_r(side, sl, rr):
    """Net R-multiple of the trade after costs."""
    return f"R_{side}_sl{sl:g}_rr{rr:g}"


def col_o(side, sl, rr):
    """Outcome code: 1 target first, -1 stop first, 0 timed out."""
    return f"O_{side}_sl{sl:g}_rr{rr:g}"