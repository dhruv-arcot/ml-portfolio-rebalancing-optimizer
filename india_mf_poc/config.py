"""Central configuration for the India mutual-fund tax-aware rebalancing PoC."""

import os

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")
NAV_DIR = os.path.join(DATA_DIR, "nav")
SCHEME_META_PATH = os.path.join(DATA_DIR, "scheme_meta.json")
MODEL_OUTPUT_DIR = os.path.join(HERE, "model_output")
RESULTS_DIR = os.path.join(HERE, "results")

# Curated fund basket spanning every post-Budget-2024 tax bucket.
# Scheme codes verified live against https://api.mfapi.in/mf/{code}.
FUND_UNIVERSE = [
    {"label": "large_cap", "scheme_code": 120586},
    {"label": "flexi_cap", "scheme_code": 122639},
    {"label": "small_cap", "scheme_code": 125354},
    {"label": "corp_bond", "scheme_code": 118987},
    {"label": "balanced_adv", "scheme_code": 118968},
    {"label": "gold", "scheme_code": 119788},
]

# Static target allocation exercised by the rebalancer (allocation research is
# out of scope for this PoC — the point is tax-aware *execution*).
TARGET_WEIGHTS = {
    "large_cap": 0.25,
    "flexi_cap": 0.20,
    "small_cap": 0.15,
    "corp_bond": 0.20,
    "balanced_adv": 0.10,
    "gold": 0.10,
}

# ---------------------------------------------------------------------------
# Forecasting (reuses ../preprocessor.py, ../lstm.py, ../rnn.py, ../xgb.py, ../lr.py)
# ---------------------------------------------------------------------------
SEQ_LEN = 180
HORIZON_DAYS = 182
MIN_TRAIN_EXAMPLES = 50
TRAIN_START = "2020-01-01"
TRAIN_END = "2022-12-31"
BACKTEST_START = "2023-01-01"
# ../preprocessor.py's process_file() hardcodes its train/test split to
# 2020-01-01..2022-12-31 / 2023-01-01..2025-12-31 internally (not driven by
# the min/max_target_date args) -- we reuse it unmodified rather than editing
# the shared US pipeline, so the backtest window is capped here to match,
# even though NAV history actually extends to today.
BACKTEST_END = "2025-12-31"

# ---------------------------------------------------------------------------
# India capital-gains tax rules for mutual funds, FY2025-26 (post Budget 2024).
# Sources (see india_mf_poc/README.md for full citations):
#   - Equity-oriented funds: s.111A/112A — STCG 20%, LTCG 12.5% with a pooled
#     Rs 1.25L/financial-year exemption under s.112A.
#   - "Specified mutual funds" (>=65% debt/money-market, units acquired on or
#     after 1 Apr 2023): always taxed at slab rate, no LTCG concept at all.
#   - Hybrid (35-65% equity) and other non-equity-oriented funds (incl. FoFs
#     such as gold funds): STCG at slab rate if held < 24 months, else flat
#     12.5% LTCG with NO exemption (s.112).
# ---------------------------------------------------------------------------
EQUITY_STCG_RATE = 0.20
EQUITY_LTCG_RATE = 0.125
EQUITY_LTCG_EXEMPTION_PER_FY = 125_000.0
EQUITY_LTCG_HOLDING_DAYS = 365

OTHER_LTCG_RATE = 0.125
OTHER_LTCG_HOLDING_DAYS = 730

# Assumed flat income-tax slab rate applied to anything taxed "at slab rate"
# (specified-debt gains of any holding period, and short-term hybrid/other
# gains). A PoC simplification — no progressive bracket modeling.
SLAB_RATE = 0.30

# Indian financial year runs Apr 1 - Mar 31.
FY_START_MONTH = 4

# Rebalancing behaviour: only trim/top-up a scheme once its weight has
# drifted this many percentage points away from target. Calibrated against
# this basket's actual 2023-2025 drift (max ~5.2 points, rarely both an
# over- and under-weight breach at once) -- 0.05 produced ~0 trades, which
# demonstrates nothing; 0.015 produces a handful of trades and a clear tax
# gap vs the naive monthly-rebalance baseline.
DRIFT_BAND = 0.015

# Rebalance check frequency.
REBALANCE_FREQ = "MS"  # monthly, pandas offset alias

INITIAL_INVESTMENT = 1_000_000.0
