# Engineering Doc: India Mutual-Fund Tax-Aware Rebalancing System

This document traces the system end to end: where the data comes from, how it's
turned into training examples, how the forecasters are trained, how Indian
mutual-fund tax law is encoded, and — the part that's easy to assume
incorrectly — exactly where tax rules do and don't touch the ML models. For
tax-rate citations and known simplifications, see [README.md](README.md); this
doc is about the *mechanics*, not the source law.

## 1. System map

```
data_fetcher.py  --pulls-->  data/nav/{label}.csv, data/scheme_meta.json
                                        |
                                        v
forecast.py  --reuses ../preprocessor.py, ../xgb.py, ../lstm.py-->
                                        |
                                        v
                          model_output/{label}_xgb.pkl (+ _predictions.csv)
                                        |
                                        v
rebalancer.py  <--tax rules from-- tax_rules.py (FIFOLotBook, FYExemptionTracker, compute_tax)
                                        |
                                        v
main_poc.py  --orchestrates-->  results/{ledger_naive,ledger_tax_aware,summary}.csv
```

Two pipelines that only meet at one point (§7): a **forecaster** that predicts
a fund's future NAV, and a **tax engine** that prices the cost of any given
trade. The rebalancer is the only module that talks to both.

## 2. Data: sources and collection

### 2.1 Fund universe

Six real, live AMFI-registered schemes (`config.FUND_UNIVERSE`), chosen to hit
every tax bucket the Finance Act 2024 created, not for return characteristics:

| Label | Scheme code | SEBI category | Tax bucket |
|---|---|---|---|
| large_cap | 120586 | Equity Scheme - Large Cap Fund | equity |
| flexi_cap | 122639 | Equity Scheme - Flexi Cap Fund | equity |
| small_cap | 125354 | Equity Scheme - Small Cap Fund | equity |
| corp_bond | 118987 | Debt Scheme - Corporate Bond Fund | specified_debt |
| balanced_adv | 118968 | Hybrid Scheme - Dynamic Asset Allocation / Balanced Advantage | hybrid_35_65 |
| gold | 119788 | Other Scheme - FoF Domestic | other_nonequity |

Target weights (`config.TARGET_WEIGHTS`) are static — this system is about
tax-aware *execution* of a fixed allocation, not allocation research.

### 2.2 Collection mechanism (`data_fetcher.py`)

`fetch_scheme(scheme_code)` hits `https://api.mfapi.in/mf/{scheme_code}` — a
free, unauthenticated JSON wrapper over AMFI's official daily NAV disclosures.
The response has two parts that get used for two different purposes:

- `data`: a plain list of `{date, nav}` records → parsed into `[Date, Close]`,
  deduplicated, sorted. Local NAV history currently runs **2013-01-02 through
  2026-07-03** (~3,100–3,320 rows per fund).
- `meta.scheme_category`: the SEBI-defined category string (e.g. `"Equity
  Scheme - Large Cap Fund"`) — this is the *only* field that later determines
  tax treatment (§6.1). It is fetched once and cached to
  `data/scheme_meta.json`, not re-derived from price behavior.

`_fetch_json` retries 3x with a 1s delay on any exception before raising.
`fetch_universe()` loops the six schemes, writes one CSV per label to
`data/nav/`, and writes the `{label: scheme_category}` map to
`data/scheme_meta.json`. `main_poc.ensure_data()` only calls this if any
output file is missing, or `--refetch` is passed.

### 2.3 The OHLCV synthesis workaround

The shared feature pipeline (`../preprocessor.py`) was written for US stock
OHLCV data. NAVs are a single daily print with no intraday open/high/low/volume,
so `fetch_scheme()` synthesizes `Open = High = Low = Close` and `Volume = 1.0`
before the CSV ever reaches `preprocessor.py` — chosen so the shared pipeline
runs completely unmodified rather than forking it for a NAV-only data shape.

This has a real, traceable cost in the resulting feature set (see §4.1): two
of the engineered features become mathematically constant and one triplet
becomes fully redundant. It's a deliberate trade — code reuse over feature
purity — documented here so it isn't mistaken for an oversight later.

### 2.4 Tax-bucket classification of the data

`rebalancer.load_scheme_buckets()` reads `scheme_meta.json` back at backtest
time and runs each `scheme_category` string through
`tax_rules.classify_tax_bucket()` (substring match, see §6.1) to produce the
`{label: bucket}` map the `Portfolio` is constructed with. So the tax
treatment of a fund is decided once, from metadata fetched alongside price
history — not inferred from anything the model produces.

## 3. Feature engineering (shared, unmodified, from `../preprocessor.py`)

### 3.1 `create_features()`

Per ticker, computes: 1-day and 5-day returns, log-return, moving averages
(5/21/63-day), rolling volatility of log-returns (21/63-day), 21-day momentum,
intraday range and its 21-day mean, a volume ratio, and cyclical
day-of-week/month encodings (sin/cos pairs).

Because of the OHLCV synthesis in §2.3, two of these are dead weight for this
dataset specifically:
- `range = (High - Low) / (Open + eps)` → always `0` (High=Low=Open here).
- `vol_ratio = Volume / rolling_mean(Volume)` → always `1.0` (Volume is
  constant `1.0`).

Additionally, the raw `Open`, `High`, `Low` columns (kept as features since
`preprocessor.py` excludes only `{Date, Price, Close}`) are exact duplicates
of `Close` for this data, so they add collinear-but-not-constant noise rather
than signal. The features actually carrying information for NAV series are
the return/MA/volatility/momentum/cyclical ones.

### 3.2 Windowing (`build_supervised()`) and forecast horizon

For each candidate target date `t` (the date whose `Close`/NAV is the label):
1. Compute `desired_input_end = t - horizon_days` (182 days, ~6 months).
2. Find the latest actual trading date at or before that point.
3. Take the `seq_len` = 180 trading days ending there as the input window `X`.
4. Label `y` = NAV on date `t`.

So every training example is genuinely forward-looking: the model sees a
180-day window that ends **six months before** the date it's asked to predict
— there is no leakage of near-term price action into the label, by
construction of the horizon gap, not just by train/test split discipline.

### 3.3 Train/test split and scaling

`process_file()` hardcodes calendar-based splits (not touched by
`india_mf_poc`, since it reuses the parent module as-is):
- **Train**: target dates 2020-01-01 → 2022-12-31.
- **Test/backtest**: target dates 2023-01-01 → 2025-12-31.

`config.py` mirrors this cap explicitly (`BACKTEST_END = "2025-12-31"`) even
though live NAV history now extends to mid-2026, because the shared
preprocessor's split is internal, not parameterized — see the comment at
`config.py:43-48`.

A single `StandardScaler` is fit per fund on the flattened **training**
window only, then applied to both train and test windows — no scaler is fit
across funds or on test data.

Funds with fewer than `min_train_examples` (50) training rows would be
dropped by `process_file()` returning `None`; all six funds in the current
universe clear this given ~13 years of history.

## 4. Forecast model training (`forecast.py`)

`forecast.py` is an adapter, not a new model: it calls `../preprocessor.py`'s
`process_file()` and the parent repo's model modules unmodified, just pointed
at `data/nav/` with India-appropriate date bounds from `config.py`.

### 4.1 XGBoost path (the default, `--model xgb`)

XGBoost can't consume a `(seq_len, n_features)` tensor, so
`_aggregate_window_features()` collapses each 180-day window per feature to
`[last_step, mean, std, min, max]`, concatenated into one flat vector (5×
the feature count). `xgb_module.train_xgb()` fits an `XGBRegressor` with
`n_estimators=400, max_depth=6, learning_rate=0.05,
objective=reg:squarederror` (`forecast.py:42-47`).

### 4.2 LSTM path (`--model lstm`)

Consumes the full `(n, 180, n_features)` tensor directly. 2-layer LSTM,
hidden size 128, dropout 0.2, trained up to 50 epochs with early stopping
(patience 8) on a 90/10 train/validation split of the training examples
(`forecast.py:34-41, 111-158`). Much slower than XGBoost for the same
backtest, per the README, hence XGBoost being the default.

### 4.3 What gets written out

Per fund: a model file (`model_output/{label}_xgb.pkl` — a joblib dump of the
fitted model plus `feature_cols` metadata, or `{label}_lstm.pt`) and a
predictions CSV (`target_date, y_true, y_pred`) for every test-window target
date. `train_universe()` returns an in-memory `{label: {preds_by_date, ...}}`
map that `rebalancer.py` consumes directly in the same process — the CSVs are
for inspection, not how predictions reach the rebalancer.

### 4.4 One local environment workaround worth knowing about

On this machine, importing `torch` before `xgboost` in the same process
segfaults (conflicting bundled OpenMP runtimes on macOS). `forecast.py`
imports `xgb` first for this reason (`forecast.py:21-29`). Nothing tax- or
model-logic related — purely an import-order landmine.

**Nothing about tax law appears anywhere in §3 or §4.** The forecaster's
features, labels, loss function, and evaluation are 100% price-derived. This
matters for §7.

## 5. The taxation model (`tax_rules.py`)

This is a standalone module with zero dependency on the forecasters — it
would work identically if every prediction in `model_output/` were deleted.

### 5.1 Tax bucket classification

`classify_tax_bucket(scheme_category)` runs the SEBI category string through
an ordered substring match (`tax_rules.py:22-35`):

| Match (first wins) | Bucket |
|---|---|
| `"Equity Scheme"` | `equity` |
| `"Hybrid Scheme"` | `hybrid_35_65` |
| `"Debt Scheme"` | `specified_debt` |
| `"Other Scheme"` | `other_nonequity` |
| `"Solution Oriented"` | `equity` |

This is a one-time, deterministic string match at data-load time (§2.4) — the
bucket is fixed for the whole backtest per fund, never recomputed from actual
daily portfolio composition (the README flags `balanced_adv` specifically as
a case where the real-world bucket can drift across the 65%-equity line;
here it's hardcoded).

### 5.2 Holding-period ("term") classification

`classify_term(bucket, purchase_date, sale_date)` (`tax_rules.py:46-51`):
- `equity`: long-term (`LT`) if held ≥ 365 days, else short-term (`ST`).
- `specified_debt`: **always `ST`** — no long-term concept exists for this
  bucket at all under the post-1-Apr-2023 rule, regardless of holding period.
- `hybrid_35_65` / `other_nonequity`: `LT` if held ≥ 730 days (24 months).

### 5.3 FIFO lot accounting

`FIFOLotBook` (`tax_rules.py:71-108`) is a per-fund ledger of `TaxLot(purchase_date,
units, cost_nav)`. `buy()` appends a new lot. `sell()` always consumes from
`self.lots[0]` (oldest first) until the requested units are filled, returning
`(units_from_lot, realized_gain, purchase_date)` per lot touched — gain is
`units × (sale_nav - lot.cost_nav)`. This is the mechanism that makes
lot-selection *not* a lever available to this system, unlike the US
average-cost model in `../fund_net_returns_calculator.py`: you cannot pick a
high-cost-basis lot to minimize gain, because Indian law mandates FIFO.

### 5.4 Financial year and the pooled LTCG exemption

`financial_year(date)` labels any date into India's Apr–Mar FY (e.g.
`"FY2024-25"`). `FYExemptionTracker` (`tax_rules.py:111-127`) tracks, per FY
string, how much of the Rs 1,25,000 equity-LTCG exemption (s.112A) has
already been consumed. `consume(date, gain)` applies whatever's left of that
FY's exemption against the incoming gain and returns the taxable residual —
critically, this is **stateful across the whole backtest and shared across
all equity funds**, not a per-trade or per-fund allowance, matching the real
law's "pooled across all equity LTCG in the FY" rule.

### 5.5 `compute_tax()` — the decision table

Given `(bucket, term, gain, fy_tracker, date)` (`tax_rules.py:130-157`):

| Bucket | Term | Tax |
|---|---|---|
| equity | ST | `gain × 20%` |
| equity | LT | `fy_tracker.consume(date, gain) × 12.5%` |
| specified_debt | (any) | `gain × slab_rate` (30% default) |
| hybrid_35_65 / other_nonequity | ST | `gain × slab_rate` |
| hybrid_35_65 / other_nonequity | LT | `gain × 12.5%` (no exemption) |

Losses (`gain <= 0`) owe zero tax and are **not** offset against other gains
— `total_realized_gain` bookkeeping includes losses, but there's no
loss-harvesting/set-off modeling (documented limitation, README §Known
simplifications).

## 6. Where the forecast and the tax engine actually meet (`rebalancer.py`)

This is the section worth reading carefully if the assumption is that tax
rules shaped how the models were trained — **they didn't.** The two systems
are wired together only inside `run_backtest()`, and only at decision time,
never at training time.

### 6.1 State

`Portfolio` (`rebalancer.py:52-95`) holds one `FIFOLotBook` per fund, one
shared `FYExemptionTracker`, and a running ledger. `Portfolio.sell()` is the
only place `tax_rules.classify_term` and `tax_rules.compute_tax` get called
during the live backtest; `Portfolio.buy()` never touches tax logic (buys
aren't taxable events).

### 6.2 Two strategies, same market path, same tax engine

`run_backtest(tax_aware: bool, ...)` walks monthly rebalance dates
(`config.REBALANCE_FREQ = "MS"`) from 2023-01-01 to 2025-12-31:

- **naive**: at every checkpoint, rebalances every fund back to its exact
  static target weight. Overweight funds are sorted by "most overweight
  first" and sold down to target regardless of how small the drift is.
- **tax_aware**: only considers a fund for trimming/topping-up once its
  weight has drifted more than `config.DRIFT_BAND` (1.5 percentage points)
  from target. Among overweight candidates past that threshold, it sells the
  one the trained forecaster expects to perform **worst** going forward
  first (`rebalancer.py:191-193`: `sorted(overweight, key=lambda l:
  pred_ret.get(l) ...)`, ascending — lowest predicted return sold first).

Both strategies call the exact same `Portfolio.sell()` → FIFO → term →
`compute_tax()` chain once a fund and unit count are chosen. **The tax engine
does not know or care which strategy picked the trade.**

### 6.3 The precise role of the forecast — and what it does *not* do

`predicted_return_at(date)` (`rebalancer.py:146-156`) converts each fund's
latest available model prediction into a simple forward return:
`predicted_nav / current_price - 1`. That single number is used for exactly
one thing: **ranking which overweight fund to sell first** when the
tax-aware strategy has already decided (via the drift band) that a
rebalance is due. It has no influence on:
- *whether* a rebalance happens at all (that's the drift band, tax-agnostic),
- *how much* to sell (that's target-weight math),
- or *the tax cost* of the sale (that's `tax_rules.py`, computed strictly
  after the fund and units are already chosen).

So concretely: the forecast is a **pure sell-ordering heuristic** (trim the
fund expected to underperform before trimming one expected to do well), and
the tax engine is a **pure cost-measurement layer** applied after that
choice. Nothing in `rebalancer.py` optimizes the sell order *for* tax
efficiency directly (e.g. preferring the fund whose FIFO lots would realize
the smallest gain, or whose bucket/term combination is cheapest) — the tax
saving the backtest shows up (§8) is a second-order effect of trading far
less often (drift-band gating) and, secondarily, of which funds happen to
get picked by the forecast ranking, not of the tax engine feeding back into
the selection itself. Worth knowing if the next iteration is "make the
rebalancer tax-optimal" rather than "measure the tax cost of a
forecast-informed rebalancer."

### 6.4 Full-liquidation tax, for an apples-to-apples comparison

`_simulate_full_liquidation_tax()` (`rebalancer.py:98-115`) is a
non-mutating, deep-copied simulation: "if every position were sold today,
what tax would be owed?" This exists so a strategy that merely *defers* tax
into the future (still owed on the same eventual gain) isn't scored as
better than one that *reduces* tax outright — both `market_value` (pre-tax,
mark-to-market) and `after_liquidation_value` (post-tax-if-liquidated) are
reported per strategy.

## 7. Orchestration (`main_poc.py`)

`main()`: ensure NAV data exists (fetch if not) → load NAV series and tax
buckets → train forecasters for all 6 funds (`forecast.train_universe`,
skippable via `--no-forecast`, which makes the tax-aware strategy fall back
to magnitude-of-overweight ranking — isolating the drift-band/FIFO/exemption
tax effect from the forecast-ranking effect) → run both backtests → write
`results/ledger_naive.csv`, `results/ledger_tax_aware.csv` (every buy/sell
with realized gain, term, tax due), and `results/summary.csv`.

## 8. Observed backtest results

From the checked-in `results/summary.csv` (2023-01-01 → 2025-12-31, Rs
10,00,000 initial investment, XGBoost forecaster):

| Metric | Naive | Tax-aware |
|---|---|---|
| Trades executed | 99 | 5 |
| Total realized gain | Rs 1,03,310.51 | Rs 41,011.50 |
| Tax paid (in-period) | Rs 10,740.48 | Rs 3,814.74 |
| Terminal market value | Rs 16,78,009.84 | Rs 16,95,537.24 |
| Tax owed if liquidated now | Rs 75,413.01 | Rs 76,757.94 |
| Terminal after-tax value | Rs 16,02,596.83 | Rs 16,18,779.30 |

Delta: **~64% less in-period tax paid**, **94 fewer trades (95% fewer)**, and
a Rs 16,182 higher after-liquidation terminal value for the tax-aware
strategy over this window — consistent with §6.3's read that most of the
saving comes from trading far less often (drift-band gating), not from any
explicit tax-minimizing trade selection.

## 9. File-by-file reference

| File | Role |
|---|---|
| `config.py` | Fund universe, target weights, forecast/backtest date bounds, all tax-rate constants, drift band |
| `data_fetcher.py` | mfapi.in client; writes `data/nav/*.csv` + `data/scheme_meta.json` |
| `forecast.py` | Adapter over `../preprocessor.py` + `../xgb.py`/`../lstm.py`; trains per-fund forecasters, writes `model_output/` |
| `tax_rules.py` | Tax bucket classifier, FIFO lots, FY exemption tracker, `compute_tax()` — no dependency on forecasts |
| `rebalancer.py` | `Portfolio` (FIFO books + ledger) and `run_backtest()`; only place forecast output and tax engine are both used |
| `main_poc.py` | End-to-end CLI orchestration |

For tax-rate sources, holding-period thresholds' legal basis, and the full
list of known simplifications (no STT/surcharge/cess, no loss set-off,
hardcoded `balanced_adv` bucket, etc.), see [README.md](README.md).
