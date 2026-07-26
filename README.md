# India Mutual Fund Tax-Aware Rebalancing PoC

A proof of concept for the same idea as the parent repo -- forecast forward
returns, then decide sell/hold accounting for tax -- rebalanced against
**Indian mutual fund taxation** instead of US capital-gains rules. Lives as a
sibling to the US pipeline; reuses `../preprocessor.py`, `../lstm.py`,
`../rnn.py`, `../xgb.py`, `../lr.py` unmodified.

## Why this isn't a straight port of the US logic

- **Fund-category-dependent tax treatment**, not one flat regime. See
  [Tax rules implemented](#tax-rules-implemented) below.
- **FIFO is legally mandated** for which mutual-fund units are deemed sold on
  redemption -- you cannot cherry-pick tax-favorable lots the way
  `../fund_net_returns_calculator.py` implicitly can with its single
  average-cost basis. `tax_rules.FIFOLotBook` always consumes the oldest lot
  first. That means the optimization lever here is **which scheme to trim and
  when**, not lot selection within a scheme.
- **Financial year is Apr-Mar**, not calendar year -- matters for the annual
  LTCG exemption bucket.
- Mutual funds publish a daily **NAV**, not OHLCV. See
  [Data simplifications](#data-simplifications).

## Fund universe

A curated basket spanning every post-Budget-2024 tax bucket, pulled live from
[mfapi.in](https://www.mfapi.in/) (free, unauthenticated AMFI NAV data):

| Label | Scheme code | Category (from API) | Tax bucket |
|---|---|---|---|
| large_cap | 120586 | Equity Scheme - Large Cap Fund | equity |
| flexi_cap | 122639 | Equity Scheme - Flexi Cap Fund | equity |
| small_cap | 125354 | Equity Scheme - Small Cap Fund | equity |
| corp_bond | 118987 | Debt Scheme - Corporate Bond Fund | specified_debt |
| balanced_adv | 118968 | Hybrid Scheme - Dynamic Asset Allocation or Balanced Advantage | hybrid_35_65 (assumption -- see below) |
| gold | 119788 | Other Scheme - FoF Domestic (SBI Gold Fund) | other_nonequity |

Allocation targets are static (`config.TARGET_WEIGHTS`) -- this PoC is about
tax-aware *execution*, not allocation research.

## Tax rules implemented

Current for FY2025-26, per the Finance Act 2024 (Budget 2024) changes:

- **Equity-oriented** (>=65% equity): STCG **20%** if held <12 months; LTCG
  **12.5%** if held >=12 months, with a **Rs 1.25 lakh/financial-year
  exemption** pooled across all equity LTCG (s.112A) -- not per-trade, tracked
  in `tax_rules.FYExemptionTracker`.
- **Specified debt funds** (>=65% debt/money-market, units acquired on/after
  1 Apr 2023): **always** taxed at the investor's income-tax **slab rate**
  (`config.SLAB_RATE`, default 30%), no LTCG concept regardless of holding
  period.
- **Hybrid (35-65% equity) and other non-equity funds** (incl. gold FoFs):
  STCG at slab rate if held <24 months; flat **12.5% LTCG** if >=24 months,
  with **no exemption** (s.112).

Sources: [Finnovate — MF taxation FY2025-26](https://www.finnovate.in/learn/blog/mutual-fund-taxation-india-fy-2025-26),
[PrimeInvestor — Budget 2024 equity/debt taxation](https://primeinvestor.in/reports/budget-2024-equity-and-debt-investments-taxation/),
[Bajaj AMC — Budget 2024 MF capital gains changes](https://www.bajajamc.com/knowledge-centre/union-budget-2024-new-mutual-funds-capital-gains-tax-explained).

## Known simplifications

- No STT, exit loads, surcharge/cess, or progressive slab brackets -- one
  configurable flat slab rate for anything taxed at slab rate.
- NAV-only data means `../preprocessor.py`'s `range` and `vol_ratio` features
  are inert constants for this dataset (`data_fetcher.py` synthesizes
  `Open=High=Low=Close`, `Volume=1.0` so the shared feature pipeline runs
  unmodified). Returns, moving averages, volatility, momentum, and cyclical
  time features are the ones actually carrying signal here.
- Balanced Advantage Fund's post-Budget-2024 tax bucket is genuinely debated
  in practice (its equity allocation floats and can cross the 65% line either
  way); we hardcode it to `hybrid_35_65` rather than reconstruct actual daily
  portfolio equity % (not available for free).
- `../preprocessor.py`'s `process_file()` hardcodes its train/test split
  internally (2020-2022 / 2023-2025) rather than taking it as a parameter, so
  the backtest window here is capped at 2025-12-31 even though NAV history
  extends further -- reused as-is rather than editing the shared US pipeline.
- No loss-harvesting or loss set-off modeling -- realized losses reduce
  `total_realized_gain` bookkeeping but don't offset other gains' tax.
- Rebalance target weights are static; no allocation research.

## How it works

```
data_fetcher.py   -> data/nav/{label}.csv, data/scheme_meta.json
forecast.py       -> model_output/ (predictions + saved models per fund)
tax_rules.py      -> FIFOLotBook, FYExemptionTracker, compute_tax()
rebalancer.py     -> Portfolio + run_backtest(tax_aware: bool, ...)
main_poc.py       -> orchestrates all of the above, prints/saves the comparison
```

`rebalancer.run_backtest` implements two strategies on the same market path:

- **naive**: rebalances to exact target weight every month, sells picked by
  "most overweight first" -- no drift tolerance, no forecast input.
- **tax_aware**: only trims/tops-up a scheme once it has drifted past
  `config.DRIFT_BAND` (2026-07 default: 1.5 points -- see the comment in
  `config.py` for how this was calibrated against this basket's actual
  drift), and among overweight candidates prefers trimming the one the
  trained forecaster expects to perform worst going forward.

Both strategies pay FIFO-correct tax per `tax_rules.py`. The comparison
reports realized gain, tax paid, trade count, and terminal value both at
mark-to-market and after simulating a full liquidation (so a strategy that
merely *defers* tax isn't mistaken for one that *reduces* it).

## Running it

```bash
pip install -r ../requirements.txt   # + xgboost if not already installed

python data_fetcher.py     # pull NAV history + scheme categories from mfapi.in
python forecast.py         # train xgboost forecasters, write model_output/
python main_poc.py         # fetch (if missing) -> forecast -> compare -> results/
```

`main_poc.py` flags:
- `--refetch` -- force re-pull NAV data.
- `--no-forecast` -- skip ML forecasting; tax-aware sell ranking falls back
  to magnitude-of-overweight, isolating the pure drift-band/FIFO/exemption
  tax effect from the forecast-based scheme-selection effect.
- `--model {xgb,lstm}` -- which forecaster to train (xgb is much faster).

Note: on this machine, importing `torch` before `xgboost` in the same
process segfaults (conflicting bundled OpenMP runtimes on macOS) --
`forecast.py` imports `xgb` first for this reason. The same latent issue
exists in `../main.py`, which imports `torch` before `xgb`; it just never
surfaced there because `xgboost` wasn't installed in this environment before
now.

Outputs land in `results/`: `ledger_naive.csv`, `ledger_tax_aware.csv` (every
buy/sell with realized gain, term, tax due), and `summary.csv`.
