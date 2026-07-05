# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

A CS229 course project: ML-based forecasting of stock returns (LSTM, RNN, XGBoost, Linear Regression) feeding into a tax-aware portfolio rebalancing simulator. This is a flat script collection, not a package — there is no `src/` layout, test suite, or CI despite what the README's architecture diagram implies.

`india_mf_poc/` is a sibling PoC applying the same forecast → tax-aware-rebalance idea to Indian mutual funds under Indian tax law (FIFO-mandated lot accounting, equity/debt/hybrid tax buckets, the Rs 1.25L/FY LTCG exemption). It reuses `preprocessor.py`/`lstm.py`/`rnn.py`/`xgb.py`/`lr.py` from this directory unmodified via `sys.path` import — see `india_mf_poc/README.md` for the full design and tax-rule citations.

## Commands

```bash
pip install -r requirements.txt

python main.py                          # train all 4 model types over every CSV in data/csv_files/
python fund_net_returns_calculator.py   # tax-aware rebalancing simulation
python decide_sell.py                   # single sell/hold decision for a hardcoded ticker/date
```

There is no test runner, linter, or build step configured in this repo (a Pylint GitHub Actions workflow was added and then reverted — see `3ac55db`/`7a4a733`).

## Architecture

### Pipeline shape

All four modeling paths (`main.py`) share one preprocessing step and diverge only in model/model-input shape:

1. **`preprocessor.py`** — `process_file(csv_path, seq_len, horizon_days, ...)` is the single entry point used by every model. For one ticker's OHLCV CSV it:
   - engineers features via `create_features()` (returns, moving averages, volatility, momentum, volume ratios, cyclical day/month encodings),
   - builds sliding windows via `build_supervised()`: for each candidate target date, looks back `horizon_days` (182) days to find the input window's end, then takes `seq_len` (180) days ending there as `X`, with `Close` on the target date as `y` — i.e. the model sees a 180-day window that ends ~6 months before the date it's predicting,
   - splits by **fixed calendar ranges**, not a random split: train = target dates in 2020-01-01..2022-12-31, test = 2023-01-01..2025-12-31,
   - fits a single `StandardScaler` on flattened training windows and applies it to both train/test,
   - returns `None` if the file lacks required columns or has fewer than `min_train_examples` (default 50) training rows — callers must handle `None`.
2. Per-model training in `main.py` (`run_lstm`, `run_rnn`, `run_lr`, `run_xgb`) calls `process_file` per CSV in a directory, then:
   - LSTM/RNN consume the full `(n, seq_len, n_features)` windows directly.
   - LR/XGBoost instead call `aggregate_window_features()`, which collapses each window to `[last_step, mean, std, min, max]` per feature (concatenated) since these models can't consume sequences.
3. Each model module (`lstm.py`, `rnn.py`, `lr.py`, `xgb.py`) exposes a uniform shape: a train function, a predict/evaluate function, and a save function. Results per ticker (predictions CSV + saved model) plus one `summary_{model}.csv` land under an output directory.

### Known inconsistencies to watch for

- **Directory paths differ between README and code.** The README describes `stock_dataset/processed_stocks/`, `results/{model}/`, `funds/`, `prices/`. The actual code uses different defaults: `main.py` reads from `data/csv_files/` and writes to `model_output/{lstm,rnn,lr,xgb}/`; `fund_net_returns_calculator.py`'s `main()` hardcodes `prices_dir = "dataset/stocks"` and `funds_dir = ""` (current directory). Don't assume the README's paths are current — check the `main()` of the script you're touching.
- **`decide_sell.py` and `lstm.py` disagree on the checkpoint format.** `lstm.save()` writes `{"state_dict": ..., "metadata": {...}}`, but `decide_sell.load_checkpoint()` reads keys `model_state_dict`, `scaler`, `feature_cols`, `seq_len` directly off the top-level payload. Loading a checkpoint produced by `main.py`'s LSTM run into `decide_sell.py` as-is will `KeyError`. If asked to fix inference, this mismatch is the first thing to reconcile (either change what `lstm.save()` writes, or what `decide_sell.py` reads — check which callers exist before picking a direction).
- `fund_net_returns_calculator.py`'s tax logic (`tax_due = realized_gain * short_term_tax_rate` for both gains and losses) applies the same rate to losses as gains rather than crediting a loss offset — worth confirming intent before "fixing" it.

### Tax-aware rebalancing model (`fund_net_returns_calculator.py`)

`process_fund_file()` walks a fund's target-weight CSV (`Date, Stock, Weight`) date by date: at each date it marks positions to the current price, computes each ticker's target dollar allocation from the (re-normalized) weights, sells overweight positions down to target (realizing gain/loss against a running average cost basis) before buying underweight positions up to target (updating average cost basis on buys), and records portfolio value / realized gain / tax due per date. Prices are looked up via `pd.merge_asof(..., direction="backward")` in `get_prices_for_dates()`, so a fund date with no exact price match uses the most recent prior price.

### Data contracts

- Stock price CSVs need `Date, Close, High, Low, Open, Volume` (or `Price` as a `Close` fallback) — `preprocessor.create_features()` will raise/return `None` otherwise.
- Fund CSVs need `Date, Stock, Weight` — weights are auto-normalized per date (values >1 are treated as percentages and divided by 100, then renormalized to sum to 1).
