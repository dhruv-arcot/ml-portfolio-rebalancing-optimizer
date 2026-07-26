"""End-to-end PoC: fetch NAVs -> train forecasters -> run tax-aware vs naive
rebalancing -> report the tax/terminal-value gap between them.

Usage:
    python main_poc.py              # fetch data if missing, train xgb forecasters, compare
    python main_poc.py --refetch    # force re-fetch NAV data
    python main_poc.py --no-forecast  # skip ML forecasting, tax-aware ranks sells by
                                       # magnitude-of-overweight only (isolates the pure
                                       # drift-band/FIFO/exemption tax effect from the
                                       # forecast-based scheme-selection effect)
"""

import argparse
import glob
import os

import pandas as pd

import config
import data_fetcher
import forecast
import rebalancer as rb


def ensure_data(refetch: bool = False) -> None:
    have_all = all(
        os.path.exists(os.path.join(config.NAV_DIR, f"{fund['label']}.csv"))
        for fund in config.FUND_UNIVERSE
    ) and os.path.exists(config.SCHEME_META_PATH)
    if refetch or not have_all:
        data_fetcher.fetch_universe()
    else:
        print("NAV data already present, skipping fetch (use --refetch to force).")


def print_comparison(naive: dict, aware: dict) -> None:
    def fmt(r):
        return (
            f"  trades executed:            {r['num_trades']}\n"
            f"  total realized gain:        Rs {r['total_realized_gain']:,.2f}\n"
            f"  total tax paid (in-period): Rs {r['total_tax_paid']:,.2f}\n"
            f"  terminal market value:      Rs {r['market_value']:,.2f}\n"
            f"  tax owed if liquidated now: Rs {r['liquidation_tax']:,.2f}\n"
            f"  terminal after-tax value:   Rs {r['after_liquidation_value']:,.2f}\n"
        )

    print("\n=== NAIVE (rebalance to exact target every month, no drift band, no forecast) ===")
    print(fmt(naive))
    print("=== TAX-AWARE (drift-band deferral + forecast-ranked trims + FIFO/exemption tax) ===")
    print(fmt(aware))

    tax_saved = naive["total_tax_paid"] - aware["total_tax_paid"]
    value_gap = aware["after_liquidation_value"] - naive["after_liquidation_value"]
    print("=== DELTA (tax-aware vs naive) ===")
    print(f"  in-period tax saved:         Rs {tax_saved:,.2f}")
    print(f"  fewer trades:                {naive['num_trades'] - aware['num_trades']}")
    print(f"  after-tax terminal value gap: Rs {value_gap:,.2f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--refetch", action="store_true", help="Force re-fetch NAV data from mfapi.in")
    parser.add_argument("--no-forecast", action="store_true", help="Skip ML forecasting for the tax-aware run")
    parser.add_argument("--model", default="xgb", choices=["xgb", "lstm"], help="Forecaster to train")
    args = parser.parse_args()

    ensure_data(refetch=args.refetch)

    nav_data = rb.load_universe_nav()
    buckets = rb.load_scheme_buckets()
    print("Tax buckets:", buckets)

    forecasts = None
    if not args.no_forecast:
        forecasts = forecast.train_universe(args.model)

    os.makedirs(config.RESULTS_DIR, exist_ok=True)

    naive = rb.run_backtest(tax_aware=False, nav_data=nav_data, buckets=buckets)
    aware = rb.run_backtest(tax_aware=True, nav_data=nav_data, buckets=buckets, forecasts=forecasts)

    naive["ledger"].to_csv(os.path.join(config.RESULTS_DIR, "ledger_naive.csv"), index=False)
    aware["ledger"].to_csv(os.path.join(config.RESULTS_DIR, "ledger_tax_aware.csv"), index=False)

    summary = pd.DataFrame([
        {
            "mode": r["mode"],
            "num_trades": r["num_trades"],
            "total_realized_gain": r["total_realized_gain"],
            "total_tax_paid": r["total_tax_paid"],
            "market_value": r["market_value"],
            "liquidation_tax": r["liquidation_tax"],
            "after_liquidation_value": r["after_liquidation_value"],
        }
        for r in (naive, aware)
    ])
    summary_path = os.path.join(config.RESULTS_DIR, "summary.csv")
    summary.to_csv(summary_path, index=False)
    print(f"\nWrote ledgers and {summary_path}")

    print_comparison(naive, aware)


if __name__ == "__main__":
    main()
