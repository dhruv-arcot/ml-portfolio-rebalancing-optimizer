"""Tax-aware rebalancing engine for the India mutual-fund basket.

Because FIFO is legally mandated for which mutual-fund units are deemed
sold (see tax_rules.py), the optimization lever here is *which scheme to
trim and when* -- not lot selection within a scheme. Two strategies are
implemented so main_poc.py can compare them on the same market path:

- tax_aware: only rebalances a scheme once it has drifted past
  config.DRIFT_BAND from target, and among several overweight candidates
  prefers trimming the one the trained forecaster expects to perform worst
  going forward (least opportunity cost to sell).
- naive: rebalances every checkpoint back to the exact target weight,
  picking sells purely by "most overweight first" -- no drift tolerance,
  no forecast input. Still pays FIFO-correct tax, just trades more often
  and without regard to forecasted returns.
"""

import copy
import json
import os
from typing import Dict, Optional

import pandas as pd

import config
import tax_rules as tr


def load_nav_series(label: str) -> pd.DataFrame:
    path = os.path.join(config.NAV_DIR, f"{label}.csv")
    df = pd.read_csv(path, parse_dates=["Date"])
    return df[["Date", "Close"]].sort_values("Date").reset_index(drop=True)


def load_universe_nav() -> Dict[str, pd.DataFrame]:
    return {f["label"]: load_nav_series(f["label"]) for f in config.FUND_UNIVERSE}


def load_scheme_buckets() -> Dict[str, str]:
    with open(config.SCHEME_META_PATH) as fh:
        meta = json.load(fh)
    return {label: tr.classify_tax_bucket(category) for label, category in meta.items()}


def _price_on_or_before(nav_df: pd.DataFrame, date: pd.Timestamp) -> Optional[float]:
    sub = nav_df[nav_df["Date"] <= date]
    if sub.empty:
        return None
    return float(sub["Close"].iloc[-1])


class Portfolio:
    def __init__(self, buckets: Dict[str, str]):
        self.books = {label: tr.FIFOLotBook() for label in buckets}
        self.buckets = buckets
        self.fy_tracker = tr.FYExemptionTracker()
        self.total_tax_paid = 0.0
        self.total_realized_gain = 0.0
        self.ledger = []

    def value(self, prices: Dict[str, float]) -> float:
        return sum(self.books[l].total_units() * prices[l] for l in self.books)

    def weights(self, prices: Dict[str, float]) -> Dict[str, float]:
        v = self.value(prices)
        if v <= 0:
            return {l: 0.0 for l in self.books}
        return {l: self.books[l].total_units() * prices[l] / v for l in self.books}

    def sell(self, label: str, date: pd.Timestamp, units: float, price: float) -> (float, float):
        consumed = self.books[label].sell(date, units, price)
        bucket = self.buckets[label]
        proceeds = units * price
        tax_due = 0.0
        for lot_units, gain, purchase_date in consumed:
            term = tr.classify_term(bucket, purchase_date, date)
            t = tr.compute_tax(bucket, term, gain, self.fy_tracker, date)
            tax_due += t
            self.total_realized_gain += gain
            self.ledger.append({
                "date": date, "label": label, "action": "SELL", "units": lot_units,
                "price": price, "purchase_date": purchase_date, "term": term,
                "realized_gain": gain, "tax_due": t,
            })
        self.total_tax_paid += tax_due
        return proceeds, tax_due

    def buy(self, label: str, date: pd.Timestamp, dollars: float, price: float) -> None:
        units = dollars / price
        self.books[label].buy(date, units, price)
        self.ledger.append({
            "date": date, "label": label, "action": "BUY", "units": units,
            "price": price, "purchase_date": date, "term": None,
            "realized_gain": 0.0, "tax_due": 0.0,
        })


def _simulate_full_liquidation_tax(portfolio: Portfolio, prices: Dict[str, float], date: pd.Timestamp) -> float:
    """Non-mutating: what tax would be owed if every holding were sold today?
    Used to report an apples-to-apples post-tax terminal value alongside the
    raw mark-to-market value (which otherwise flatters strategies that merely
    defer tax rather than reduce it)."""
    books_copy = copy.deepcopy(portfolio.books)
    fy_copy = copy.deepcopy(portfolio.fy_tracker)
    total_tax = 0.0
    for label, book in books_copy.items():
        units = book.total_units()
        if units <= 1e-9:
            continue
        price = prices[label]
        bucket = portfolio.buckets[label]
        for lot_units, gain, purchase_date in book.sell(date, units, price):
            term = tr.classify_term(bucket, purchase_date, date)
            total_tax += tr.compute_tax(bucket, term, gain, fy_copy, date)
    return total_tax


def run_backtest(
    tax_aware: bool,
    nav_data: Dict[str, pd.DataFrame],
    buckets: Dict[str, str],
    forecasts: Optional[dict] = None,
    target_weights: Dict[str, float] = None,
    initial_investment: float = config.INITIAL_INVESTMENT,
    drift_band: float = config.DRIFT_BAND,
) -> dict:
    target_weights = target_weights or config.TARGET_WEIGHTS
    labels = list(target_weights.keys())

    rebalance_dates = pd.date_range(start=config.BACKTEST_START, end=config.BACKTEST_END, freq=config.REBALANCE_FREQ)

    empty_series = pd.Series(dtype=float, index=pd.DatetimeIndex([]))
    pred_series = {}
    if forecasts:
        for label in labels:
            items = sorted(forecasts.get(label, {}).get("preds_by_date", {}).items())
            if items:
                dates, vals = zip(*items)
                pred_series[label] = pd.Series(vals, index=pd.DatetimeIndex(dates))
            else:
                pred_series[label] = empty_series

    def price_at(date):
        return {label: _price_on_or_before(nav_data[label], date) for label in labels}

    def predicted_return_at(date):
        out = {}
        for label in labels:
            s = pred_series.get(label, empty_series)
            s = s[s.index <= date]
            cur = _price_on_or_before(nav_data[label], date)
            if s.empty or not cur:
                out[label] = None
            else:
                out[label] = float(s.iloc[-1]) / cur - 1.0
        return out

    portfolio = Portfolio(buckets)

    first_date = rebalance_dates[0]
    prices0 = price_at(first_date)
    for label in labels:
        portfolio.buy(label, first_date, target_weights[label] * initial_investment, prices0[label])

    for date in rebalance_dates[1:]:
        prices = price_at(date)
        if any(p is None for p in prices.values()):
            continue

        value = portfolio.value(prices)
        weights = portfolio.weights(prices)
        target_dollars = {l: target_weights[l] * value for l in labels}
        current_dollars = {l: portfolio.books[l].total_units() * prices[l] for l in labels}
        drift = {l: weights[l] - target_weights[l] for l in labels}

        if tax_aware:
            overweight = [l for l in labels if drift[l] > drift_band]
            underweight = [l for l in labels if drift[l] < -drift_band]
        else:
            overweight = [l for l in labels if current_dollars[l] > target_dollars[l] + 1e-6]
            underweight = [l for l in labels if current_dollars[l] < target_dollars[l] - 1e-6]

        if not overweight or not underweight:
            continue

        buy_need = {l: target_dollars[l] - current_dollars[l] for l in underweight}
        total_buy_need = sum(buy_need.values())
        if total_buy_need <= 1e-6:
            continue

        if tax_aware:
            pred_ret = predicted_return_at(date)
            overweight_sorted = sorted(overweight, key=lambda l: pred_ret.get(l) if pred_ret.get(l) is not None else 0.0)
        else:
            overweight_sorted = sorted(overweight, key=lambda l: -drift[l])

        cash_raised = 0.0
        for l in overweight_sorted:
            if cash_raised >= total_buy_need - 1e-6:
                break
            available_excess = current_dollars[l] - target_dollars[l]
            sell_dollars = min(available_excess, total_buy_need - cash_raised)
            if sell_dollars <= 1e-6:
                continue
            units = min(sell_dollars / prices[l], portfolio.books[l].total_units())
            if units <= 1e-9:
                continue
            proceeds, tax_due = portfolio.sell(l, date, units, prices[l])
            cash_raised += proceeds - tax_due

        if cash_raised <= 1e-6:
            continue

        for l in underweight:
            share = buy_need[l] / total_buy_need if total_buy_need > 0 else 0.0
            dollars = cash_raised * share
            if dollars > 1e-6:
                portfolio.buy(l, date, dollars, prices[l])

    final_date = rebalance_dates[-1]
    final_prices = price_at(final_date)
    market_value = portfolio.value(final_prices)
    liquidation_tax = _simulate_full_liquidation_tax(portfolio, final_prices, final_date)

    return {
        "mode": "tax_aware" if tax_aware else "naive",
        "portfolio": portfolio,
        "final_date": final_date,
        "final_prices": final_prices,
        "market_value": market_value,
        "liquidation_tax": liquidation_tax,
        "after_liquidation_value": market_value - liquidation_tax,
        "total_tax_paid": portfolio.total_tax_paid,
        "total_realized_gain": portfolio.total_realized_gain,
        "num_trades": sum(1 for row in portfolio.ledger if row["action"] == "SELL"),
        "ledger": pd.DataFrame(portfolio.ledger),
    }
