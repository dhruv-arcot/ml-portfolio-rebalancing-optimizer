"""India mutual-fund capital-gains tax engine (FY2025-26, post Budget 2024).

Key difference from the US model in ../fund_net_returns_calculator.py: India
mandates FIFO for which mutual-fund units are deemed sold on a redemption --
lots cannot be cherry-picked for tax efficiency. So `FIFOLotBook.sell()`
always consumes the oldest lot first; the only optimization levers left to
a rebalancer are *which scheme* to trim and *when*.
"""

from dataclasses import dataclass
from typing import List, Literal, Tuple

import pandas as pd

import config

TaxBucket = Literal["equity", "hybrid_35_65", "specified_debt", "other_nonequity"]
Term = Literal["ST", "LT"]

# Substring rules over the SEBI scheme_category string returned by mfapi.in.
# Order matters: first match wins.
_CATEGORY_RULES = [
    ("Equity Scheme", "equity"),
    ("Hybrid Scheme", "hybrid_35_65"),
    ("Debt Scheme", "specified_debt"),
    ("Other Scheme", "other_nonequity"),
    ("Solution Oriented", "equity"),
]


def classify_tax_bucket(scheme_category: str) -> TaxBucket:
    for substr, bucket in _CATEGORY_RULES:
        if substr.lower() in scheme_category.lower():
            return bucket
    raise ValueError(f"Cannot classify tax bucket for scheme_category={scheme_category!r}")


def _holding_days_threshold(bucket: TaxBucket) -> int:
    if bucket == "equity":
        return config.EQUITY_LTCG_HOLDING_DAYS
    if bucket == "specified_debt":
        return None  # never long-term
    return config.OTHER_LTCG_HOLDING_DAYS  # hybrid_35_65, other_nonequity


def classify_term(bucket: TaxBucket, purchase_date: pd.Timestamp, sale_date: pd.Timestamp) -> Term:
    threshold = _holding_days_threshold(bucket)
    if threshold is None:
        return "ST"
    held_days = (pd.Timestamp(sale_date) - pd.Timestamp(purchase_date)).days
    return "LT" if held_days >= threshold else "ST"


def financial_year(date: pd.Timestamp) -> str:
    """Returns e.g. 'FY2024-25' for any date in India's Apr-Mar financial year."""
    date = pd.Timestamp(date)
    if date.month >= config.FY_START_MONTH:
        start_year = date.year
    else:
        start_year = date.year - 1
    return f"FY{start_year}-{str(start_year + 1)[-2:]}"


@dataclass
class TaxLot:
    purchase_date: pd.Timestamp
    units: float
    cost_nav: float


class FIFOLotBook:
    """Per-scheme FIFO cost-basis ledger, as mandated for Indian MF units."""

    def __init__(self):
        self.lots: List[TaxLot] = []

    def buy(self, date: pd.Timestamp, units: float, nav: float) -> None:
        if units <= 0:
            return
        self.lots.append(TaxLot(purchase_date=pd.Timestamp(date), units=units, cost_nav=nav))

    def total_units(self) -> float:
        return sum(lot.units for lot in self.lots)

    def sell(self, date: pd.Timestamp, units: float, nav: float) -> List[Tuple[float, float, Term, pd.Timestamp]]:
        """Consumes lots oldest-first. Returns a list of
        (units_from_lot, realized_gain, term, original_purchase_date) tuples,
        one per lot touched."""
        date = pd.Timestamp(date)
        remaining = units
        consumed = []

        while remaining > 1e-9 and self.lots:
            lot = self.lots[0]
            take = min(remaining, lot.units)
            gain = take * (nav - lot.cost_nav)
            # term is computed by the caller (needs the scheme's tax bucket);
            # we return the purchase date and let compute_lot_tax classify it.
            consumed.append((take, gain, lot.purchase_date))
            lot.units -= take
            remaining -= take
            if lot.units <= 1e-9:
                self.lots.pop(0)

        if remaining > 1e-9:
            raise ValueError(f"Attempted to sell {units} units but only {units - remaining} available")

        return consumed


class FYExemptionTracker:
    """Tracks the pooled Rs 1.25L/financial-year LTCG exemption under s.112A,
    which applies only to equity-bucket LTCG (not hybrid/debt/other)."""

    def __init__(self, exemption_per_fy: float = config.EQUITY_LTCG_EXEMPTION_PER_FY):
        self.exemption_per_fy = exemption_per_fy
        self._used: dict = {}

    def consume(self, date: pd.Timestamp, gain: float) -> float:
        """Applies remaining exemption for the FY containing `date` against
        `gain`, returns the taxable residual (>= 0)."""
        fy = financial_year(date)
        used_so_far = self._used.get(fy, 0.0)
        remaining = max(0.0, self.exemption_per_fy - used_so_far)
        applied = min(remaining, max(0.0, gain))
        self._used[fy] = used_so_far + applied
        return max(0.0, gain) - applied


def compute_tax(
    bucket: TaxBucket,
    term: Term,
    gain: float,
    fy_tracker: FYExemptionTracker,
    date: pd.Timestamp,
    slab_rate: float = config.SLAB_RATE,
) -> float:
    """Computes tax due on one realized-gain event. Losses (gain < 0) owe no
    tax here; they are still returned upstream so callers can track them
    (no loss-harvesting/set-off modeling in this PoC)."""
    if gain <= 0:
        return 0.0

    if bucket == "equity":
        if term == "ST":
            return gain * config.EQUITY_STCG_RATE
        taxable = fy_tracker.consume(date, gain)
        return taxable * config.EQUITY_LTCG_RATE

    if bucket == "specified_debt":
        return gain * slab_rate

    # hybrid_35_65, other_nonequity
    if term == "ST":
        return gain * slab_rate
    return gain * config.OTHER_LTCG_RATE
