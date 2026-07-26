"""Pulls historical NAV series for the fund universe from mfapi.in.

mfapi.in (https://www.mfapi.in/) is a free, unauthenticated JSON API over
AMFI mutual fund data. GET /mf/{scheme_code} returns scheme metadata
(including the SEBI scheme_category string used for tax-bucket
classification) plus the full daily NAV history.

The existing ML pipeline (../preprocessor.py) expects OHLCV columns. NAVs
have no intraday open/high/low/volume, so we synthesize
Open = High = Low = Close and Volume = 1.0 before handing data to that
pipeline unmodified. This makes the 'range' and 'vol_ratio' features
degenerate constants (zero variance) for this dataset -- harmless, but
carrying no signal. The features that matter for NAV series (returns,
moving averages, volatility, momentum, cyclical time) are unaffected.
"""

import json
import os
import time
import urllib.request
from typing import Dict, Tuple

import pandas as pd

import config

API_BASE = "https://api.mfapi.in/mf"


def _fetch_json(url: str, retries: int = 3, delay: float = 1.0) -> dict:
    last_err = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=30) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception as e:
            last_err = e
            time.sleep(delay)
    raise RuntimeError(f"Failed to fetch {url}: {last_err}")


def fetch_scheme(scheme_code: int) -> Tuple[pd.DataFrame, str]:
    """Returns (DataFrame[Date, Open, High, Low, Close, Volume], scheme_category)."""
    payload = _fetch_json(f"{API_BASE}/{scheme_code}")
    meta = payload.get("meta", {})
    rows = payload.get("data", [])
    if not rows:
        raise ValueError(f"No NAV data returned for scheme {scheme_code}")

    df = pd.DataFrame(rows)
    df["Date"] = pd.to_datetime(df["date"], format="%d-%m-%Y")
    df["Close"] = df["nav"].astype(float)
    df = df[["Date", "Close"]].sort_values("Date").drop_duplicates("Date").reset_index(drop=True)

    df["Open"] = df["Close"]
    df["High"] = df["Close"]
    df["Low"] = df["Close"]
    df["Volume"] = 1.0

    return df[["Date", "Open", "High", "Low", "Close", "Volume"]], meta.get("scheme_category", "Unknown")


def fetch_universe(out_dir: str = config.NAV_DIR, meta_path: str = config.SCHEME_META_PATH) -> Dict[str, str]:
    """Fetches every fund in config.FUND_UNIVERSE, writes one CSV per label plus
    a scheme_meta.json mapping label -> scheme_category. Returns that mapping."""
    os.makedirs(out_dir, exist_ok=True)
    scheme_meta = {}

    for fund in config.FUND_UNIVERSE:
        label = fund["label"]
        code = fund["scheme_code"]
        print(f"Fetching {label} (scheme {code})...")
        df, category = fetch_scheme(code)
        out_path = os.path.join(out_dir, f"{label}.csv")
        df.to_csv(out_path, index=False)
        scheme_meta[label] = category
        print(f"  -> {out_path} ({len(df)} rows, category={category!r})")

    with open(meta_path, "w") as f:
        json.dump(scheme_meta, f, indent=2)
    print(f"Wrote scheme metadata to {meta_path}")

    return scheme_meta


if __name__ == "__main__":
    fetch_universe()
