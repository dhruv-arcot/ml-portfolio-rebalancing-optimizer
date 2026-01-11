import os
import glob
import argparse
import warnings
from typing import Dict, Tuple
import numpy as np
import pandas as pd


def load_price_db(prices_dir: str) -> Dict[str, pd.DataFrame]:
    price_db = {}
    if prices_dir is None:
        return price_db
    for p in glob.glob(os.path.join(prices_dir, "*.csv")):
        ticker = os.path.splitext(os.path.basename(p))[0]
        try:
            dfp = pd.read_csv(p, parse_dates=["Date"])
            if "Close" not in dfp.columns:
                warnings.warn(f"{p} missing Close column — skipping price file")
                continue
            dfp = dfp[["Date", "Close"]].copy()
            dfp["Date"] = pd.to_datetime(dfp["Date"]).dt.normalize()
            dfp = dfp.sort_values("Date").drop_duplicates("Date").reset_index(drop=True)
            price_db[ticker] = dfp
        except Exception as e:
            warnings.warn(f"Failed reading {p}: {e}")
    return price_db


def get_prices_for_dates(price_db: Dict[str, pd.DataFrame], tickers: list, dates: pd.DatetimeIndex) -> pd.DataFrame:
    out = pd.DataFrame(index=dates)
    for ticker in tickers:
        if ticker not in price_db:
            out[ticker] = np.nan
            continue
        dfp = price_db[ticker].copy()
        # ensure Date sorted
        dfp = dfp.sort_values("Date").reset_index(drop=True)
        # prepare left frame
        left = pd.DataFrame({"Date": dates})
        # merge_asof: left must be sorted, right sorted
        merged = pd.merge_asof(left, dfp, on="Date", direction="backward")
        out[ticker] = merged["Close"].values
    return out


def process_fund_file(fpath: str, price_db: Dict[str, pd.DataFrame], investment: float, short_term_tax_rate: float, out_dir: str):
    fund_name = os.path.splitext(os.path.basename(fpath))[0]
    df = pd.read_csv(fpath)
    
    expected_cols = {"Date", "Stock", "Weight"}
    if not expected_cols.issubset(set(df.columns)):
        raise ValueError(f"{fpath} must contain columns: {expected_cols}")

    
    df["Date"] = pd.to_datetime(df["Date"]).dt.normalize()
    df = df.sort_values(["Date", "Stock"]).reset_index(drop=True)

    
    def normalize_weights(gdf):
        w = gdf["Weight"].astype(float).copy()
        
        if w.max() > 1.0:
            w = w / 100.0
        
        s = w.sum()
        if s == 0:
            return w
        return w / s

    
    holdings_by_date = {}
    for date, grp in df.groupby("Date"):
        grp2 = grp.copy().reset_index(drop=True)
        grp2["Weight"] = normalize_weights(grp2)
        holdings_by_date[pd.to_datetime(date).normalize()] = grp2[["Stock", "Weight"]].reset_index(drop=True)

    dates = sorted(holdings_by_date.keys())
    if len(dates) == 0:
        raise ValueError(f"No dates found in {fpath}")

    
    all_tickers = sorted(set(df["Stock"].unique()))

    
    price_table = pd.DataFrame(index=pd.to_datetime(dates))
    if price_db:
        price_table = get_prices_for_dates(price_db, all_tickers, pd.DatetimeIndex(dates))
    else:
        price_table = pd.DataFrame(index=pd.DatetimeIndex(dates), columns=all_tickers, dtype=float)

    
    shares = {t: 0.0 for t in all_tickers}
    avg_cost = {t: 0.0 for t in all_tickers}

    
    first_date = dates[0]
    first_weights = holdings_by_date[first_date]
    
    missing_prices = []
    for idx, row in first_weights.iterrows():
        t = row["Stock"]
        p = price_table.at[pd.to_datetime(first_date), t] if t in price_table.columns else np.nan
        if np.isnan(p):
            missing_prices.append(t)
    if missing_prices:
        warnings.warn(f"Fund {fund_name}: missing prices on first date {first_date} for tickers {missing_prices}. Those tickers will be skipped in initial allocation.")

    
    total_value = investment
    for _, r in first_weights.iterrows():
        t = r["Stock"]
        w = float(r["Weight"])
        price = price_table.at[pd.to_datetime(first_date), t] if t in price_table.columns else np.nan
        if np.isnan(price) or price <= 0:
            shares[t] = 0.0
            avg_cost[t] = 0.0
            continue
        dollars = total_value * w
        s = dollars / price
        shares[t] = s
        avg_cost[t] = price  

    
    results = []
    results.append({
        "Date": pd.to_datetime(first_date),
        "Total_Portfolio_Value": sum((shares[t] * (price_table.at[pd.to_datetime(first_date), t] if t in price_table.columns else 0.0)) for t in all_tickers),
        "Total_Sold_Dollars": 0.0,
        "Realized_Gain": 0.0,
        "Tax_Due": 0.0
    })

    
    for i in range(1, len(dates)):
        tdate = dates[i]
        prev_date = dates[i-1]
        prices_at_t = price_table.loc[pd.to_datetime(tdate)]
        portfolio_value = 0.0
        for t in all_tickers:
            p = prices_at_t.get(t) if t in prices_at_t.index or t in prices_at_t.index else np.nan
            try:
                pval = float(prices_at_t[t])
            except Exception:
                pval = np.nan
            if np.isnan(pval):
                pval = np.nan
            portfolio_value += shares.get(t, 0.0) * (pval if not np.isnan(pval) else 0.0)

        if portfolio_value <= 0 or np.isnan(portfolio_value):
            warnings.warn(f"{fund_name} on {tdate}: portfolio market value is zero or unknown (prices missing) — skipping tax calc for this date.")
            results.append({
                "Date": pd.to_datetime(tdate),
                "Total_Portfolio_Value": float(portfolio_value) if not np.isnan(portfolio_value) else np.nan,
                "Total_Sold_Dollars": 0.0,
                "Realized_Gain": 0.0,
                "Tax_Due": 0.0
            })
            continue

        desired = holdings_by_date[pd.to_datetime(tdate)]
        wsum = desired["Weight"].astype(float).sum()
        if wsum == 0:
            warnings.warn(f"{fund_name} on {tdate}: weights sum to zero — skipping")
            continue
        desired = desired.copy()
        desired["Weight"] = desired["Weight"].astype(float) / wsum

        realized_gain = 0.0
        total_sold = 0.0

        current_value = {}
        for t in all_tickers:
            pval = prices_at_t[t] if t in prices_at_t.index else np.nan
            try:
                pval = float(pval)
            except Exception:
                pval = np.nan
            current_value[t] = shares.get(t, 0.0) * (pval if not np.isnan(pval) else 0.0)


        for t in desired["Stock"].unique():
            if t not in all_tickers:
                all_tickers.append(t)
                shares[t] = 0.0
                avg_cost[t] = 0.0
                if t not in price_table.columns:
                    price_table[t] = np.nan
                    current_value[t] = 0.0

        target_dollars = {row["Stock"]: float(row["Weight"]) * portfolio_value for _, row in desired.iterrows()}
        for t in all_tickers:
            if t not in target_dollars:
                target_dollars[t] = 0.0

        for t in all_tickers:
            cur_val = current_value.get(t, 0.0)
            tgt = target_dollars.get(t, 0.0)
            if cur_val > tgt + 1e-8:
                sell_dollars = cur_val - tgt
                sell_price = prices_at_t[t] if t in prices_at_t.index else np.nan
                try:
                    sell_price = float(sell_price)
                except Exception:
                    sell_price = np.nan
                if np.isnan(sell_price) or sell_price <= 0:
                    warnings.warn(f"{fund_name} on {tdate}: missing sell price for {t}. Skipping sells for this ticker.")
                    continue
                shares_to_sell = sell_dollars / sell_price
                shares_available = shares.get(t, 0.0)
                shares_sold = min(shares_to_sell, shares_available)
                cost = avg_cost.get(t, 0.0)
                gain = shares_sold * (sell_price - cost)
                if gain < 0:
                    pass
                shares[t] = shares.get(t, 0.0) - shares_sold
                total_sold += shares_sold * sell_price
                realized_gain += gain
        
        for t in all_tickers:
            pval = prices_at_t[t] if t in prices_at_t.index else np.nan
            try:
                pval = float(pval)
            except Exception:
                pval = np.nan
            current_value[t] = shares.get(t, 0.0) * (pval if not np.isnan(pval) else 0.0)

        for t, tgt in target_dollars.items():
            cur_val = current_value.get(t, 0.0)
            if tgt > cur_val + 1e-8:
                buy_dollars = tgt - cur_val
                buy_price = prices_at_t[t] if t in prices_at_t.index else np.nan
                try:
                    buy_price = float(buy_price)
                except Exception:
                    buy_price = np.nan
                if np.isnan(buy_price) or buy_price <= 0:
                    warnings.warn(f"{fund_name} on {tdate}: missing buy price for {t}. Skipping buys for this ticker.")
                    continue
                shares_bought = buy_dollars / buy_price
                old_shares = shares.get(t, 0.0)
                old_cost = avg_cost.get(t, 0.0)
                new_total_shares = old_shares + shares_bought
                if new_total_shares > 0:
                    new_avg = (old_shares * old_cost + shares_bought * buy_price) / new_total_shares
                else:
                    new_avg = 0.0
                avg_cost[t] = new_avg
                shares[t] = new_total_shares

        new_portfolio_value = 0.0
        for t in all_tickers:
            pval = prices_at_t[t] if t in prices_at_t.index else np.nan
            try:
                pval = float(pval)
            except Exception:
                pval = np.nan
            if np.isnan(pval):
                pval = 0.0
            new_portfolio_value += shares.get(t, 0.0) * pval

        tax_due = 0.0
        if realized_gain > 0:
            tax_due = realized_gain * short_term_tax_rate
        elif realized_gain < 0:
            tax_due = realized_gain * short_term_tax_rate 

        results.append({
            "Date": pd.to_datetime(tdate),
            "Total_Portfolio_Value": float(new_portfolio_value),
            "Total_Sold_Dollars": float(total_sold),
            "Realized_Gain": float(realized_gain),
            "Tax_Due": float(tax_due)
        })


    res_df = pd.DataFrame(results)
    out_path = os.path.join(out_dir, f"{fund_name}_tax_report.csv")
    os.makedirs(out_dir, exist_ok=True)
    res_df.to_csv(out_path, index=False)
    print(f"Wrote {out_path}")
    return res_df


def main():
    

    funds_dir = ""
    prices_dir = "dataset/stocks"
    investment = 100000.0
    tax_rate = 0.30
    out_dir = "./rebalance_tax_results"

    
    out_dir = "./rebalance_tax_results"

    
    price_db = {}
    if prices_dir:
        print(f"Loading prices from {prices_dir}...")
        price_db = load_price_db(prices_dir)
        print(f"Loaded prices for {len(price_db)} tickers.")

    
    fund_files = glob.glob(os.path.join(funds_dir, "*.csv"))
    if len(fund_files) == 0:
        raise ValueError(f"No fund CSVs found in {funds_dir}")

    summary_rows = []
    for f in fund_files:
        try:
            df_report = process_fund_file(f, price_db, investment, tax_rate, out_dir)
            total_tax = df_report["Tax_Due"].sum()
            total_gain = df_report["Realized_Gain"].sum()
            final_value = df_report["Total_Portfolio_Value"].iloc[-1] if df_report.shape[0] > 0 else np.nan
            summary_rows.append({
                "fund": os.path.splitext(os.path.basename(f))[0],
                "total_realized_gain": float(total_gain),
                "total_tax_due": float(total_tax),
                "final_portfolio_value": float(final_value)
            })
        except Exception as e:
            warnings.warn(f"Error processing {f}: {e}")

    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(out_dir, "summary_funds_tax.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"Summary written to {summary_path}")

if __name__ == "__main__":
    main()
