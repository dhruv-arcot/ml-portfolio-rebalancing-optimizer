#!/usr/bin/env python3
"""
decide_sell.py

Load pretrained LSTM model:
    base_path/models/{TICKER}.pt

Read prices from:
    prices/{TICKER}.csv

Apply:
    SELL if predicted_price < current_price - tax_on_sale
    ELSE HOLD

Prints only a single boolean (True/False).
"""

import os
import argparse
import pandas as pd
import torch
import numpy as np

from preprocessor import create_features   
from lstm import LSTMModel                


def load_checkpoint(base_path, ticker, device="cpu"):
    ckpt_path = os.path.join(base_path, "models", f"{ticker}.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Missing model: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device)

    model_state = ckpt["model_state_dict"]
    scaler = ckpt["scaler"]
    feature_cols = ckpt["feature_cols"]
    seq_len = ckpt["seq_len"]

    n_features = len(feature_cols)

    model = LSTMModel(n_features=n_features)
    model.load_state_dict(model_state)
    model.eval()

    return model, scaler, feature_cols, seq_len


def build_input_window(prices_df, feature_cols, seq_len, decision_date):
    prices_df = prices_df.copy()
    prices_df = prices_df.sort_values("Date").reset_index(drop=True)

    feats = create_features(prices_df).dropna().reset_index(drop=True)

    feats = feats[feats["Date"] <= decision_date]
    if len(feats) < seq_len:
        return None

    window = feats.tail(seq_len)

    X = window[feature_cols].values.astype(float)
    return X


def predict_price(model, scaler, X):
    seq_len, n_features = X.shape
    X_scaled = scaler.transform(X)

    X_scaled = X_scaled.reshape(1, seq_len, n_features)

    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    with torch.no_grad():
        pred = model(X_tensor).item()

    return float(pred)


def should_sell(current_price, predicted_price, tax_rate):
    tax = max(0, (current_price * tax_rate))
    net_after_tax = current_price - tax

    return predicted_price < net_after_tax



def main():

    ticker = "NVDA"
    date = "2024-12-01"
    decision_date = pd.to_datetime(date).normalize()
    tax_rate = 0.3

    base_path = "results/models"

    model, scaler, feature_cols, seq_len = load_checkpoint(
        base_path, ticker
    )

    prices_dir = "prices"

    price_path = os.path.join(prices_dir, f"{ticker}.csv")
    if not os.path.exists(price_path):
        raise FileNotFoundError(f"Missing price file: {price_path}")

    df = pd.read_csv(price_path, parse_dates=["Date"])
    df["Date"] = df["Date"].dt.normalize()

    row = df[df["Date"] <= decision_date].tail(1)
    if row.empty:
        raise ValueError("No price available for or prior to decision_date")

    current_price = float(row["Close"].iloc[0])

 
    X = build_input_window(df, feature_cols, seq_len, decision_date)
    if X is None:
        print(False)
        return

    predicted_price = predict_price(model, scaler, X)

    decision_flag = should_sell(
        current_price=current_price,
        predicted_price=predicted_price,
        tax_rate=tax_rate
    )

    print(decision_flag)


if __name__ == "__main__":
    main()
