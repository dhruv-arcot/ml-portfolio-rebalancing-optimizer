# Intelligent Tax Aware Portfolio Rebalancing Solution using Long-Horizon forecasting models
An ML solution to perform tax-aware stock portfolio rebalancing 


This directory contains scripts for stock-level feature engineering, time-series forecasting (LSTM / RNN), classical ML baselines (XGBoost, Linear Regression), mutual-fund rebalancing tax simulation, and a simple LSTM-based sell/no-sell decisioner.

project/
│
├── preprocessor.py                       # Feature engineering + supervised window building + scaling
├── lstm.py                               # LSTM model architecture + training + evaluation
├── rnn.py                                # Vanilla RNN/GRU architecture + training + evaluation
├── xgb.py                                # XGBoost training + prediction
├── lr.py                                 # Linear Regression training + prediction
├── main.py                               # Orchestration (run_lstm, run_rnn, run_lr, run_xgb)
├── decide_sell.py                        # Inference-only: load model -> predict -> boolean sell decision
├── fund_net_returns_calculator.py        # Monthly rebalancing + tax simulation for fund CSVs
├── results/                              # Output models / predictions / summaries (created by runs)
└── stock_dataset/processed_stocks/       # Per-ticker CSVs (Date,Close,High,Low,Open,Volume)

Data format

Place historical stock CSVs under:
stock_dataset/processed_stocks/


CSV schema for each individual stock

"""
Date,Close,High,Low,Open,Volume
2015-01-02,24.23,24.705,23.79,24.69,212818400
2015-01-05,23.55,24.086,23.36,24.00,257142000
"""


High-level pipeline (what each run_* does)

1. Load stock CSV
2. Feature engineering (returns, moving averages, volatility, cyclical time features…)
3. Supervised window building:
    Input window: 180 trading days
    Forecast horizon: ~6 calendar months ≈ 182 days

4. Automatic alignment to previous trading day
5. Train/Test split:
    Train: 2020–2022
    Test: 2023–2025

6. Feature scaling (per stock)
7. Model training
8. Save:

    Model

    Predictions

    Summary CSV

NOTE: Everything before model training is handled only inside preprocessor.py.



Output layout example (LSTM run)

results_lstm/
├── models/
│   ├── AAPL.pt
│   ├── MSFT.pt
│   └── ...
├── AAPL_lstm_predictions.csv
├── MSFT_lstm_predictions.csv
└── summary_lstm.csv


Mutual Fund Tax Simulator

Folder layout:

funds/
  FundA.csv
  FundB.csv

prices/
  AAPL.csv
  MSFT.csv
  ...


CSV schema for each individual FUND

"""
Date,Stock,Weight
2025-01-31,AAPL,30
2025-01-31,MSFT,70
2025-02-28,AAPL,28
"""


tax_results/
├── FundA_tax_report.csv
├── FundB_tax_report.csv
└── summary_funds_tax.csv


Sell/no-sell decision: decide_sell.py
Running this as a standalone script for each individual inference.
For each constituent stock, perform inference using the stocks LSTM model to predict 6 months/ 12 months returns
The script will return a binary flag indicating wether to perofrm re-balancing or not based on net returns post taxation