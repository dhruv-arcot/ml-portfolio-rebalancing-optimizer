# ML Portfolio Rebalancing Optimizer

An intelligent machine learning solution for tax-aware stock portfolio rebalancing using long-horizon forecasting models.

## Overview

This system combines deep learning (LSTM/RNN) and classical ML approaches (XGBoost, Linear Regression) to predict stock returns and optimize portfolio rebalancing decisions while accounting for tax implications.

## Features

- **Multi-Model Forecasting**: LSTM, RNN, XGBoost, and Linear Regression models
- **Tax-Aware Rebalancing**: Optimizes trades considering capital gains tax impact
- **Feature Engineering**: Automated extraction of technical indicators and time-based features
- **Production-Ready**: Structured for deployment with proper error handling and validation
- **Comprehensive Evaluation**: Model performance tracking and comparison

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Train all models
python main.py

# Run tax-aware portfolio optimization
python fund_net_returns_calculator.py

# Make individual sell/hold decisions
python decide_sell.py
```

## Architecture

```
src/
├── preprocessor.py          # Feature engineering and data preparation
├── models/
│   ├── lstm.py             # LSTM model architecture
│   ├── rnn.py              # RNN model architecture  
│   ├── xgb.py              # XGBoost model
│   └── lr.py               # Linear Regression model
├── main.py                 # Training orchestration
├── decide_sell.py          # Inference for sell/hold decisions
└── fund_net_returns_calculator.py  # Tax-aware portfolio optimization
```

## Data Requirements

### Stock Price Data
Place historical stock CSV files in `stock_dataset/processed_stocks/`:

```csv
Date,Close,High,Low,Open,Volume
2015-01-02,24.23,24.705,23.79,24.69,212818400
2015-01-05,23.55,24.086,23.36,24.00,257142000
```

### Portfolio Data
Fund composition files in `funds/`:

```csv
Date,Stock,Weight
2025-01-31,AAPL,30
2025-01-31,MSFT,70
2025-02-28,AAPL,28
```

## Model Configuration

### Training Parameters
- **Input Window**: 180 trading days (~6 months)
- **Forecast Horizon**: 182 days (~6 months)
- **Training Period**: 2020-2022
- **Test Period**: 2023-2025
- **Features**: Returns, moving averages, volatility, momentum, volume, cyclical time

### Model Architecture
- **LSTM**: 2 layers, 128 hidden units, dropout 0.2
- **RNN**: 2 layers, 128 hidden units, dropout 0.2  
- **XGBoost**: 400 estimators, max depth 6
- **Linear Regression**: Standard sklearn implementation



## Usage

### Training Models
```bash
# Train all models with default settings
python main.py

# Models are saved in results/{model_type}/
# - models/ : Trained model files
# - {ticker}_predictions.csv : Test set predictions
# - summary_{model_type}.csv : Performance metrics
```

### Portfolio Optimization
```bash
# Run tax-aware rebalancing simulation
python fund_net_returns_calculator.py

# Requires:
# - funds/ : Portfolio composition files
# - prices/ : Current stock price data
# - Output: tax_results/ with detailed tax impact analysis
```

### Individual Stock Decisions
```bash
# Make sell/hold decision for a specific stock
python decide_sell.py

# Returns boolean flag:
# - True : Sell (predicted return < current price - tax)
# - False : Hold
```

## Output Structure

### Training Results
```
results_lstm/
├── models/
│   ├── AAPL.pt
│   ├── MSFT.pt
│   └── ...
├── AAPL_lstm_predictions.csv
├── MSFT_lstm_predictions.csv
└── summary_lstm.csv
```

### Tax Analysis Results
```
tax_results/
├── FundA_tax_report.csv
├── FundB_tax_report.csv
└── summary_funds_tax.csv
```

## Key Features

### Tax-Aware Decision Logic
The system optimizes rebalancing decisions by:
1. Predicting 6-month forward returns using trained models
2. Calculating after-tax returns: `predicted_return < current_price - capital_gains_tax`
3. Only recommending trades that improve after-tax portfolio value

### Feature Engineering
Automated extraction of:
- **Returns**: 1-day, 5-day, and log returns
- **Moving Averages**: 5, 21, 63-day periods
- **Volatility**: Rolling standard deviations
- **Momentum**: Price momentum indicators
- **Volume**: Volume ratios and averages
- **Time Features**: Cyclical day/month encoding

## Dependencies

See `requirements.txt` for full dependency list:
- numpy, pandas, scikit-learn
- torch (PyTorch for deep learning)
- xgboost (gradient boosting)
- joblib (model serialization)

## Performance Notes

- **LSTM/RNN**: Best for capturing temporal patterns
- **XGBoost**: Strong baseline with feature importance
- **Linear Regression**: Fast, interpretable benchmark
- All models use 180-day input windows with 182-day forecast horizon