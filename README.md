# ML Portfolio Rebalancing Optimizer - US ETF Implementation

An intelligent machine learning solution for tax-aware stock portfolio rebalancing using US capital gains taxation rules.

## Overview

This system combines deep learning (LSTM/RNN) and classical ML approaches (XGBoost, Linear Regression) to predict stock returns and optimize portfolio rebalancing decisions while accounting for US federal capital gains tax impact.

## Features

- **Multi-Model Forecasting**: LSTM, RNN, XGBoost, and Linear Regression models for stock return prediction
- **US Tax-Aware Rebalancing**: Optimizes trades considering short-term vs long-term capital gains rates
- **Feature Engineering**: Automated extraction of technical indicators and time-based features
- **Production-Ready**: Structured for deployment with proper error handling and validation
- **Comprehensive Evaluation**: Model performance tracking and comparison

## US Tax Rules Implemented

This implementation accounts for US federal capital gains taxation:

- **Short-term Capital Gains (holding ≤ 1 year)**: Taxed as ordinary income (varies by bracket, typically 20-37% federal)
- **Long-term Capital Gains (holding > 1 year)**: Preferential rates (0%, 15%, or 20% federal)
- **Tax-Loss Harvesting**: Strategy considers realized losses to offset gains
- **Average-Cost Basis**: Implicit lot selection using average-cost method

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
├── preprocessor.py              # Feature engineering and data preparation
├── lstm.py                      # LSTM model (2 layers, 128 units)
├── rnn.py                       # RNN model (2 layers, 128 units)
├── xgb.py                       # XGBoost model (400 estimators)
├── lr.py                        # Linear Regression model
├── main.py                      # Training orchestration
├── decide_sell.py               # Inference for sell/hold decisions
└── fund_net_returns_calculator.py  # Tax-aware portfolio optimization
```

## Data Requirements

### Stock Price Data
Place historical stock CSV files in `data/csv_files/`:

```csv
Date,Close,High,Low,Open,Volume
2015-01-02,24.23,24.705,23.79,24.69,212818400
2015-01-05,23.55,24.086,23.36,24.00,257142000
```

### Portfolio Composition
Fund composition files in `funds/`:

```csv
Date,Stock,Weight
2025-01-31,AAPL,30
2025-01-31,MSFT,70
2025-02-28,AAPL,28
2025-02-28,MSFT,72
```

### Price Data for Rebalancing
Current prices in `prices/`:

```csv
Date,Stock,Price
2025-01-31,AAPL,150.25
2025-01-31,MSFT,320.45
```

## Model Configuration

### Training Parameters
- **Input Window**: 180 trading days (~6 months)
- **Forecast Horizon**: 182 days (~6 months)
- **Training Period**: 2020-01-01 to 2022-12-31
- **Test Period**: 2023-01-01 to 2025-12-31
- **Features**: Returns, moving averages, volatility, momentum, volume, cyclical time

### Model Architectures
- **LSTM**: 2 layers, 128 hidden units, dropout 0.2
- **RNN**: 2 layers, 128 hidden units, dropout 0.2
- **XGBoost**: 400 estimators, max depth 6
- **Linear Regression**: Standard sklearn implementation

## Usage

### Training Models

```bash
# Train all models with default settings
python main.py

# Models are saved in model_output/{lstm,rnn,lr,xgb}/
# Output includes:
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
# Output: Detailed tax impact analysis with realized gains and tax due
```

### Individual Stock Decisions

```bash
# Make sell/hold decision for a specific stock
python decide_sell.py

# Returns boolean flag:
# - True : Sell (predicted return < threshold after tax)
# - False : Hold (expected performance justifies holding)
```

## Output Structure

### Training Results
```
model_output/{lstm,rnn,lr,xgb}/
├── models/
│   ├── AAPL.pt
│   ├── MSFT.pt
│   └── ...
├── AAPL_{model_type}_predictions.csv
├── MSFT_{model_type}_predictions.csv
└── summary_{model_type}.csv
```

### Tax Analysis Results
```
tax_results/
├── {fund}_tax_report.csv
├── {fund}_detailed_ledger.csv
└── summary_funds_tax.csv
```

## Tax-Aware Decision Logic

The system optimizes rebalancing decisions by:

1. **Predict**: Calculate 6-month forward returns using trained models
2. **Calculate After-Tax Returns**: Determine tax impact based on holding period
3. **Compare**: Evaluate predicted return vs capital gains tax cost
4. **Decide**: Only recommend trades that improve after-tax portfolio value

### Example Calculation
```
Current Position: AAPL at $100 (basis $80, +$20 gain)
Holding Period: 8 months (short-term)
Predicted 6-month Return: +15%

Tax Calculation (STCG = 37% federal + state):
- Gain: $20
- Tax if sold: $20 × 37% = $7.40
- After-tax proceeds: $100 + $20 - $7.40 = $112.60

Decision:
- If predicted return (+15%) × $100 = $115 > $112.60 → HOLD
- If predicted return (+5%) × $100 = $105 < $112.60 → SELL
```

## Key Features

### Feature Engineering

Automated extraction of:
- **Returns**: 1-day, 5-day, and log returns
- **Moving Averages**: 5, 21, 63-day periods
- **Volatility**: Rolling standard deviations
- **Momentum**: Price momentum indicators
- **Volume**: Volume ratios and averages
- **Time Features**: Cyclical day/month encoding

### Model Comparison

All models are evaluated on:
- **Mean Squared Error (MSE)**: Magnitude of prediction errors
- **Mean Absolute Error (MAE)**: Average absolute deviation
- **Directional Accuracy**: Percentage of correctly predicted direction
- **Sharpe Ratio**: Risk-adjusted performance
- **Cumulative Returns**: Total backtest performance

## Dependencies

See `requirements.txt` for full dependency list:
- numpy, pandas, scikit-learn
- torch (PyTorch for deep learning models)
- xgboost (gradient boosting)
- joblib (model serialization)

## Performance Notes

- **LSTM/RNN**: Best for capturing temporal patterns in stock returns
- **XGBoost**: Strong baseline with feature importance analysis
- **Linear Regression**: Fast, interpretable benchmark
- All models use 180-day input windows with 182-day forecast horizon

## Known Limitations

- No transaction fees or commissions modeled
- Average-cost basis assumption (not spec-id or specific lot selection)
- State and local taxes not included
- No alternative minimum tax (AMT) considerations
- Assumes static portfolio composition (no new contributions/withdrawals)

## Related Documentation

For comparison with Indian mutual fund tax rules, see the `indian-mf` branch:
- Different tax regime (equity funds, debt funds, hybrids)
- FIFO-mandated lot accounting (vs US average-cost basis)
- Tax-efficient rebalancing strategies for Indian investors

## License

CS229 Course Project - Comparative Tax-Aware Portfolio Rebalancing

