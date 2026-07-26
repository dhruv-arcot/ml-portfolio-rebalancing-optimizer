# ML Portfolio Rebalancing Optimizer

**A comprehensive machine learning solution for tax-aware portfolio rebalancing across different investment markets and tax regimes.**

## Project Overview

This is a CS229 (Stanford Machine Learning) course project that combines deep learning and classical ML approaches to forecast stock returns and optimize portfolio rebalancing decisions while accounting for tax implications. The project explores tax-aware rebalancing across **two distinct investment markets** with different tax structures:

- **US ETF Portfolio** – US capital gains taxation
- **Indian Mutual Fund Portfolio** – Indian mutual fund taxation rules

---

## 🎯 Key Features

- **Multi-Model Forecasting**: LSTM, RNN, XGBoost, and Linear Regression models for stock return prediction
- **Dual Tax Regime Support**: Separate implementations for US and Indian tax rules
- **Tax-Aware Rebalancing**: Optimizes trades considering capital gains tax impact and holding periods
- **Feature Engineering**: Automated extraction of technical indicators and time-based features
- **Production-Ready**: Structured for deployment with proper error handling and validation
- **Comprehensive Evaluation**: Model performance tracking and comparison across regimes

---

## 📁 Branch Organization

This repository is organized into three branches for clarity and specialization:

### 1. **main** (This branch)
Overview, documentation, and reference materials explaining both US-ETF and Indian-MF approaches.
- Project architecture and comparison
- Poster information and project context
- Quick reference guides

### 2. **us-etf** 
US-based portfolio rebalancing using US capital gains tax rules.

**Key files:**
- `main.py` – Train all 4 model types over stock CSVs
- `fund_net_returns_calculator.py` – Tax-aware rebalancing simulation (US capital gains)
- `decide_sell.py` – Single sell/hold decision for a ticker/date
- Core models: `preprocessor.py`, `lstm.py`, `rnn.py`, `xgb.py`, `lr.py`
- `requirements.txt` – Dependencies

**Tax Logic:**
- Short-term capital gains (holding ≤ 1 year): Ordinary income rates (typically 20-37%)
- Long-term capital gains (holding > 1 year): Preferential rates (0%, 15%, or 20%)

### 3. **indian-mf** 
Indian mutual fund portfolio rebalancing using Indian mutual fund tax rules.

**Key files:**
- `main_poc.py` – Orchestrates the complete pipeline
- `forecast.py` – Train XGBoost forecasters for mutual funds
- `rebalancer.py` – Portfolio rebalancing engine with drift bands
- `tax_rules.py` – Indian MF taxation rules (FIFO lot accounting, exemptions)
- `data_fetcher.py` – Pulls NAV data from mfapi.in
- `config.py` – Configuration for target weights and tax parameters
- Indian-specific README and engineering documentation

**Tax Logic:**
- Equity funds: 20% STCG (< 12 months), 12.5% LTCG (≥ 12 months) with Rs 1.25L exemption
- Specified debt funds: Always at investor's income tax slab rate
- Hybrid/Other funds: Slab rate STCG (< 24 months), 12.5% LTCG (≥ 24 months)
- FIFO-mandated lot accounting (unlike US average-cost basis)

---

## 🏗️ High-Level Architecture

### Shared Core (Both branches use)
```
preprocessor.py          # Feature engineering and data preparation
├── create_features()    # Returns, moving averages, volatility, momentum
└── build_supervised()   # Sliding window generation
```

### ML Models (Training & Inference)
```
lstm.py                  # LSTM architecture (2 layers, 128 units, dropout 0.2)
rnn.py                   # RNN architecture (2 layers, 128 units, dropout 0.2)
xgb.py                   # XGBoost (400 estimators, max depth 6)
lr.py                    # Linear Regression (sklearn)
```

### Tax-Aware Rebalancing (Regime-Specific)
```
US-ETF Branch:
  fund_net_returns_calculator.py    # US capital gains logic
  
Indian-MF Branch:
  rebalancer.py                     # Indian FIFO + exemption logic
  tax_rules.py                      # Tax computation engine
```

---

## 📊 Model Configuration

### Training Parameters
- **Input Window**: 180 trading days (~6 months)
- **Forecast Horizon**: 182 days (~6 months)
- **Training Period**: 2020-2022
- **Test Period**: 2023-2025
- **Feature Set**: Returns, moving averages, volatility, momentum, volume, cyclical time

### Model Architectures
| Model | Architecture | Best For |
|-------|--------------|----------|
| **LSTM** | 2 layers, 128 hidden units, dropout 0.2 | Temporal pattern recognition |
| **RNN** | 2 layers, 128 hidden units, dropout 0.2 | Sequential dependencies |
| **XGBoost** | 400 estimators, max depth 6 | Fast, interpretable baseline |
| **Linear Regression** | Standard sklearn | Benchmark, interpretability |

---

## 🚀 Quick Start

### US-ETF Branch
```bash
# Checkout the US-ETF branch
git checkout us-etf

# Install dependencies
pip install -r requirements.txt

# Train all models
python main.py

# Run tax-aware portfolio optimization
python fund_net_returns_calculator.py

# Make individual sell/hold decisions
python decide_sell.py
```

### Indian-MF Branch
```bash
# Checkout the Indian-MF branch
git checkout indian-mf

# Install dependencies
pip install -r requirements.txt

# Fetch mutual fund NAV data
python data_fetcher.py

# Train forecasters
python forecast.py

# Run complete pipeline with tax comparison
python main_poc.py
```

---

## 📈 Key Insights & Comparisons

### US-ETF Approach
- **Focus**: Predicting individual stock returns and optimizing rebalancing under US tax rules
- **Advantage**: Simple tax regime (long-term vs short-term rates)
- **Challenge**: Multiple realized gains/losses in single calendar year
- **Lot Strategy**: Average-cost basis (implicit lot selection)

### Indian-MF Approach
- **Focus**: Tax-efficient rebalancing across mutual fund schemes with drift bands
- **Advantage**: Pre-tax consolidated NAV data, predictable tax treatment by category
- **Challenge**: Complex multi-bucket tax regime, FIFO-mandated lot selection, financial-year boundaries
- **Lot Strategy**: FIFO-enforced (legally mandated in India)

### Key Tax Differences
| Aspect | US | India |
|--------|----|----|
| **Long-term threshold** | 1 year | 12 months (equity), 24 months (hybrid) |
| **Tax rate structure** | 0%, 15%, 20% (federal) + state | 12.5%, 20%, or slab rate by category |
| **Lot selection** | Flexible | FIFO-mandated for MFs |
| **Tax year** | Calendar | Financial year (Apr-Mar) |
| **Special exemptions** | None for trades | Rs 1.25L equity LTCG exemption (India) |

---

## 📚 Documentation

- **US-ETF Branch**: See `README.md` for detailed US portfolio setup and usage
- **Indian-MF Branch**: See `README.md` for India-specific setup and `ENGINEERING.md` for tax rules deep-dive
- **This branch (`main`)**: High-level overview and project context

---

## 👤 Poster & Project Context

**Course**: CS229 – Machine Learning (Stanford University)  
**Project Type**: Comparative study of ML-driven tax-aware portfolio rebalancing  
**Focus**: Validating that forecast-informed sell decisions outperform baseline drift-based strategies under realistic tax constraints

---

## 🔧 Technical Stack

- **Core**: Python 3.8+
- **Data Processing**: pandas, numpy
- **ML Models**: PyTorch (LSTM/RNN), scikit-learn (Linear Regression), XGBoost
- **Utilities**: joblib (serialization), pandas, scipy
- **Data Sources**: 
  - US: Yahoo Finance or similar OHLCV data
  - India: mfapi.in (free AMFI NAV data)

---

## 📋 Data Requirements

### US-ETF Branch
**Stock Price Data** (`data/csv_files/`):
```csv
Date,Close,High,Low,Open,Volume
2015-01-02,24.23,24.705,23.79,24.69,212818400
```

**Fund Composition** (`funds/`):
```csv
Date,Stock,Weight
2025-01-31,AAPL,30
2025-01-31,MSFT,70
```

### Indian-MF Branch
**NAV Data** (auto-fetched from mfapi.in):
- Mutual fund NAV history for 6 schemes across equity/debt/hybrid/gold categories

**Fund Composition** (auto-generated from config):
```csv
Date,Scheme,Weight
2025-01-31,large_cap,25
2025-01-31,flexi_cap,25
2025-01-31,small_cap,15
...
```

---

## 📊 Output Artifacts

### US-ETF Branch
```
model_output/{lstm,rnn,lr,xgb}/
├── models/                       # Trained model files
├── {ticker}_predictions.csv      # Test set predictions
└── summary_{model}.csv           # Performance metrics

tax_results/
├── {fund}_tax_report.csv         # Detailed trade-by-trade analysis
└── summary_funds_tax.csv         # Aggregated results
```

### Indian-MF Branch
```
model_output/
└── {scheme}_xgb_predictions.csv  # Predictions per fund

results/
├── ledger_naive.csv              # Baseline strategy trades
├── ledger_tax_aware.csv          # Tax-optimized strategy trades
└── summary.csv                   # Comparison metrics
```

---

## ⚠️ Known Limitations

### Both Branches
- No transaction fees or expense ratios modeled
- No STT (India) or commissions (US)
- Static target allocations (not dynamic optimization)

### US-ETF Branch
- Assumes average-cost basis for lot selection
- No alternative minimum tax (AMT) considerations
- State and local taxes not included

### Indian-MF Branch
- No surcharge or cess modeling
- Balanced Advantage Fund tax category hardcoded (not dynamically determined)
- NAV-only data means high/low/open/volume are synthetic
- Loss harvesting not modeled
- Backtest window capped at 2025-12-31

---

## 🔗 Useful Links

- [mfapi.in](https://www.mfapi.in/) – Free AMFI Mutual Fund NAV data
- [Finnovate – MF Taxation FY2025-26](https://www.finnovate.in/learn/blog/mutual-fund-taxation-india-fy-2025-26)
- [US IRS Capital Gains Rates](https://www.irs.gov/taxtopics/tc409)

---

## 📝 License

CS229 Course Project  
Authored by Dhruv Arcot

---

## 🤝 Contributing

For branches:
- **us-etf**: Report issues specific to US portfolio optimization or stock data
- **indian-mf**: Report issues specific to Indian MF rebalancing or tax logic

---

**Last Updated**: July 2026  
**Maintained**: Dhruv Arcot
