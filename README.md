# Automated Stock Trading Bot

An automated stock trading system that compares a **Reinforcement Learning (PPO)** agent against a **Supervised Learning (LSTM)** baseline and a traditional **Buy-and-Hold** strategy.

## Project Structure

```
project/
├── data/
│   ├── data_pipeline.py      # Download, indicators, normalize, split
│   └── processed/             # Generated CSV outputs (git-ignored)
├── models/
│   ├── lstm_model.py          # LSTM next-day price predictor
│   └── ppo_agent.py           # PPO trading agent (Stable-Baselines3)
├── env/
│   └── trading_env.py         # Custom Gymnasium trading environment
├── evaluation/
│   └── metrics.py             # Cumulative Return, Sharpe Ratio, Max Drawdown
├── requirements.txt
└── README.md
```

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the data pipeline

```bash
# Default: downloads AAPL, MSFT, GOOGL (2015-2024)
python -m data.data_pipeline

# Use a preset group of tickers
python -m data.data_pipeline --preset faang
python -m data.data_pipeline --preset sp500_top30 --start 2020-01-01
python -m data.data_pipeline --preset tech

# Custom tickers and date range
python -m data.data_pipeline --tickers TSLA AMZN META --start 2018-01-01 --end 2024-06-30

# Mix presets with extra tickers
python -m data.data_pipeline --preset mag7 --tickers NFLX AMD

# See all available presets
python -m data.data_pipeline --list-presets
```

**Available presets:**

| Preset | Count | Description |
|--------|-------|-------------|
| `faang` | 5 | META, AAPL, AMZN, NFLX, GOOGL |
| `mag7` | 7 | Magnificent 7 mega-caps |
| `tech` | 15 | Broad technology sector |
| `finance` | 14 | Major banks and financial services |
| `healthcare` | 14 | Pharma, biotech, med devices |
| `energy` | 12 | Oil, gas, energy services |
| `consumer` | 13 | Consumer staples and discretionary |
| `sp500_top30` | 30 | Largest S&P 500 constituents |
| `all` | 72 | Union of all presets above |

The pipeline will:
- Download daily OHLCV data from Yahoo Finance
- Compute technical indicators (RSI, MACD, SMA, EMA)
- Split into train/test sets (80/20 chronological)
- Normalize features (fit on train, transform both)
- Save outputs to `data/processed/<TICKER>/`

### 3. Pipeline Output

For each ticker, the pipeline produces:

| File | Description |
|------|-------------|
| `train_raw.csv` | Training set with original price scale |
| `test_raw.csv` | Test set with original price scale |
| `train.csv` | Normalized training set |
| `test.csv` | Normalized test set |
| `scalers.pkl` | Fitted scalers for inverse-transforming predictions |

### 4. Feature Set

| Feature | Description |
|---------|-------------|
| Open, High, Low, Close | Daily OHLCV prices |
| Volume | Daily trading volume |
| RSI | Relative Strength Index (14-period) |
| MACD, MACD_Signal, MACD_Hist | Moving Average Convergence Divergence |
| SMA_7, SMA_21, SMA_50 | Simple Moving Averages |
| EMA_7, EMA_21, EMA_50 | Exponential Moving Averages |

## Evaluation Metrics

- **Cumulative Return** — total percentage gain over the test period
- **Sharpe Ratio** — risk-adjusted return (annualized)
- **Maximum Drawdown** — largest peak-to-trough decline

## Tech Stack

Python, PyTorch, Gymnasium, Stable-Baselines3, Pandas, NumPy, Matplotlib, scikit-learn
