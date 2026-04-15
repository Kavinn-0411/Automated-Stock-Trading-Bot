# Automated Stock Trading Bot

An automated stock trading system that compares a **Reinforcement Learning (PPO)** agent against a **Supervised Learning (LSTM)** baseline and a traditional **Buy-and-Hold** strategy.

## Project Structure

```
project/
├── data/
│   ├── data_pipeline.py          # Download, indicators, normalize, split
│   └── processed/                # Generated CSV outputs (git-ignored)
├── models/
│   ├── lstm_model.py             # LSTM architectures (single + multi-ticker)
│   ├── train_lstm.py             # Single-ticker LSTM training script
│   ├── train_lstm_multi.py       # Multi-ticker pooled LSTM training script
│   ├── predict_test_lstm.py      # Test-set prediction & metrics
│   ├── ppo_agent.py              # PPO factory (Stable-Baselines3)
│   └── saved/                    # LSTM checkpoints (.pkl) + loss curves
├── env/
│   └── trading_env.py            # Custom Gymnasium trading environment
├── evaluation/
│   ├── metrics.py                # Metrics + Buy-and-Hold helper
│   ├── harness.py                # Validate arrays, compare, plots
│   ├── baseline.py               # Run Buy-and-Hold on test_raw.csv
│   ├── lstm_backtest.py          # LSTM heuristic portfolio simulation
│   └── compare_all.py            # Three-way comparison (B&H, LSTM, PPO)
├── outputs/
│   ├── models/                   # Saved PPO model weights + VecNormalize stats
│   └── portfolios/               # <TICKER>_ppo.npy, <TICKER>_ppo_meta.json
├── train_rl.py                   # PPO train + inference end-to-end script
├── train_faang.py                # Train PPO on all FAANG + generate charts
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

### Compare all strategies (Buy-and-Hold, LSTM, PPO)

1. Run the data pipeline for your ticker (see above).
2. Train the LSTM and save a checkpoint (default path `models/saved/<TICKER>_lstm.pkl`):

   ```bash
   python -m models.train_lstm --ticker AAPL
   ```

3. Train PPO and write test-set portfolio values:

   ```bash
   python train_rl.py --ticker AAPL
   ```

   This saves `outputs/portfolios/<TICKER>_ppo.npy` (and a legacy `ppo.npy`).

4. Run the comparison (table + optional plot):

   ```bash
   python -m evaluation.compare_all --ticker AAPL
   ```

   Outputs: `evaluation/outputs/<TICKER>_comparison.csv` and `<TICKER>_comparison.png` (one chart: $ on the left axis, normalized wealth on the right).  
   If the LSTM checkpoint or PPO file is missing, that strategy is skipped and noted in the log.

   **Phase 2 harness (programmatic use):** if you already have three aligned `numpy` arrays of daily portfolio values, call `evaluation.harness.compare_three_strategies(...)` or `compare_validated_strategies({...})` after optional `trim_to_common_length`. Arrays must be 1-D, equal length, and start within `$1` of each other by default (`StrategyArrayError` otherwise).

---

## LSTM price forecaster (supervised learning)

The LSTM component predicts next-day closing prices using a multi-ticker architecture trained jointly on all FAANG stocks. Predictions are converted to trading signals via a threshold-based heuristic.

### LSTM-related files

| File | Description |
|------|-------------|
| `models/lstm_model.py` | Model classes: `LSTMPricePredictor`, `MultiTickerLSTMPricePredictor`, `TemporalAttention` |
| `models/train_lstm.py` | Single-ticker training script and data loading utilities |
| `models/train_lstm_multi.py` | Multi-ticker pooled LSTM training script (used for latest model) |
| `models/predict_test_lstm.py` | Test-set predictions and error metrics (MSE, MAE, RMSE) |
| `evaluation/lstm_backtest.py` | Heuristic portfolio simulation from LSTM predictions |
| `evaluation/compare_lstm_vs_baseline.py` | LSTM vs Buy-and-Hold directional comparison |
| `models/saved/multi_lstm.pkl` | Trained multi-ticker checkpoint |
| `models/saved/multi_lstm_loss_curve.png` | Training/validation loss curve |

### Step 1: Prepare the data

Download and preprocess FAANG data (skip if already done):

```bash
python -m data.data_pipeline --preset faang
```

### Step 2: Train the multi-ticker LSTM

```bash
python -m models.train_lstm_multi \
    --tickers META AAPL AMZN NFLX GOOGL \
    --target close \
    --window-size 45 \
    --hidden-dim 64 \
    --embedding-dim 16 \
    --dropout 0.3 \
    --lr 5e-4 \
    --weight-decay 1e-4 \
    --grad-clip 1.0 \
    --epochs 80 \
    --patience 15 \
    --batch-size 64
```

This saves the checkpoint to `models/saved/multi_lstm.pkl` and the loss curve to `models/saved/multi_lstm_loss_curve.png`. Training will stop early if validation loss does not improve for 15 consecutive epochs.

To train with log-return targets instead of normalized close prices, change `--target close` to `--target return`.

### Step 3: Evaluate predictions on the test set

```bash
python -m models.predict_test_lstm --ticker AAPL --checkpoint models/saved/multi_lstm.pkl
```

This prints MSE, MAE, and RMSE in both normalized and dollar scales. Add `--output-csv predictions.csv` to save per-day predicted vs actual values.

Run for all FAANG tickers:

```bash
for t in META AAPL AMZN NFLX GOOGL; do
    python -m models.predict_test_lstm --ticker "$t" --checkpoint models/saved/multi_lstm.pkl
done
```

### Step 4: Run the heuristic backtest

Compare the LSTM heuristic strategy against Buy-and-Hold and PPO:

```bash
python -m evaluation.compare_all \
    --ticker AAPL \
    --lstm-checkpoint models/saved/multi_lstm.pkl \
    --auto-threshold
```

The `--auto-threshold` flag tests multiple signal thresholds (0.5%, 0.3%, 0.2%, 0.1%, 0.05%) and picks the one that maximizes final portfolio value. Results are saved to `evaluation/outputs/AAPL_comparison.csv` and `AAPL_comparison.png`.

Run across all FAANG:

```bash
for t in META AAPL AMZN NFLX GOOGL; do
    python -m evaluation.compare_all \
        --ticker "$t" \
        --lstm-checkpoint models/saved/multi_lstm.pkl \
        --auto-threshold
done
```

### Training a single-ticker LSTM (alternative)

If you prefer a per-ticker model instead of the shared multi-ticker model:

```bash
python -m models.train_lstm --ticker AAPL --epochs 50 --hidden-dim 128
```

This saves to `models/saved/AAPL_lstm.pkl`. The comparison scripts will auto-detect single-ticker checkpoints when you omit `--lstm-checkpoint`.

---

## PPO trading agent (reinforcement learning)

This component implements a **Proximal Policy Optimization (PPO)** agent that learns to trade a single stock by interacting with a custom **Gymnasium** environment. The agent observes **15** normalized market features plus **3** portfolio scalars each step and chooses **Buy**, **Sell**, or **Hold**. Training uses the chronological train split; evaluation produces a daily portfolio value array compatible with `evaluation.compare_all` and `evaluation.harness`.

### PPO-related files

| File | Description |
|------|-------------|
| `env/trading_env.py` | Custom environment — executes trades, tracks portfolio, rewards |
| `models/ppo_agent.py` | PPO factory (Stable-Baselines3) with tuned hyperparameters |
| `train_rl.py` | Train, run test inference, save model + portfolio arrays |
| `outputs/models/ppo_<TICKER>.zip` | Saved policy (after training) |
| `outputs/models/vecnorm_<TICKER>.pkl` | VecNormalize stats (must pair with the same policy) |
| `outputs/portfolios/<TICKER>_ppo.npy` | Daily portfolio values on the test window |
| `outputs/portfolios/<TICKER>_ppo_meta.json` | Metadata (ticker, length, final value, return) |
| `outputs/portfolios/ppo.npy` | Legacy copy of the last run (do not use for another ticker) |

### PPO dependencies

Install from the root `requirements.txt`. Key packages: `stable-baselines3`, `gymnasium`, `numpy`, `pandas`, `torch`.

### Running PPO

**Prerequisite:** run the data pipeline for your ticker first:

```bash
python -m data.data_pipeline --tickers AAPL
```

**Option A — Inference only (if a trained model already exists)**

```bash
python train_rl.py --ticker AAPL --skip-train
```

**Option B — Train (examples)**

```bash
# Shorter run
python train_rl.py --ticker AAPL --timesteps 500000

# Longer run (higher quality, more wall time)
python train_rl.py --ticker AAPL --timesteps 2000000
```

Checkpoints are also written under `outputs/checkpoints/` when training completes.

**Another ticker**

```bash
python -m data.data_pipeline --tickers MSFT
python train_rl.py --ticker MSFT --timesteps 2000000
```
**Train all FAANG tickers at once (saves to Final_Submission/)**
```bash
python -m data.data_pipeline --preset faang
python train_faang.py --timesteps 2000000
```

**Skip training, regenerate charts/portfolios from saved models**
```bash
python train_faang.py --skip-train
```

### Environment design (`env/trading_env.py`)

**Observation (18 dimensions):**  
`[Open, High, Low, Close, Volume, RSI, MACD, MACD_Signal, MACD_Hist, SMA_7, SMA_21, SMA_50, EMA_7, EMA_21, EMA_50, current_balance, shares_held, net_worth]`  

- First **15** features: normalized market columns from the data pipeline.  
- Last **3**: live portfolio state (VecNormalize may further scale observations at runtime).

**Actions**

| Action | Index | Behaviour |
|--------|-------|-----------|
| Hold | 0 | No trade |
| Buy | 1 | Invest all cash in shares (with fee) |
| Sell | 2 | Liquidate all shares (with fee) |

**Reward:** `(net_worth_t - net_worth_{t-1}) / initial_balance` (normalized step return).

**Fees:** 0.1% on buys and sells (consistent with common RL finance benchmarks, e.g. Yang et al., 2020).

**Episode end:** last row of data, or net worth ≤ 0.

### Agent design (`models/ppo_agent.py`)

PPO with **MlpPolicy**; training often uses **VecNormalize** (`norm_obs=True`, `norm_reward=False`, `clip_obs=10.0`). Example tuned settings (see source for current defaults):

| Hyperparameter | Example value | Note |
|----------------|---------------|------|
| Policy network | e.g. [256, 256] | Larger than SB3 default [64, 64] for tabular features |
| Learning rate | e.g. 1e-4 | May be reduced vs 3e-4 to limit policy oscillation |
| n_steps | 2048 | Rollout length |
| batch_size | 64 | |
| n_epochs | 10 | |
| gamma | 0.99 | |
| gae_lambda | 0.95 | |
| clip_range | 0.2 | |

### Training notes (example ablations)

| Stage | Timesteps | Notes |
|-------|-----------|--------|
| Early / debugging | 500k | Faster iteration |
| Stronger fit | 2M | Often better test behavior before overfitting |
| Very long | 5M+ | Risk of overfitting to the training trajectory |

*Exact numbers depend on seed, ticker, and data range.*

### Output contract (integration with evaluation)

`outputs/portfolios/<TICKER>_ppo.npy` is a **1-D** `float` array of **daily portfolio value** on the test split (length = number of test rows from `train_rl` inference loop). Use the **same initial balance** as Buy-and-Hold / LSTM when running `evaluation.compare_all`, or the harness will reject mismatched day-0 values.

Recommended end-to-end comparison:

```bash
python -m evaluation.compare_all --ticker AAPL
```

Programmatic merge with other strategies (arrays must be aligned and same starting capital):

```python
from evaluation.harness import compare_validated_strategies
import numpy as np

results = {
    "PPO": np.load("outputs/portfolios/AAPL_ppo.npy"),
    # LSTM / Buy-and-Hold: produce via evaluation pipeline or teammates’ scripts
}
summary = compare_validated_strategies(results)
print(summary)
```

### Relation to literature (Yang et al., 2020)

Inspired by *Deep Reinforcement Learning for Automated Stock Trading: An Ensemble Strategy* (ICAIF ’20): MDP framing, PPO, technical indicators, transaction costs, and evaluation via cumulative return, Sharpe, and max drawdown. This repo uses a **single-asset** setup and does not implement their full ensemble or turbulence index; reward normalization and VecNormalize are practical additions for stable value-function learning.

---

## Tech Stack

Python, PyTorch, Gymnasium, Stable-Baselines3, Pandas, NumPy, Matplotlib, scikit-learn
