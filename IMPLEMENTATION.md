# Implementation Guide

## Team Assignments

### Prakash -- Data Pipeline + Evaluation Framework

**Status: COMPLETE**

- [x] Download OHLCV data from Yahoo Finance (71 tickers across 7 sectors)
- [x] Compute technical indicators (RSI, MACD, SMA, EMA)
- [x] Normalize features (MinMaxScaler fit on train, transform both)
- [x] Chronological train/test split (80/20)
- [x] Evaluation module (Cumulative Return, Sharpe Ratio, Max Drawdown)
- [ ] Implement Buy-and-Hold baseline
- [ ] Wire evaluation module into a comparison harness that all 3 strategies feed into

**Buy-and-Hold baseline:** Simulate buying on day 1 of the test set and holding until the last day. Track daily portfolio value as an array, then pass it to `evaluation/metrics.py` like every other strategy.

**Comparison harness:** A single script that loads test results from all 3 strategies (Buy-and-Hold, LSTM, PPO), runs them through `compare_strategies()`, and outputs a summary table + chart.

---

### Sarvesh -- PPO Agent + Gym Environment

**Status: Scaffolded, needs implementation**

Files to work in:
- `env/trading_env.py` -- the Gymnasium environment (skeleton exists)
- `models/ppo_agent.py` -- PPO config and training (skeleton exists)

#### Step 1: Complete the Trading Environment

The skeleton already has observation/action spaces, `reset()`, and `_get_obs()` wired up. What's missing is the core `step()` logic:

- **Buy action:** Spend available cash to buy shares at current Close price, subtract transaction fee (0.1%)
- **Sell action:** Sell all held shares at current Close price, subtract transaction fee
- **Hold action:** Do nothing
- **Reward:** Use change in net worth from one step to the next (net_worth_new - net_worth_old). Optionally penalize excessive trading to discourage churn.
- **Termination:** Episode ends when the agent reaches the last row of data

Use the normalized CSVs from `data/processed/<TICKER>/train.csv` as input. The raw CSVs have actual prices needed to compute dollar trades -- consider loading both.

#### Step 2: Train the PPO Agent

- Use `create_ppo_agent()` from `models/ppo_agent.py` (defaults already set)
- Train for ~100k-500k timesteps, monitor with SB3's built-in logging
- Save the trained model with `model.save()`

#### Step 3: Hyperparameter Tuning

Key knobs to experiment with: learning rate, n_steps, clip_range, reward shaping, and whether to use a larger MLP policy. Start with defaults, then iterate.

#### Step 4: Evaluation

Run the trained policy on the test set, record daily portfolio values as an array, and hand it to Prakash's `evaluation/metrics.py`.

---

### Kavinn -- LSTM Baseline + Final Comparative Analysis

**Status: Scaffolded, needs implementation**

Files to work in:
- `models/lstm_model.py` -- LSTM architecture (skeleton exists with model class)
- A new training script (e.g. `models/train_lstm.py`)

#### Step 1: Data Preparation for LSTM

Load `data/processed/<TICKER>/train.csv`. Create sliding windows of N consecutive rows (e.g. 30 days) as input sequences, with the next day's normalized Close price as the target. Use PyTorch `Dataset` and `DataLoader`.

#### Step 2: Train the LSTM

- The `LSTMPredictor` class is already defined (2-layer LSTM + linear head)
- Use MSE loss and Adam optimizer
- Train for ~50-100 epochs, track validation loss
- Save best model checkpoint

#### Step 3: Heuristic Trading Strategy

At inference time, the LSTM predicts tomorrow's Close price. Convert that into a trading signal:
- If predicted price > current price by some threshold (e.g. +0.5%) --> **Buy**
- If predicted price < current price by threshold --> **Sell**
- Otherwise --> **Hold**

Simulate this on the test set with an initial balance, tracking daily portfolio value.

#### Step 4: Final Comparison

- Collect portfolio value arrays from all 3 strategies: Buy-and-Hold (from Prakash), PPO (from Sarvesh), LSTM (yours)
- Pass all three to `compare_strategies()` in `evaluation/metrics.py`
- Generate a line chart of portfolio value over time (all 3 overlaid) and a summary table with Cumulative Return, Sharpe Ratio, and Max Drawdown

---

## Suggested Timeline

| Week | Prakash | Sarvesh | Kavinn |
|------|---------|---------|--------|
| 1 | ~~Data pipeline~~ (done). Build Buy-and-Hold baseline + comparison harness | Implement `step()` in trading env, verify with random agent | Build sliding window dataset, train LSTM, validate loss |
| 2 | Help debug integration issues | Train PPO, tune hyperparameters | Implement heuristic trading, simulate on test set |
| 3 | Final evaluation runs | Final evaluation runs | Run comparison, generate charts + report tables |

## Key Integration Contract

All three strategies must produce the same output format so the evaluation module can compare them:

```
np.ndarray of daily portfolio values, shape (n_test_days,)
starting from the same initial balance (e.g. $10,000)
```

This array gets passed to `evaluation/metrics.py :: compare_strategies()` which handles the rest.
