from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from env.trading_env import TradingEnv
from models.ppo_agent import create_ppo_agent


INITIAL_BALANCE = 10_000.0
DATA_DIR = Path("data/processed")
MODEL_DIR = Path("outputs/models")
CHECKPOINT_DIR = Path("outputs/checkpoints")
PORTFOLIO_DIR = Path("outputs/portfolios")


def _load_split(ticker: str, split: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    d = DATA_DIR / ticker
    norm = pd.read_csv(d / f"{split}.csv", index_col=0)
    raw = pd.read_csv(d / f"{split}_raw.csv", index_col=0)
    return norm, raw


def _make_vec_env(df: pd.DataFrame, raw_df: pd.DataFrame) -> DummyVecEnv:
    return DummyVecEnv([lambda: TradingEnv(df, raw_df, initial_balance=INITIAL_BALANCE)])


def train_agent(ticker: str, timesteps: int) -> Path:
    train_norm, train_raw = _load_split(ticker, "train")

    vec_env = _make_vec_env(train_norm, train_raw)
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=False, clip_obs=10.0)

    model = create_ppo_agent(vec_env)
    model.learn(total_timesteps=timesteps, progress_bar=True)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    save_path = MODEL_DIR / f"ppo_{ticker}"
    vecnorm_path = MODEL_DIR / f"vecnorm_{ticker}.pkl"

    model.save(str(save_path))
    vec_env.save(str(vecnorm_path))

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_model = CHECKPOINT_DIR / f"ppo_{ticker}_{ts}.zip"
    ckpt_norm = CHECKPOINT_DIR / f"vecnorm_{ticker}_{ts}.pkl"
    shutil.copy2(str(save_path) + ".zip", ckpt_model)
    shutil.copy2(str(vecnorm_path), ckpt_norm)

    print(f"\n[train] Model saved    -> {save_path}.zip")
    print(f"[train] VecNorm saved  -> {vecnorm_path}")
    print(f"[train] Checkpoint     -> {ckpt_model}")
    return save_path


def run_inference(ticker: str, model_path: Path) -> np.ndarray:
    test_norm, test_raw = _load_split(ticker, "test")
    n_days = len(test_norm)

    test_vec_env = _make_vec_env(test_norm, test_raw)
    vecnorm_path = MODEL_DIR / f"vecnorm_{ticker}.pkl"
    test_vec_env = VecNormalize.load(str(vecnorm_path), test_vec_env)
    test_vec_env.training = False
    test_vec_env.norm_reward = False

    model = PPO.load(str(model_path), device="cpu")

    obs = test_vec_env.reset()
    portfolio: list[float] = [INITIAL_BALANCE]

    for _ in range(n_days - 1):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, info = test_vec_env.step(action)
        portfolio.append(float(info[0]["net_worth"]))
        if done[0]:
            last = portfolio[-1]
            portfolio.extend([last] * (n_days - len(portfolio)))
            break

    return np.array(portfolio, dtype=np.float64)


def save_outputs(ticker: str, portfolio: np.ndarray) -> None:
    PORTFOLIO_DIR.mkdir(parents=True, exist_ok=True)

    npy_path = PORTFOLIO_DIR / "ppo.npy"
    np.save(npy_path, portfolio)

    meta = {
        "ticker": ticker,
        "window_size_used": None,
        "test_start_index": 0,
        "n_test_days": int(portfolio.shape[0]),
        "initial_balance": INITIAL_BALANCE,
        "final_portfolio_value": float(portfolio[-1]),
        "cumulative_return_pct": round((float(portfolio[-1]) / float(portfolio[0]) - 1.0) * 100, 4),
    }
    meta_path = PORTFOLIO_DIR / "ppo_meta.json"
    with open(meta_path, "w") as fh:
        json.dump(meta, fh, indent=2)

    print(f"Output: Portfolio array -> {npy_path}  shape={portfolio.shape}")
    print(f"Output: Metadata -> {meta_path}")
    print(f"Output: Final value: ${portfolio[-1]:,.2f}  ({meta['cumulative_return_pct']:+.2f}%)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate a PPO trading agent.")
    parser.add_argument("--ticker", required=True, help="Ticker symbol must exist in data/processed/")
    parser.add_argument(
        "--timesteps",
        type=int,
        default=200_000,
        help="PPO training timesteps (default: 200000)",
    )
    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="Skip training and load an existing saved model for inference only",
    )
    args = parser.parse_args()

    ticker = args.ticker.upper()
    model_path = MODEL_DIR / f"ppo_{ticker}"
    vecnorm_path = MODEL_DIR / f"vecnorm_{ticker}.pkl"
    ticker_data_dir = DATA_DIR / ticker

    if not ticker_data_dir.exists():
        raise FileNotFoundError(
            f"No processed data found for {ticker}. "
            f"Run: python -m data.data_pipeline --tickers {ticker}"
        )

    if not args.skip_train:
        train_agent(ticker, args.timesteps)

    model_zip = Path(str(model_path) + ".zip")
    if not model_zip.exists():
        raise FileNotFoundError(
            f"Model not found at {model_zip}. Train first"
        )
    if not vecnorm_path.exists():
        raise FileNotFoundError(
            f"VecNormalize stats not found at {vecnorm_path}. Train first"
        )

    portfolio = run_inference(ticker, model_path)
    save_outputs(ticker, portfolio)


if __name__ == "__main__":
    main()
