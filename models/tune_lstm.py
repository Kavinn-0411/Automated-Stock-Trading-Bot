"""
Hyperparameter search for the LSTM price predictor.

Uses validation MSE from the same train/val split as `run_training`
(fit on train.csv only — no test-set leakage).

Modes
-----
  random  — sample hyperparameters for `--trials` runs (default)
  grid    — small exhaustive grid (faster to estimate total runtime)

After search, retrains the best config with `--final-epochs` and saves
`{ticker}_lstm.pkl` plus a CSV log of all trials.

Usage
-----
  python -m models.tune_lstm --ticker AAPL --trials 15 --seed 42
  python -m models.tune_lstm --ticker AAPL --mode grid
  python -m models.tune_lstm --ticker AAPL --trials 8 --epochs 30 --final-epochs 80
"""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd

from models.lstm_model import run_training
from models.train_lstm import load_ticker_data


def sample_random_config(rng: np.random.Generator) -> dict:
    """One random hyperparameter draw."""
    return {
        "window_size": int(rng.choice([15, 20, 30, 45, 60])),
        "hidden_dim": int(rng.choice([64, 96, 128, 192, 256])),
        "num_layers": int(rng.choice([1, 2])),
        "dropout": float(rng.uniform(0.0, 0.35)),
        "lr": float(10 ** rng.uniform(-4.0, -2.45)),
        "batch_size": int(rng.choice([32, 64, 128])),
    }


def grid_configs() -> list[dict]:
    """Small grid: 3 x 2 x 2 x 2 = 24 configs (lr and batch fixed for speed)."""
    windows = [20, 30, 45]
    hiddens = [64, 128]
    layers = [1, 2]
    dropouts = [0.0, 0.2]
    lr = 1e-3
    batch_size = 64
    configs = []
    for w, h, L, d in itertools.product(windows, hiddens, layers, dropouts):
        configs.append(
            {
                "window_size": w,
                "hidden_dim": h,
                "num_layers": L,
                "dropout": d,
                "lr": lr,
                "batch_size": batch_size,
            }
        )
    return configs


def run_one_trial(
    features: np.ndarray,
    targets: np.ndarray,
    cfg: dict,
    *,
    epochs: int,
    train_ratio: float,
    verbose: bool,
) -> dict:
    out = run_training(
        features,
        targets,
        window_size=cfg["window_size"],
        hidden_dim=cfg["hidden_dim"],
        num_layers=cfg["num_layers"],
        dropout=cfg["dropout"],
        lr=cfg["lr"],
        batch_size=cfg["batch_size"],
        epochs=epochs,
        train_ratio=train_ratio,
        save_path=None,
        verbose=verbose,
    )
    row = {**cfg, "epochs": epochs, "best_val_loss": out["best_val_loss"], "best_epoch": out["best_epoch"]}
    return row


def main() -> None:
    p = argparse.ArgumentParser(description="Hyperparameter search for LSTM")
    p.add_argument("--ticker", default="AAPL")
    p.add_argument("--data-dir", default="data/processed")
    p.add_argument("--output-dir", default="models/saved")
    p.add_argument("--mode", choices=["random", "grid"], default="random")
    p.add_argument("--trials", type=int, default=15, help="Random-search trial count")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=40, help="Epochs per tuning trial (keep moderate)")
    p.add_argument(
        "--final-epochs",
        type=int,
        default=None,
        help="Epochs for final train of best config (default: max(epochs, 50))",
    )
    p.add_argument("--train-ratio", type=float, default=0.8)
    p.add_argument("--verbose-trials", action="store_true", help="Print per-epoch logs each trial")
    args = p.parse_args()

    if args.final_epochs is None:
        args.final_epochs = max(args.epochs, 50)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{args.ticker}_lstm_tuning.csv"
    best_ckpt = out_dir / f"{args.ticker}_lstm.pkl"

    features, targets = load_ticker_data(args.ticker, args.data_dir)
    print(f"Loaded {args.ticker}: {features.shape[0]} rows x {features.shape[1]} features\n")

    rng = np.random.default_rng(args.seed)
    rows: list[dict] = []

    if args.mode == "random":
        print(f"Random search: {args.trials} trials, {args.epochs} epochs/trial, seed={args.seed}\n")
        for t in range(args.trials):
            cfg = sample_random_config(rng)
            print(
                f"Trial {t + 1}/{args.trials}  |  w={cfg['window_size']} h={cfg['hidden_dim']} "
                f"L={cfg['num_layers']} drop={cfg['dropout']:.3f} lr={cfg['lr']:.2e} bs={cfg['batch_size']}"
            )
            row = run_one_trial(
                features,
                targets,
                cfg,
                epochs=args.epochs,
                train_ratio=args.train_ratio,
                verbose=args.verbose_trials,
            )
            row["trial"] = t + 1
            rows.append(row)
            print(f"  -> best_val_mse={row['best_val_loss']:.6f} @ epoch {row['best_epoch']}\n")
    else:
        configs = grid_configs()
        print(f"Grid search: {len(configs)} configs, {args.epochs} epochs each\n")
        for t, cfg in enumerate(configs):
            if (t + 1) % 10 == 1 or t == 0:
                print(f"--- Grid progress {t + 1}/{len(configs)} ---")
            row = run_one_trial(
                features,
                targets,
                cfg,
                epochs=args.epochs,
                train_ratio=args.train_ratio,
                verbose=args.verbose_trials,
            )
            row["trial"] = t + 1
            rows.append(row)
            if (t + 1) % 10 == 0 or t == len(configs) - 1:
                print(f"  last: val_mse={row['best_val_loss']:.6f}")

    df = pd.DataFrame(rows)
    df = df.sort_values("best_val_loss").reset_index(drop=True)
    df.to_csv(csv_path, index=False)
    print(f"\nSaved tuning log: {csv_path}")

    best = df.iloc[0].to_dict()
    print("\nBest config (by validation MSE):")
    for k in ("window_size", "hidden_dim", "num_layers", "dropout", "lr", "batch_size"):
        print(f"  {k}: {best[k]}")
    print(f"  best_val_loss: {best['best_val_loss']:.6f} (trial {int(best['trial'])})")

    best_cfg = {
        "window_size": int(best["window_size"]),
        "hidden_dim": int(best["hidden_dim"]),
        "num_layers": int(best["num_layers"]),
        "dropout": float(best["dropout"]),
        "lr": float(best["lr"]),
        "batch_size": int(best["batch_size"]),
    }

    print(f"\nFinal training with best config, {args.final_epochs} epochs -> {best_ckpt}\n")
    run_training(
        features,
        targets,
        window_size=best_cfg["window_size"],
        hidden_dim=best_cfg["hidden_dim"],
        num_layers=best_cfg["num_layers"],
        dropout=best_cfg["dropout"],
        lr=best_cfg["lr"],
        batch_size=best_cfg["batch_size"],
        epochs=args.final_epochs,
        train_ratio=args.train_ratio,
        save_path=str(best_ckpt),
        verbose=True,
    )
    print("\nTuning complete.")


if __name__ == "__main__":
    main()
