"""
Training script for the LSTM price predictor.

Loads preprocessed CSVs from the data pipeline, trains the LSTM model,
and saves the best checkpoint as a .pkl file (via torch.save).

Usage
-----
  # 1. First, run the data pipeline to download & preprocess:
  python -m data.data_pipeline --tickers AAPL

  # 2. Then train the LSTM on that ticker:
  python -m models.train_lstm --ticker AAPL

  # 3. Or train on multiple tickers sequentially:
  python -m models.train_lstm --ticker AAPL MSFT GOOGL

  # 4. Custom hyperparams:
  python -m models.train_lstm --ticker AAPL --epochs 100 --hidden-dim 256 --window-size 60
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from models.lstm_model import run_training

TARGET_COL = "Close"


def load_ticker_data(
    ticker: str,
    data_dir: str = "data/processed",
    target_mode: str = "close",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Read the normalised train.csv for a ticker and return
    (features, targets) as numpy arrays.

    target_mode
    -----------
    "close"  — target is the normalized Close price (original behaviour).
    "return" — target is the next-day log-return of the *raw* Close price.
               Features are still the normalized columns from train.csv.
               The first row is dropped (no prior day for return).
    """
    csv_path = Path(data_dir) / ticker / "train.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"{csv_path} not found. Run the data pipeline first:\n"
            f"  python -m data.data_pipeline --tickers {ticker}"
        )

    df = pd.read_csv(csv_path, index_col="Date", parse_dates=True)
    if TARGET_COL not in df.columns:
        raise ValueError(f"'{TARGET_COL}' column missing from {csv_path}")

    if target_mode == "return":
        raw_path = Path(data_dir) / ticker / "train_raw.csv"
        if not raw_path.exists():
            raise FileNotFoundError(
                f"{raw_path} not found — needed for return targets"
            )
        raw_df = pd.read_csv(raw_path, index_col="Date", parse_dates=True)
        log_ret = np.log(raw_df["Close"] / raw_df["Close"].shift(1)).values.astype(np.float32)
        # drop first row (NaN return)
        features = df.values[1:].astype(np.float32)
        targets = log_ret[1:]
        return features, targets

    targets = df[TARGET_COL].values.astype(np.float32)
    features = df.values.astype(np.float32)
    return features, targets


def plot_loss_curves(
    train_losses: list[float],
    val_losses: list[float],
    save_path: str,
) -> None:
    """Save a train/val loss curve plot."""
    fig, ax = plt.subplots(figsize=(8, 4))
    epochs = range(1, len(train_losses) + 1)
    ax.plot(epochs, train_losses, label="Train MSE")
    ax.plot(epochs, val_losses, label="Val MSE")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.set_title("LSTM Training Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Loss curve saved to {save_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train LSTM price predictor")
    parser.add_argument(
        "--ticker", nargs="+", default=["AAPL"],
        help="Ticker(s) to train on (default: AAPL)",
    )
    parser.add_argument("--data-dir", default="data/processed")
    parser.add_argument("--output-dir", default="models/saved")
    parser.add_argument("--window-size", type=int, default=30)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--train-ratio", type=float, default=0.8,
                        help="Fraction of train.csv used for fitting (rest = validation)")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for ticker in args.ticker:
        print(f"\n{'='*60}")
        print(f"  Training LSTM for {ticker}")
        print(f"{'='*60}\n")

        features, targets = load_ticker_data(ticker, args.data_dir)
        print(f"Loaded {ticker}: {features.shape[0]} rows, {features.shape[1]} features")

        save_path = str(out_dir / f"{ticker}_lstm.pkl")
        results = run_training(
            features,
            targets,
            window_size=args.window_size,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            dropout=args.dropout,
            lr=args.lr,
            epochs=args.epochs,
            batch_size=args.batch_size,
            train_ratio=args.train_ratio,
            save_path=save_path,
        )

        plot_path = str(out_dir / f"{ticker}_loss_curve.png")
        plot_loss_curves(results["train_losses"], results["val_losses"], plot_path)

        print(f"\n  {ticker} done — best epoch: {results['best_epoch']}")
        print(f"  Model : {save_path}")
        print(f"  Chart : {plot_path}")


if __name__ == "__main__":
    main()
