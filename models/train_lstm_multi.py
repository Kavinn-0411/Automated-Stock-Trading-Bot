"""
Train one pooled LSTM on multiple tickers (shared trunk + learned embeddings
+ temporal attention).

Each ticker must already have `data/processed/<TICKER>/train.csv` from the
pipeline.  Checkpoints default to `models/saved/multi_lstm.pkl`.

Supports two target modes:
  --target close   (default v1 behaviour — predict normalized Close price)
  --target return  (predict next-day log-return — solves OOD extrapolation)

Usage
-----
  python -m data.data_pipeline --preset faang

  python -m models.train_lstm_multi --tickers META AAPL AMZN NFLX GOOGL \\
         --target return --epochs 80 --window-size 45
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from models.lstm_model import run_training_multi
from models.train_lstm import load_ticker_data, plot_loss_curves


def main() -> None:
    p = argparse.ArgumentParser(description="Train pooled multi-ticker LSTM")
    p.add_argument(
        "--tickers",
        nargs="+",
        required=True,
        help="Two or more symbols (order defines embedding indices in the checkpoint)",
    )
    p.add_argument("--data-dir", default="data/processed")
    p.add_argument("--output", default="models/saved/multi_lstm.pkl", help="Checkpoint path")
    p.add_argument(
        "--target",
        choices=["close", "return"],
        default="return",
        help="Prediction target: 'close' (norm price) or 'return' (log-return)",
    )
    p.add_argument("--window-size", type=int, default=45)
    p.add_argument("--embedding-dim", type=int, default=16)
    p.add_argument("--hidden-dim", type=int, default=64)
    p.add_argument("--num-layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--patience", type=int, default=15, help="Early-stopping patience")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--train-ratio", type=float, default=0.8)
    args = p.parse_args()

    tickers = [t.upper() for t in args.tickers]
    if len(tickers) != len(set(tickers)):
        raise ValueError("Duplicate tickers are not allowed")
    if len(tickers) < 2:
        raise ValueError("Provide at least two distinct tickers")

    print(f"Target mode: {args.target}")

    feats: list = []
    tgts: list = []
    for sym in tickers:
        f, t = load_ticker_data(sym, args.data_dir, target_mode=args.target)
        feats.append(f)
        tgts.append(t)
        print(f"  {sym}: {f.shape[0]} train rows, {f.shape[1]} features")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\nTraining pooled model on {len(tickers)} tickers: {', '.join(tickers)}\n")
    results = run_training_multi(
        feats,
        tgts,
        tickers,
        window_size=args.window_size,
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        lr=args.lr,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        epochs=args.epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        train_ratio=args.train_ratio,
        target_mode=args.target,
        save_path=str(out_path),
    )

    plot_path = str(out_path.parent / f"{out_path.stem}_loss_curve.png")
    plot_loss_curves(results["train_losses"], results["val_losses"], plot_path)
    print(f"\nBest epoch: {results['best_epoch']}")
    print(f"Checkpoint: {out_path}")
    print(f"Loss plot:  {plot_path}")


if __name__ == "__main__":
    main()
