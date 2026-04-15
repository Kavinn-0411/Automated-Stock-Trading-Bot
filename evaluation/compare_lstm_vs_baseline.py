"""
Compare LSTM signal-based trading against Buy-and-Hold on the test set.

Supports both checkpoint modes (auto-detected):

  target_mode="close"  — use predicted normalized-price direction as signal
  target_mode="return" — use predicted log-return sign as signal

Trading rule (long-only):
  - predicted next-day price UP   →  BUY / stay long
  - predicted next-day price DOWN →  SELL / stay in cash

Usage
-----
  python -m evaluation.compare_lstm_vs_baseline
  python -m evaluation.compare_lstm_vs_baseline --checkpoint models/saved/multi_lstm.pkl
  python -m evaluation.compare_lstm_vs_baseline --tickers META AAPL
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from models.predict_test_lstm import (
    load_model,
    predict_test_set,
    predict_test_set_returns,
)
from evaluation.metrics import (
    buy_and_hold_portfolio_values,
    compare_strategies,
    cumulative_return,
    sharpe_ratio,
    max_drawdown,
)


def lstm_signal_portfolio(
    pred_direction: np.ndarray,
    actual_close: np.ndarray,
    initial_balance: float = 10_000.0,
    tx_cost: float = 0.001,
) -> np.ndarray:
    """
    Long-only strategy driven by LSTM directional predictions.

    pred_direction[i] > 0  →  want to hold shares on day i
    pred_direction[i] <= 0 →  want to be in cash on day i
    """
    n = len(actual_close)
    portfolio = np.empty(n, dtype=np.float64)

    cash = initial_balance
    shares = 0
    in_position = False

    for i in range(n):
        signal = pred_direction[i] if i < len(pred_direction) else 0.0
        price = actual_close[i]

        if signal > 0 and not in_position:
            shares = int(cash // (price * (1 + tx_cost)))
            cost = shares * price * (1 + tx_cost)
            cash -= cost
            in_position = True
        elif signal <= 0 and in_position:
            proceeds = shares * price * (1 - tx_cost)
            cash += proceeds
            shares = 0
            in_position = False

        portfolio[i] = cash + shares * price

    return portfolio


def plot_comparison(
    dates: pd.DatetimeIndex,
    bh_values: np.ndarray,
    lstm_values: np.ndarray,
    ticker: str,
    save_path: str,
) -> None:
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(dates, bh_values, label="Buy & Hold", linewidth=1.5)
    ax.plot(dates, lstm_values, label="LSTM Signal", linewidth=1.5)
    ax.set_xlabel("Date")
    ax.set_ylabel("Portfolio Value ($)")
    ax.set_title(f"{ticker} — LSTM Signal vs Buy & Hold (test set)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description="LSTM vs Buy-and-Hold comparison")
    p.add_argument("--checkpoint", default="models/saved/multi_lstm.pkl")
    p.add_argument("--tickers", nargs="+", default=None,
                   help="Subset of tickers (default: all in checkpoint)")
    p.add_argument("--data-dir", default="data/processed")
    p.add_argument("--initial-balance", type=float, default=10_000.0)
    p.add_argument("--tx-cost", type=float, default=0.001,
                   help="Transaction cost fraction (0.001 = 0.1%%)")
    p.add_argument("--output-dir", default="evaluation/outputs")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ckpt = load_model(args.checkpoint, device)
    target_mode = ckpt.get("target_mode", "close")
    W = int(ckpt["window_size"])

    tickers = args.tickers or ckpt.get("tickers")
    if not tickers:
        raise ValueError("No tickers found in checkpoint and none provided via --tickers")
    tickers = [t.upper() for t in tickers]

    tmap = ckpt.get("ticker_to_idx") or {t: i for i, t in enumerate(ckpt["tickers"])}
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results: dict[str, np.ndarray] = {}

    print(f"Checkpoint : {args.checkpoint}")
    print(f"Target mode: {target_mode}")

    for ticker in tickers:
        print(f"\n{'='*60}")
        print(f"  {ticker}")
        print(f"{'='*60}")

        data_dir = Path(args.data_dir) / ticker
        train_df = pd.read_csv(data_dir / "train.csv", index_col="Date", parse_dates=True)
        test_df = pd.read_csv(data_dir / "test.csv", index_col="Date", parse_dates=True)
        test_raw_df = pd.read_csv(data_dir / "test_raw.csv", index_col="Date", parse_dates=True)

        tid = int(tmap[ticker]) if ckpt.get("multi_ticker") else None
        actual_close = test_raw_df["Close"].values.astype(np.float64)

        if target_mode == "return":
            train_raw_df = pd.read_csv(
                data_dir / "train_raw.csv", index_col="Date", parse_dates=True
            )
            pred_ret, actual_ret, pred_dollar, actual_dollar, dates = predict_test_set_returns(
                model, train_df, test_df, train_raw_df, test_raw_df,
                W, device, ticker_id=tid,
            )
            pred_direction = pred_ret
        else:
            preds_norm, actual_norm, dates = predict_test_set(
                model, train_df, test_df, W, device, ticker_id=tid,
            )
            pred_direction = np.diff(preds_norm, prepend=preds_norm[0])

        up_pct = (pred_direction > 0).sum() / len(pred_direction) * 100
        print(f"  Signal stats: {up_pct:.1f}% BUY days, {100 - up_pct:.1f}% SELL/HOLD days")

        bh = buy_and_hold_portfolio_values(actual_close, args.initial_balance)
        lstm_pv = lstm_signal_portfolio(
            pred_direction, actual_close, args.initial_balance, args.tx_cost,
        )

        bh_key = f"Buy & Hold ({ticker})"
        lstm_key = f"LSTM Signal ({ticker})"
        all_results[bh_key] = bh
        all_results[lstm_key] = lstm_pv

        print(f"  Period: {test_raw_df.index[0].date()} -> {test_raw_df.index[-1].date()}")
        print(f"  Initial balance: ${args.initial_balance:,.2f}")
        print()
        print(f"  {'':20s} {'Buy & Hold':>14s}  {'LSTM Signal':>14s}")
        print(f"  {'Final Value':20s} ${bh[-1]:>13,.2f}  ${lstm_pv[-1]:>13,.2f}")
        print(f"  {'Cumulative Return':20s} {cumulative_return(bh):>13.2%}  {cumulative_return(lstm_pv):>13.2%}")
        print(f"  {'Sharpe Ratio':20s} {sharpe_ratio(bh):>13.3f}  {sharpe_ratio(lstm_pv):>13.3f}")
        print(f"  {'Max Drawdown':20s} {max_drawdown(bh):>13.2%}  {max_drawdown(lstm_pv):>13.2%}")

        plot_path = str(out_dir / f"{ticker}_lstm_vs_bh.png")
        plot_comparison(test_raw_df.index, bh, lstm_pv, ticker, plot_path)
        print(f"\n  Plot saved: {plot_path}")

    print(f"\n\n{'='*60}")
    print("  SUMMARY TABLE")
    print(f"{'='*60}\n")
    df = compare_strategies(all_results)
    print(df.to_string())
    print()


if __name__ == "__main__":
    main()
