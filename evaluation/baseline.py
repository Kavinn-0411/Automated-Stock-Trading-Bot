"""
Buy-and-Hold baseline runner.

Loads a test set CSV (expects a Close column), simulates Buy-and-Hold,
and returns/saves daily portfolio values.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from evaluation.metrics import (
    buy_and_hold_portfolio_values,
    evaluation_summary,
)


def run_buy_and_hold_from_csv(
    csv_path: str,
    initial_balance: float = 10_000.0,
) -> np.ndarray:
    """Run Buy-and-Hold from a test_raw.csv file and return portfolio values."""
    df = pd.read_csv(csv_path)
    if "Close" not in df.columns:
        raise ValueError(f"'Close' column not found in {csv_path}")

    close_prices = df["Close"].to_numpy(dtype=float)
    return buy_and_hold_portfolio_values(
        close_prices=close_prices,
        initial_balance=initial_balance,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Buy-and-Hold baseline.")
    parser.add_argument(
        "--ticker",
        default="AAPL",
        help="Ticker folder under data/processed (default: AAPL)",
    )
    parser.add_argument(
        "--input-csv",
        default=None,
        help="Optional explicit path to test_raw.csv",
    )
    parser.add_argument(
        "--initial-balance",
        type=float,
        default=10_000.0,
        help="Starting cash amount (default: 10000)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional .npy output path for portfolio values",
    )
    args = parser.parse_args()

    input_csv = args.input_csv or f"data/processed/{args.ticker}/test_raw.csv"
    values = run_buy_and_hold_from_csv(
        csv_path=input_csv,
        initial_balance=args.initial_balance,
    )

    if args.output:
        output_path = Path(args.output)
    else:
        output_path = Path(f"evaluation/outputs/{args.ticker}_buy_and_hold.npy")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, values)

    summary = evaluation_summary(values, label=f"Buy-and-Hold ({args.ticker})")
    print(f"[buy_and_hold] Input: {input_csv}")
    print(f"[buy_and_hold] Output: {output_path}")
    print(f"[buy_and_hold] Days: {len(values)}")
    print(f"[buy_and_hold] Final Value: ${values[-1]:,.2f}")
    print("[buy_and_hold] Metrics:")
    print(f"  Cumulative Return: {summary['Cumulative Return']}")
    print(f"  Sharpe Ratio:      {summary['Sharpe Ratio']}")
    print(f"  Max Drawdown:      {summary['Max Drawdown']}")


if __name__ == "__main__":
    main()
