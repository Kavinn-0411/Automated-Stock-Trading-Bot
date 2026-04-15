"""
Compare Buy-and-Hold, LSTM (heuristic), and PPO on the same test window.

Uses :mod:`evaluation.harness` for validation (Phase 2), metrics table, and charts.

Requires:
  - data/processed/<TICKER>/test_raw.csv (and train/test CSVs)
  - models/saved/<TICKER>_lstm.pkl  (run models.train_lstm first)
  - outputs/portfolios/<TICKER>_ppo.npy  (run train_rl after training)

Usage
-----
  python -m evaluation.compare_all --ticker AAPL
  python -m evaluation.compare_all --ticker AAPL --no-plot
  python -m evaluation.compare_all --ticker AAPL --no-trim   # fail if lengths differ
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from evaluation.baseline import run_buy_and_hold_from_csv
from evaluation.harness import (
    StrategyArrayError,
    compare_validated_strategies,
    plot_strategy_comparison,
    trim_to_common_length,
)
from evaluation.lstm_backtest import simulate_lstm_heuristic_portfolio


def _load_ppo_portfolio(ticker: str, portfolio_dir: Path) -> tuple[np.ndarray | None, str | None]:
    """
    Return (array, warning_message). ``ppo.npy`` is only used as fallback and may
    belong to a different ticker if you last trained another symbol.
    """
    p = portfolio_dir / f"{ticker}_ppo.npy"
    if p.exists():
        return np.load(p), None
    legacy = portfolio_dir / "ppo.npy"
    if legacy.exists():
        return np.load(legacy), (
            f"Using legacy {legacy} (no {p.name}). "
            f"Re-run: python train_rl.py --ticker {ticker} for a {ticker}-specific PPO curve."
        )
    return None, None


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare all trading strategies on test set.")
    parser.add_argument("--ticker", default="AAPL")
    parser.add_argument("--data-dir", default="data/processed")
    parser.add_argument("--portfolio-dir", default="outputs/portfolios")
    parser.add_argument("--lstm-checkpoint", default=None)
    parser.add_argument("--lstm-threshold", type=float, default=0.005)
    parser.add_argument(
        "--auto-threshold",
        action="store_true",
        help="Select the best LSTM threshold from --threshold-grid by final portfolio value.",
    )
    parser.add_argument(
        "--threshold-grid",
        nargs="+",
        type=float,
        default=[0.005, 0.003, 0.002, 0.001, 0.0005],
        help="Candidate thresholds for --auto-threshold (default: 0.005 0.003 0.002 0.001 0.0005).",
    )
    parser.add_argument("--initial-balance", type=float, default=10_000.0)
    parser.add_argument(
        "--balance-tol",
        type=float,
        default=1.0,
        help="Max allowed $ difference in day-0 portfolio value across strategies (default: 1.0)",
    )
    parser.add_argument(
        "--no-trim",
        action="store_true",
        help="Fail if strategy series have different lengths instead of trimming to the shortest.",
    )
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--output-dir", default="evaluation/outputs")
    args = parser.parse_args()

    ticker = args.ticker.upper()
    test_raw_path = Path(args.data_dir) / ticker / "test_raw.csv"
    if not test_raw_path.exists():
        raise FileNotFoundError(
            f"Missing {test_raw_path}. Run: python -m data.data_pipeline --tickers {ticker}"
        )

    results: dict[str, np.ndarray] = {}
    notes: dict[str, str] = {}

    bh = run_buy_and_hold_from_csv(str(test_raw_path), initial_balance=args.initial_balance)
    results["Buy-and-Hold"] = bh

    lstm_ckpt = Path(
        args.lstm_checkpoint or f"models/saved/{ticker}_lstm.pkl"
    )
    if not lstm_ckpt.exists():
        notes["lstm"] = (
            f"skipped: no LSTM checkpoint at {lstm_ckpt}. "
            "Train one per ticker (weights are not shared across symbols)."
        )
        print(
            f"[hint] LSTM missing — run: python -m models.train_lstm --ticker {ticker}",
            flush=True,
        )
    else:
        try:
            if args.auto_threshold:
                if not args.threshold_grid:
                    raise ValueError("--threshold-grid must have at least one value")

                best_vals: np.ndarray | None = None
                best_meta: dict | None = None
                best_threshold: float | None = None
                best_final_value = -np.inf
                tried: list[dict[str, float]] = []

                for th in args.threshold_grid:
                    vals, meta = simulate_lstm_heuristic_portfolio(
                        ticker,
                        data_dir=args.data_dir,
                        checkpoint=str(lstm_ckpt),
                        initial_balance=args.initial_balance,
                        threshold=float(th),
                    )
                    final_value = float(vals[-1])
                    tried.append(
                        {
                            "threshold": float(th),
                            "final_value": final_value,
                            "cumulative_return_pct": (final_value / float(vals[0]) - 1.0) * 100.0,
                        }
                    )
                    if final_value > best_final_value:
                        best_final_value = final_value
                        best_vals = vals
                        best_meta = meta
                        best_threshold = float(th)

                assert best_vals is not None and best_meta is not None and best_threshold is not None
                lstm_vals = best_vals
                lstm_meta = dict(best_meta)
                lstm_meta["auto_threshold"] = True
                lstm_meta["selected_threshold"] = best_threshold
                lstm_meta["threshold_grid"] = [float(x) for x in args.threshold_grid]
                lstm_meta["threshold_trials"] = tried
            else:
                lstm_vals, lstm_meta = simulate_lstm_heuristic_portfolio(
                    ticker,
                    data_dir=args.data_dir,
                    checkpoint=str(lstm_ckpt),
                    initial_balance=args.initial_balance,
                    threshold=args.lstm_threshold,
                )
            results["LSTM (heuristic)"] = lstm_vals
            notes["lstm"] = json.dumps(lstm_meta, indent=2)
        except (ImportError, ModuleNotFoundError) as e:
            notes["lstm"] = f"skipped: {e}"
        except Exception as e:
            notes["lstm"] = f"skipped (error during LSTM backtest): {type(e).__name__}: {e}"
            print(f"[warn] LSTM backtest failed: {e}", flush=True)

    ppo_dir = Path(args.portfolio_dir)
    ppo_arr, ppo_warn = _load_ppo_portfolio(ticker, ppo_dir)
    if ppo_arr is not None:
        results["PPO"] = ppo_arr.astype(np.float64)
        if ppo_warn:
            print(f"[warn] PPO: {ppo_warn}", flush=True)
            notes["ppo"] = ppo_warn
    else:
        notes["ppo"] = f"skipped: no {ppo_dir / f'{ticker}_ppo.npy'} (train with train_rl.py)"

    lengths = [len(v) for v in results.values()]
    if len(set(lengths)) > 1:
        if args.no_trim:
            raise StrategyArrayError(
                f"Length mismatch among strategies: {dict(zip(results.keys(), lengths))}. "
                "Omit --no-trim to align by trimming to the shortest series."
            )
        results, prev_max = trim_to_common_length(results)
        print(
            f"[compare_all] Trimmed all series to length {min(lengths)} "
            f"(was up to {prev_max}) for alignment."
        )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== Strategy comparison: {ticker} ===\n")
    if notes:
        for k, v in notes.items():
            print(f"[note] {k}: {v}\n")

    try:
        table = compare_validated_strategies(
            results,
            initial_balance_tolerance=args.balance_tol,
        )
    except StrategyArrayError as e:
        raise SystemExit(f"[compare_all] Validation failed: {e}") from e
    print(table.to_string())
    table_path = out_dir / f"{ticker}_comparison.csv"
    table.to_csv(table_path)
    print(f"\n[compare_all] Wrote {table_path}")

    for name, arr in results.items():
        print(f"  {name}: final ${arr[-1]:,.2f}")

    if not args.no_plot and len(results) >= 1:
        plot_path = out_dir / f"{ticker}_comparison.png"
        saved = plot_strategy_comparison(
            results,
            title=f"{ticker} — test period",
            save_path=plot_path,
        )
        if saved:
            print(f"[compare_all] Wrote chart: {saved}")


if __name__ == "__main__":
    main()
