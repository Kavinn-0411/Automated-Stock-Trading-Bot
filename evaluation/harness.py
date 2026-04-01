"""
Phase 2 comparison harness: validate arrays, compare metrics, plot charts.

Use this when you already have numpy arrays of daily portfolio values
(Buy-and-Hold, LSTM, PPO) over the same test window.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from evaluation.metrics import compare_strategies


DEFAULT_INITIAL_BALANCE_TOLERANCE = 1.0  # dollars


class StrategyArrayError(ValueError):
    """Raised when portfolio arrays fail validation."""


def validate_strategy_arrays(
    *arrays: np.ndarray,
    labels: Sequence[str] | None = None,
    initial_balance_tolerance: float = DEFAULT_INITIAL_BALANCE_TOLERANCE,
) -> None:
    """
    Fail-fast checks before computing metrics or plotting.

    - Each array is 1-D and non-empty
    - All arrays share the same length
    - First value of each array matches the others within ``initial_balance_tolerance``
      (same starting portfolio value, e.g. $10,000)
    """
    if not arrays:
        raise StrategyArrayError("At least one portfolio array is required.")

    names = list(labels) if labels is not None else [f"series_{i}" for i in range(len(arrays))]
    if len(names) != len(arrays):
        raise StrategyArrayError("labels must have the same length as arrays.")

    converted: list[np.ndarray] = []
    for i, a in enumerate(arrays):
        arr = np.asarray(a, dtype=np.float64)
        if arr.ndim != 1:
            raise StrategyArrayError(
                f"{names[i]}: expected a 1-D array, got shape {arr.shape}."
            )
        if arr.size == 0:
            raise StrategyArrayError(f"{names[i]}: array is empty.")
        converted.append(arr)

    n = len(converted[0])
    for arr, name in zip(converted, names):
        if len(arr) != n:
            raise StrategyArrayError(
                f"All series must have equal length. {name} has length {len(arr)}, "
                f"expected {n}."
            )

    starts = np.array([arr[0] for arr in converted], dtype=np.float64)
    spread = float(np.max(starts) - np.min(starts))
    if spread > initial_balance_tolerance:
        raise StrategyArrayError(
            "Starting portfolio values differ beyond tolerance "
            f"(${initial_balance_tolerance:.2f}). Values: "
            + ", ".join(f"{names[j]}={starts[j]:,.2f}" for j in range(len(names)))
        )


def trim_to_common_length(
    results: dict[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], int | None]:
    """
    Trim every series to the minimum length (suffix truncation).

    Returns (trimmed dict, original max length if any trim occurred, else None).
    """
    if not results:
        return {}, None
    lengths = [len(v) for v in results.values()]
    m = min(lengths)
    if max(lengths) == m:
        return {k: np.asarray(v, dtype=np.float64) for k, v in results.items()}, None

    trimmed = {k: np.asarray(v, dtype=np.float64)[:m].copy() for k, v in results.items()}
    return trimmed, max(lengths)


def compare_three_strategies(
    buy_and_hold_portfolio: np.ndarray,
    lstm_portfolio: np.ndarray,
    ppo_portfolio: np.ndarray,
    *,
    labels: tuple[str, str, str] = ("Buy-and-Hold", "LSTM", "PPO"),
    initial_balance_tolerance: float = DEFAULT_INITIAL_BALANCE_TOLERANCE,
) -> pd.DataFrame:
    """
    Compare exactly three strategies after validation.

    Arrays must be shape ``(n_test_days,)`` with aligned calendar and matching
    starting portfolio value.
    """
    validate_strategy_arrays(
        buy_and_hold_portfolio,
        lstm_portfolio,
        ppo_portfolio,
        labels=labels,
        initial_balance_tolerance=initial_balance_tolerance,
    )
    return compare_strategies(
        dict(zip(labels, (buy_and_hold_portfolio, lstm_portfolio, ppo_portfolio)))
    )


def compare_validated_strategies(
    results: dict[str, np.ndarray],
    *,
    initial_balance_tolerance: float = DEFAULT_INITIAL_BALANCE_TOLERANCE,
) -> pd.DataFrame:
    """Validate any number of series, then build the summary table."""
    arrays = tuple(np.asarray(v, dtype=np.float64) for v in results.values())
    validate_strategy_arrays(
        *arrays,
        labels=tuple(results.keys()),
        initial_balance_tolerance=initial_balance_tolerance,
    )
    return compare_strategies(results)


def plot_strategy_comparison(
    results: dict[str, np.ndarray],
    *,
    title: str = "Strategy comparison",
    save_path: str | Path | None = None,
    figsize: tuple[float, float] = (11, 5.5),
    dpi: int = 150,
) -> Path | None:
    """
    Single chart: portfolio value ($) on the left y-axis, normalized wealth
    (day 0 = 1.0) on the right via ``secondary_yaxis`` — same lines, two scales.

    Works when all series share the same starting value (as after harness validation).
    """
    if not results:
        return None

    fig, ax = plt.subplots(figsize=figsize)
    days = np.arange(next(iter(results.values())).shape[0])

    starts = [float(np.asarray(v, dtype=np.float64)[0]) for v in results.values()]
    s0 = starts[0]
    if s0 == 0:
        s0 = 1.0

    for name, arr in results.items():
        ax.plot(days, np.asarray(arr, dtype=np.float64), label=name, linewidth=1.8)

    ax.set_xlabel("Trading day (test set)")
    ax.set_ylabel("Portfolio value ($)")
    ax.set_title(title)
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    ax.ticklabel_format(axis="y", style="plain")

    sec = ax.secondary_yaxis(
        "right",
        functions=(lambda x, _s=s0: x / _s, lambda x, _s=s0: x * _s),
    )
    sec.set_ylabel("Normalized wealth (day 0 = 1.0)")

    fig.tight_layout()

    if save_path is None:
        plt.close(fig)
        return None

    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return path
