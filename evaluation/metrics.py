"""
Quantitative evaluation metrics for comparing trading strategies.

TODO (Prakash):
  - Hook these into the evaluation harness that runs each model
    on the test set and produces a comparison table / plots.
"""

import numpy as np
import pandas as pd


def cumulative_return(portfolio_values: np.ndarray) -> float:
    """Total return over the evaluation period."""
    return (portfolio_values[-1] / portfolio_values[0]) - 1.0


def sharpe_ratio(
    portfolio_values: np.ndarray,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    """Annualized Sharpe ratio from daily portfolio values."""
    daily_returns = np.diff(portfolio_values) / portfolio_values[:-1]
    excess = daily_returns - risk_free_rate / periods_per_year
    if excess.std() == 0:
        return 0.0
    return float(np.sqrt(periods_per_year) * excess.mean() / excess.std())


def max_drawdown(portfolio_values: np.ndarray) -> float:
    """Maximum peak-to-trough decline (returned as a positive fraction)."""
    cumulative_max = np.maximum.accumulate(portfolio_values)
    drawdowns = (cumulative_max - portfolio_values) / cumulative_max
    return float(drawdowns.max())


def evaluation_summary(portfolio_values: np.ndarray, label: str = "Model") -> dict:
    """Return a summary dict for a single strategy."""
    return {
        "Strategy": label,
        "Cumulative Return": f"{cumulative_return(portfolio_values):.2%}",
        "Sharpe Ratio": f"{sharpe_ratio(portfolio_values):.3f}",
        "Max Drawdown": f"{max_drawdown(portfolio_values):.2%}",
    }


def compare_strategies(results: dict[str, np.ndarray]) -> pd.DataFrame:
    """
    Compare multiple strategies side by side.
    `results` maps strategy name → array of daily portfolio values.
    """
    rows = [evaluation_summary(v, label=k) for k, v in results.items()]
    return pd.DataFrame(rows).set_index("Strategy")
