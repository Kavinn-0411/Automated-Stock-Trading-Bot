from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

from train_rl import train_agent, run_inference, save_outputs, DATA_DIR, INITIAL_BALANCE

FAANG = ["META", "AAPL", "AMZN", "NFLX", "GOOGL"]
OUTPUT_DIR = Path("Final_Submission")
COLORS = {
    "META": "#1877F2",
    "AAPL": "#555555",
    "AMZN": "#FF9900",
    "NFLX": "#E50914",
    "GOOGL": "#34A853",
}

def _format_axes(ax):
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax.grid(True, alpha=0.25, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_individual(ticker: str, portfolio: np.ndarray, output_dir: Path) -> None:
    charts_dir = output_dir / "charts"
    charts_dir.mkdir(parents=True, exist_ok=True)

    ret = (portfolio[-1] / portfolio[0] - 1.0) * 100
    color = COLORS.get(ticker, "steelblue")
    days = np.arange(len(portfolio))

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(days, portfolio, color=color, linewidth=2, label=ticker)
    ax.axhline(INITIAL_BALANCE, color="gray", linestyle="--", linewidth=1, alpha=0.7, label="Initial ($10,000)")
    ax.fill_between(days, INITIAL_BALANCE, portfolio,
                    where=(portfolio >= INITIAL_BALANCE), alpha=0.08, color="green")
    ax.fill_between(days, INITIAL_BALANCE, portfolio,
                    where=(portfolio < INITIAL_BALANCE), alpha=0.08, color="red")

    ax.set_title(f"PPO Agent — {ticker}   |   Cumulative Return: {ret:+.2f}%", fontsize=13, pad=12)
    ax.set_xlabel("Trading Day (Test Set)", fontsize=10)
    ax.set_ylabel("Portfolio Value", fontsize=10)
    ax.legend(fontsize=10)
    _format_axes(ax)

    plt.tight_layout()
    out = charts_dir / f"portfolio_{ticker}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[chart] Saved {out.name}")


def plot_faang_comparison(portfolios: dict[str, np.ndarray], output_dir: Path) -> None:
    charts_dir = output_dir / "charts"
    charts_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.axhline(INITIAL_BALANCE, color="gray", linestyle="--", linewidth=1,
               alpha=0.7, label="Initial ($10,000)")

    for ticker, portfolio in portfolios.items():
        ret = (portfolio[-1] / portfolio[0] - 1.0) * 100
        ax.plot(np.arange(len(portfolio)), portfolio,
                color=COLORS.get(ticker, "gray"),
                linewidth=1.8,
                label=f"{ticker}  ({ret:+.1f}%)")

    ax.set_title("PPO Agent — FAANG Portfolio Comparison (Test Set)", fontsize=13, pad=12)
    ax.set_xlabel("Trading Day (Test Set)", fontsize=10)
    ax.set_ylabel("Portfolio Value", fontsize=10)
    ax.legend(fontsize=10, loc="upper left")
    _format_axes(ax)

    plt.tight_layout()
    out = charts_dir / "faang_comparison.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[chart] Saved {out.name}")


def plot_returns_bar(portfolios: dict[str, np.ndarray], output_dir: Path) -> None:
    charts_dir = output_dir / "charts"
    charts_dir.mkdir(parents=True, exist_ok=True)

    tickers = list(portfolios.keys())
    returns = [(portfolios[t][-1] / portfolios[t][0] - 1.0) * 100 for t in tickers]
    bar_colors = [COLORS.get(t, "steelblue") for t in tickers]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(tickers, returns, color=bar_colors, edgecolor="white", width=0.5, zorder=3)
    ax.axhline(0, color="black", linewidth=0.8)

    for bar, ret in zip(bars, returns):
        ypos = bar.get_height() + 0.4 if ret >= 0 else bar.get_height() - 1.8
        ax.text(bar.get_x() + bar.get_width() / 2, ypos,
                f"{ret:+.1f}%", ha="center", va="bottom", fontsize=11, fontweight="bold")

    ax.set_title("PPO Agent — Cumulative Return by Ticker (Test Set)", fontsize=13, pad=12)
    ax.set_ylabel("Cumulative Return (%)", fontsize=10)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}%"))
    ax.grid(axis="y", alpha=0.25, linestyle="--", zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    out = charts_dir / "cumulative_returns_faang.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[chart] Saved {out.name}")


def plot_average_portfolio(portfolios: dict[str, np.ndarray], output_dir: Path) -> None:
    charts_dir = output_dir / "charts"
    charts_dir.mkdir(parents=True, exist_ok=True)

    min_len = min(len(p) for p in portfolios.values())
    stacked = np.stack([p[:min_len] for p in portfolios.values()])
    avg = stacked.mean(axis=0)

    ret = (avg[-1] / avg[0] - 1.0) * 100

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(np.arange(min_len), avg, color="darkblue", linewidth=2, label=f"FAANG Average ({ret:+.1f}%)")
    ax.axhline(INITIAL_BALANCE, color="gray", linestyle="--", linewidth=1, alpha=0.7, label="Initial ($10,000)")
    ax.fill_between(np.arange(min_len), INITIAL_BALANCE, avg,
                    where=(avg >= INITIAL_BALANCE), alpha=0.08, color="green")
    ax.fill_between(np.arange(min_len), INITIAL_BALANCE, avg,
                    where=(avg < INITIAL_BALANCE), alpha=0.08, color="red")

    ax.set_title(f"PPO Agent — Equal-Weight FAANG Average Portfolio", fontsize=13, pad=12)
    ax.set_xlabel("Trading Day (Test Set)", fontsize=10)
    ax.set_ylabel("Portfolio Value", fontsize=10)
    ax.legend(fontsize=10)
    _format_axes(ax)

    plt.tight_layout()
    out = charts_dir / "faang_average_portfolio.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[chart] Saved {out.name}")

    return avg



def save_summary(portfolios: dict[str, np.ndarray], output_dir: Path) -> None:
    summary = {}
    for ticker, portfolio in portfolios.items():
        ret = (portfolio[-1] / portfolio[0] - 1.0) * 100
        summary[ticker] = {
            "n_test_days": int(portfolio.shape[0]),
            "initial_balance": INITIAL_BALANCE,
            "final_portfolio_value": round(float(portfolio[-1]), 2),
            "cumulative_return_pct": round(ret, 4),
        }

    out = output_dir / "portfolios" / "faang_summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"[output] Summary -> {out}")

def main() -> None:
    parser = argparse.ArgumentParser(description="Train PPO on all FAANG tickers and generate visualizations.")
    parser.add_argument(
        "--timesteps", type=int, default=2_000_000,
        help="Training timesteps per ticker",
    )
    parser.add_argument(
        "--skip-train", action="store_true",
        help="Skip training and run inference only",
    )
    args = parser.parse_args()

    missing = [t for t in FAANG if not (DATA_DIR / t).exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing processed data for: {missing}. "
            f"Run: python -m data.data_pipeline --preset faang"
        )

    portfolios: dict[str, np.ndarray] = {}

    for ticker in FAANG:
        print(f"\n{'='*55}")
        print(f"  {ticker}")
        print(f"{'='*55}")

        model_exists = (OUTPUT_DIR / "models" / f"ppo_{ticker}.zip").exists()

        if args.skip_train or model_exists:
            if model_exists:
                print(f"[skip] Model already exists for {ticker}, skipping training.")
        else:
            train_agent(ticker, args.timesteps, OUTPUT_DIR)

        portfolio = run_inference(ticker, OUTPUT_DIR)
        save_outputs(ticker, portfolio, OUTPUT_DIR)
        portfolios[ticker] = portfolio

    print(f"\n{'='*55}")
    print("  Generating visualizations")
    print(f"{'='*55}")

    for ticker, portfolio in portfolios.items():
        plot_individual(ticker, portfolio, OUTPUT_DIR)

    plot_faang_comparison(portfolios, OUTPUT_DIR)
    plot_returns_bar(portfolios, OUTPUT_DIR)
    avg_portfolio = plot_average_portfolio(portfolios, OUTPUT_DIR)

    save_summary(portfolios, OUTPUT_DIR)

    integration_path = OUTPUT_DIR / "portfolios" / "ppo.npy"
    np.save(integration_path, avg_portfolio)
    avg_ret = (avg_portfolio[-1] / avg_portfolio[0] - 1.0) * 100
    print(f"\n[output] Integration file (FAANG avg) -> {integration_path}")
    print(f"[output] FAANG average return: {avg_ret:+.2f}%")
    print(f"\n[done] All outputs saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
