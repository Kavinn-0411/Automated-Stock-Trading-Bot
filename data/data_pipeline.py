"""
Data pipeline for the Automated Stock Trading Bot.

Downloads OHLCV data via yfinance, computes technical indicators
(RSI, MACD, Moving Averages), normalizes features, and performs
a train/test split. Outputs are saved as CSV files ready for
consumption by the LSTM model and the PPO trading environment.
"""

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
import joblib


# ──────────────────────────────────────────────
# Preset Ticker Groups
# ──────────────────────────────────────────────

TICKER_PRESETS: dict[str, list[str]] = {
    "faang": [
        "META", "AAPL", "AMZN", "NFLX", "GOOGL",
    ],
    "mag7": [
        "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA",
    ],
    "tech": [
        "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA",
        "AMD", "INTC", "CRM", "ADBE", "ORCL", "CSCO", "AVGO", "QCOM",
    ],
    "finance": [
        "JPM", "BAC", "WFC", "GS", "MS", "C", "BLK", "SCHW",
        "AXP", "USB", "PNC", "TFC", "COF", "BK",
    ],
    "healthcare": [
        "JNJ", "UNH", "PFE", "ABBV", "MRK", "TMO", "ABT", "LLY",
        "BMY", "AMGN", "GILD", "MDT", "ISRG", "CVS",
    ],
    "energy": [
        "XOM", "CVX", "COP", "SLB", "EOG", "MPC", "PSX", "VLO",
        "OXY", "HAL", "DVN", "HES",
    ],
    "consumer": [
        "WMT", "PG", "KO", "PEP", "COST", "MCD", "NKE", "SBUX",
        "TGT", "HD", "LOW", "DIS", "CMCSA",
    ],
    "sp500_top30": [
        "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA",
        "BRK-B", "UNH", "JNJ", "JPM", "V", "XOM", "PG", "MA",
        "HD", "CVX", "MRK", "ABBV", "LLY", "PEP", "KO", "AVGO",
        "COST", "WMT", "MCD", "CSCO", "TMO", "ABT", "CRM",
    ],
    "all": [],  # populated dynamically below
}

# "all" = union of every other preset (deduplicated, sorted)
_all_tickers: set[str] = set()
for _group, _syms in TICKER_PRESETS.items():
    if _group != "all":
        _all_tickers.update(_syms)
TICKER_PRESETS["all"] = sorted(_all_tickers)


# ──────────────────────────────────────────────
# 1. Data Download
# ──────────────────────────────────────────────

def download_ohlcv(
    tickers: list[str],
    start: str = "2015-01-01",
    end: str = "2024-12-31",
    interval: str = "1d",
) -> dict[str, pd.DataFrame]:
    """Download daily OHLCV data from Yahoo Finance for each ticker."""
    data = {}
    for ticker in tickers:
        print(f"[download] Fetching {ticker} ({start} -> {end}) ...")
        df = yf.download(ticker, start=start, end=end, interval=interval, progress=False)
        if df.empty:
            print(f"[download] WARNING: No data returned for {ticker}")
            continue

        # yfinance may return MultiIndex columns for single tickers; flatten
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df.index.name = "Date"
        df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
        df.dropna(inplace=True)
        data[ticker] = df
        print(f"[download] {ticker}: {len(df)} rows ({df.index[0].date()} -> {df.index[-1].date()})")
    return data


# ──────────────────────────────────────────────
# 2. Technical Indicators
# ──────────────────────────────────────────────

def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index."""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def compute_macd(
    series: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> pd.DataFrame:
    """MACD line, signal line, and histogram."""
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return pd.DataFrame({
        "MACD": macd_line,
        "MACD_Signal": signal_line,
        "MACD_Hist": histogram,
    })


def compute_moving_averages(
    series: pd.Series,
    windows: list[int] | None = None,
) -> pd.DataFrame:
    """Simple and exponential moving averages for given windows."""
    if windows is None:
        windows = [7, 21, 50]
    cols = {}
    for w in windows:
        cols[f"SMA_{w}"] = series.rolling(window=w).mean()
        cols[f"EMA_{w}"] = series.ewm(span=w, adjust=False).mean()
    return pd.DataFrame(cols)


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Attach RSI, MACD, and moving averages to OHLCV DataFrame."""
    close = df["Close"]
    df["RSI"] = compute_rsi(close)

    macd_df = compute_macd(close)
    df = pd.concat([df, macd_df], axis=1)

    ma_df = compute_moving_averages(close)
    df = pd.concat([df, ma_df], axis=1)

    # Drop warm-up rows where indicators are NaN
    df.dropna(inplace=True)
    return df


# ──────────────────────────────────────────────
# 3. Normalization
# ──────────────────────────────────────────────

PRICE_COLS = [
    "Open", "High", "Low", "Close",
    "SMA_7", "SMA_21", "SMA_50",
    "EMA_7", "EMA_21", "EMA_50",
    "MACD", "MACD_Signal", "MACD_Hist",
]
VOLUME_COLS = ["Volume"]
BOUNDED_COLS = ["RSI"]  # already 0-100, scale to 0-1


def normalize_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    output_dir: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Fit scalers on train data, transform both train and test.
    Returns normalized copies and a dict of fitted scalers.
    """
    train = train_df.copy()
    test = test_df.copy()
    scalers: dict[str, MinMaxScaler] = {}

    # Price columns → MinMaxScaler
    price_scaler = MinMaxScaler()
    train[PRICE_COLS] = price_scaler.fit_transform(train[PRICE_COLS])
    test[PRICE_COLS] = price_scaler.transform(test[PRICE_COLS])
    scalers["price"] = price_scaler

    # Volume → separate MinMaxScaler
    vol_scaler = MinMaxScaler()
    train[VOLUME_COLS] = vol_scaler.fit_transform(train[VOLUME_COLS])
    test[VOLUME_COLS] = vol_scaler.transform(test[VOLUME_COLS])
    scalers["volume"] = vol_scaler

    # RSI → simple divide by 100
    train["RSI"] = train["RSI"] / 100.0
    test["RSI"] = test["RSI"] / 100.0

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(scalers, output_dir / "scalers.pkl")
        print(f"[normalize] Scalers saved to {output_dir / 'scalers.pkl'}")

    return train, test, scalers


# ──────────────────────────────────────────────
# 4. Train / Test Split
# ──────────────────────────────────────────────

def train_test_split_temporal(
    df: pd.DataFrame,
    test_ratio: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Chronological split — no shuffling.
    The last `test_ratio` fraction of rows becomes the test set.
    """
    split_idx = int(len(df) * (1 - test_ratio))
    train = df.iloc[:split_idx].copy()
    test = df.iloc[split_idx:].copy()
    print(f"[split] Train: {len(train)} rows | Test: {len(test)} rows")
    return train, test


# ──────────────────────────────────────────────
# 5. Full Pipeline
# ──────────────────────────────────────────────

def run_pipeline(
    tickers: list[str],
    start: str = "2015-01-01",
    end: str = "2024-12-31",
    test_ratio: float = 0.2,
    output_dir: str = "data/processed",
) -> dict[str, dict[str, pd.DataFrame]]:
    """
    End-to-end pipeline:
      download → indicators → split → normalize → save
    Returns a dict mapping ticker → {"train": df, "test": df}.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    raw_data = download_ohlcv(tickers, start=start, end=end)
    results = {}

    for ticker, df in raw_data.items():
        print(f"\n{'='*50}")
        print(f"  Processing {ticker}")
        print(f"{'='*50}")

        df = add_technical_indicators(df)
        train_raw, test_raw = train_test_split_temporal(df, test_ratio=test_ratio)

        ticker_dir = out / ticker
        train_norm, test_norm, _ = normalize_features(train_raw, test_raw, output_dir=ticker_dir)

        # Save both raw (for price-reference) and normalized versions
        train_raw.to_csv(ticker_dir / "train_raw.csv")
        test_raw.to_csv(ticker_dir / "test_raw.csv")
        train_norm.to_csv(ticker_dir / "train.csv")
        test_norm.to_csv(ticker_dir / "test.csv")

        print(f"[save] Files written to {ticker_dir}/")
        print(f"       Features: {list(train_norm.columns)}")

        results[ticker] = {"train": train_norm, "test": test_norm}

    return results


# ──────────────────────────────────────────────
# CLI Entry Point
# ──────────────────────────────────────────────

def _print_presets():
    """Display all available ticker presets."""
    print("\nAvailable presets:\n")
    for name, syms in TICKER_PRESETS.items():
        count = len(syms)
        preview = ", ".join(syms[:6])
        suffix = " ..." if count > 6 else ""
        print(f"  {name:16s} ({count:3d} tickers)  {preview}{suffix}")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Stock data pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Examples:\n"
               "  python -m data.data_pipeline --preset mag7\n"
               "  python -m data.data_pipeline --preset tech --start 2020-01-01\n"
               "  python -m data.data_pipeline --preset sp500_top30\n"
               "  python -m data.data_pipeline --tickers AAPL TSLA --preset finance\n"
               "  python -m data.data_pipeline --list-presets\n",
    )
    parser.add_argument(
        "--tickers",
        nargs="+",
        default=[],
        help="Individual ticker symbols to download",
    )
    parser.add_argument(
        "--preset",
        choices=list(TICKER_PRESETS.keys()),
        help="Use a predefined group of tickers (see --list-presets)",
    )
    parser.add_argument(
        "--list-presets",
        action="store_true",
        help="Show all available preset ticker groups and exit",
    )
    parser.add_argument("--start", default="2015-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", default="2024-12-31", help="End date (YYYY-MM-DD)")
    parser.add_argument("--test-ratio", type=float, default=0.2, help="Fraction of data for test set")
    parser.add_argument("--output-dir", default="data/processed", help="Output directory")
    args = parser.parse_args()

    if args.list_presets:
        _print_presets()
        return

    # Merge --preset tickers with any explicit --tickers, deduplicate
    tickers = list(args.tickers)
    if args.preset:
        tickers.extend(TICKER_PRESETS[args.preset])
    # Deduplicate while preserving order
    seen: set[str] = set()
    unique_tickers = []
    for t in tickers:
        t_upper = t.upper()
        if t_upper not in seen:
            seen.add(t_upper)
            unique_tickers.append(t_upper)
    tickers = unique_tickers

    if not tickers:
        tickers = ["AAPL", "MSFT", "GOOGL"]
        print("[pipeline] No tickers specified, using defaults: AAPL, MSFT, GOOGL")

    print(f"[pipeline] {len(tickers)} tickers queued: {', '.join(tickers)}\n")

    results = run_pipeline(
        tickers=tickers,
        start=args.start,
        end=args.end,
        test_ratio=args.test_ratio,
        output_dir=args.output_dir,
    )

    # Summary
    print("\n" + "=" * 50)
    print("  Pipeline complete!")
    print("=" * 50)
    successes = 0
    for ticker, splits in results.items():
        tr, te = splits["train"], splits["test"]
        print(f"  {ticker:8s}: train={tr.shape}, test={te.shape}")
        successes += 1
    failed = len(tickers) - successes
    if failed:
        print(f"\n  {failed} ticker(s) failed to download (see warnings above)")


if __name__ == "__main__":
    main()
