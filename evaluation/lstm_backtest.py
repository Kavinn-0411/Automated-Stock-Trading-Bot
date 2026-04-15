"""
Heuristic LSTM trading simulation on the test set (dollar prices).

Uses next-day Close predictions vs a reference price (previous day close) with
a relative threshold; executes full buy / sell / hold with the same fee model
as TradingEnv (default 0.1%).
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd


def _apply_buy(
    balance: float,
    shares: float,
    price: float,
    fee_pct: float,
) -> tuple[float, float]:
    if balance <= 0 or price <= 0:
        return balance, shares
    shares_to_buy = balance / (price * (1.0 + fee_pct))
    fee = shares_to_buy * price * fee_pct
    new_shares = shares + shares_to_buy
    new_balance = balance - shares_to_buy * price - fee
    return new_balance, new_shares


def _apply_sell(
    balance: float,
    shares: float,
    price: float,
    fee_pct: float,
) -> tuple[float, float]:
    if shares <= 0 or price <= 0:
        return balance, shares
    gross = shares * price
    fee = gross * fee_pct
    return balance + gross - fee, 0.0


def simulate_lstm_heuristic_portfolio(
    ticker: str,
    *,
    data_dir: str | Path = "data/processed",
    checkpoint: str | Path | None = None,
    initial_balance: float = 10_000.0,
    threshold: float = 0.005,
    transaction_fee_percent: float = 0.001,
) -> tuple[np.ndarray, dict]:
    """
    Run heuristic LSTM strategy on test period; return daily portfolio values.

    Reference price for the signal on test day i is the previous day's actual
    close (train's last close on day 0). Trades execute at that day's actual
    close from test_raw.csv.
    """
    data_dir = Path(data_dir) / ticker
    train_norm = pd.read_csv(data_dir / "train.csv", index_col="Date", parse_dates=True)
    test_norm = pd.read_csv(data_dir / "test.csv", index_col="Date", parse_dates=True)
    train_raw = pd.read_csv(data_dir / "train_raw.csv", index_col="Date", parse_dates=True)
    test_raw = pd.read_csv(data_dir / "test_raw.csv", index_col="Date", parse_dates=True)
    scalers = joblib.load(data_dir / "scalers.pkl")

    ckpt_path = Path(checkpoint or f"models/saved/{ticker}_lstm.pkl")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"LSTM checkpoint not found: {ckpt_path}")

    import torch
    from models.predict_test_lstm import (
        inverse_close_column,
        load_model,
        predict_test_set,
        predict_test_set_returns,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ckpt = load_model(str(ckpt_path), device)
    window_size = int(ckpt["window_size"])
    target_mode = ckpt.get("target_mode", "close")

    ticker_id = None
    if ckpt.get("multi_ticker"):
        ticker_map = ckpt.get("ticker_to_idx") or {
            t: i for i, t in enumerate(ckpt["tickers"])
        }
        if ticker not in ticker_map:
            raise KeyError(
                f"Ticker {ticker!r} not in checkpoint; trained on: {ckpt['tickers']}"
            )
        ticker_id = int(ticker_map[ticker])

    if target_mode == "return":
        _pred_ret, _actual_ret, pred_dollar, actual_dollar, _dates = predict_test_set_returns(
            model,
            train_norm,
            test_norm,
            train_raw,
            test_raw,
            window_size,
            device,
            ticker_id=ticker_id,
        )
    else:
        preds_norm, actual_norm, _dates = predict_test_set(
            model, train_norm, test_norm, window_size, device, ticker_id=ticker_id
        )
        price_scaler = scalers["price"]
        pred_dollar = inverse_close_column(preds_norm, price_scaler)
        actual_dollar = inverse_close_column(actual_norm, price_scaler)

    close_prices = test_raw["Close"].to_numpy(dtype=np.float64)
    n = len(close_prices)
    if len(pred_dollar) != n:
        raise ValueError(f"Prediction length {len(pred_dollar)} != test rows {n}")

    ref_prev = float(train_raw["Close"].iloc[-1])

    balance = float(initial_balance)
    shares = 0.0
    values = np.empty(n, dtype=np.float64)

    for i in range(n):
        price = float(close_prices[i])
        pred = float(pred_dollar[i])
        ref = ref_prev

        if pred > ref * (1.0 + threshold):
            balance, shares = _apply_buy(balance, shares, price, transaction_fee_percent)
        elif pred < ref * (1.0 - threshold):
            balance, shares = _apply_sell(balance, shares, price, transaction_fee_percent)

        values[i] = balance + shares * price
        ref_prev = float(actual_dollar[i])

    meta = {
        "ticker": ticker,
        "checkpoint": str(ckpt_path),
        "threshold": threshold,
        "transaction_fee_percent": transaction_fee_percent,
        "initial_balance": initial_balance,
        "final_value": float(values[-1]),
    }
    return values, meta
