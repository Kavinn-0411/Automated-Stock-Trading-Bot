"""
Evaluate LSTM next-day Close predictions on the held-out test set.

Uses the last `window_size` rows of *train* (normalized) plus all of *test*
so the first prediction targets the first test day — no peeking at future
test rows when forming inputs.

Metrics: MSE / MAE / RMSE in normalized space, and MAE / RMSE in dollars
(via inverse MinMaxScaler for the Close column).

Usage
-----
  python -m models.predict_test_lstm --ticker AAPL
  python -m models.predict_test_lstm --ticker AAPL --checkpoint models/saved/AAPL_lstm.pkl
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

from models.lstm_model import LSTMPricePredictor
from data.data_pipeline import PRICE_COLS


def inverse_close_column(norm_values: np.ndarray, price_scaler) -> np.ndarray:
    """Map normalized Close back to dollars using the fitted price MinMaxScaler."""
    close_j = PRICE_COLS.index("Close")
    n = len(norm_values)
    X = np.zeros((n, len(PRICE_COLS)), dtype=np.float64)
    X[:, close_j] = norm_values
    return price_scaler.inverse_transform(X)[:, close_j]


def load_model(path: str, device: torch.device) -> tuple[LSTMPricePredictor, dict]:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = LSTMPricePredictor(
        input_dim=ckpt["input_dim"],
        hidden_dim=ckpt["hidden_dim"],
        num_layers=ckpt["num_layers"],
        dropout=ckpt["dropout"],
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt


@torch.no_grad()
def predict_test_set(
    model: LSTMPricePredictor,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    window_size: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
    """One next-day normalized Close prediction per test row."""
    if list(train_df.columns) != list(test_df.columns):
        raise ValueError("train.csv and test.csv columns must match")

    close_i = train_df.columns.get_loc("Close")
    full = np.vstack([train_df.values, test_df.values]).astype(np.float32)
    T_train = len(train_df)
    T_test = len(test_df)

    preds_list: list[float] = []
    actual_list: list[float] = []
    dates_list: list[pd.Timestamp] = []

    for idx in range(T_train - window_size, T_train + T_test - window_size):
        tgt_idx = idx + window_size
        window = full[idx : idx + window_size]
        x = torch.tensor(window, dtype=torch.float32).unsqueeze(0).to(device)
        pred = model(x).cpu().item()
        preds_list.append(pred)
        actual_list.append(float(full[tgt_idx, close_i]))
        dates_list.append(test_df.index[tgt_idx - T_train])

    return (
        np.array(preds_list, dtype=np.float64),
        np.array(actual_list, dtype=np.float64),
        pd.DatetimeIndex(dates_list),
    )


def main() -> None:
    p = argparse.ArgumentParser(description="Test-set prediction metrics for LSTM")
    p.add_argument("--ticker", default="AAPL")
    p.add_argument("--data-dir", default="data/processed")
    p.add_argument("--checkpoint", default=None, help="Default: models/saved/<TICKER>_lstm.pkl")
    p.add_argument("--output-csv", default=None, help="Optional path to save pred vs actual")
    args = p.parse_args()

    data_dir = Path(args.data_dir) / args.ticker
    train_path = data_dir / "train.csv"
    test_path = data_dir / "test.csv"
    scaler_path = data_dir / "scalers.pkl"
    ckpt_path = Path(args.checkpoint or f"models/saved/{args.ticker}_lstm.pkl")

    for path in (train_path, test_path, scaler_path, ckpt_path):
        if not path.exists():
            raise FileNotFoundError(path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ckpt = load_model(str(ckpt_path), device)
    W = int(ckpt["window_size"])

    train_df = pd.read_csv(train_path, index_col="Date", parse_dates=True)
    test_df = pd.read_csv(test_path, index_col="Date", parse_dates=True)

    preds_norm, actual_norm, dates = predict_test_set(
        model, train_df, test_df, W, device
    )

    err = preds_norm - actual_norm
    mse = float(np.mean(err**2))
    mae_norm = float(np.mean(np.abs(err)))
    rmse_norm = float(np.sqrt(mse))

    scalers = joblib.load(scaler_path)
    price_scaler = scalers["price"]
    pred_dollar = inverse_close_column(preds_norm, price_scaler)
    actual_dollar = inverse_close_column(actual_norm, price_scaler)
    err_d = pred_dollar - actual_dollar
    mae_dollar = float(np.mean(np.abs(err_d)))
    rmse_dollar = float(np.sqrt(np.mean(err_d**2)))

    print(f"Checkpoint: {ckpt_path}")
    print(f"Test rows:  {len(test_df)}  |  Predictions: {len(preds_norm)}  |  window_size={W}")
    print(f"Period:     {dates[0].date()} -> {dates[-1].date()}")
    print()
    print("Normalized Close (next-day):")
    print(f"  MSE   = {mse:.6f}")
    print(f"  MAE   = {mae_norm:.6f}")
    print(f"  RMSE  = {rmse_norm:.6f}")
    print()
    print("Dollar Close (inverse MinMax on Close):")
    print(f"  MAE   = ${mae_dollar:.4f}")
    print(f"  RMSE  = ${rmse_dollar:.4f}")

    if args.output_csv:
        out = Path(args.output_csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            {
                "Date": dates,
                "actual_close_norm": actual_norm,
                "pred_close_norm": preds_norm,
                "actual_close_usd": actual_dollar,
                "pred_close_usd": pred_dollar,
            }
        ).to_csv(out, index=False)
        print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
