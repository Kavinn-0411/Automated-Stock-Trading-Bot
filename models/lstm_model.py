"""
LSTM-based stock price predictor.

Predicts next-day closing price from a sliding window of historical features
(OHLCV + technical indicators). Uses a heuristic threshold on the predicted
price change to emit buy / sell / hold signals.

Owner: Kavinn
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# ---------------------------------------------------------------------------
# 1.  Model
# ---------------------------------------------------------------------------

class LSTMPricePredictor(nn.Module):
    """
    Multi-layer LSTM followed by a fully-connected head that outputs a single
    scalar — the predicted next-day closing price (normalised scale).

    Parameters
    ----------
    input_dim  : int   – number of features per timestep
    hidden_dim : int   – LSTM hidden-state size
    num_layers : int   – stacked LSTM layers
    dropout    : float – dropout between LSTM layers (ignored when num_layers=1)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor of shape (batch, seq_len, input_dim)

        Returns
        -------
        Tensor of shape (batch, 1) — predicted next-day close (normalised).
        """
        # lstm_out: (batch, seq_len, hidden_dim)
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]          # take the last timestep
        return self.fc(last_hidden)               # (batch, 1)


# ---------------------------------------------------------------------------
# 2.  Dataset
# ---------------------------------------------------------------------------

class StockSequenceDataset(Dataset):
    """
    Converts a 2-D feature array into overlapping (window, target) pairs.

    Parameters
    ----------
    features     : np.ndarray of shape (T, F) — all features incl. close
    targets      : np.ndarray of shape (T,)   — the value to predict (e.g. normalised close)
    window_size  : int — number of past timesteps per sample
    """

    def __init__(self, features: np.ndarray, targets: np.ndarray, window_size: int = 30):
        assert len(features) == len(targets)
        self.features = torch.tensor(features, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)
        self.window_size = window_size

    def __len__(self) -> int:
        return len(self.features) - self.window_size

    def __getitem__(self, idx: int):
        x = self.features[idx : idx + self.window_size]       # (window, F)
        y = self.targets[idx + self.window_size]               # scalar
        return x, y


# ---------------------------------------------------------------------------
# 3.  Training utilities
# ---------------------------------------------------------------------------

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """Run one pass over the dataloader; return mean loss."""
    model.train()
    total_loss = 0.0
    for x_batch, y_batch in loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device).unsqueeze(1)  # (batch, 1)

        preds = model(x_batch)
        loss = criterion(preds, y_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x_batch.size(0)

    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """Return mean loss on the given loader (no gradient)."""
    model.eval()
    total_loss = 0.0
    for x_batch, y_batch in loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device).unsqueeze(1)

        preds = model(x_batch)
        total_loss += criterion(preds, y_batch).item() * x_batch.size(0)

    return total_loss / len(loader.dataset)


# ---------------------------------------------------------------------------
# 4.  End-to-end training loop (callable or __main__)
# ---------------------------------------------------------------------------

def run_training(
    features: np.ndarray,
    targets: np.ndarray,
    *,
    window_size: int = 30,
    hidden_dim: int = 128,
    num_layers: int = 2,
    dropout: float = 0.2,
    lr: float = 1e-3,
    epochs: int = 50,
    batch_size: int = 64,
    train_ratio: float = 0.8,
    device: torch.device | None = None,
) -> dict:
    """
    Splits data, builds model, trains, and returns results dict.

    Parameters
    ----------
    features    : (T, F) array of input features
    targets     : (T,) array of prediction targets
    Other params are hyperparameters with sensible defaults.

    Returns
    -------
    dict with keys: model, train_losses, val_losses, device
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    split = int(len(features) * train_ratio)
    train_ds = StockSequenceDataset(features[:split], targets[:split], window_size)
    val_ds = StockSequenceDataset(features[split:], targets[split:], window_size)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    input_dim = features.shape[1]
    model = LSTMPricePredictor(input_dim, hidden_dim, num_layers, dropout).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_losses, val_losses = [], []

    for epoch in range(1, epochs + 1):
        t_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        v_loss = evaluate(model, val_loader, criterion, device)
        train_losses.append(t_loss)
        val_losses.append(v_loss)

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:>3}/{epochs}  |  train MSE: {t_loss:.6f}  |  val MSE: {v_loss:.6f}")

    return {
        "model": model,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "device": device,
    }


# ---------------------------------------------------------------------------
# 5.  Quick smoke-test with synthetic data
# ---------------------------------------------------------------------------

def _demo():
    """Generate fake data and verify the full pipeline runs end-to-end."""
    np.random.seed(42)
    T, F = 500, 8          # 500 days, 8 features
    features = np.random.randn(T, F).astype(np.float32)
    targets = np.random.randn(T).astype(np.float32)

    results = run_training(
        features,
        targets,
        window_size=20,
        hidden_dim=64,
        num_layers=2,
        epochs=20,
        batch_size=32,
    )

    print(f"\nFinal train MSE: {results['train_losses'][-1]:.6f}")
    print(f"Final val   MSE: {results['val_losses'][-1]:.6f}")
    print(f"Model device   : {results['device']}")
    print("Smoke-test passed.")


if __name__ == "__main__":
    _demo()
