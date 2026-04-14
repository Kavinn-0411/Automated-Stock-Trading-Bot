"""
LSTM-based stock price predictor.

Predicts next-day closing price (or next-day return) from a sliding window of
historical features (OHLCV + technical indicators).  Supports both single-ticker
and multi-ticker (shared trunk + learned embeddings + temporal attention) modes.

Owner: Kavinn
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# ── Temporal Attention ───────────────────────────────────────────────

class TemporalAttention(nn.Module):
    """Additive (Bahdanau-style) attention over LSTM timesteps."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attn = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, lstm_out: torch.Tensor) -> torch.Tensor:
        """
        lstm_out : (batch, seq_len, hidden_dim)
        returns  : (batch, hidden_dim) — attention-weighted context vector
        """
        scores = self.attn(lstm_out).squeeze(-1)       # (batch, seq_len)
        weights = F.softmax(scores, dim=-1)             # (batch, seq_len)
        context = torch.bmm(weights.unsqueeze(1), lstm_out).squeeze(1)
        return context


# 1.  Model

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
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]
        return self.fc(last_hidden)


# 2.  Dataset
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


# 2b.  Multi-ticker pooled model (shared LSTM + learned ticker embedding + attention)

class MultiTickerLSTMPricePredictor(nn.Module):
    """
    One LSTM trained on many tickers.  Each timestep is
    [market_features || embedding(ticker_id)], so the network can learn
    both shared dynamics and a per-symbol offset in embedding space.

    Uses temporal attention to weight all timesteps instead of only the last.

    Forward
    -------
    x : (batch, seq_len, feature_dim)  — same normalized columns as `train.csv`
    ticker_ids : (batch,) int64 — index in the fixed ticker list used at train time
    """

    def __init__(
        self,
        feature_dim: int,
        num_tickers: int,
        embedding_dim: int = 16,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_tickers = num_tickers
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.ticker_emb = nn.Embedding(num_tickers, embedding_dim)
        lstm_in = feature_dim + embedding_dim
        self.lstm = nn.LSTM(
            input_size=lstm_in,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.attention = TemporalAttention(hidden_dim)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor, ticker_ids: torch.Tensor) -> torch.Tensor:
        if ticker_ids.dim() == 0:
            ticker_ids = ticker_ids.unsqueeze(0)
        e = self.ticker_emb(ticker_ids.long())
        e = e.unsqueeze(1).expand(-1, x.size(1), -1)
        z = torch.cat([x, e], dim=-1)
        lstm_out, _ = self.lstm(z)
        context = self.attention(lstm_out)
        return self.fc(context)


class MultiTickerStockSequenceDataset(Dataset):
    """Pre-built windows with integer ticker indices."""

    def __init__(self, x: np.ndarray, y: np.ndarray, ticker_ids: np.ndarray):
        if not (len(x) == len(y) == len(ticker_ids)):
            raise ValueError("x, y, ticker_ids must have the same length")
        self.x = torch.from_numpy(np.ascontiguousarray(x.astype(np.float32)))
        self.y = torch.from_numpy(np.ascontiguousarray(y.astype(np.float32)))
        self.ticker_ids = torch.from_numpy(np.ascontiguousarray(ticker_ids.astype(np.int64)))

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, idx: int):
        return self.x[idx], self.y[idx], self.ticker_ids[idx]


def _ticker_train_val_windows(
    features: np.ndarray,
    targets: np.ndarray,
    window_size: int,
    train_ratio: float,
    ticker_idx: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Per-ticker chronological split matching `StockSequenceDataset` on train
    then val slices: train windows live only in the train region; val only in val.
    """
    t, f = features.shape
    split = int(t * train_ratio)
    n_tr = max(0, split - window_size)
    if n_tr > 0:
        tx = np.stack([features[i : i + window_size] for i in range(n_tr)], axis=0)
        ty = np.array([targets[i + window_size] for i in range(n_tr)], dtype=np.float32)
        tt = np.full(n_tr, ticker_idx, dtype=np.int64)
    else:
        tx = np.zeros((0, window_size, f), dtype=np.float32)
        ty = np.zeros(0, dtype=np.float32)
        tt = np.zeros(0, dtype=np.int64)

    tv = t - split
    n_va = max(0, tv - window_size)
    if n_va > 0:
        vx = np.stack(
            [features[split + i : split + i + window_size] for i in range(n_va)],
            axis=0,
        )
        vy = np.array(
            [targets[split + i + window_size] for i in range(n_va)],
            dtype=np.float32,
        )
        vt = np.full(n_va, ticker_idx, dtype=np.int64)
    else:
        vx = np.zeros((0, window_size, f), dtype=np.float32)
        vy = np.zeros(0, dtype=np.float32)
        vt = np.zeros(0, dtype=np.int64)

    return tx, ty, tt, vx, vy, vt


# 3.  Training utilities

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


def train_one_epoch_multi(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip: float = 0.0,
) -> float:
    model.train()
    total_loss = 0.0
    for x_batch, y_batch, tid_batch in loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device).unsqueeze(1)
        tid_batch = tid_batch.to(device)

        preds = model(x_batch, tid_batch)
        loss = criterion(preds, y_batch)

        optimizer.zero_grad()
        loss.backward()
        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total_loss += loss.item() * x_batch.size(0)

    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate_multi(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.eval()
    total_loss = 0.0
    for x_batch, y_batch, tid_batch in loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device).unsqueeze(1)
        tid_batch = tid_batch.to(device)
        preds = model(x_batch, tid_batch)
        total_loss += criterion(preds, y_batch).item() * x_batch.size(0)

    return total_loss / len(loader.dataset)


# 4.  End-to-end training loop (callable or __main__)

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
    save_path: str | None = None,
    verbose: bool = True,
    device: torch.device | None = None,
) -> dict:
    """
    Splits data, builds model, trains, and returns results dict.

    Parameters
    ----------
    features    : (T, F) array of input features
    targets     : (T,) array of prediction targets
    save_path   : if provided, the best model (by val loss) is saved here
    Other params are hyperparameters with sensible defaults.

    Returns
    -------
    dict with keys: model, train_losses, val_losses, device, best_epoch, best_val_loss
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
    best_val_loss = float("inf")
    best_epoch = 0
    best_state = None

    for epoch in range(1, epochs + 1):
        t_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        v_loss = evaluate(model, val_loader, criterion, device)
        train_losses.append(t_loss)
        val_losses.append(v_loss)

        if v_loss < best_val_loss:
            best_val_loss = v_loss
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if verbose and (epoch % 10 == 0 or epoch == 1):
            print(f"Epoch {epoch:>3}/{epochs}  |  train MSE: {t_loss:.6f}  |  val MSE: {v_loss:.6f}")

    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)
        if verbose:
            print(f"\nRestored best model from epoch {best_epoch} (val MSE: {best_val_loss:.6f})")

    if save_path is not None:
        from pathlib import Path
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "input_dim": input_dim,
            "hidden_dim": hidden_dim,
            "num_layers": num_layers,
            "dropout": dropout,
            "window_size": window_size,
            "best_epoch": best_epoch,
            "best_val_loss": best_val_loss,
            "train_losses": train_losses,
            "val_losses": val_losses,
        }
        torch.save(checkpoint, save_path)
        if verbose:
            print(f"Model saved to {save_path}")

    return {
        "model": model,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "device": device,
        "best_epoch": best_epoch,
        "best_val_loss": float(best_val_loss),
    }


def run_training_multi(
    ticker_features: list[np.ndarray],
    ticker_targets: list[np.ndarray],
    tickers: list[str],
    *,
    window_size: int = 45,
    embedding_dim: int = 16,
    hidden_dim: int = 64,
    num_layers: int = 2,
    dropout: float = 0.3,
    lr: float = 5e-4,
    weight_decay: float = 1e-4,
    grad_clip: float = 1.0,
    epochs: int = 80,
    patience: int = 15,
    batch_size: int = 64,
    train_ratio: float = 0.8,
    save_path: str | None = None,
    target_mode: str = "close",
    verbose: bool = True,
    device: torch.device | None = None,
) -> dict:
    """
    Pool sliding-window samples from multiple tickers (each split chronologically
    like `run_training`), train one `MultiTickerLSTMPricePredictor`, return results.

    target_mode : "close" — predict normalized close price (original behaviour)
                  "return" — predict next-day log-return (stationary, no OOD issue)
    patience    : early-stop after this many epochs with no val-loss improvement
    grad_clip   : max gradient norm (0 = disabled)
    weight_decay: L2 regularisation in Adam
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if len(ticker_features) != len(ticker_targets) or len(ticker_features) != len(tickers):
        raise ValueError("ticker_features, ticker_targets, and tickers must align")
    if len(tickers) < 2:
        raise ValueError("Need at least two tickers for multi-ticker training")

    feature_dims = {a.shape[1] for a in ticker_features}
    if len(feature_dims) != 1:
        raise ValueError(
            f"All tickers must have the same feature column count; got {feature_dims}"
        )
    feature_dim = ticker_features[0].shape[1]
    num_tickers = len(tickers)
    ticker_to_idx = {name: i for i, name in enumerate(tickers)}

    tx_list, ty_list, tt_list = [], [], []
    vx_list, vy_list, vt_list = [], [], []
    for i, (feat, tgt) in enumerate(zip(ticker_features, ticker_targets)):
        a, b, c, d, e, f_ = _ticker_train_val_windows(
            feat, tgt, window_size, train_ratio, i
        )
        tx_list.append(a)
        ty_list.append(b)
        tt_list.append(c)
        vx_list.append(d)
        vy_list.append(e)
        vt_list.append(f_)

    tx = np.concatenate(tx_list, axis=0)
    ty = np.concatenate(ty_list, axis=0)
    tt = np.concatenate(tt_list, axis=0)
    vx = np.concatenate(vx_list, axis=0)
    vy = np.concatenate(vy_list, axis=0)
    vt = np.concatenate(vt_list, axis=0)

    if tx.shape[0] == 0:
        raise ValueError("No training windows — increase train.csv length or lower window_size")
    if vx.shape[0] == 0:
        raise ValueError("No validation windows — check train_ratio and window_size")

    train_ds = MultiTickerStockSequenceDataset(tx, ty, tt)
    val_ds = MultiTickerStockSequenceDataset(vx, vy, vt)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    model = MultiTickerLSTMPricePredictor(
        feature_dim=feature_dim,
        num_tickers=num_tickers,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5,
    )

    train_losses, val_losses = [], []
    best_val_loss = float("inf")
    best_epoch = 0
    best_state = None
    epochs_no_improve = 0

    for epoch in range(1, epochs + 1):
        t_loss = train_one_epoch_multi(
            model, train_loader, criterion, optimizer, device, grad_clip=grad_clip,
        )
        v_loss = evaluate_multi(model, val_loader, criterion, device)
        train_losses.append(t_loss)
        val_losses.append(v_loss)
        scheduler.step(v_loss)

        if v_loss < best_val_loss:
            best_val_loss = v_loss
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if verbose and (epoch % 5 == 0 or epoch == 1):
            cur_lr = optimizer.param_groups[0]["lr"]
            print(
                f"Epoch {epoch:>3}/{epochs}  |  train MSE: {t_loss:.8f}"
                f"  |  val MSE: {v_loss:.8f}  |  lr: {cur_lr:.1e}"
            )

        if epochs_no_improve >= patience:
            if verbose:
                print(f"\nEarly stopping at epoch {epoch} (no improvement for {patience} epochs)")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)
        if verbose:
            print(f"Restored best model from epoch {best_epoch} (val MSE: {best_val_loss:.8f})")

    if save_path is not None:
        from pathlib import Path

        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "multi_ticker": True,
            "target_mode": target_mode,
            "model_state_dict": model.state_dict(),
            "tickers": list(tickers),
            "ticker_to_idx": ticker_to_idx,
            "feature_dim": feature_dim,
            "num_tickers": num_tickers,
            "embedding_dim": embedding_dim,
            "hidden_dim": hidden_dim,
            "num_layers": num_layers,
            "dropout": dropout,
            "window_size": window_size,
            "best_epoch": best_epoch,
            "best_val_loss": best_val_loss,
            "train_losses": train_losses,
            "val_losses": val_losses,
        }
        torch.save(checkpoint, save_path)
        if verbose:
            print(f"Model saved to {save_path}")

    return {
        "model": model,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "device": device,
        "best_epoch": best_epoch,
        "best_val_loss": float(best_val_loss),
        "ticker_to_idx": ticker_to_idx,
    }


# 5.  Quick smoke-test with synthetic data

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
