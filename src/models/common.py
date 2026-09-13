"""
Shared pieces for the autoencoder and contrastive models: feature-block columns,
the training loop, latent diagnostics and NumPy export.
"""

import pickle
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.eval.harness import PCA_VARIANTS, EvalData

VARIANTS = PCA_VARIANTS


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def variant_cols(data: EvalData, variant: str) -> list:
    """Column indices into data.X for one feature-block combination, in feature_cols order."""
    return sorted(i for b in VARIANTS[variant] for i in data.blocks[b])


def split_rows(data: EvalData, split: str) -> np.ndarray:
    return np.flatnonzero(data.meta["split"].to_numpy() == split)


def loader(*arrays, batch_size: int, shuffle: bool, seed: int = 0, drop_last: bool = False) -> DataLoader:
    tensors = [torch.from_numpy(np.asarray(a, dtype=np.float32)) for a in arrays]
    return DataLoader(TensorDataset(*tensors), batch_size=batch_size, shuffle=shuffle,
                      drop_last=drop_last, generator=torch.Generator().manual_seed(seed))


def encoder_mlp(d_in: int, d_hidden: list, d_out: int, dropout: float) -> nn.Sequential:
    """Linear -> GELU -> Dropout for each hidden size, then a plain Linear to d_out."""
    layers, d = [], d_in
    for h in d_hidden:
        layers += [nn.Linear(d, h), nn.GELU(), nn.Dropout(dropout)]
        d = h
    layers.append(nn.Linear(d, d_out))
    return nn.Sequential(*layers)


def train_loop(model: nn.Module, loss_fn, train_loader: DataLoader, val_loader: DataLoader,
               cfg: dict, device: str) -> dict:
    """
    Adam with early stopping on val loss; restores the best epoch's weights before returning.
    loss_fn(model, batch) -> scalar tensor, where batch is the list of tensors a loader yields.
    """
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    history = {"train_loss": [], "val_loss": []}
    best_loss, best_epoch, best_state = float("inf"), 0, None

    for epoch in range(cfg["epochs_max"]):
        model.train()
        total = 0.0
        for batch in train_loader:
            loss = loss_fn(model, [t.to(device) for t in batch])
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()
        train_loss = total / len(train_loader)

        model.eval()
        with torch.no_grad():
            val_loss = sum(loss_fn(model, [t.to(device) for t in b]).item() for b in val_loader) / len(val_loader)
        assert np.isfinite(train_loss) and np.isfinite(val_loss), f"loss went non-finite at epoch {epoch + 1}"
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if epoch == 0 or (epoch + 1) % 20 == 0:
            print(f"  epoch {epoch + 1:3d}  train {train_loss:.4f}  val {val_loss:.4f}")
        if val_loss < best_loss:
            best_loss, best_epoch = val_loss, epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        elif epoch - best_epoch >= cfg["early_stopping_patience"]:
            break

    model.load_state_dict(best_state)
    history["best_epoch"] = best_epoch + 1
    print(f"  stopped after {epoch + 1} epochs; best val {best_loss:.4f} at epoch {best_epoch + 1}")
    return history


def encode(encoder: nn.Module, X: np.ndarray) -> np.ndarray:
    encoder.cpu().eval()
    with torch.no_grad():
        return encoder(torch.from_numpy(X.astype(np.float32))).numpy().astype(np.float64)


def latent_diagnostics(Z: np.ndarray) -> dict:
    """Dead dims (std < 0.01) and how many dims hold 95% of the variance."""
    Z = Z.astype(np.float64)
    std = Z.std(axis=0)
    S = np.linalg.svd(Z - Z.mean(axis=0), compute_uv=False)
    energy = np.cumsum(S ** 2) / np.sum(S ** 2)
    out = {"latent_std": std.round(4).tolist(), "dead_dims": int((std < 0.01).sum()),
           "effective_rank_95": int(np.searchsorted(energy, 0.95) + 1)}
    print(f"  latent std {out['latent_std']}")
    print(f"  dead dims {out['dead_dims']}; 95% of variance in {out['effective_rank_95']} of {Z.shape[1]} dims")
    return out


def export_encoder(encoder: nn.Sequential, data: EvalData, cols: list, config: dict, path: Path) -> None:
    """
    Save the encoder as plain arrays (W0, b0, W1, b1, ...) plus the scaler for exactly these
    columns, so NumPy inference takes raw feature values and returns the embedding.
    """
    with open(Path(config["paths"]["snapshots"]) / "scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    arrays = {}
    linears = [m for m in encoder if isinstance(m, nn.Linear)]
    for k, layer in enumerate(linears):
        arrays[f"W{k}"] = layer.weight.detach().cpu().numpy().astype(np.float64)
        arrays[f"b{k}"] = layer.bias.detach().cpu().numpy().astype(np.float64)
    np.savez(path, n_layers=len(linears), scaler_mean=scaler.mean_[cols], scaler_scale=scaler.scale_[cols],
             feature_cols=np.array([data.feature_cols[i] for i in cols]), **arrays)
