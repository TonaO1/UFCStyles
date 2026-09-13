"""
Autoencoder style embedding (days 6-7).

Trains one model per feature-block combination, prints diagnostics, runs each
embedding through the harness and exports the encoder for NumPy serving.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.eval.harness import assert_rows_aligned, evaluate, load_eval_data
from src.features.snapshots import load_config
from src.models.common import (VARIANTS, encode, encoder_mlp, export_encoder, latent_diagnostics,
                               loader, set_seed, split_rows, train_loop, variant_cols)


class StyleAE(nn.Module):
    """Encoder d_in -> 32 -> 16 -> d_latent; the decoder mirrors it."""

    def __init__(self, d_in: int, d_latent: int = 8, d_hidden=(32, 16), dropout: float = 0.15):
        super().__init__()
        self.enc = encoder_mlp(d_in, list(d_hidden), d_latent, dropout)
        self.dec = encoder_mlp(d_latent, list(reversed(d_hidden)), d_in, dropout)

    def forward(self, x):
        z = self.enc(x)
        return self.dec(z), z


def reconstruction_loss(model: StyleAE, batch: list) -> torch.Tensor:
    (x,) = batch
    x_hat, _ = model(x)
    return F.mse_loss(x_hat, x)


def run_diagnostics(model: StyleAE, X: np.ndarray, feature_names: list) -> dict:
    """Latent health plus the five features the decoder rebuilds worst. X: scaled test rows."""
    model.cpu().eval()
    with torch.no_grad():
        x_hat, z = model(torch.from_numpy(X.astype(np.float32)))
    out = latent_diagnostics(z.numpy())

    err = ((X - x_hat.numpy()) ** 2).mean(axis=0)
    out["recon_mse"] = float(err.mean())
    out["worst_features"] = {feature_names[i]: round(float(err[i]), 4) for i in np.argsort(-err)[:5]}
    print(f"  test recon MSE {out['recon_mse']:.3f}; hardest: {out['worst_features']}")
    return out


def train_variant(data, variant: str, config: dict, eval_dir: str) -> dict:
    cfg = config["training"]["autoencoder"]
    seed = config["training"]["seed"]
    set_seed(seed)

    cols = variant_cols(data, variant)
    X = data.X[:, cols]
    train, val, test = (split_rows(data, s) for s in ("train", "val", "test"))
    name = f"ae_{variant}_{cfg['d_latent']}"
    print(f"\n{'=' * 70}\n{name}: {len(cols)} features, {len(train)} train / {len(val)} val rows\n{'=' * 70}")

    model = StyleAE(len(cols), cfg["d_latent"], cfg["d_hidden"], cfg["dropout"])
    history = train_loop(model, reconstruction_loss,
                         loader(X[train], batch_size=cfg["batch_size"], shuffle=True, seed=seed),
                         loader(X[val], batch_size=cfg["batch_size"], shuffle=False),
                         cfg, config["training"]["device"])
    diagnostics = run_diagnostics(model, X[test], [data.feature_cols[i] for i in cols])

    results = evaluate(encode(model.enc, X), data, name, config, eval_dir)

    model_dir = Path(config["paths"]["models"]) / name
    model_dir.mkdir(parents=True, exist_ok=True)
    export_encoder(model.enc, data, cols, config, model_dir / "encoder.npz")
    torch.save(model.state_dict(), model_dir / "model.pt")
    run = {"variant": variant, "config": cfg, "history": history, "diagnostics": diagnostics}
    (model_dir / "run.json").write_text(json.dumps(run, indent=2))
    return results


def main(args):
    """config -> load -> [rows aligned] -> per variant: train -> diagnose -> evaluate -> export"""
    config = load_config()
    data = load_eval_data(config)
    assert_rows_aligned(data)
    for variant in args.variants:
        train_variant(data, variant, config, config["paths"]["eval"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--variants", nargs="+", choices=list(VARIANTS), default=list(VARIANTS))
    main(parser.parse_args())
