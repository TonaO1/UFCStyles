"""
Contrastive style embedding (day 8).

Positive pair: the same fighter's snapshots `gap` fights apart. Every other pair in the
batch is a negative. The loss reads the projector output; the embedding is the encoder output.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.eval.harness import assert_rows_aligned, evaluate, load_eval_data
from src.features.snapshots import load_config
from src.models.common import (VARIANTS, encode, encoder_mlp, export_encoder, latent_diagnostics,
                               loader, set_seed, split_rows, train_loop, variant_cols)


class StyleContrastive(nn.Module):
    """Same encoder shape as the autoencoder, plus a small projector used only by the loss."""

    def __init__(self, d_in: int, d_latent: int = 8, d_hidden=(32, 16), d_proj: int = 16, dropout: float = 0.15):
        super().__init__()
        self.encoder = encoder_mlp(d_in, list(d_hidden), d_latent, dropout)
        self.projector = nn.Sequential(nn.Linear(d_latent, d_proj), nn.GELU(), nn.Linear(d_proj, d_proj))

    def forward(self, x):
        z = self.encoder(x)
        return z, self.projector(z)


def nt_xent_loss(p1: torch.Tensor, p2: torch.Tensor, tau: float) -> torch.Tensor:
    """Each row of p1 must pick its own partner out of all of p2 by cosine similarity, and back."""
    p1, p2 = F.normalize(p1, dim=1), F.normalize(p2, dim=1)
    logits = p1 @ p2.T / tau
    labels = torch.arange(len(p1), device=p1.device)
    return (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2


def gapped_pairs(meta: pd.DataFrame, gap: int) -> dict:
    """
    (earlier_row, later_row) for the same fighter, `gap` snapshots apart. meta must be date-sorted.
    train: both rows in train. val: the later row in val, so no pair reaches into test.
    """
    m = meta[["fighter_id", "split"]].assign(k=meta.groupby("fighter_id").cumcount(), row=np.arange(len(meta)))
    p = m.merge(m.assign(k=m["k"] - gap), on=["fighter_id", "k"], suffixes=("_i", "_j"))
    dates = meta["date"].to_numpy()
    assert (dates[p["row_i"]] < dates[p["row_j"]]).all(), "pair is not earlier -> later"

    train = p[(p["split_i"] == "train") & (p["split_j"] == "train")]
    val = p[p["split_j"] == "val"]
    return {"train": train[["row_i", "row_j"]].to_numpy(), "val": val[["row_i", "row_j"]].to_numpy()}


def train_variant(data, variant: str, config: dict, eval_dir: str) -> dict:
    cfg = config["training"]["contrastive"]
    assert cfg["pair_strategy"] == "gapped", "only the gapped pair strategy is implemented"
    seed = config["training"]["seed"]
    set_seed(seed)

    cols = variant_cols(data, variant)
    X = data.X[:, cols]
    pairs = gapped_pairs(data.meta, cfg["gap"])
    name = f"contrastive_{variant}_{cfg['d_latent']}"
    print(f"\n{'=' * 70}\n{name}: {len(cols)} features, gap {cfg['gap']}, "
          f"{len(pairs['train'])} train / {len(pairs['val'])} val pairs\n{'=' * 70}")

    tau = cfg["nt_xent_tau"]
    loss_fn = lambda model, batch: nt_xent_loss(model(batch[0])[1], model(batch[1])[1], tau)
    tr, va = pairs["train"], pairs["val"]
    model = StyleContrastive(len(cols), cfg["d_latent"], cfg["d_hidden"], cfg["d_proj"], cfg["dropout"])
    history = train_loop(model, loss_fn,
                         loader(X[tr[:, 0]], X[tr[:, 1]], batch_size=cfg["batch_size"], shuffle=True,
                                seed=seed, drop_last=True),
                         loader(X[va[:, 0]], X[va[:, 1]], batch_size=cfg["batch_size"], shuffle=False),
                         cfg, config["training"]["device"])

    Z = encode(model.encoder, X)
    diagnostics = latent_diagnostics(Z[split_rows(data, "test")])
    results = evaluate(Z, data, name, config, eval_dir)

    model_dir = Path(config["paths"]["models"]) / name
    model_dir.mkdir(parents=True, exist_ok=True)
    export_encoder(model.encoder, data, cols, config, model_dir / "encoder.npz")
    torch.save(model.state_dict(), model_dir / "model.pt")
    run = {"variant": variant, "config": cfg, "history": history, "diagnostics": diagnostics}
    (model_dir / "run.json").write_text(json.dumps(run, indent=2))
    return results


def main(args):
    """config -> load -> [rows aligned] -> per variant: pairs -> train -> evaluate -> export"""
    config = load_config()
    data = load_eval_data(config)
    assert_rows_aligned(data)
    for variant in args.variants:
        train_variant(data, variant, config, config["paths"]["eval"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--variants", nargs="+", choices=list(VARIANTS), default=list(VARIANTS))
    main(parser.parse_args())
