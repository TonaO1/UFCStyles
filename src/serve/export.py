"""
Build the serving bundle for one trained model (days 9-10, the non-AWS half).

Checks NumPy inference against PyTorch, refits the harness fight model on the NumPy
embedding, and writes one record per fighter (latest snapshot) to
data/models/{name}/serving/. Loading those records into DynamoDB is not done here.
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.eval.harness import (FightPairs, _logits, fit_fight_model, load_eval_data, pair_matrix)
from src.features.snapshots import load_config
from src.models.common import encoder_mlp
from src.serve.inference import encode, load_npz, most_similar, p_a_wins


def torch_embedding(model_dir: Path, enc: dict, X_scaled: np.ndarray) -> np.ndarray:
    """Rebuild the encoder from model.pt and run it, to compare against the NumPy version."""
    state = torch.load(model_dir / "model.pt")
    prefix = "enc." if any(k.startswith("enc.") for k in state) else "encoder."
    n = int(enc["n_layers"])
    hidden = [enc[f"W{k}"].shape[0] for k in range(n - 1)]
    encoder = encoder_mlp(enc["W0"].shape[1], hidden, enc[f"W{n - 1}"].shape[0], dropout=0.0)
    encoder.load_state_dict({k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)})
    encoder.eval()
    with torch.no_grad():
        return encoder(torch.from_numpy(X_scaled.astype(np.float32))).numpy().astype(np.float64)


def main(args):
    config = load_config()
    model_dir = Path(config["paths"]["models"]) / args.model
    out_dir = model_dir / "serving"
    out_dir.mkdir(parents=True, exist_ok=True)

    data = load_eval_data(config)
    enc = load_npz(model_dir / "encoder.npz")
    cols = [data.feature_cols.index(c) for c in enc["feature_cols"]]
    X_raw = data.X[:, cols] * enc["scaler_scale"] + enc["scaler_mean"]

    Z = encode(enc, X_raw)
    gap = np.abs(Z - torch_embedding(model_dir, enc, data.X[:, cols])).max()
    assert gap < 1e-4, f"NumPy encoder disagrees with PyTorch by {gap:.2e}"
    print(f"✓ NumPy encoder matches PyTorch on {len(Z)} rows (max difference {gap:.1e})")

    model = fit_fight_model(data.X, Z, {s: data.pairs[s] for s in ("train", "val")}, data.blocks)
    fight_model = {"W": pair_matrix(model), "z_mean": model["z_mean"], "z_std": model["z_std"]}
    strength = data.X[:, model["quality_cols"]] @ model["full"].coef_[0][:model["n_strength"]]

    p = data.pairs["test"]
    k = min(100, len(p.fight_id))
    sub = FightPairs(p.idx_a[:k], p.idx_b[:k], p.a_won[:k], p.fight_id[:k])
    expected = 1 / (1 + np.exp(-_logits(model, data.X, Z, sub)))
    served = [p_a_wins(fight_model, strength[a], strength[b], Z[a], Z[b]) for a, b in zip(sub.idx_a, sub.idx_b)]
    assert np.allclose(served, expected), "served win probability disagrees with the harness"
    print(f"✓ served win probabilities match the harness on {k} test fights "
          f"(pair weight {model['pair_scale']}, C {model['C']})")

    raw_dir = Path(config["paths"]["raw"]) / config["paths"]["snapshot_date"]
    names = pd.read_csv(raw_dir / "fighters.csv").set_index("fighter_id")["fighter_name"]
    background = pd.read_csv(Path(config["paths"]["labels"]) / "background.csv").set_index("fighter_id")["background"]
    latest = data.meta.groupby("fighter_id").tail(1)
    records = [{
        "fighter_id": r.fighter_id,
        "fighter_name": names[r.fighter_id],
        "snapshot_before_fight": r.fight_id,
        "snapshot_date": str(pd.Timestamp(r.date).date()),
        "weight_class": r.weight_class if isinstance(r.weight_class, str) else None,
        "background": background.get(r.fighter_id),
        "embedding": Z[i].round(6).tolist(),
        "strength": round(float(strength[i]), 6),
    } for i, r in zip(latest.index, latest.itertuples())]

    (out_dir / "fighters.json").write_text(json.dumps(records, indent=1))
    np.savez(out_dir / "fight_model.npz", **fight_model)
    shutil.copy(model_dir / "encoder.npz", out_dir / "encoder.npz")
    print(f"✓ wrote {len(records)} fighter records, fight_model.npz and encoder.npz to {out_dir}")

    by_name = {rec["fighter_name"]: n for n, rec in enumerate(records)}
    Z_latest = np.array([rec["embedding"] for rec in records])
    for who in args.demo:
        if who not in by_name:
            continue
        top, sims = most_similar(Z_latest, by_name[who], k=5)
        print(f"  most similar to {who}: " + ", ".join(f"{records[t]['fighter_name']} ({s:.2f})" for t, s in zip(top, sims)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="directory name under data/models, e.g. ae_all_8")
    parser.add_argument("--demo", nargs="*", default=["Islam Makhachev", "Alex Pereira", "Merab Dvalishvili"])
    main(parser.parse_args())
