"""
NumPy-only inference: no PyTorch or scikit-learn, so it fits in a Lambda.

encoder.npz: W0, b0, W1, b1, ... plus the scaler for its columns (src/models/common.py).
fight_model.npz: pair matrix W and the embedding centering the harness used (src/serve/export.py).
"""

import math

import numpy as np

_erf = np.vectorize(math.erf)


def load_npz(path) -> dict:
    with np.load(path) as f:
        return {k: f[k] for k in f.files}


def gelu(x: np.ndarray) -> np.ndarray:
    """Exact GELU, matching torch.nn.GELU's default."""
    return 0.5 * x * (1.0 + _erf(x / math.sqrt(2.0)))


def encode(enc: dict, x_raw: np.ndarray) -> np.ndarray:
    """x_raw: (n, d_in) unscaled feature values in enc["feature_cols"] order -> (n, d_latent)."""
    h = (np.atleast_2d(x_raw) - enc["scaler_mean"]) / enc["scaler_scale"]
    n_layers = int(enc["n_layers"])
    for k in range(n_layers):
        h = h @ enc[f"W{k}"].T + enc[f"b{k}"]
        if k < n_layers - 1:
            h = gelu(h)
    return h


def most_similar(Z: np.ndarray, i: int, k: int = 10) -> tuple:
    """Rows of the k embeddings closest to row i by cosine similarity, and those similarities."""
    U = Z / np.linalg.norm(Z, axis=1, keepdims=True)
    sims = U @ U[i]
    sims[i] = -np.inf
    top = np.argsort(-sims)[:k]
    return top, sims[top]


def p_a_wins(fight_model: dict, strength_a: float, strength_b: float,
             z_a: np.ndarray, z_b: np.ndarray) -> float:
    """Strength gap plus the style pair term, through a sigmoid. Swapping A and B gives 1 - p."""
    za = (z_a - fight_model["z_mean"]) / fight_model["z_std"]
    zb = (z_b - fight_model["z_mean"]) / fight_model["z_std"]
    logit = strength_a - strength_b + za @ fight_model["W"] @ zb
    return float(1.0 / (1.0 + np.exp(-logit)))
