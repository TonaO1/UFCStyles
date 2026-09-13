"""
Evaluation harness: one fixed test every embedding takes.

Four checks: probe recovery, matchup AUC against strength-only and raw-stat models,
predicted cycles, and dispersion vs cycles. Writes data/eval/{name}.json.
"""

import argparse
import json
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, log_loss
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neighbors import KNeighborsClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.features.snapshots import load_config

SPLITS = ("train", "val", "test")
BLOCK_PREFIXES = {"style": "prop_", "quality": "rate_", "physical": "phys_", "dispersion": "disp_"}
PCA_VARIANTS = {
    "style": ["style"],
    "style_quality": ["style", "quality"],
    "style_physical_dispersion": ["style", "physical", "dispersion"],
    "all": list(BLOCK_PREFIXES),
}
CHANCE_TOL = 0.08   # how far a random or fingerprint embedding may drift from chance
C_GRID = [1e-4, 3e-4, 1e-3, 3e-3, 0.01, 0.1, 1]   # penalty strengths tried on val
PAIR_SCALES = [0.0, 0.1, 0.3, 1.0]   # pair-term weight tried on val; 0 means strength only


@dataclass
class FightPairs:
    idx_a: np.ndarray      # row of fighter A's pre-fight snapshot (index into X and Z)
    idx_b: np.ndarray
    a_won: np.ndarray      # 1 if A won, 0 if B won
    fight_id: np.ndarray


@dataclass
class EvalData:
    meta: pd.DataFrame     # one row per snapshot; row i matches X[i] and every Z[i]
    X: np.ndarray          # (n_rows, 36) features, scaled with the train-fit scaler
    feature_cols: list
    blocks: dict           # block name -> column indices into X
    labels: dict           # probe name -> (n_rows,) object array, None where unlabeled
    pairs: dict            # split -> FightPairs


# ============================================================================
# SECTION 1: MATH (yours)
# ============================================================================

def check_probe_recovery(Z: np.ndarray, y: np.ndarray, groups: np.ndarray,
                         folds: list, k: int) -> dict:
    """
    Can a fighter's label be guessed from nearest neighbors in the embedding?

    Z: (n, d) labeled test rows. y: (n,) labels. groups: (n,) fighter ids.
    folds: list of (train_idx, test_idx); no fighter is ever on both sides.
    Must return a dict with "balanced_accuracy" (the trust check reads it).
    """
    output = dict()
    for metric in ["cosine", "euclidean"]: # 2 ways for comparing embeddings
        guesses = np.empty(len(y), dtype=object)

        for voter_rows, guess_rows in folds:
            voter = KNeighborsClassifier(n_neighbors=k, metric=metric)
            voter.fit(Z[voter_rows], y[voter_rows]) # stores voters embeddings and voter's actual labels
            guesses[guess_rows] = voter.predict(Z[guess_rows]) # use k closest neighbors for each guess row to predict guess row label

        score = balanced_accuracy_score(y,guesses)
        if metric == "cosine":
            output["balanced_accuracy"] = score
        else:
            output["balanced_accuracy_euclidean"] = score

    output["chance"] = 1 / len(np.unique(y))   # 1 / how many different labels are in y
    return output


def fit_fight_model(X: np.ndarray, Z: np.ndarray, fit_pairs: dict, blocks: dict):
    """
    Fit P(A beats B): strength term plus a pair term that flips sign when A and B swap.

    fit_pairs: {"train": FightPairs, "val": FightPairs}. Test fights never reach this function.
    Z is centered and scaled on train-fight rows first, so the pair term carries no strength.
    Penalty C and pair weight are picked by val log loss; the strength-only model picks its own C.
    """
    quality_cols = blocks["quality"]
    n_strength = len(quality_cols)
    train, val = fit_pairs["train"], fit_pairs["val"]

    train_rows = np.unique(np.concatenate([train.idx_a, train.idx_b]))
    z_mean, z_std = Z[train_rows].mean(axis=0), Z[train_rows].std(axis=0) + 1e-8
    Zs = (Z - z_mean) / z_std

    table_train = _fight_table(X, Zs, train, quality_cols)
    table_val = _fight_table(X, Zs, val, quality_cols)
    weights = lambda scale, n: np.r_[np.ones(n_strength), np.full(n - n_strength, scale)]

    best = (np.inf, None, None, None)
    for scale in PAIR_SCALES:
        w = weights(scale, table_train.shape[1])
        model, C, loss = _fit_best_c(table_train * w, train.a_won, table_val * w, val.a_won)
        if loss < best[0]:
            best = (loss, model, C, scale)
    _, full, c_full, pair_scale = best

    strength_only, c_strength, _ = _fit_best_c(table_train[:, :n_strength], train.a_won,
                                               table_val[:, :n_strength], val.a_won)

    return {"full": full, "strength_only": strength_only, "C": c_full, "C_strength": c_strength,
            "pair_scale": pair_scale, "quality_cols": quality_cols, "n_strength": n_strength,
            "z_mean": z_mean, "z_std": z_std}


def _fit_best_c(table_train, y_train, table_val, y_val):
    """Logistic regression with no intercept, so swapping A and B flips the prediction."""
    best, best_c, best_loss = None, None, np.inf
    for C in C_GRID:
        model = LogisticRegression(C=C, fit_intercept=False, max_iter=5000).fit(table_train, y_train)
        loss = log_loss(y_val, model.predict_proba(table_val)[:, 1])
        if loss < best_loss:
            best, best_c, best_loss = model, C, loss
    return best, best_c, best_loss


def _fight_table(X: np.ndarray, Z: np.ndarray, pairs: FightPairs, quality_cols: list) -> np.ndarray:
    """One row per fight: A-minus-B quality columns, then A-vs-B pair columns."""
    strength = X[pairs.idx_a][:, quality_cols] - X[pairs.idx_b][:, quality_cols]

    za, zb = Z[pairs.idx_a], Z[pairs.idx_b]
    i, j = np.triu_indices(Z.shape[1], k=1)
    pair = za[:, i] * zb[:, j] - za[:, j] * zb[:, i]

    return np.hstack([strength, pair])


def _logits(model: dict, X: np.ndarray, Z: np.ndarray, pairs: FightPairs, which: str = "full") -> np.ndarray:
    Zs = (Z - model["z_mean"]) / model["z_std"]
    table = _fight_table(X, Zs, pairs, model["quality_cols"])
    if which == "strength_only":
        return model[which].decision_function(table[:, :model["n_strength"]])
    table[:, model["n_strength"]:] *= model["pair_scale"]
    return model[which].decision_function(table)


def pair_matrix(model: dict) -> np.ndarray:
    """The pair term as a d x d matrix W with W = -W.T: pair logit = za @ W @ zb on scaled embeddings."""
    d = len(model["z_mean"])
    i, j = np.triu_indices(d, k=1)
    pair_coef = model["full"].coef_[0][model["n_strength"]:] * model["pair_scale"]
    W = np.zeros((d, d))
    W[i, j], W[j, i] = pair_coef, -pair_coef
    return W


def _logit_matrix(model: dict, X: np.ndarray, Z: np.ndarray, rows: np.ndarray, which: str = "full") -> np.ndarray:
    """L[a, b] = log-odds that fighter a beats fighter b, for every pair of rows at once."""
    coef = model[which].coef_[0]
    s = X[rows][:, model["quality_cols"]] @ coef[:model["n_strength"]]
    L = s[:, None] - s[None, :]
    if which == "full":
        Zs = (Z[rows] - model["z_mean"]) / model["z_std"]
        L = L + Zs @ pair_matrix(model) @ Zs.T
    assert np.allclose(L, -L.T), "fight model is not antisymmetric"
    return L


def _auc(y: np.ndarray, score: np.ndarray) -> float:
    """Chance a random winner-side fight scores above a random loser-side one (ties count half)."""
    ranks = rankdata(score)
    n_pos = y.sum()
    n_neg = len(y) - n_pos
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def check_matchup_auc(model, X: np.ndarray, Z: np.ndarray, fit_pairs: dict,
                      test_pairs: FightPairs, blocks: dict, n_boot: int, seed: int) -> dict:
    """
    Does the embedding predict test fights better than the comparison models?

    Compares against the strength-only model and a model on all 36 raw A-minus-B stats.
    Ranges are 2.5-97.5 percentiles over n_boot resamples of test fights.
    """
    train, val, test = fit_pairs["train"], fit_pairs["val"], test_pairs
    raw, _, _ = _fit_best_c(X[train.idx_a] - X[train.idx_b], train.a_won,
                         X[val.idx_a] - X[val.idx_b], val.a_won)

    y = test.a_won
    scores = {
        "full": _logits(model, X, Z, test, "full"),
        "strength_only": _logits(model, X, Z, test, "strength_only"),
        "raw_stats": raw.decision_function(X[test.idx_a] - X[test.idx_b]),
    }
    out = {f"auc_{k}": _auc(y, s) for k, s in scores.items()}

    rng = np.random.default_rng(seed)
    gaps = {"gap_vs_strength": [], "gap_vs_raw": []}
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        full = _auc(y[idx], scores["full"][idx])
        gaps["gap_vs_strength"].append(full - _auc(y[idx], scores["strength_only"][idx]))
        gaps["gap_vs_raw"].append(full - _auc(y[idx], scores["raw_stats"][idx]))

    out["gap_vs_strength"] = out["auc_full"] - out["auc_strength_only"]
    out["gap_vs_raw"] = out["auc_full"] - out["auc_raw_stats"]
    for k, v in gaps.items():
        out[f"{k}_lo"], out[f"{k}_hi"] = np.percentile(v, [2.5, 97.5])
    out["n_test_fights"] = len(y)
    return out


def _cycles_through(beats: np.ndarray) -> np.ndarray:
    """Per fighter, how many predicted A > B > C > A loops he is in. beats[a, b] is 0/1."""
    return np.diag(beats @ beats @ beats)


def check_cycles(model, X: np.ndarray, Z: np.ndarray, rows: np.ndarray, margin: float = 0.05) -> dict:
    """
    How often does the fight model predict A > B > C > A?

    Counts every triple exactly. "confident" only uses predictions at least margin away from 50%.
    rows: each test fighter's most recent snapshot, so all compared fighters are contemporaries.
    """
    n = len(rows)
    n_triples = n * (n - 1) * (n - 2) / 6
    threshold = np.log((0.5 + margin) / (0.5 - margin))

    strength_beats = (_logit_matrix(model, X, Z, rows, "strength_only") > 0).astype(float)
    assert _cycles_through(strength_beats).sum() == 0, "strength-only model produced a cycle"

    L = _logit_matrix(model, X, Z, rows, "full")
    cycles = _cycles_through((L > 0).astype(float)).sum() / 3
    confident = _cycles_through((L > threshold).astype(float)).sum() / 3
    return {"n_fighters": n, "n_triples": int(n_triples),
            "n_cycles": int(cycles), "cycle_rate": cycles / n_triples,
            "n_confident_cycles": int(confident), "confident_cycle_rate": confident / n_triples,
            "confident_margin": margin}


def check_dispersion_correlation(model, X: np.ndarray, Z: np.ndarray, rows: np.ndarray,
                                 D: np.ndarray, dispersion_cols: list) -> dict:
    """
    Do high-dispersion fighters show up in predicted cycles more often?

    D: (n_rows, 4) scaled disp_ columns, named by dispersion_cols. rows as in check_cycles.
    A fighter who beats (or loses to) almost everyone cannot be in many loops, so the
    "share" version divides by wins x losses, the most loops he could be in.
    """
    beats = (_logit_matrix(model, X, Z, rows, "full") > 0).astype(float)
    in_cycles = _cycles_through(beats)
    wins = beats.sum(axis=1)
    possible = wins * (len(rows) - 1 - wins)
    share = np.divide(in_cycles, possible, out=np.zeros_like(in_cycles), where=possible > 0)

    Dr = D[rows]
    columns = {**{c: Dr[:, k] for k, c in enumerate(dispersion_cols)}, "disp_mean": Dr.mean(axis=1)}
    out = {}
    for name, values in columns.items():
        for target, y in (("cycles", in_cycles), ("cycle_share", share)):
            r, p = spearmanr(values, y) if np.ptp(y) > 0 else (0.0, 1.0)   # no cycles at all
            out[f"{name}.{target}_r"] = float(r)
            out[f"{name}.{target}_p"] = float(p)
    return out


# ============================================================================
# SECTION 2: LOADING
# ============================================================================

def load_eval_data(config: dict) -> EvalData:
    """Load snapshots, scaled features, probe labels and fight pairs, all in one row order."""
    paths = config["paths"]
    snap_dir = Path(paths["snapshots"])
    raw_dir = Path(paths["raw"]) / paths["snapshot_date"]

    snaps = pd.concat([pd.read_parquet(snap_dir / f"{s}.parquet") for s in SPLITS], ignore_index=True)
    snaps = snaps.sort_values(["date", "fight_id", "fighter_id"]).reset_index(drop=True)

    feature_cols = json.loads((snap_dir / "feature_cols.json").read_text())
    with open(snap_dir / "scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    X = scaler.transform(snaps[feature_cols]).astype(np.float64)

    blocks = {name: [i for i, c in enumerate(feature_cols) if c.startswith(prefix)]
              for name, prefix in BLOCK_PREFIXES.items()}
    assert sum(map(len, blocks.values())) == len(feature_cols), "a feature column has no block prefix"

    meta = snaps[["fighter_id", "fight_id", "opponent_id", "date", "split", "weight_class"]].copy()

    fighters = pd.read_csv(raw_dir / "fighters.csv")
    background = pd.read_csv(Path(paths["labels"]) / "background.csv")
    assert fighters["fighter_id"].is_unique and background["fighter_id"].is_unique
    labels = _probe_labels(meta, fighters, background)

    fights = pd.read_csv(raw_dir / "fights.csv")
    assert fights["fight_id"].is_unique
    seed = config["training"]["seed"]
    print("Fight pairs:")
    pairs = {s: build_fight_pairs(meta, fights, s, seed + i) for i, s in enumerate(SPLITS)}

    return EvalData(meta, X, feature_cols, blocks, labels, pairs)


def _probe_labels(meta: pd.DataFrame, fighters: pd.DataFrame, background: pd.DataFrame) -> dict:
    """Per-row probe labels; None marks rows a probe skips. "unclear" is never a class."""
    bg = meta["fighter_id"].map(background.set_index("fighter_id")["background"])
    bg = bg.where(bg.notna() & (bg != "unclear"))
    probes = {
        "background": bg,
        "background_no_hybrid": bg.where(bg != "hybrid"),
        "stance": meta["fighter_id"].map(fighters.set_index("fighter_id")["stance"]),
        "weight_class": meta["weight_class"],
    }
    return {name: s.astype(object).where(s.notna(), None).to_numpy() for name, s in probes.items()}


def build_fight_pairs(meta: pd.DataFrame, fights: pd.DataFrame, split: str, seed: int) -> FightPairs:
    """
    Fights in one split where both fighters have a snapshot and someone won.

    Sides are swapped at random: the source lists the winner as fighter A about 54% of the time.
    """
    rows = meta.loc[meta["split"] == split, ["fight_id", "fighter_id"]].rename_axis("row").reset_index()
    f = fights.loc[fights["fight_id"].isin(rows["fight_id"]),
                   ["fight_id", "fighter_a_id", "fighter_b_id", "winner_id"]]
    n_fights = len(f)

    decided = f["winner_id"].eq(f["fighter_a_id"]) | f["winner_id"].eq(f["fighter_b_id"])
    f = f[decided]
    for side in ("a", "b"):
        f = f.merge(rows.rename(columns={"fighter_id": f"fighter_{side}_id", "row": f"idx_{side}"}),
                    on=["fight_id", f"fighter_{side}_id"], how="inner")
    f = f.sort_values("fight_id").reset_index(drop=True)

    flip = np.random.default_rng(seed).random(len(f)) < 0.5
    a, b = f["idx_a"].to_numpy(), f["idx_b"].to_numpy()
    a_won = f["winner_id"].eq(f["fighter_a_id"]).to_numpy()

    print(f"  {split}: {len(f)} pairs ({n_fights - int(decided.sum())} draws/NC dropped, "
          f"{int(decided.sum()) - len(f)} missing a snapshot)")
    return FightPairs(idx_a=np.where(flip, b, a), idx_b=np.where(flip, a, b),
                      a_won=(a_won ^ flip).astype(int), fight_id=f["fight_id"].to_numpy())


def fighter_folds(y: np.ndarray, groups: np.ndarray, n_splits: int = 5, seed: int = 42) -> list:
    """Stratified folds in which no fighter appears on both sides."""
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(cv.split(np.zeros(len(y)), y, groups))


def probe_inputs(Z: np.ndarray, data: EvalData, probe: str, split: str = "test",
                 seed: int = 42) -> dict:
    """Labeled rows of one split for one probe, with fighter-disjoint folds."""
    labeled = np.array([v is not None for v in data.labels[probe]])
    rows = np.flatnonzero((data.meta["split"].to_numpy() == split) & labeled)
    y = data.labels[probe][rows].astype(str)
    groups = data.meta["fighter_id"].to_numpy()[rows]

    folds = fighter_folds(y, groups, seed=seed)
    assert_folds_fighter_disjoint(folds, groups)
    return {"Z": Z[rows], "y": y, "groups": groups, "folds": folds, "rows": rows}


def latest_rows(meta: pd.DataFrame, split: str = "test") -> np.ndarray:
    """Each fighter's most recent snapshot in the split."""
    m = meta[meta["split"] == split]
    return np.sort(m.sort_values("date").groupby("fighter_id").tail(1).index.to_numpy())


# ============================================================================
# SECTION 3: BASELINES
# ============================================================================

def baseline_embeddings(data: EvalData, which: str = "all", d: int = 8, seed: int = 42) -> dict:
    """Random, raw-feature and PCA embeddings. PCA is fit on train rows only."""
    out = {}
    if which in ("random", "all"):
        out[f"random_{d}"] = np.random.default_rng(seed).standard_normal((len(data.meta), d))
    if which in ("raw", "all"):
        out[f"raw_{data.X.shape[1]}"] = data.X
    if which in ("pca", "all"):
        train = data.meta["split"].to_numpy() == "train"
        for name, block_names in PCA_VARIANTS.items():
            cols = sorted(i for b in block_names for i in data.blocks[b])
            pca = PCA(n_components=d, random_state=seed).fit(data.X[train][:, cols])
            out[f"pca_{name}_{d}"] = pca.transform(data.X[:, cols])
    return out


def fingerprint_embedding(meta: pd.DataFrame, d: int = 8, seed: int = 0) -> np.ndarray:
    """The same random vector for every row of a fighter: knows who, nothing about style."""
    codes, _ = pd.factorize(meta["fighter_id"])
    return np.random.default_rng(seed).standard_normal((codes.max() + 1, d))[codes]


# ============================================================================
# SECTION 4: TRUST CHECKS
# ============================================================================

def assert_rows_aligned(data: EvalData) -> bool:
    """Row i means the same snapshot in meta, X and every label array."""
    n = len(data.meta)
    assert data.meta.index.equals(pd.RangeIndex(n)), "meta index is not 0..n-1"
    assert len(data.X) == n and np.isfinite(data.X).all(), "X misaligned or non-finite"
    assert not data.meta.duplicated(["fight_id", "fighter_id"]).any(), "duplicate snapshot rows"
    for name, y in data.labels.items():
        assert len(y) == n, f"label array {name} has {len(y)} rows, meta has {n}"

    labeled = {name: int(sum(v is not None for v in y)) for name, y in data.labels.items()}
    print(f"✓ Rows aligned: {n} snapshots, {data.X.shape[1]} features, labeled rows {labeled}")
    return True


def assert_fight_pairs(data: EvalData) -> bool:
    """Both rows of a pair are the same fight, opposite fighters, same split; sides balanced; splits disjoint."""
    fight = data.meta["fight_id"].to_numpy()
    fighter = data.meta["fighter_id"].to_numpy()
    opponent = data.meta["opponent_id"].to_numpy()
    split_of = data.meta["split"].to_numpy()
    dates = data.meta["date"].to_numpy()

    for split, p in data.pairs.items():
        assert (fight[p.idx_a] == p.fight_id).all() and (fight[p.idx_b] == p.fight_id).all(), \
            f"{split}: pair rows point at the wrong fight"
        assert (fighter[p.idx_a] != fighter[p.idx_b]).all(), f"{split}: fighter paired with himself"
        assert (opponent[p.idx_a] == fighter[p.idx_b]).all(), f"{split}: A's opponent is not B"
        assert (split_of[p.idx_a] == split).all() and (split_of[p.idx_b] == split).all(), \
            f"{split}: pair row from another split"
        assert len(np.unique(p.fight_id)) == len(p.fight_id), f"{split}: fight used twice"
        assert set(np.unique(p.a_won)) <= {0, 1}

        limit = 3 * 0.5 / np.sqrt(len(p.a_won))
        assert abs(p.a_won.mean() - 0.5) < limit, \
            f"{split}: A wins {p.a_won.mean():.3f} after swapping; side swap is not working"

    ids = {s: set(p.fight_id) for s, p in data.pairs.items()}
    assert not (ids["train"] & ids["test"]) and not (ids["val"] & ids["test"]) and not (ids["train"] & ids["val"])
    for earlier, later in (("train", "val"), ("val", "test")):
        assert dates[data.pairs[earlier].idx_a].max() < dates[data.pairs[later].idx_a].min(), \
            f"{earlier} fights are not all before {later} fights"

    counts = {s: len(p.fight_id) for s, p in data.pairs.items()}
    a_rate = {s: round(float(p.a_won.mean()), 3) for s, p in data.pairs.items()}
    print(f"✓ Fight pairs valid: {counts}, A-win rate after swap {a_rate}, splits disjoint and in date order")
    return True


def assert_folds_fighter_disjoint(folds: list, groups: np.ndarray) -> bool:
    """No fighter on both sides of any fold; otherwise a fighter's own rows give his label away."""
    for i, (tr, te) in enumerate(folds):
        shared = set(groups[tr]) & set(groups[te])
        assert not shared, f"fold {i}: {len(shared)} fighters on both sides, e.g. {sorted(shared)[:2]}"
    return True


def assert_harness_not_fooled(data: EvalData, config: dict) -> bool:
    """A random embedding and a who-is-this fingerprint must both score near chance on background."""
    k = config["eval"]["probe_k"]
    candidates = {
        "random": [np.random.default_rng(s).standard_normal((len(data.meta), 8)) for s in range(5)],
        "fingerprint": [fingerprint_embedding(data.meta, seed=s) for s in range(5)],
    }
    for name, embeddings in candidates.items():
        scores = []
        for Z in embeddings:
            inp = probe_inputs(Z, data, "background")
            try:
                result = check_probe_recovery(inp["Z"], inp["y"], inp["groups"], inp["folds"], k)
            except NotImplementedError:
                print("  (skipped harness-not-fooled check: check_probe_recovery is not written yet)")
                return False
            scores.append(result["balanced_accuracy"])

        chance = 1 / len(np.unique(inp["y"]))
        mean = float(np.mean(scores))
        assert abs(mean - chance) < CHANCE_TOL, (
            f"HARNESS FOOLED: {name} embedding scores {mean:.3f} on background over 5 seeds; "
            f"chance is {chance:.3f}"
        )
        print(f"✓ {name} embedding scores {mean:.3f} on background (chance {chance:.3f})")

    Z = candidates["random"][0]
    fit_pairs = {s: data.pairs[s] for s in ("train", "val")}
    model = fit_fight_model(data.X, Z, fit_pairs, data.blocks)
    m = check_matchup_auc(model, data.X, Z, fit_pairs, data.pairs["test"], data.blocks, 200, 0)
    assert m["gap_vs_strength_lo"] <= 0, (
        f"HARNESS FOOLED: random embedding beats strength-only by {m['gap_vs_strength']:.3f} AUC "
        f"(range {m['gap_vs_strength_lo']:.3f} to {m['gap_vs_strength_hi']:.3f})"
    )
    print(f"✓ random embedding adds {m['gap_vs_strength']:+.3f} AUC over strength-only "
          f"(range {m['gap_vs_strength_lo']:+.3f} to {m['gap_vs_strength_hi']:+.3f})")
    return True


def assert_fight_model_consistent(model: dict, data: EvalData, Z: np.ndarray) -> bool:
    """The all-pairs matrix used for cycles must agree with the per-fight predictions used for AUC."""
    p = data.pairs["val"]
    k = min(200, len(p.fight_id))
    sub = FightPairs(p.idx_a[:k], p.idx_b[:k], p.a_won[:k], p.fight_id[:k])
    rows = np.concatenate([sub.idx_a, sub.idx_b])
    for which in ("full", "strength_only"):
        L = _logit_matrix(model, data.X, Z, rows, which)
        assert np.allclose(L[np.arange(k), k + np.arange(k)], _logits(model, data.X, Z, sub, which)), \
            f"{which}: logit matrix disagrees with per-fight logits"
    return True


# ============================================================================
# SECTION 5: EVALUATE + MAIN
# ============================================================================

def _run(fn, *args):
    try:
        return fn(*args)
    except NotImplementedError:
        return {"pending": fn.__name__}


def _is_pending(result) -> bool:
    return isinstance(result, dict) and "pending" in result


def _json_default(o):
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"not JSON-serializable: {type(o).__name__}")


def evaluate(Z: np.ndarray, data: EvalData, name: str, config: dict, eval_dir: str) -> dict:
    """Score one embedding. Z[i] must describe data.meta row i. Writes {eval_dir}/{name}.json."""
    assert Z.shape[0] == len(data.meta), f"Z has {Z.shape[0]} rows, meta has {len(data.meta)}"
    assert np.isfinite(Z).all(), f"{name}: non-finite values in embedding"
    seed = config["training"]["seed"]
    ev = config["eval"]

    print(f"\n{'=' * 70}\nEvaluating: {name}  {Z.shape}\n{'=' * 70}")
    results = {"model": name, "embedding_dim": int(Z.shape[1]), "n_rows": len(data.meta),
               "snapshot_date": config["paths"]["snapshot_date"], "probes": {}}

    for probe in data.labels:
        inp = probe_inputs(Z, data, probe, seed=seed)
        out = _run(check_probe_recovery, inp["Z"], inp["y"], inp["groups"], inp["folds"], ev["probe_k"])
        results["probes"][probe] = {**out, "n_rows": len(inp["rows"]),
                                    "n_fighters": int(len(np.unique(inp["groups"])))}
        print(f"  probe {probe}: {out}")

    fit_pairs = {s: data.pairs[s] for s in ("train", "val")}
    model = _run(fit_fight_model, data.X, Z, fit_pairs, data.blocks)
    if _is_pending(model):
        for check in ("matchup", "cycles", "dispersion"):
            results[check] = {"pending": "fit_fight_model"}
    else:
        assert_fight_model_consistent(model, data, Z)
        results["fight_model"] = {k: model[k] for k in ("C", "C_strength", "pair_scale")}
        rows = latest_rows(data.meta)
        disp = data.blocks["dispersion"]
        results["matchup"] = _run(check_matchup_auc, model, data.X, Z, fit_pairs, data.pairs["test"],
                                  data.blocks, ev["matchup_bootstrap_resamples"], seed)
        results["cycles"] = _run(check_cycles, model, data.X, Z, rows)
        results["dispersion"] = _run(check_dispersion_correlation, model, data.X, Z, rows,
                                     data.X[:, disp], [data.feature_cols[i] for i in disp])
    for check in ("matchup", "cycles", "dispersion"):
        print(f"  {check}: {results[check]}")

    out_dir = Path(eval_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"{name}.json", "w") as f:
        json.dump(results, f, indent=2, default=_json_default)
    print(f"  saved {out_dir / (name + '.json')}")
    return results


def print_comparison(eval_dir: str) -> None:
    """One column per saved model, one row per numeric metric."""
    files = sorted(Path(eval_dir).glob("*.json"))
    if not files:
        return
    table = pd.json_normalize([json.loads(p.read_text()) for p in files], sep=".").set_index("model")
    table = table.select_dtypes("number")
    print(f"\n{'=' * 70}\nComparison ({eval_dir})\n{'=' * 70}")
    print(table.T.round(3).to_string())


def main(args):
    """config -> load -> [rows, pairs] -> [not fooled] -> baselines -> evaluate -> compare"""
    config = load_config()
    eval_dir = args.eval_dir or config["paths"]["eval"]
    print("[Day 5b] Evaluation harness")

    data = load_eval_data(config)
    assert_rows_aligned(data)
    assert_fight_pairs(data)
    assert_harness_not_fooled(data, config)

    for name, Z in baseline_embeddings(data, args.baseline, seed=config["training"]["seed"]).items():
        evaluate(Z, data, name, config, eval_dir)
    print_comparison(eval_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", choices=["random", "raw", "pca", "all"], default="all")
    parser.add_argument("--eval-dir", default=None, help="defaults to paths.eval in configs/v1.yaml")
    main(parser.parse_args())
