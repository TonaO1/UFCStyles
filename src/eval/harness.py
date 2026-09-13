"""
Evaluation harness: one fixed test every embedding takes.

Plumbing (loading, row alignment, fight pairs, fighter-disjoint folds, baselines,
trust checks) is done. The MATH section is left to write; until then evaluate()
records those checks as pending and keeps going.
"""

import argparse
import json
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import balanced_accuracy_score
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
    Index rows with X[p.idx_a], Z[p.idx_b]. blocks["quality"] gives the rate_ columns of X.
    Return any object; it is handed to the three checks below.
    """
    raise NotImplementedError


def check_matchup_auc(model, X: np.ndarray, Z: np.ndarray, fit_pairs: dict,
                      test_pairs: FightPairs, blocks: dict, n_boot: int, seed: int) -> dict:
    """
    Does the embedding predict test fights better than the comparison model(s)?

    Report the AUC gap with a range from resampling test fights n_boot times.
    """
    raise NotImplementedError


def check_cycles(model, X: np.ndarray, Z: np.ndarray, rows: np.ndarray,
                 sample_rate: float, seed: int) -> dict:
    """
    How often does the fight model predict A > B > C > A?

    rows: each test fighter's most recent snapshot, so all compared fighters are contemporaries.
    """
    raise NotImplementedError


def check_dispersion_correlation(model, X: np.ndarray, Z: np.ndarray, rows: np.ndarray,
                                 D: np.ndarray, dispersion_cols: list) -> dict:
    """
    Do high-dispersion fighters show up in predicted cycles more often?

    D: (n_rows, 4) scaled disp_ columns, named by dispersion_cols. rows as in check_cycles.
    """
    raise NotImplementedError


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
        rows = latest_rows(data.meta)
        disp = data.blocks["dispersion"]
        results["matchup"] = _run(check_matchup_auc, model, data.X, Z, fit_pairs, data.pairs["test"],
                                  data.blocks, ev["matchup_bootstrap_resamples"], seed)
        results["cycles"] = _run(check_cycles, model, data.X, Z, rows, ev["cycle_sample_rate"], seed)
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
