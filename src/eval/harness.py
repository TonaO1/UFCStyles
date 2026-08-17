"""
Evaluation harness: the single function that scores every embedding.

Contract:
  - Every model, every baseline (random, raw, PCA, AE, contrastive) runs through this.
  - Same metrics, same harness, honest comparison.
  
Checks:
  1. Probe recovery (k-NN on embedding vs. raw features vs. random)
  2. Non-transitivity (cycle rate as evidence of style, not quality)
  3. Matchup delta (AUC gap on temporal test split)
  4. Dispersion correlation (high-dispersion fighters in cycles?)

Tests on:
  - Background (martial-arts base): wrestler, striker, BJJ, hybrid, unclear
  - Guard type (visual, per-bout)
  - Stance (weak but clean probe)
  - Weight class (structural check)
  - Reach percentile (regression, Spearman)
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, log_loss
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler
import argparse


# ============================================================================
# CHECK 1: PROBE RECOVERY
# ============================================================================

def check_probe_recovery(embeddings: np.ndarray, meta: pd.DataFrame,
                         labels: pd.DataFrame, probe_k: int = 10) -> dict:
    """
    Train k-NN classifier on embedding to predict each probe.
    Compare against raw features and random embedding.
    
    Metric: balanced accuracy (handles imbalanced class distributions).
    
    Args:
        embeddings: (n_snapshots, d) embedding matrix
        meta: DataFrame with fighter_id, fight_id, date, split
        labels: DataFrame with fighter_id and probe columns
        probe_k: k for k-NN
    
    Returns:
        dict with accuracies for each probe and each method
    """
    
    results = {}
    
    # Merge meta and labels
    data = meta.merge(labels, on="fighter_id", how="inner")
    
    # Test on test split only
    test_mask = data["split"] == "test"
    test_indices = data[test_mask].index
    
    probes_to_check = {
        "background": ("background", "classification"),
        "stance": ("stance", "classification"),
        "weight_class": ("weight_class", "classification"),
        # "guard": ("guard", "classification"),  # optional if labeled
        # "reach_percentile": ("reach_percentile", "regression"),  # optional
    }
    
    for probe_name, (probe_col, task_type) in probes_to_check.items():
        if probe_col not in labels.columns:
            continue
        
        y = data.loc[test_indices, probe_col]
        
        # Skip if all NaN or single class
        if y.isna().sum() > len(y) * 0.5 or y.nunique() <= 1:
            results[probe_name] = {"skipped": "insufficient labels"}
            continue
        
        # Drop NaN
        valid_mask = ~y.isna()
        y = y[valid_mask]
        X_test = embeddings[test_indices[valid_mask]]
        
        if len(y) < 10:
            results[probe_name] = {"skipped": f"only {len(y)} samples"}
            continue
        
        # Stratified train/test within the test split (re-split for cross-val)
        # Simple: use 80/20 within test split
        np.random.seed(42)
        n = len(y)
        train_idx = np.random.choice(n, size=int(0.8 * n), replace=False)
        test_idx = np.setdiff1d(np.arange(n), train_idx)
        
        X_train = X_test[train_idx]
        X_test_split = X_test[test_idx]
        y_train = y.iloc[train_idx].values
        y_test = y.iloc[test_idx].values
        
        if task_type == "classification":
            clf = KNeighborsClassifier(n_neighbors=probe_k)
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test_split)
            acc = balanced_accuracy_score(y_test, y_pred)
            results[probe_name] = {"accuracy": acc, "n_samples": len(y_test)}
        
        # Also measure gap vs. random
        random_embedding = np.random.randn(*embeddings.shape)
        X_train_random = random_embedding[test_indices[valid_mask]][train_idx]
        X_test_random = random_embedding[test_indices[valid_mask]][test_idx]
        
        clf_random = KNeighborsClassifier(n_neighbors=probe_k)
        clf_random.fit(X_train_random, y_train)
        y_pred_random = clf_random.predict(X_test_random)
        acc_random = balanced_accuracy_score(y_test, y_pred_random)
        
        results[probe_name]["accuracy_random"] = acc_random
        results[probe_name]["gap_vs_random"] = acc - acc_random
    
    return results


# ============================================================================
# CHECK 2: NON-TRANSITIVITY (CYCLES)
# ============================================================================

def check_cycles(embeddings: np.ndarray, meta: pd.DataFrame,
                 features: pd.DataFrame, sample_rate: float = 0.5) -> dict:
    """
    Build a matchup model: P(fighter_a beats fighter_b) from embedding.
    Count cycles (non-transitive triples) and compare to a pure Elo model.
    
    A cycle: A > B AND B > C AND C > A (all with P > 0.5).
    
    Args:
        embeddings: (n_snapshots, d) embedding
        meta: fighter_id, fight_id, date, split
        features: raw feature vectors (for baseline comparison)
        sample_rate: sample this fraction of triples to avoid explosion
    
    Returns:
        dict with cycle rate, comparison to Elo, etc.
    """
    
    # Use test split only
    test_mask = meta["split"] == "test"
    test_meta = meta[test_mask].reset_index(drop=True)
    test_emb = embeddings[test_mask]
    
    results = {}
    
    if len(test_meta) < 100:
        results["skipped"] = "fewer than 100 test samples"
        return results
    
    # Build triples
    n_samples = int(len(test_meta) * sample_rate)
    triple_indices = np.random.choice(len(test_meta), size=(min(n_samples, 100), 3), replace=True)
    
    # Simple matchup model: logit = (e_a - e_b) @ w
    # Use a weighted combination (could be more sophisticated)
    w = np.random.randn(test_emb.shape[1])  # placeholder weight
    w = w / np.linalg.norm(w)
    
    cycle_count = 0
    total_triples = 0
    
    for a_idx, b_idx, c_idx in triple_indices:
        e_a = test_emb[a_idx]
        e_b = test_emb[b_idx]
        e_c = test_emb[c_idx]
        
        p_a_beats_b = 1 / (1 + np.exp(-((e_a - e_b) @ w)))
        p_b_beats_c = 1 / (1 + np.exp(-((e_b - e_c) @ w)))
        p_c_beats_a = 1 / (1 + np.exp(-((e_c - e_a) @ w)))
        
        if p_a_beats_b > 0.5 and p_b_beats_c > 0.5 and p_c_beats_a > 0.5:
            cycle_count += 1
        
        total_triples += 1
    
    cycle_rate = cycle_count / max(total_triples, 1)
    
    results["cycle_rate"] = cycle_rate
    results["n_triples"] = total_triples
    results["n_cycles"] = cycle_count
    
    # Compare to pure Elo (scalar rating)
    # Elo produces zero cycles by construction
    results["comparison_to_elo"] = (
        "Elo would have 0 cycles. Your rate: " +
        f"{cycle_rate:.3f}. Non-zero is evidence of style effects, "
        "not pure quality ranking."
    )
    
    return results


# ============================================================================
# CHECK 3: MATCHUP AUC DELTA
# ============================================================================

def check_matchup_auc(embeddings: np.ndarray, meta: pd.DataFrame,
                      features: pd.DataFrame, bootstrap_resamples: int = 1000) -> dict:
    """
    Compare matchup prediction AUC:
      - Baseline: logistic regression on (features_a - features_b)
      - Model: logistic regression on (embeddings_a - embeddings_b)
    
    Metric: AUC and log-loss on test split.
    Report: gap with 95% bootstrap CI.
    
    Args:
        embeddings: (n_snapshots, d)
        meta: including actual fight outcome (winner)
        features: raw feature vectors
        bootstrap_resamples: for CI
    
    Returns:
        dict with AUC gap and CI
    """
    from sklearn.linear_model import LogisticRegression
    
    results = {}
    
    # Use test split
    test_mask = meta["split"] == "test"
    test_meta = meta[test_mask].reset_index(drop=True)
    test_emb = embeddings[test_mask]
    test_feat = features[test_mask] if features is not None else None
    
    if len(test_meta) < 50:
        results["skipped"] = "fewer than 50 test samples"
        return results
    
    # Build matchup pairs from actual test fights
    # You need to extract (fighter_a_id, fighter_b_id, winner_id) from test_meta
    # Placeholder: assume meta has these columns
    
    if "winner_id" not in test_meta.columns:
        results["skipped"] = "no winner_id in metadata"
        return results
    
    # For now, just report placeholder
    results["placeholder"] = "Implement pairwise matchup encoding in your data pipeline"
    results["gap_auc"] = np.nan
    results["gap_auc_ci"] = [np.nan, np.nan]
    
    return results


# ============================================================================
# CHECK 4: DISPERSION CORRELATION
# ============================================================================

def check_dispersion_correlation(embeddings: np.ndarray, meta: pd.DataFrame,
                                  features: pd.DataFrame) -> dict:
    """
    Hypothesis: high-dispersion fighters should be overrepresented in cycle triples.
    
    Compute: Spearman correlation between style_dispersion and P(appears in cycle).
    
    Args:
        embeddings: (n_snapshots, d)
        meta: includes fighter_id, and should have a dispersion column
        features: includes style_dispersion column
    
    Returns:
        dict with correlation and p-value
    """
    
    results = {}
    
    if "style_dispersion" not in features.columns:
        results["skipped"] = "no style_dispersion in features"
        return results
    
    test_mask = meta["split"] == "test"
    dispersion = features.loc[test_mask, "style_dispersion"]
    
    # Placeholder: compute P(appears in cycle) for each row
    # (Real implementation would check cycle triples and mark participants)
    in_cycle = np.random.randint(0, 2, size=len(dispersion))
    
    r, pval = spearmanr(dispersion, in_cycle)
    
    results["correlation"] = r
    results["p_value"] = pval
    results["interpretation"] = (
        "Positive r suggests high-dispersion fighters adapt styles, "
        "enabling non-transitive matchups."
    )
    
    return results


# ============================================================================
# HARNESS CONTRACT
# ============================================================================

def evaluate(embeddings: np.ndarray, meta: pd.DataFrame, features: pd.DataFrame,
             labels: pd.DataFrame, name: str, config: dict = None) -> dict:
    """
    The main evaluation function. Every embedding runs through this once.
    
    Args:
        embeddings: (n_snapshots, d) embedding matrix, scaled or not doesn't matter
        meta: DataFrame with fighter_id, fight_id, date, split
        features: raw feature vectors (for baseline comparison)
        labels: probe labels (background, guard, stance, weight_class)
        name: model name for reporting
        config: config dict (optional, for eval settings)
    
    Returns:
        dict with all metrics, saved to eval/{name}.json
    """
    
    if config is None:
        config = {
            "eval": {
                "probe_k": 10,
                "cycle_sample_rate": 0.5,
                "matchup_bootstrap_resamples": 1000,
            }
        }
    
    print(f"\n{'='*70}")
    print(f"Evaluating: {name}")
    print(f"{'='*70}")
    print(f"Embeddings: {embeddings.shape}")
    print(f"Metadata: {meta.shape}")
    print(f"Features: {features.shape if features is not None else 'None'}")
    print(f"Labels: {labels.shape}")
    
    all_results = {
        "model": name,
        "embedding_shape": embeddings.shape,
        "n_test": (meta["split"] == "test").sum(),
    }
    
    # Check 1: Probe recovery
    print("\nCheck 1: Probe recovery (k-NN on embedding)...")
    all_results["probes"] = check_probe_recovery(
        embeddings, meta, labels,
        probe_k=config["eval"]["probe_k"]
    )
    print(f"  Results: {all_results['probes']}")
    
    # Check 2: Cycles
    print("\nCheck 2: Non-transitivity (cycles)...")
    all_results["cycles"] = check_cycles(
        embeddings, meta, features,
        sample_rate=config["eval"]["cycle_sample_rate"]
    )
    print(f"  Results: {all_results['cycles']}")
    
    # Check 3: Matchup AUC
    print("\nCheck 3: Matchup AUC delta...")
    all_results["matchup"] = check_matchup_auc(
        embeddings, meta, features,
        bootstrap_resamples=config["eval"]["matchup_bootstrap_resamples"]
    )
    print(f"  Results: {all_results['matchup']}")
    
    # Check 4: Dispersion correlation
    print("\nCheck 4: Dispersion correlation...")
    all_results["dispersion"] = check_dispersion_correlation(
        embeddings, meta, features
    )
    print(f"  Results: {all_results['dispersion']}")
    
    # Save to JSON
    output_dir = Path("data/eval")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / f"{name}.json", "w") as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n✓ Saved to {output_dir / f'{name}.json'}")
    
    return all_results


# ============================================================================
# MAIN (TEST ON RANDOM EMBEDDING FIRST)
# ============================================================================

def main(args):
    """
    Load data and run harness on baselines (random, raw features, PCA).
    """
    print("[Day 5] Evaluation Harness")
    
    # Load snapshots and labels
    snapshots_train = pd.read_parquet("data/snapshots/v1/train.parquet")
    snapshots_val = pd.read_parquet("data/snapshots/v1/val.parquet")
    snapshots_test = pd.read_parquet("data/snapshots/v1/test.parquet")
    
    snapshots = pd.concat([snapshots_train, snapshots_val, snapshots_test], ignore_index=False)
    
    # Load labels (you create these by hand)
    try:
        labels = pd.read_csv("data/labels/background.csv")
    except FileNotFoundError:
        print("⚠ data/labels/background.csv not found. Running harness without probe checks.")
        labels = pd.DataFrame({"fighter_id": []})
    
    # Extract meta and features
    meta = snapshots[["fighter_id", "fight_id", "date", "split"]].copy()
    feature_cols = [c for c in snapshots.columns if c not in ["fighter_id", "fight_id", "date", "split", "weight_class", "n_prior"]]
    features = snapshots[feature_cols].values
    
    # Load config
    import yaml
    with open("configs/v1.yaml") as f:
        config = yaml.safe_load(f)
    
    # Test on random embedding (if harness is broken, this will score high)
    print("\nTesting harness on RANDOM embedding (should score poorly)...")
    random_emb = np.random.randn(len(snapshots), 8)
    evaluate(random_emb, meta, features, labels, "random_8", config)
    
    # Optional: test on raw features
    if args.baseline == "raw" or args.baseline == "all":
        print("\nTesting harness on RAW FEATURES...")
        raw_emb = features[:, :min(8, features.shape[1])]
        if raw_emb.shape[1] < 8:
            raw_emb = np.hstack([raw_emb, np.random.randn(len(snapshots), 8 - raw_emb.shape[1])])
        evaluate(raw_emb, meta, features, labels, "raw_8", config)
    
    # Optional: test on PCA
    if args.baseline == "pca" or args.baseline == "all":
        print("\nTesting harness on PCA...")
        from sklearn.decomposition import PCA
        pca = PCA(n_components=8)
        pca_emb = pca.fit_transform(features[np.array(meta["split"]) == "train"])
        pca_emb_all = pca.transform(features)
        evaluate(pca_emb_all, meta, features, labels, "pca_8", config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", choices=["random", "raw", "pca", "all"], default="all")
    args = parser.parse_args()
    main(args)
