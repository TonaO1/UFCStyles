"""
Build leak-free fighter snapshots.

For fighter F at fight N: every feature comes from fights 1 through N-1 only.
This module is where the data-quality contract lives.

Output:
  - data/per_bout/v1/per_bout_vectors.parquet
  - data/snapshots/v1/{train,val,test}.parquet
  - data/snapshots/v1/scaler.pkl (fitted on train only)
"""

import pandas as pd
import numpy as np
import yaml
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import pickle

# ============================================================================
# LOAD CONFIG
# ============================================================================

def load_config(config_path: str = "configs/v1.yaml") -> dict:
    """Load configuration."""
    with open(config_path) as f:
        return yaml.safe_load(f)


# ============================================================================
# LOAD DATA
# ============================================================================

def load_raw_data(config: dict):
    """Load the four raw CSVs."""
    raw_dir = Path(config["paths"]["raw"])
    dates = sorted([d for d in raw_dir.iterdir() if d.is_dir()])
    latest = dates[-1]
    
    events = pd.read_csv(latest / "events.csv")
    fights = pd.read_csv(latest / "fights.csv")
    fight_stats = pd.read_csv(latest / "fight_stats.csv")
    fighters = pd.read_csv(latest / "fighters.csv")
    
    return events, fights, fight_stats, fighters


# ============================================================================
# PER-BOUT VECTORS
# ============================================================================

def compute_per_bout_vectors(fights: pd.DataFrame, fight_stats: pd.DataFrame) -> pd.DataFrame:
    """
    Compute style proportions for each individual bout.
    
    This is the foundation: before aggregating into career vectors,
    compute the same features per fight. This enables:
      1. Aggregation (mean/weighted per fighter)
      2. Dispersion measurement (spread across bouts)
    
    Returns:
        DataFrame with columns:
        fight_id, fighter_id, round_type (total or number),
        [all proportion features]
    """
    # Aggregate by fight and fighter (total across rounds)
    stats_total = fight_stats.groupby(["fight_id", "fighter_id"]).agg({
        "KD": "sum",
        "sig_str_landed": "sum",
        "sig_str_att": "sum",
        "head_landed": "sum",
        "head_att": "sum",
        "body_landed": "sum",
        "body_att": "sum",
        "leg_landed": "sum",
        "leg_att": "sum",
        "distance_landed": "sum",
        "distance_att": "sum",
        "clinch_landed": "sum",
        "clinch_att": "sum",
        "ground_landed": "sum",
        "ground_att": "sum",
        "td_landed": "sum",
        "td_att": "sum",
        "sub_att": "sum",
        "rev": "sum",
        "ctrl_seconds": "sum",
    }).reset_index()
    
    # Compute proportions (style block)
    # Target distribution
    stats_total["head_share"] = stats_total.eval("head_landed / (head_landed + body_landed + leg_landed + 0.001)")
    stats_total["body_share"] = stats_total.eval("body_landed / (head_landed + body_landed + leg_landed + 0.001)")
    stats_total["leg_share"] = stats_total.eval("leg_landed / (head_landed + body_landed + leg_landed + 0.001)")
    
    # Position distribution
    stats_total["distance_share"] = stats_total.eval("distance_landed / (distance_landed + clinch_landed + ground_landed + 0.001)")
    stats_total["clinch_share"] = stats_total.eval("clinch_landed / (distance_landed + clinch_landed + ground_landed + 0.001)")
    stats_total["ground_share"] = stats_total.eval("ground_landed / (distance_landed + clinch_landed + ground_landed + 0.001)")
    
    # Grappling
    stats_total["td_att_share"] = stats_total.eval("td_att / (td_att + sig_str_att + 0.001)")
    stats_total["ctrl_minutes"] = stats_total["ctrl_seconds"] / 60
    stats_total["sub_att_per_ctrl_min"] = stats_total.eval("sub_att / (ctrl_minutes + 0.1)")
    
    # Pace
    stats_total["total_fight_seconds"] = stats_total.eval("ctrl_seconds + 1")  # placeholder
    stats_total["pace_per_min"] = stats_total.eval("sig_str_landed / total_fight_seconds * 60")
    
    # Keep key columns
    per_bout = stats_total[[
        "fight_id", "fighter_id",
        "head_share", "body_share", "leg_share",
        "distance_share", "clinch_share", "ground_share",
        "td_att_share", "sub_att_per_ctrl_min",
        "pace_per_min",
        "sig_str_landed", "td_landed", "sub_att", "KD", "ctrl_seconds"
    ]].copy()
    
    return per_bout


# ============================================================================
# SNAPSHOT BUILDER (THE LEAK-FREE CORE)
# ============================================================================

def build_snapshots(fights: pd.DataFrame, fight_stats: pd.DataFrame,
                    events: pd.DataFrame, fighters: pd.DataFrame,
                    config: dict) -> pd.DataFrame:
    """
    Build one row per fighter-at-fight, with all features computed from PRIOR fights only.
    
    This is the critical function. Every assertion protects against leakage.
    
    Args:
        fights, fight_stats, events, fighters: raw tables
        config: config dict
    
    Returns:
        DataFrame with one row per snapshot, ready for train/val/test split.
        Columns: fighter_id, fight_id, date, n_prior, [all features]
    """
    
    # Filter to non-DWCS fights and configured era
    era_start = pd.to_datetime(config["roster"]["era_start"])
    min_prior = config["history"]["min_prior_fights"]
    
    events_filtered = events[~events.get("is_dwcs", False) & (pd.to_datetime(events["date"]) >= era_start)]
    fights_filtered = fights[fights["event_id"].isin(events_filtered["event_id"])].copy()
    fights_filtered["date"] = pd.to_datetime(fights_filtered["date"])
    
    # Per-bout vectors (computed once, used for both aggregation and dispersion)
    per_bout = compute_per_bout_vectors(fights_filtered, fight_stats)
    
    snapshots = []
    
    # For each fighter, build one row per fight (if enough prior fights)
    all_fighter_ids = pd.concat([
        fights_filtered["fighter_a_id"],
        fights_filtered["fighter_b_id"]
    ]).unique()
    
    for fighter_id in all_fighter_ids:
        # Get all fights for this fighter
        fighter_fights = pd.concat([
            fights_filtered[fights_filtered["fighter_a_id"] == fighter_id],
            fights_filtered[fights_filtered["fighter_b_id"] == fighter_id]
        ]).drop_duplicates("fight_id").sort_values("date").reset_index(drop=True)
        
        # Build snapshots: one per fight, using only PRIOR fights
        for i, fight_row in fighter_fights.iterrows():
            if i < min_prior:
                # Not enough prior fights
                continue
            
            prior_fights = fighter_fights.iloc[:i]  # strictly before
            current_fight = fight_row
            current_fight_date = current_fight["date"]
            
            # LEAK ASSERTION
            assert all(prior_fights["date"] < current_fight_date), \
                f"LEAK DETECTED: fighter {fighter_id} fight {current_fight['fight_id']}"
            
            # Build feature row from prior_fights
            feature_row = build_feature_row(
                fighter_id=fighter_id,
                fight_id=current_fight["fight_id"],
                date=current_fight_date,
                prior_fights=prior_fights,
                prior_stats=fight_stats[fight_stats["fight_id"].isin(prior_fights["fight_id"])],
                per_bout=per_bout,
                weight_class=current_fight.get("weight_class", "unknown"),
                config=config
            )
            
            snapshots.append(feature_row)
    
    snapshots_df = pd.DataFrame(snapshots)
    
    print(f"✓ Built {len(snapshots_df)} leak-free snapshots")
    print(f"  Unique fighters: {snapshots_df['fighter_id'].nunique()}")
    print(f"  Date range: {snapshots_df['date'].min()} to {snapshots_df['date'].max()}")
    
    return snapshots_df


def build_feature_row(fighter_id: str, fight_id: str, date: pd.Timestamp,
                      prior_fights: pd.DataFrame, prior_stats: pd.DataFrame,
                      per_bout: pd.DataFrame, weight_class: str, config: dict) -> dict:
    """
    Compute all features for one fighter-at-fight snapshot.
    
    Args:
        fighter_id, fight_id, date: identifiers
        prior_fights, prior_stats: fights and stats strictly before this fight
        per_bout: per-bout style vectors
        weight_class: for percentile normalization
        config: config dict
    
    Returns:
        dict with all features
    """
    row = {
        "fighter_id": fighter_id,
        "fight_id": fight_id,
        "date": date,
        "n_prior": len(prior_fights),
        "weight_class": weight_class,
    }
    
    # If no prior fights, return sparse row (shouldn't happen with min_prior check)
    if len(prior_fights) == 0:
        return row
    
    # === STYLE BLOCK: PROPORTIONS ===
    if config["features"]["blocks"]["proportions"]:
        # You hand-code these aggregations
        # Start with per_bout vectors for prior fights, then aggregate
        prior_per_bout = per_bout[per_bout["fight_id"].isin(prior_fights["fight_id"])]
        
        if len(prior_per_bout) > 0:
            # Mean proportions across prior bouts
            row["head_share_mean"] = prior_per_bout["head_share"].mean()
            row["body_share_mean"] = prior_per_bout["body_share"].mean()
            row["distance_share_mean"] = prior_per_bout["distance_share"].mean()
            row["clinch_share_mean"] = prior_per_bout["clinch_share"].mean()
            row["pace_per_min_mean"] = prior_per_bout["pace_per_min"].mean()
            # ... add all proportion features
    
    # === QUALITY BLOCK: RATES ===
    if config["features"]["blocks"]["rates"]:
        stats_agg = prior_stats.groupby("fighter_id").agg({
            "sig_str_landed": "sum",
            "sig_str_att": "sum",
            "ctrl_seconds": "sum",
            "td_landed": "sum",
            # ... add all columns needed
        }).loc[fighter_id]
        
        # Compute rates
        total_fight_seconds = 1  # placeholder; derive from fights
        row["sig_str_per_min"] = stats_agg["sig_str_landed"] / (total_fight_seconds / 60 + 1)
        row["td_landed_per_15min"] = stats_agg["td_landed"] / (total_fight_seconds / 60 + 1) * 15
        # ... add all rate features
    
    # === PHYSICAL BLOCK ===
    if config["features"]["blocks"]["physical"]:
        # Requires fighter metadata and weight-class stats
        row["reach_percentile_in_wc"] = 0.5  # placeholder; compute from fighters df
        row["height_percentile_in_wc"] = 0.5
    
    # === DISPERSION BLOCK ===
    if config["features"]["blocks"]["dispersion"]:
        # Compute centroid of prior per-bout vectors, then distance from each
        prior_per_bout = per_bout[per_bout["fight_id"].isin(prior_fights["fight_id"])]
        if len(prior_per_bout) > 1:
            # You compute this: mean Euclidean distance from each bout to the centroid
            row["style_dispersion"] = 0.0  # placeholder
    
    return row


# ============================================================================
# SPLITS
# ============================================================================

def apply_temporal_split(snapshots_df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Assign train/val/test split based on temporal cutoffs (not random).
    
    Args:
        snapshots_df: all snapshots
        config: config dict
    
    Returns:
        Same DataFrame with new "split" column
    """
    train_range = [pd.to_datetime(d) for d in config["splits"]["train"]["date_range"]]
    val_range = [pd.to_datetime(d) for d in config["splits"]["val"]["date_range"]]
    test_range = [pd.to_datetime(d) for d in config["splits"]["test"]["date_range"]]
    
    snapshots_df["split"] = "unknown"
    snapshots_df.loc[(snapshots_df["date"] >= train_range[0]) & (snapshots_df["date"] <= train_range[1]), "split"] = "train"
    snapshots_df.loc[(snapshots_df["date"] >= val_range[0]) & (snapshots_df["date"] <= val_range[1]), "split"] = "val"
    snapshots_df.loc[(snapshots_df["date"] >= test_range[0]) & (snapshots_df["date"] <= test_range[1]), "split"] = "test"
    
    print(f"\nTemporal split:")
    print(f"  Train: {(snapshots_df['split'] == 'train').sum()} rows")
    print(f"  Val:   {(snapshots_df['split'] == 'val').sum()} rows")
    print(f"  Test:  {(snapshots_df['split'] == 'test').sum()} rows")
    
    return snapshots_df


def fit_and_save_scaler(snapshots_df: pd.DataFrame, feature_cols: list, config: dict):
    """
    Fit StandardScaler on train split, persist alongside model weights later.
    """
    train_data = snapshots_df[snapshots_df["split"] == "train"][feature_cols]
    
    scaler = StandardScaler()
    scaler.fit(train_data)
    
    output_dir = Path(config["paths"]["snapshots"])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / "scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    
    print(f"✓ Scaler fitted and saved to {output_dir / 'scaler.pkl'}")
    
    return scaler


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("[Days 3-4] Building leak-free snapshots...")
    
    config = load_config()
    events, fights, fight_stats, fighters = load_raw_data(config)
    
    # Build snapshots
    snapshots = build_snapshots(fights, fight_stats, events, fighters, config)
    
    # Apply split
    snapshots = apply_temporal_split(snapshots, config)
    
    # Fit scaler on train
    feature_cols = [c for c in snapshots.columns if c not in ["fighter_id", "fight_id", "date", "split", "weight_class", "n_prior"]]
    scaler = fit_and_save_scaler(snapshots, feature_cols, config)
    
    # Scale all data
    snapshots_scaled = snapshots.copy()
    snapshots_scaled[feature_cols] = scaler.transform(snapshots[feature_cols])
    
    # Save parquets
    output_dir = Path(config["paths"]["snapshots"])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for split in ["train", "val", "test"]:
        split_data = snapshots_scaled[snapshots_scaled["split"] == split]
        split_data.to_parquet(output_dir / f"{split}.parquet", index=False)
        print(f"✓ {output_dir / f'{split}.parquet'} ({len(split_data)} rows)")
    
    print("\n✓ Snapshots complete. Next: label background and guard type.")


if __name__ == "__main__":
    main()
