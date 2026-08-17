"""
Day 2: Analyze raw data to make empirical decisions on roster scope and era cutoff.

Output:
  - Three key numbers: fighters at 3+, 5+, and feature completeness by year
  - Plots saved to notebooks/scope_analysis.md
  - Updated configs/v1.yaml with era cutoff decision
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import json

# ============================================================================
# LOAD DATA
# ============================================================================

def load_raw_csvs(raw_dir: str = "data/raw"):
    """Load the four scraped CSVs from the most recent date."""
    raw_path = Path(raw_dir)
    dates = sorted([d for d in raw_path.iterdir() if d.is_dir()])
    if not dates:
        raise FileNotFoundError(f"No raw data found in {raw_dir}")
    
    latest = dates[-1]
    print(f"Loading from {latest}/")
    
    events = pd.read_csv(latest / "events.csv")
    fights = pd.read_csv(latest / "fights.csv")
    fight_stats = pd.read_csv(latest / "fight_stats.csv")
    fighters = pd.read_csv(latest / "fighters.csv")
    
    return events, fights, fight_stats, fighters


# ============================================================================
# ANALYSIS 1: Roster Scope (3+ vs 5+)
# ============================================================================

def analyze_roster_scope(events: pd.DataFrame, fights: pd.DataFrame) -> pd.DataFrame:
    """
    Count non-DWCS UFC bouts per fighter.
    
    Returns:
        DataFrame with columns: fighter_id, n_bouts_3plus, n_bouts_5plus, ...
    """
    # Filter out DWCS events
    non_dwcs_events = events[~events.get("is_dwcs", False)]["event_id"].unique()
    non_dwcs_fights = fights[fights["event_id"].isin(non_dwcs_events)].copy()
    
    # Count bouts per fighter
    fighter_a_counts = non_dwcs_fights.groupby("fighter_a_id").size()
    fighter_b_counts = non_dwcs_fights.groupby("fighter_b_id").size()
    
    all_fighters = pd.concat([fighter_a_counts, fighter_b_counts]).groupby(level=0).sum()
    all_fighters = all_fighters.to_frame("n_bouts")
    
    # Roster thresholds
    at_3plus = (all_fighters["n_bouts"] >= 3).sum()
    at_5plus = (all_fighters["n_bouts"] >= 5).sum()
    
    print(f"\nROSTER SCOPE:")
    print(f"  Fighters with 3+ non-DWCS UFC bouts: {at_3plus}")
    print(f"  Fighters with 5+ non-DWCS UFC bouts: {at_5plus}")
    print(f"  Ratio: {at_3plus / at_5plus:.2f}x")
    
    return all_fighters


# ============================================================================
# ANALYSIS 2: Bouts per Year
# ============================================================================

def analyze_bouts_by_year(events: pd.DataFrame, fights: pd.DataFrame) -> pd.DataFrame:
    """
    Breakdown of bouts by year. Identify when the "modern era" stabilizes.
    """
    events["date"] = pd.to_datetime(events["date"])
    events["year"] = events["date"].dt.year
    
    # Filter non-DWCS
    non_dwcs = events[~events.get("is_dwcs", False)][["event_id", "year"]]
    fight_years = fights.merge(non_dwcs, on="event_id", how="inner")
    
    bouts_by_year = fight_years.groupby("year").size()
    
    # Key statistics
    total_bouts = bouts_by_year.sum()
    post_2014 = bouts_by_year[bouts_by_year.index >= 2014].sum()
    pct_post_2014 = 100 * post_2014 / total_bouts
    
    print(f"\nBOUTS BY YEAR:")
    print(f"  Total non-DWCS bouts: {total_bouts}")
    print(f"  Bouts after 2014-01-01: {post_2014} ({pct_post_2014:.1f}%)")
    print(f"  Year range: {bouts_by_year.index.min()} to {bouts_by_year.index.max()}")
    
    return bouts_by_year


# ============================================================================
# ANALYSIS 3: Feature Completeness by Year
# ============================================================================

def analyze_feature_completeness(events: pd.DataFrame, fights: pd.DataFrame,
                                  fight_stats: pd.DataFrame) -> pd.DataFrame:
    """
    For each year, compute the fraction of fights with complete strike data.
    
    This identifies where target/position data becomes consistent.
    UFCStats improved over time; old fights have missing position splits.
    """
    events["date"] = pd.to_datetime(events["date"])
    events["year"] = events["date"].dt.year
    
    # Filter non-DWCS
    non_dwcs = events[~events.get("is_dwcs", False)][["event_id", "year"]]
    fight_years = fights.merge(non_dwcs, on="event_id", how="inner")
    
    # Merge with fight_stats to check completeness
    fight_years_with_stats = fight_years.merge(
        fight_stats,
        on="fight_id",
        how="left"
    )
    
    # Define "complete" as having both position and location breakdowns
    completeness = fight_years_with_stats.groupby("year").apply(
        lambda df: df[["distance_landed", "clinch_landed", "ground_landed"]].notna().all(axis=1).mean(),
        include_groups=False
    )
    
    # Find inflection: where completeness > 80% consistently
    stable_complete = completeness[completeness > 0.8].index.min()
    
    print(f"\nFEATURE COMPLETENESS BY YEAR:")
    print(f"  First year >80% complete: {stable_complete}")
    print("\n  Year-by-year:")
    for year, pct in completeness.items():
        marker = " <-- use this as era_start?" if year == stable_complete else ""
        print(f"    {year}: {100*pct:.1f}%{marker}")
    
    return completeness


# ============================================================================
# DECISION POINT
# ============================================================================

def make_decisions() -> dict:
    """
    Interactively guide you to update configs/v1.yaml with empirical choices.
    """
    print("\n" + "="*70)
    print("DECISION POINT")
    print("="*70)
    
    decisions = {}
    
    print("\n1. Roster scope: use 3+ or 5+ fighters?")
    print("   (5+ = smaller, cleaner sample; 3+ = more fighters, noisier per-fighter stats)")
    roster_choice = input("   Enter: 3 or 5 > ")
    decisions["min_ufc_bouts"] = int(roster_choice)
    
    print("\n2. Era cutoff: from the completeness plot above, when does data stabilize?")
    print("   (Recommendation: first year >80% complete)")
    era_choice = input("   Enter year (YYYY) > ")
    decisions["era_start"] = f"{era_choice}-01-01"
    
    print("\n3. History strategy: extend or truncate?")
    print("   extend    = a 2015-debuting fighter can use pre-2014 bouts")
    print("   truncate  = reset history at era_start cutoff")
    strategy = input("   Enter: extend or truncate > ")
    decisions["history_strategy"] = strategy
    
    print("\n✓ Decisions recorded:")
    print(f"  Min UFC bouts: {decisions['min_ufc_bouts']}")
    print(f"  Era start: {decisions['era_start']}")
    print(f"  History strategy: {decisions['history_strategy']}")
    
    return decisions


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("[Day 2] Scope Analysis")
    
    events, fights, fight_stats, fighters = load_raw_csvs()
    
    # Run analyses
    roster_df = analyze_roster_scope(events, fights)
    bouts_by_year = analyze_bouts_by_year(events, fights)
    completeness = analyze_feature_completeness(events, fights, fight_stats)
    
    # Get decisions
    decisions = make_decisions()
    
    # Save summary
    summary = {
        "date": datetime.now().isoformat(),
        "roster_3plus": int((roster_df["n_bouts"] >= 3).sum()),
        "roster_5plus": int((roster_df["n_bouts"] >= 5).sum()),
        "decisions": decisions
    }
    
    with open("data/scope_analysis.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n✓ Analysis saved to data/scope_analysis.json")
    print("\nNext step: Update configs/v1.yaml manually with the decisions above,")
    print("then run src/features/build_snapshots.py")


if __name__ == "__main__":
    main()
