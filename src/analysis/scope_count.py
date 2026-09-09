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
import yaml

ERA_CANDIDATES = (2010, 2012, 2014, 2016, 2017)
THRESHOLDS = (3, 5)
MIN_PRIOR = 3  # configs/v1.yaml: history.min_prior_fights

# ============================================================================
# LOAD DATA
# ============================================================================

REPO_ROOT = Path(__file__).resolve().parents[2]


def load_raw_csvs(raw_dir=None):
    """Load the four scraped CSVs from the most recent date."""
    raw_path = Path(raw_dir) if raw_dir else REPO_ROOT / "data" / "raw"
    dates = sorted([d for d in raw_path.iterdir() if d.is_dir()])
    if not dates:
        raise FileNotFoundError(f"No raw data found in {raw_dir}")
    
    latest = dates[-1]
    print(f"Loading from {latest}/")
    
    events = pd.read_csv(latest / "events.csv")
    fights = pd.read_csv(latest / "fights.csv")
    fight_stats = pd.read_csv(latest / "fight_stats.csv")
    fighters = pd.read_csv(latest / "fighters.csv")

    # fights carries its own date + is_dwcs, so nothing needs the events table.
    # (events.csv uses DATE, not date -- that was the KeyError.)
    fights["date"] = pd.to_datetime(fights["date"])
    fights["year"] = fights["date"].dt.year
    assert not fights["is_dwcs"].any(), "DWCS bouts appeared -- revisit scope filter"

    return events, fights, fight_stats, fighters


def build_fighter_bouts(fights):
    """One row per fighter per bout, in career order.

    career_idx = number of PRIOR career bouts, i.e. exactly what
    history.min_prior_fights is compared against under the extend strategy.
    """
    cols = ["fight_id", "date", "year"]
    a = fights[cols + ["fighter_a_id"]].rename(columns={"fighter_a_id": "fighter_id"})
    b = fights[cols + ["fighter_b_id"]].rename(columns={"fighter_b_id": "fighter_id"})
    fb = pd.concat([a, b], ignore_index=True).sort_values(["fighter_id", "date"])
    fb["career_idx"] = fb.groupby("fighter_id").cumcount()
    return fb.reset_index(drop=True)


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
    # non_dwcs_events = events[events.get("is_dwcs", False)]["event_id"].unique()
    non_dwcs_fights = fights
    
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
    bouts_by_year = fights.groupby("year").size()
    
    # Key statistics
    total_bouts = bouts_by_year.sum()
    post_2014 = bouts_by_year[bouts_by_year.index >= 2014].sum()
    pct_post_2014 = 100 * post_2014 / total_bouts
    
    print(f"\nBOUTS BY YEAR:")
    print(f"  Total non-DWCS bouts: {total_bouts}")
    print(f"  Bouts after 2014-01-01: {post_2014} ({pct_post_2014:.1f}%)")
    print(f"  Year range: {bouts_by_year.index.min()} to {bouts_by_year.index.max()}")
    for yr in ERA_CANDIDATES:
        kept = int(bouts_by_year[bouts_by_year.index >= yr].sum())
        print(f"    {yr}+: {kept:>5} bouts ({100 * kept / total_bouts:.1f}%)")

    return bouts_by_year


# ============================================================================
# ANALYSIS 3: Feature Completeness by Year
# ============================================================================

def analyze_feature_completeness(fights, fight_stats, fighter_bouts, fighters):
    """
    Which completeness signal actually constrains the era choice?

    Stat completeness does NOT: fetch_ufcstats.py already dropped the statless
    rows, and the only bouts with no stats at all are 1994-1998. Recorded as an
    explicit negative so the question stays answered.

    Reach coverage DOES -- it climbs across the whole candidate range and is the
    binding constraint on the physical feature block.
    """
    have_stats = set(fight_stats["fight_id"].unique())
    has = fights["fight_id"].isin(have_stats)
    gaps = fights[~has]["year"]

    print("\nSTAT COMPLETENESS:")
    print(f"  Bouts with zero stat rows: {(~has).sum()} of {len(fights)}")
    print(f"  All of them fall in {gaps.min()}-{gaps.max()}; complete from {gaps.max() + 1} on.")
    print("  ==> does NOT discriminate between era candidates.")

    phys = fighter_bouts.merge(
        fighters[["fighter_id", "reach_in", "height_in", "stance"]],
        on="fighter_id", how="left",
    ).groupby("year").agg(
        n=("fighter_id", "size"),
        reach=("reach_in", lambda s: 100 * s.notna().mean()),
        height=("height_in", lambda s: 100 * s.notna().mean()),
        stance=("stance", lambda s: 100 * s.notna().mean()),
    )

    print("\nPHYSICAL COVERAGE (per fighter-bout):")
    print("    year      n   reach%  height%  stance%")
    for year, r in phys[phys.index >= 2010].iterrows():
        mark = "  <-- era candidate" if year in ERA_CANDIDATES else ""
        print(f"    {year}  {int(r.n):>5}   {r.reach:>5.1f}   {r.height:>6.1f}   {r.stance:>6.1f}{mark}")
    print("  Reach is the binding constraint; height/stance are ~100% throughout.")
    print("  NOTE: the dip in the last two years is scrape freshness (recent debutants")
    print("  missing from fighter_tott), and it lands in the TEST split. Day 3 problem.")

    return phys


# ============================================================================
# ANALYSIS 4: Era x roster -- the decision table
# ============================================================================

def snapshot_yield(fighter_bouts, era_year, min_bouts, min_prior=MIN_PRIOR,
                   strategy="extend"):
    """
    Fighters and snapshot rows surviving one (era, threshold, strategy) combo.

    The roster threshold counts bouts INSIDE the era -- those are the bouts that
    produce snapshot rows. The strategy decides what counts as prior history:

      extend   - career-wide; a 2015 snapshot may draw on 2012 bouts
      truncate - history resets at the era cutoff
    """
    era_rows = fighter_bouts[fighter_bouts["year"] >= era_year].copy()
    era_rows["era_idx"] = era_rows.groupby("fighter_id").cumcount()

    n = era_rows.groupby("fighter_id").size()
    qualified = n[n >= min_bouts].index
    era_rows = era_rows[era_rows["fighter_id"].isin(qualified)]

    prior = era_rows["career_idx"] if strategy == "extend" else era_rows["era_idx"]
    return len(qualified), int((prior >= min_prior).sum())


def analyze_roster_by_era(fighter_bouts):
    """Read one row off this table and questions 1, 2 and 3 are all answered."""
    print(f"\nERA x ROSTER DECISION TABLE (min_prior_fights={MIN_PRIOR}):")
    print("\n     era  bouts  fighters   extend  truncate     gain")
    rows = []
    for era in ERA_CANDIDATES:
        for thresh in THRESHOLDS:
            n_f, n_ext = snapshot_yield(fighter_bouts, era, thresh, strategy="extend")
            _, n_trunc = snapshot_yield(fighter_bouts, era, thresh, strategy="truncate")
            gain = n_ext - n_trunc
            print(f"    {era}    {thresh}+     {n_f:>5}    {n_ext:>5}     {n_trunc:>5}   "
                  f"+{gain} ({100 * gain / n_trunc:.0f}%)")
            rows.append({"era_start": era, "min_bouts": thresh, "n_fighters": n_f,
                         "snapshots_extend": n_ext, "snapshots_truncate": n_trunc})
    print("\n  'gain' = rows that exist only because pre-era bouts count as history.")
    return pd.DataFrame(rows)


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
    fighter_bouts = build_fighter_bouts(fights)

    roster_df = analyze_roster_scope(events, fights)
    bouts_by_year = analyze_bouts_by_year(events, fights)
    analyze_feature_completeness(fights, fight_stats, fighter_bouts, fighters)
    table = analyze_roster_by_era(fighter_bouts)

    # Read the live decision back out of the config so the two cannot drift.
    cfg = yaml.safe_load((REPO_ROOT / "configs" / "v1.yaml").read_text())
    era_yr = int(cfg["roster"]["era_start"][:4])
    min_bouts = int(cfg["roster"]["min_ufc_bouts"])
    strategy = cfg["history"]["strategy"]

    chosen = table[(table.era_start == era_yr) & (table.min_bouts == min_bouts)].iloc[0]
    n_snap = chosen.snapshots_extend if strategy == "extend" else chosen.snapshots_truncate
    print(f"\nCONFIG: era {era_yr}, {min_bouts}+ bouts, {strategy} -> "
          f"{chosen.n_fighters} fighters, {n_snap} snapshot rows.")
    print("  NB: the plan assumed ~25k rows. min_prior_fights=3 drops every")
    print("  fighter's first three bouts, so the real figure is ~1/3 of that.")
    print("  Shrinkage and dropout are load-bearing on Days 6-7.")

    summary = {
        "date": datetime.now().isoformat(),
        "roster_3plus": int((roster_df["n_bouts"] >= 3).sum()),
        "roster_5plus": int((roster_df["n_bouts"] >= 5).sum()),
        "bouts_by_year": {int(k): int(v) for k, v in bouts_by_year.items()},
        "decision_table": table.to_dict(orient="records"),
        "decisions": {"min_ufc_bouts": min_bouts,
                      "era_start": cfg["roster"]["era_start"],
                      "history_strategy": strategy},
    }
    out = REPO_ROOT / "data" / "scope_analysis.json"
    out.write_text(json.dumps(summary, indent=2))
    print(f"\n[OK] Saved to {out.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
