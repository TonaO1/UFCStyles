"""
Scrape UFC fighter and fight data from UFCStats.

Source: https://github.com/Greco1899/scrape_ufc_stats
You implement the main loop and validation.

This script outputs four tables to raw/{YYYY-MM-DD}/:
  - events.csv       (one row per event)
  - fights.csv       (one row per bout)
  - fight_stats.csv  (one row per fighter per fight, optionally per round)
  - fighters.csv     (one row per fighter with profile data)

Contract (you ensure these hold before uploading to S3):
  - head + body + leg LANDED == sig_str_landed
  - distance + clinch + ground LANDED == sig_str_landed (two independent partitions of total)
  - No NaN in mandatory columns (event_id, fighter_id, fight_id, round, ...)
  - DWCS events identifiable and marked (see flag_dwcs_events)
"""

import os
import sys
import requests
import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd
import numpy as np
import boto3

# ============================================================================
# SECTION 1: FETCH DATA
# ============================================================================

def fetch_events() -> pd.DataFrame:
    """
    Fetch events from UFCStats.
    
    Returns:
        DataFrame with columns: event_id, event_name, date, location, promotion
    
    You write this using Greco1899/scrape_ufc_stats or similar.
    The key is marking DWCS events so we can exclude them.
    """
    
    # Pseudo-code:
    # 1. Load or scrape events
    # 2. Call flag_dwcs_events()
    # 3. Validate date format (YYYY-MM-DD)
    # 4. Return
    raise NotImplementedError("You implement the scraper. Use Greco1899 as reference.")


def flag_dwcs_events(events_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a boolean column 'is_dwcs' to mark Dana White's Contender Series events.
    
    DWCS has a different opponent pool and sits as separate events in UFCStats.
    We exclude it from feature aggregation to keep fighter profiles clean.
    
    Args:
        events_df: DataFrame with event_name column
    
    Returns:
        Same DataFrame with new 'is_dwcs' column (bool)
    
    Example:
        >>> flag_dwcs_events({"event_name": ["UFC 123", "DWCS 5", "UFC 124"]})
        # Returns with is_dwcs: [False, True, False]
    """
    events_df["is_dwcs"] = events_df["event_name"].str.contains(
        r"(?:Dana White|DWCS|Contender Series)",
        case=False,
        na=False
    )
    return events_df


def fetch_fights(events_df: pd.DataFrame) -> pd.DataFrame:
    """
    Fetch individual bouts.
    
    Returns:
        DataFrame with columns:
        fight_id, event_id, fighter_a_id, fighter_b_id, fighter_a_name, fighter_b_name,
        winner_id, method, round, time, weight_class, title_bout
    
    Merge with events to attach event metadata and is_dwcs flag.
    """
    raise NotImplementedError("You implement the scraper.")


def fetch_fight_stats(fights_df: pd.DataFrame) -> pd.DataFrame:
    """
    Fetch strike and grappling stats per fighter per fight (and optionally per round).
    
    Returns:
        DataFrame with columns:
        fight_id, fighter_id, round (or 'total'),
        KD, sig_str_landed, sig_str_att,
        head_landed, head_att, body_landed, body_att, leg_landed, leg_att,
        distance_landed, distance_att, clinch_landed, clinch_att, ground_landed, ground_att,
        td_landed, td_att, sub_att, rev, ctrl_time (str, e.g., "5:30" or "--")
    
    Note: UFCStats records strike location and strike position as two independent partitions.
    Both sum to sig_str_landed.
    """
    raise NotImplementedError("You implement the scraper.")


def fetch_fighters() -> pd.DataFrame:
    """
    Fetch fighter metadata.
    
    Returns:
        DataFrame with columns:
        fighter_id, fighter_name, height_cm, reach_cm, stance, dob
    
    Note: height and reach are in cm or inches depending on source. Standardize.
    """
    raise NotImplementedError("You implement the scraper.")


# ============================================================================
# SECTION 2: VALIDATION
# ============================================================================

def validate_fights(fight_stats_df: pd.DataFrame) -> bool:
    """
    Verify that strike location and position partitions are internally consistent.
    
    Contract:
      (1) head_landed + body_landed + leg_landed == sig_str_landed
      (2) distance_landed + clinch_landed + ground_landed == sig_str_landed
    
    Args:
        fight_stats_df: DataFrame from fetch_fight_stats
    
    Returns:
        bool: True if all validations pass. Raises AssertionError if not.
    
    Notes:
        - Some rows may have NaN (rounds without recorded data). Skip those.
        - Underscore differences (1-2 strikes) are acceptable due to data entry quirks.
        - Call this on raw data before any filtering.
    """
    df = fight_stats_df.copy()
    
    # Skip rows with NaN in critical columns
    df = df.dropna(subset=[
        "sig_str_landed", "head_landed", "body_landed", "leg_landed",
        "distance_landed", "clinch_landed", "ground_landed"
    ])
    
    # Check 1: location partition
    location_sum = df["head_landed"] + df["body_landed"] + df["leg_landed"]
    location_error = (location_sum - df["sig_str_landed"]).abs()
    
    assert (location_error <= 1).all(), \
        f"Location partition failed for {(location_error > 1).sum()} rows. Max error: {location_error.max()}"
    
    # Check 2: position partition
    position_sum = df["distance_landed"] + df["clinch_landed"] + df["ground_landed"]
    position_error = (position_sum - df["sig_str_landed"]).abs()
    
    assert (position_error <= 1).all(), \
        f"Position partition failed for {(position_error > 1).sum()} rows. Max error: {position_error.max()}"
    
    print(f"✓ Partition validation passed ({len(df)} rows checked).")
    return True


def validate_control_time(fight_stats_df: pd.DataFrame) -> bool:
    """
    Parse control time from M:SS format to seconds. Handle "--" (zero).
    
    Contract:
      - All ctrl_time values parse to float (seconds) or NaN.
      - No weird formats.
    """
    def to_seconds(s):
        if pd.isna(s) or s == "--":
            return 0.0
        parts = str(s).split(":")
        if len(parts) != 2:
            raise ValueError(f"Bad ctrl_time format: {s}")
        return int(parts[0]) * 60 + int(parts[1])
    
    fight_stats_df["ctrl_seconds"] = fight_stats_df["ctrl_time"].apply(to_seconds)
    print(f"✓ Control time parsed. Range: {fight_stats_df['ctrl_seconds'].min():.0f}s to {fight_stats_df['ctrl_seconds'].max():.0f}s")
    return True


def validate_row_counts(events: pd.DataFrame, fights: pd.DataFrame,
                        fight_stats: pd.DataFrame, fighters: pd.DataFrame) -> bool:
    """
    Sanity check: row counts in plausible ranges.
    """
    assert len(events) > 1000, f"Too few events: {len(events)}"
    assert len(fights) > 5000, f"Too few fights: {len(fights)}"
    assert len(fight_stats) > 10000, f"Too few fight_stats rows: {len(fight_stats)}"
    assert len(fighters) > 1000, f"Too few fighters: {len(fighters)}"
    
    print(f"✓ Row counts plausible: {len(events)} events, {len(fights)} fights, "
          f"{len(fight_stats)} stats rows, {len(fighters)} fighters.")
    return True


# ============================================================================
# SECTION 3: MAIN
# ============================================================================

def main(args):
    print("[Day 1] Scraping UFC Stats...")
    
    # Fetch
    events = fetch_events()
    events = flag_dwcs_events(events)
    fights = fetch_fights(events)
    fight_stats = fetch_fight_stats(fights)
    fighters = fetch_fighters()
    
    # Validate
    validate_row_counts(events, fights, fight_stats, fighters)
    validate_fights(fight_stats)
    validate_control_time(fight_stats)
    
    # Save locally
    output_dir = Path("data/raw") / datetime.now().strftime("%Y-%m-%d")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    events.to_csv(output_dir / "events.csv", index=False)
    fights.to_csv(output_dir / "fights.csv", index=False)
    fight_stats.to_csv(output_dir / "fight_stats.csv", index=False)
    fighters.to_csv(output_dir / "fighters.csv", index=False)
    
    print(f"✓ Saved to {output_dir}/")
    
    # Upload to S3 (optional)
    if args.s3_bucket:
        s3 = boto3.client("s3")
        s3_prefix = f"raw/{datetime.now().strftime('%Y-%m-%d')}"
        for csv_file in output_dir.glob("*.csv"):
            s3.upload_file(
                str(csv_file),
                args.s3_bucket,
                f"{s3_prefix}/{csv_file.name}"
            )
        print(f"✓ Uploaded to s3://{args.s3_bucket}/{s3_prefix}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--s3-bucket", help="S3 bucket to upload to (optional)")
    args = parser.parse_args()
    main(args)
