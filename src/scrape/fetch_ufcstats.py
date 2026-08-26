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
# Absolute ile path for accessing scraped data u
SOURCE_DIR = Path(__file__).resolve().parents[2] / "data"/ "scrape_ufc_stats-main"

def fetch_events() -> pd.DataFrame:
    """
    Fetch events from UFCStats.
    
    Returns:
        DataFrame with columns: event_id, event_name, date, location, promotion
    
    You write this using Greco1899/scrape_ufc_stats or similar.
    The key is marking DWCS events so we can exclude them.
    """
    #Load events into DataFrame
    events_path : Path = SOURCE_DIR / "ufc_event_details.csv"
    events_df : pd.DataFrame = pd.read_csv(events_path) 

    #Standardize date format to (YYYY-MM-DD)
    events_df["DATE"] = pd.to_datetime(events_df["DATE"],format = "%B %d, %Y")
    return events_df


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
    events_df["is_dwcs"] = events_df["EVENT"].str.contains(
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
    #Load events into DataFrame

    results_path : Path = SOURCE_DIR / "ufc_fight_results.csv" # Use to get all other stats
    fdetails_path : Path = SOURCE_DIR / "ufc_fighter_details.csv" # Use to get fighter ID

    fight_results_df : pd.DataFrame = pd.read_csv(results_path) # Use to get all other stats
    fighter_details_df : pd.DataFrame = pd.read_csv(fdetails_path) # Use to get fighter ID
    # events_df is used to get event ID and time

    # Use URL hashes to assign IDs to each row
    fight_results_df['fight_id'] = fight_results_df["URL"].str.rsplit("/",n=1).str[-1]
    fighter_details_df['fighter_id'] = fighter_details_df["URL"].str.rsplit("/",n=1).str[-1]
    events_df['event_id'] = events_df["URL"].str.rsplit("/",n=1).str[-1]

    # Create name column for fighters
    # Outer .strip() matters: 17 fighters are mononyms with no FIRST name (Maheshate,
    # Rongzhu, Sumudaerji, ...), so the concat yields " Maheshate" and the leading space
    # would silently break the join against BOUT names.
    fighter_details_df['name'] = (
        fighter_details_df["FIRST"].fillna("").str.strip() + " "
        + fighter_details_df["LAST"].fillna("").str.strip()).str.strip()

    # create fighter a and fighter b name columns
    fight_results_df[["fighter_a_name", "fighter_b_name"]] = (
    fight_results_df["BOUT"].str.split(" vs. ", expand=True, regex=False))

    #Create winner name column --> will use to find winner id on merge
    fight_results_df["winner_name"] = np.select(
    [fight_results_df["OUTCOME"].eq("W/L"), fight_results_df["OUTCOME"].eq("L/W")],
    [fight_results_df["fighter_a_name"], fight_results_df["fighter_b_name"]],
    default=None,
    )

    # WEIGHTCLASS is messy: "UFC Welterweight Title Bout", "Interim Heavyweight Title Bout",
    # "Ultimate Fighter 14 Bantamweight Tournament". Normalise once, reuse below.
    weightclass = fight_results_df["WEIGHTCLASS"].fillna("").str.strip()

    # Label Title Bouts (also catches interim titles: "Interim Heavyweight Title Bout")
    fight_results_df["title_bout"] = weightclass.str.lower().str.contains("title")

    #Create weightclass column
    division_lookup = {
    "Women's Strawweight":115,"Women's Flyweight":125,"Women's Bantamweight":135,
    "Women's Featherweight":145, "Flyweight":125, "Bantamweight":135, "Featherweight":145,
    "Lightweight":155, "Welterweight":170, "Middleweight":185, "Light Heavyweight": 205, 
    "Heavyweight":265
    }
    sorted_divisions = sorted(list(division_lookup.keys()),key = lambda x: -len(x)) # sort in descending order

    # Weight classes like catch weight, super heavyweight and open weight don't exist in the table because they have no weight limit
    # Extract rather than strip: on TUF cards the division sits mid-string, so there is no
    # prefix/suffix to remove. Alternation resolves leftmost-first, so the suffix overlaps
    # ("Heavyweight" inside "Light Heavyweight", "Bantamweight" inside "Women's
    # Bantamweight") are already safe -- the long name starts earlier and wins on position.
    # Longest-first ordering is cheap insurance for any future key that is a true PREFIX of
    # another, where list order would decide. Keeping the women's divisions distinct matters:
    # pooling them would break reach percentiles downstream.
    division_pattern = "(" + "|".join(sorted_divisions) + ")"
    fight_results_df["weight_class"] = weightclass.str.extract(division_pattern, expand=False)

    # Nullable Int64 so unmatched rows stay null instead of coercing the column to float.
    fight_results_df["weight_lbs"] = (
        fight_results_df["weight_class"].map(division_lookup).astype("Int64"))

    


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
