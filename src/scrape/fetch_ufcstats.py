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


def _attach_fighter_id(fights_df: pd.DataFrame, roster_df: pd.DataFrame,
                       name_col: str, id_col: str) -> pd.DataFrame:
    """
    Left-join fighter_id onto a bout-name column, collapsing the fan-out that
    ambiguous names produce.

    Name is the only bridge between the bout tables and the fighter tables: no CSV
    in the snapshot pairs a fighter URL with a bout (see DATA_NOTES.md). Eight names
    in fighter_details map to two fighter_ids each and six of those appear as bout
    names, so a plain left join returns both candidates and duplicates the fight row
    -- 8,832 rows in, 8,882 out, 50 double-counted bouts.

    Resolution: keep the candidate whose listed weight sits closest to the bout's
    weight_lbs. Every colliding pair separates cleanly on weight (Bruno Silva is
    125 vs 185, Michael McDonald 135 vs 205). A candidate with no weight on file
    loses to any candidate that has one, which settles Mike Davis, where one of the
    two pages is an empty stub.

    Getting this wrong is worse than dropping the rows: merging a 125 lb flyweight
    and a 185 lb middleweight under one fighter_id invents a fighter with two
    incompatible styles, which lands in the embedding as a fake hybrid.

    Rows are unique on fight_id going in and come back unique, in the same order.
    """
    fights_df = fights_df.copy()
    fights_df["_row"] = np.arange(len(fights_df))

    candidates = roster_df[["name", "fighter_id", "fighter_weight_lbs"]].rename(
        columns={"name": name_col, "fighter_id": id_col})
    merged = fights_df.merge(candidates, on=name_col, how="left")

    # 9999 = no weight on file, loses to any real candidate. 5000 = catchweight bout
    # with no weight_lbs to compare against; the candidates tie and the stable sort
    # breaks it on source order, deterministically.
    weight_gap = (merged["fighter_weight_lbs"] - merged["weight_lbs"]).abs()
    merged["_penalty"] = np.where(
        merged["fighter_weight_lbs"].isna(), 9999.0,
        np.where(weight_gap.isna(), 5000.0, weight_gap.astype(float)))

    return (merged.sort_values(["_row", "_penalty"], kind="stable")
                  .drop_duplicates(subset="_row", keep="first")
                  .sort_values("_row")
                  .drop(columns=["_penalty", "fighter_weight_lbs", "_row"])
                  .reset_index(drop=True))


def fetch_fights(events_df: pd.DataFrame) -> pd.DataFrame:
    """
    Fetch individual bouts.
    
    Returns:
        DataFrame with columns:
        fight_id, event_id, date, fighter_a_id, fighter_b_id, fighter_a_name,
        fighter_b_name, winner_id, method, round, time, duration_seconds,
        weight_class, weight_lbs, title_bout

        Beyond the original skeleton contract:
          - date             required by snapshots.py:150 and the leak assertion at :181;
                             comes from the event join, nothing else carries it
          - duration_seconds derived from round + time + TIME FORMAT; replaces the
                             `total_fight_seconds = 1` placeholder at snapshots.py:261
                             and is the denominator for every per-minute rate feature
          - weight_lbs       numeric companion to weight_class (see DATA_NOTES.md);
                             weight_class stays the grouping key for percentiles

    Merge with events to attach event metadata and is_dwcs flag.
    """
    #Load events into DataFrame

    results_path : Path = SOURCE_DIR / "ufc_fight_results.csv" # Use to get all other stats
    fdetails_path : Path = SOURCE_DIR / "ufc_fighter_details.csv" # Use to get fighter ID
    tott_path : Path = SOURCE_DIR / "ufc_fighter_tott.csv" # "tale of the tape" -- physicals

    fight_results_df : pd.DataFrame = pd.read_csv(results_path) # Use to get all other stats
    fighter_details_df : pd.DataFrame = pd.read_csv(fdetails_path) # Use to get fighter ID
    fighter_tott_df : pd.DataFrame = pd.read_csv(tott_path) # Use to break name collisions
    # events_df is used to get event ID and time

    # Use URL hashes to assign IDs to each row
    fight_results_df['fight_id'] = fight_results_df["URL"].str.rsplit("/",n=1).str[-1]
    fighter_details_df['fighter_id'] = fighter_details_df["URL"].str.rsplit("/",n=1).str[-1]

    # Create name column for fighters
    # Outer .strip() matters: 17 fighters are mononyms with no FIRST name (Maheshate,
    # Rongzhu, Sumudaerji, ...), so the concat yields " Maheshate" and the leading space
    # would silently break the join against BOUT names.
    fighter_details_df['name'] = (
        fighter_details_df["FIRST"].fillna("").str.strip() + " "
        + fighter_details_df["LAST"].fillna("").str.strip()).str.strip()

    # Listed weight, used only to break name collisions in _attach_fighter_id.
    # details and tott are the two halves of one fighter page and share URL -- the
    # only unambiguous key anywhere in this source set.
    fighter_details_df = fighter_details_df.merge(
        fighter_tott_df[["URL", "WEIGHT"]], on="URL", how="left")
    fighter_details_df["fighter_weight_lbs"] = (
        fighter_details_df["WEIGHT"].str.extract(r"(\d+)").astype(float))

    # create fighter a and fighter b name columns
    fight_results_df[["fighter_a_name", "fighter_b_name"]] = (
    fight_results_df["BOUT"].str.split(" vs. ", expand=True, regex=False))

    # Five bout names have no fighter_details row -- 13 bouts, all post-2014. Each is
    # a different defect, and all five resolve with certainty by hand.
    # Do NOT fuzzy-match these: "Patricio Freire" is one edit from "Patricky Freire",
    # his brother and a separate fighter on the roster. An edit-distance matcher picks
    # the wrong man and nothing downstream ever notices.
    BOUT_NAME_FIXES = {
        "Kai Kamaka": "Kai Kamaka III",            # generational suffix
        "Bibulatov Magomed": "Magomed Bibulatov",  # name order reversed in BOUT
        "Tre'ston Vines": "Treston Vines",         # apostrophe
        "Rafael Cerquiera": "Rafael Cerqueira",    # transposition typo in BOUT
        "Patricio Freire": "Patricio Pitbull",     # ring name, not surname
    }
    fight_results_df[["fighter_a_name", "fighter_b_name"]] = (
        fight_results_df[["fighter_a_name", "fighter_b_name"]].replace(BOUT_NAME_FIXES))

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

    # Calculating duration using round #, time format, time. #time format doesn't matter since rounds are all 5 minutes
    fight_results_df["duration_seconds"] = (pd.to_timedelta((fight_results_df["ROUND"]-1)*300,unit = "s") + pd.to_timedelta("00:" + fight_results_df["TIME"])).dt.total_seconds().astype(int)

    # Merge events_details with fight_results
    # left merge so all fight rows exist, right merge wouldn't guarantee this
    # strip event column for both dataframes beforehand to avoid unexpected string comparison issues
    fight_results_df["EVENT"] = fight_results_df["EVENT"].str.strip()
    events_df["EVENT"] = events_df["EVENT"].str.strip()
    events_df["event_id"] = events_df["URL"].str.rsplit("/", n=1).str[-1]
    events_df : pd.DataFrame = events_df.drop(columns="URL") # hashed into event_id above
    fight_and_event_results_df : pd.DataFrame = fight_results_df.merge(events_df,on = "EVENT",how = "left")

    # Check that duplicate rows with missing dates are removed
    fight_and_event_results_df :  pd.DataFrame = fight_and_event_results_df.dropna(subset=["DATE"])
    assert not fight_and_event_results_df["DATE"].isna().any().any()
    assert not fight_and_event_results_df["fight_id"].duplicated().any()

    # Attach fighter ids. Two merges, each adding a COLUMN not a row: merge A reads
    # only fighter_a_name, merge B only fighter_b_name, so the bout stays one row.
    # _attach_fighter_id collapses the fan-out that colliding names would otherwise
    # introduce.
    fights_df : pd.DataFrame = _attach_fighter_id(
        fight_and_event_results_df, fighter_details_df, "fighter_a_name", "fighter_a_id")
    fights_df = _attach_fighter_id(
        fights_df, fighter_details_df, "fighter_b_name", "fighter_b_id")
    print('###',fights_df.iloc[0])
    # winner_id is derived, not joined -- a third merge on winner_name would be a third
    # chance to fan out. Reusing OUTCOME rather than comparing names keeps the 158
    # NC/NC and D/D fights at None instead of silently awarding them to fighter B.
    fights_df["winner_id"] = np.select(
        [fights_df["OUTCOME"].eq("W/L"), fights_df["OUTCOME"].eq("L/W")],
        [fights_df["fighter_a_id"], fights_df["fighter_b_id"]],
        default=None,
    )

    # Post-merge contract. The pre-merge assert above runs before the joins and so
    # cannot see fan-out. Both unresolved counts are frozen at zero: a refresh that
    # introduces a new unmatched or colliding name fails loudly here rather than
    # quietly dropping a fighter out of the roster.
    assert not fights_df["fight_id"].duplicated().any(), "fighter join fanned out"
    assert fights_df["fighter_a_id"].isna().sum() == 0, "unresolved fighter_a_id"
    assert fights_df["fighter_b_id"].isna().sum() == 0, "unresolved fighter_b_id"

    fights_df = fights_df.rename(columns={
        "DATE": "date", "METHOD": "method", "ROUND": "round", "TIME": "time"})

    return fights_df[[
        "fight_id", "event_id", "date", "is_dwcs",
        "fighter_a_id", "fighter_b_id", "fighter_a_name", "fighter_b_name",
        "winner_id", "method", "round", "time", "duration_seconds",
        "weight_class", "weight_lbs", "title_bout",
    ]]


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
    fight_stats_path : Path = SOURCE_DIR / "ufc_fight_stats.csv"

    fight_stats_df : pd.DataFrame = pd.read_csv(fight_stats_path)


    # Change round column to int number
    fight_stats_df["ROUND"] = fight_stats_df["ROUND"].str.split(" ").str[-1].astype('Int64')
    # Change KD column to int number
    fight_stats_df["KD"] = fight_stats_df["KD"].astype('Int64')

    # Extract sig str land and att using str split
    fight_stats_df[["sig_str_landed","sig_str_att"]] =  fight_stats_df["SIG.STR."].str.strip().str.split(" of ",expand=True)

    # Extract head landed and att using str split
    fight_stats_df[["head_landed","head_att"]] = fight_stats_df["HEAD"].str.strip().str.split(" of ",expand=True)

    # Extract body landed and att using str split
    fight_stats_df[["body_landed","body_att"]] = fight_stats_df["BODY"].str.strip().str.split(" of ",expand=True)

    # Extract leg landed and att using str split
    fight_stats_df[["leg_landed","leg_att"]] = fight_stats_df["LEG"].str.strip().str.split(" of ",expand=True)

    # Extract distance landed and att using str split
    fight_stats_df[["distance_landed","distance_att"]] = fight_stats_df["DISTANCE"].str.strip().str.split(" of ",expand=True)

    # Extract body landed and att using str split
    fight_stats_df[["clinch_landed","clinch_att"]] = fight_stats_df["CLINCH"].str.strip().str.split(" of ",expand=True)

    # Extract body landed and att using str split
    fight_stats_df[["ground_landed","ground_att"]] = fight_stats_df["GROUND"].str.strip().str.split(" of ",expand=True)

    # Extract body landed and att using str split
    fight_stats_df[["td_landed","td_att"]] = fight_stats_df["TD"].str.strip().str.split(" of ",expand=True)

    # confirm SUB.ATT column is int number
    fight_stats_df["sub_att"] = fight_stats_df["SUB.ATT"].astype('Int64')
    
    # confirm REV. column is int number
    fight_stats_df["rev"] = fight_stats_df["REV."].astype('Int64')


    '''

        Join plans:
        Read ufc_fight_details.csv and procure fight_id from there to then merge with fights_df and eradicate duplicated rows
    '''

    print("\ndude\n",fight_stats_df.iloc[0])



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
