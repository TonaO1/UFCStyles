"""
Build leak-free fighter snapshots: for fighter F at fight N, every feature comes
from fights 1..N-1 only.

Writes per-bout vectors, train/val/test parquet, the train-fitted scaler and a
manifest. Features live in compute_per_bout_vectors and build_feature_row;
everything else here is plumbing.

Column formats, data defects and the reasoning behind the feature choices are in
DATA_NOTES.md.
"""

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.preprocessing import StandardScaler


# ============================================================================
# SECTION 0: CONVENTIONS -- the contract between the plumbing and the features
# ============================================================================
# Feature keys are prefixed by block. Nothing else in the frame may use these
# prefixes, because both the day 6-7 block ablations and the shrinkage
# eligibility rule below are prefix filters:
#
#   prop_*   proportions  (style)        shrunk
#   rate_*   rates        (quality)      shrunk
#   phys_*   physical     (context)      NOT shrunk -- measurements, not estimates
#   disp_*   dispersion   (adaptation)   NOT shrunk -- already a second moment
#
# Emit prefixed keys from build_feature_row and the rest of the file wires
# itself up: selection, ablation, shrinkage, scaling, persistence.

BLOCK_PREFIXES = {
    "proportions": "prop_",
    "rates": "rate_",
    "physical": "phys_",
    "dispersion": "disp_",
}

SHRINKABLE_BLOCKS = ("proportions", "rates")

# Identifiers and bookkeeping. Never features, never scaled.
META_COLS = ["fighter_id", "fight_id", "date", "weight_class", "percentile_class",
             "opponent_id", "split", "prior_fight_ids"]

# Per-fighter evidence counts, attached by build_snapshots from the RAW tables
# (not from your per_bout columns, so they cannot drift when you rename one).
# apply_shrinkage picks one per feature: n is "how much data does THIS fighter
# have for THIS feature", and bouts / attempts / minutes are three different
# answers to that.
EVIDENCE_COLS = ["n_prior_bouts", "n_prior_sig_att", "n_prior_td_att",
                 "n_prior_minutes"]

# First matching rule wins. Override anything here by name in EVIDENCE_OVERRIDES.
EVIDENCE_RULES = [
    (("_per_min", "per_min", "per_15", "_pm", "_pace", "minutes"), "n_prior_minutes"),
    (("head", "body", "leg", "distance", "clinch", "ground", "sig"), "n_prior_sig_att"),
    (("td", "takedown", "sub", "rev", "ctrl", "control", "grappl"), "n_prior_td_att"),
]
# strike_share / sig_str_share are strike-volume ratios; nothing in EVIDENCE_RULES
# matches their names, so they would default to n_prior_bouts.
EVIDENCE_OVERRIDES: dict = {
    "prop_strike_share": "n_prior_sig_att",
}

# Exact counts, not estimates. Shrinking a number you measured toward a
# population mean corrupts it -- a 20-fight veteran is a 20-fight veteran.
NOT_SHRINKABLE = {"rate_n_prior_bouts", "rate_career_minutes"}

# k is documented in configs/v1.yaml as "attempts to shrink over" (75). Attempts
# run in the hundreds, prior bouts 3-46, prior minutes in the tens -- one k across
# three scales would shrink the bout-scale features ~90% toward the mean while
# barely touching the attempt-scale ones. Each evidence family gets its own k,
# defaulted here and overridable via features.shrinkage.k_by_evidence.
DEFAULT_K_BY_EVIDENCE = {
    "n_prior_sig_att": None,   # None => use features.shrinkage.k verbatim
    "n_prior_td_att": 12.0,    # td attempts run in the tens, not the hundreds
    "n_prior_minutes": 20.0,
    "n_prior_bouts": 5.0,
}


# ============================================================================
# SECTION 1: CONFIG AND DATA LOADING
# ============================================================================

def _as_date(value) -> pd.Timestamp:
    """
    Coerce a config date to Timestamp.

    PyYAML resolves an unquoted 2014-01-01 to datetime.date and a quoted one to
    str, so the same edit changes the type. Normalise at the boundary.
    """
    ts = pd.Timestamp(value)
    assert not pd.isna(ts), f"unparseable date in config: {value!r}"
    return ts


def _to_bool(series: pd.Series) -> pd.Series:
    """
    Parse a CSV-round-tripped boolean column.

    "False" is a non-empty string and therefore truthy, which silently inverts
    every filter written against it.
    """
    if series.dtype == bool:
        return series
    mapping = {"True": True, "true": True, "TRUE": True, "1": True,
               "False": False, "false": False, "FALSE": False, "0": False}
    out = series.astype(str).str.strip().map(mapping)
    bad = sorted(set(series[out.isna()].astype(str)))[:5]
    assert out.notna().all(), f"unparseable boolean values: {bad}"
    return out.astype(bool)


def load_config(config_path: str = "configs/v1.yaml") -> dict:
    """Load configs/v1.yaml and validate it before returning."""
    with open(config_path) as f:
        config = yaml.safe_load(f)

    assert isinstance(config, dict), f"{config_path} did not parse to a mapping"
    validate_config(config)
    return config


def validate_config(config: dict) -> bool:
    """
    Assert the config is coherent before anything reads data.

    Required keys present, dates parse, strategy known, and the three split
    ranges ordered, contiguous, and starting at era_start.
    """
    required = {
        "roster": ["min_ufc_bouts", "era_start"],
        "history": ["min_prior_fights", "strategy"],
        "features": ["blocks"],
        "splits": ["train", "val", "test"],
        "paths": ["raw", "snapshots", "per_bout"],
    }
    for section, keys in required.items():
        assert section in config, f"config missing section: {section}"
        for key in keys:
            assert key in config[section], f"config missing key: {section}.{key}"

    # --- scope ---------------------------------------------------------------
    min_bouts = config["roster"]["min_ufc_bouts"]
    min_prior = config["history"]["min_prior_fights"]
    assert isinstance(min_bouts, int) and min_bouts >= 1, f"min_ufc_bouts: {min_bouts!r}"
    assert isinstance(min_prior, int) and min_prior >= 1, f"min_prior_fights: {min_prior!r}"
    assert min_bouts > min_prior, (
        f"min_ufc_bouts ({min_bouts}) <= min_prior_fights ({min_prior}): every "
        f"eligible fighter would contribute zero or one snapshot"
    )

    strategy = config["history"]["strategy"]
    assert strategy in {"extend", "truncate_at_cutoff"}, f"history.strategy: {strategy!r}"

    scope = config["roster"].get("bout_count_scope", "in_era")
    assert scope in {"in_era", "career"}, f"roster.bout_count_scope: {scope!r}"

    era_start = _as_date(config["roster"]["era_start"])

    # --- feature blocks ------------------------------------------------------
    blocks = config["features"]["blocks"]
    unknown = set(blocks) - set(BLOCK_PREFIXES)
    assert not unknown, f"unknown feature blocks: {sorted(unknown)}"
    assert any(blocks.values()), "every feature block is disabled"
    for name, enabled in blocks.items():
        assert isinstance(enabled, bool), f"features.blocks.{name} is {enabled!r}, not a bool"

    shrink = config["features"].get("shrinkage", {})
    if shrink.get("enabled"):
        k = shrink.get("k")
        assert isinstance(k, (int, float)) and k > 0, f"features.shrinkage.k: {k!r}"

    nan_policy = config["features"].get("nan_policy", "error")
    assert nan_policy in {"error", "train_median"}, f"features.nan_policy: {nan_policy!r}"

    # --- splits: ordered, non-overlapping, gapless ---------------------------
    bounds = {}
    for name in ("train", "val", "test"):
        rng = config["splits"][name]["date_range"]
        assert isinstance(rng, list) and len(rng) == 2, f"splits.{name}.date_range: {rng!r}"
        start, end = _as_date(rng[0]), _as_date(rng[1])
        assert start <= end, f"splits.{name} runs backwards: {start.date()} > {end.date()}"
        bounds[name] = (start, end)

    for earlier, later in (("train", "val"), ("val", "test")):
        gap = (bounds[later][0] - bounds[earlier][1]).days
        assert gap == 1, (
            f"{earlier}.end {bounds[earlier][1].date()} -> {later}.start "
            f"{bounds[later][0].date()} is a {gap}-day step; must be exactly 1 "
            f"(>1 drops snapshots, <1 overlaps splits)"
        )

    assert bounds["train"][0] == era_start, (
        f"train.start {bounds['train'][0].date()} != roster.era_start "
        f"{era_start.date()}: snapshots in between land in no split"
    )

    # --- numeric scalars that YAML 1.1 mis-resolves ---------------------------
    # `lr: 1e-3` has no decimal point, so PyYAML resolves it to the STRING
    # "1e-3". torch.optim.Adam(lr="1e-3") dies on the first step. Write 1.0e-3.
    for model in ("autoencoder", "contrastive"):
        params = config.get("training", {}).get(model, {})
        for key in ("lr", "weight_decay", "nt_xent_tau", "dropout"):
            if key in params:
                assert isinstance(params[key], float), (
                    f"training.{model}.{key} = {params[key]!r} is {type(params[key]).__name__}, "
                    f"not float -- YAML needs a decimal point in exponent form (1.0e-3)"
                )

    return True


def load_raw_data(config: dict):
    """
    Load the four raw CSVs for the snapshot_date pinned in config.

    Ids are read as str and is_dwcs coerced to real bool -- CSV round-trips both
    into types that break filters silently. Dates are parsed once, here.
    """
    raw_root = Path(config["paths"]["raw"])
    snapshot_date = config["paths"].get("snapshot_date")
    assert snapshot_date, (
        "paths.snapshot_date is unset. Pin the data vintage in config; resolving "
        "it as 'newest directory under data/raw' makes every eval JSON "
        "incomparable the day you re-scrape."
    )
    raw_dir = raw_root / str(snapshot_date)
    assert raw_dir.is_dir(), f"{raw_dir} does not exist"
    for name in ("events.csv", "fights.csv", "fight_stats.csv", "fighters.csv"):
        assert (raw_dir / name).exists(), f"{raw_dir / name} missing"

    # --- events --------------------------------------------------------------
    events = pd.read_csv(
        raw_dir / "events.csv",
        dtype={"EVENT": str, "URL": str, "LOCATION": str, "event_id": str, "is_dwcs": str},
    )
    events["DATE"] = pd.to_datetime(events["DATE"], format="%Y-%m-%d")
    events["is_dwcs"] = _to_bool(events["is_dwcs"])

    # --- fights --------------------------------------------------------------
    id_cols = ["fight_id", "event_id", "fighter_a_id", "fighter_b_id", "winner_id"]
    fights = pd.read_csv(
        raw_dir / "fights.csv",
        dtype={**{c: str for c in id_cols},
               "fighter_a_name": str, "fighter_b_name": str, "method": str,
               "time": str, "weight_class": str, "is_dwcs": str, "title_bout": str},
    )
    fights["date"] = pd.to_datetime(fights["date"], format="%Y-%m-%d")
    fights["duration_seconds"] = pd.to_numeric(fights["duration_seconds"])
    fights["weight_lbs"] = pd.to_numeric(fights["weight_lbs"], errors="coerce")
    fights["round"] = pd.to_numeric(fights["round"], errors="coerce").astype("Int64")
    fights["method"] = fights["method"].str.strip()          # trailing space upstream
    fights["weight_class"] = fights["weight_class"].str.strip()
    fights["is_dwcs"] = _to_bool(fights["is_dwcs"])
    fights["title_bout"] = _to_bool(fights["title_bout"])

    # --- fight_stats ---------------------------------------------------------
    stat_cols = ["KD", "sig_str_landed", "sig_str_att", "tot_str_landed", "tot_str_att",
                 "head_landed", "head_att", "body_landed", "body_att",
                 "leg_landed", "leg_att", "distance_landed", "distance_att",
                 "clinch_landed", "clinch_att", "ground_landed", "ground_att",
                 "td_landed", "td_att", "sub_att", "rev"]
    fight_stats = pd.read_csv(
        raw_dir / "fight_stats.csv",
        dtype={"fight_id": str, "fighter_id": str, "ctrl_time": str},
    )
    for col in stat_cols:
        fight_stats[col] = pd.to_numeric(fight_stats[col], errors="coerce")
    fight_stats["ctrl_seconds"] = pd.to_numeric(fight_stats["ctrl_seconds"], errors="coerce")
    fight_stats["round"] = pd.to_numeric(fight_stats["round"]).astype(int)

    # --- fighters ------------------------------------------------------------
    fighters = pd.read_csv(
        raw_dir / "fighters.csv",
        dtype={"fighter_id": str, "fighter_name": str, "stance": str},
    )
    fighters["height_in"] = pd.to_numeric(fighters["height_in"], errors="coerce")
    fighters["reach_in"] = pd.to_numeric(fighters["reach_in"], errors="coerce")
    fighters["dob"] = pd.to_datetime(fighters["dob"], errors="coerce")

    # --- integrity, re-asserted at the boundary ------------------------------
    assert fights["fight_id"].is_unique, (
        f"{fights['fight_id'].duplicated().sum()} duplicate fight_ids -- the renamed-event "
        f"defect is back (see DATA_NOTES.md)"
    )
    assert fights["date"].notna().all(), (
        f"{fights['date'].isna().sum()} fights with no date. A NaT passes the leak "
        f"assertion without tripping it."
    )
    assert not fight_stats.duplicated(subset=["fight_id", "fighter_id", "round"]).any(), \
        "duplicate (fight_id, fighter_id, round) rows in fight_stats"
    assert fighters["fighter_id"].is_unique, "duplicate fighter_ids"

    referenced = set(fights["fighter_a_id"]) | set(fights["fighter_b_id"])
    missing = referenced - set(fighters["fighter_id"])
    assert not missing, f"{len(missing)} fighter_ids in fights absent from fighters"

    # DWCS is structurally unreachable from the completed-events index. This is a
    # tripwire, not a filter: if it ever fires, upstream changed its entry point.
    assert not events["is_dwcs"].any(), (
        f"{int(events['is_dwcs'].sum())} DWCS events appeared -- the scrape entry "
        f"point changed and exclude_dwcs stopped being a no-op"
    )

    print(f"Loaded {raw_dir}: {len(events)} events, {len(fights)} fights, "
          f"{len(fight_stats)} stat rows, {len(fighters)} fighters.")
    return events, fights, fight_stats, fighters


# ============================================================================
# SECTION 2: PER-BOUT VECTORS  <-- YOURS
# ============================================================================

def compute_per_bout_vectors(fights: pd.DataFrame,
                             fight_stats: pd.DataFrame) -> pd.DataFrame:
    """
    Style proportions for one bout, one row per (fight_id, fighter_id).

    Shares are built from ATTEMPTED, not landed -- attempts are what the fighter
    chose, landed is filtered through the opponent's defence. Accuracy goes in
    the rate block instead. Zero-attempt bouts get NaN, not 0.0. See DATA_NOTES.

    Share columns must end in `_share`; assert_partitions_sum_to_one keys on it.
    """
    # Cols that will be summed when you group together the rounds for a specific fighter in a specific fight
    addedCols: list = "KD,sig_str_landed,sig_str_att,tot_str_landed,tot_str_att,head_landed,head_att,body_landed,body_att,leg_landed,leg_att,distance_landed,distance_att,clinch_landed,clinch_att,ground_landed,ground_att,td_landed,td_att,sub_att,rev,ctrl_seconds".split(",")
    per_bout : pd.DataFrame = fight_stats.groupby(['fight_id','fighter_id'],as_index=False)[addedCols].sum(min_count=1)
    
    # Merge with fights DF to get duration seconds column, use fight_id as the merge key since both fighters have the same duration for a fight!
    per_bout: pd.DataFrame = per_bout.merge(fights[["fight_id","duration_seconds"]],on="fight_id",how="left")
    assert not per_bout.duplicated(subset=["fight_id", "fighter_id"]).any(), \
        f"{per_bout.duplicated(subset=['fight_id','fighter_id']).sum()} duplicate (fight, fighter) rows"

    # Compute head,leg,clinch,etc shares using sig_str_att
    denominator : pd.Series = per_bout["sig_str_att"].astype("float")
    for attacktype in ["head_att","body_att","leg_att","clinch_att","distance_att","ground_att"]:
        assert attacktype in per_bout.columns, f"{attacktype} not found in per_bout"
        prefix_idx = attacktype.index("_")
        newCol : str = attacktype[:prefix_idx] + "_share"
        per_bout[newCol] = per_bout[attacktype].astype("float") / denominator

    # Take into account habits that aren't instantly visible like fighting pace
    time_elapsed : pd.Series = per_bout["duration_seconds"].astype("float")/60.0
    actions : pd.Series = (per_bout["tot_str_att"] + per_bout["sub_att"] + per_bout["td_att"]).astype("float")
    total_strikes: pd.Series = per_bout["tot_str_att"].astype("float")
    # PACE - One column for total off actions per min and another for sig str per min
    per_bout["total_off_pace"] = actions / time_elapsed
    per_bout["sig_str_pace"] = per_bout["sig_str_att"].astype("float") / time_elapsed
    per_bout["td_pace"] = per_bout["td_att"].astype("float") / time_elapsed
    per_bout["kd_pace"] = per_bout["KD"].astype("float") / time_elapsed
    per_bout["rev_pace"] = per_bout["rev"].astype("float") / time_elapsed

    # offense fractions - How much of his offense is td/sub based
    per_bout["td_share"] = per_bout["td_att"].astype("float") / actions
    per_bout["sub_share"] = per_bout["sub_att"].astype("float") / actions
    per_bout["strike_share"] = total_strikes / actions

    # Control fraction - how much of the time does a fighter control their opponent
    per_bout["ctrl_share"] = per_bout["ctrl_seconds"].astype("float") / per_bout["duration_seconds"].astype("float")

    # Sig strike fraction - how much of their striking counts as significant
    per_bout["sig_str_share"] = per_bout["sig_str_att"].astype("float") / total_strikes

    # sig_str_accuracy
    per_bout["sig_str_acc"] = per_bout["sig_str_landed"].astype("float") / per_bout["sig_str_att"].astype("float") 

    # td accuracy
    per_bout["td_acc"] = per_bout["td_landed"].astype("float") / per_bout["td_att"].astype("float") 

  
    assert_partitions_sum_to_one(per_bout)
    return per_bout


# ============================================================================
# SECTION 3: THE LEAK-FREE SNAPSHOT BUILDER
# ============================================================================

def _fighter_bouts(fights: pd.DataFrame) -> pd.DataFrame:
    """
    Explode the bout table into one row per fighter-bout.

    17,664 rows out: 8,832 fights x 2 corners. `won` is NaN for the 158 fights
    with no winner (NC/D) rather than False -- "did not win" and "draw" are not
    the same claim, and a later win_rate has to choose.
    """
    cols = ["fight_id", "date", "weight_class", "duration_seconds", "title_bout", "method"]
    corners = []
    for own, opp in (("fighter_a_id", "fighter_b_id"), ("fighter_b_id", "fighter_a_id")):
        side = fights[cols + [own, opp, "winner_id"]].rename(
            columns={own: "fighter_id", opp: "opponent_id"})
        corners.append(side)

    out = pd.concat(corners, ignore_index=True)
    decided = out["winner_id"].notna()
    out["won"] = np.where(decided, out["fighter_id"] == out["winner_id"], np.nan)
    out["won"] = pd.to_numeric(out["won"], errors="coerce")
    return out.drop(columns=["winner_id"])


def _weight_class_reference(fighters: pd.DataFrame,
                            fighter_bouts: pd.DataFrame) -> dict:
    """
    Height/reach distributions per weight class, for percentile lookups.

    Keyed (weight_class, attr) -> sorted values, one per fighter. Built on the
    full roster rather than train only; see DATA_NOTES.md.
    """
    pairs = (fighter_bouts[["fighter_id", "weight_class"]]
             .drop_duplicates()
             .merge(fighters[["fighter_id", "height_in", "reach_in"]], on="fighter_id", how="left"))

    reference = {}
    for wc, group in pairs.groupby("weight_class"):
        for attr in ("height_in", "reach_in"):
            vals = group[attr].dropna().to_numpy(dtype=float)
            reference[(wc, attr)] = np.sort(vals)
    return reference


def percentile_in_wc(value: float, weight_class: str, attr: str,
                     reference: dict) -> float:
    """
    Where `value` sits in its weight class, in [0, 1]. NaN in, NaN out.

    Use this for the physical block instead of raw inches: a 74" reach is long
    at flyweight and short at heavyweight, and the raw number cannot say which.
    Thinnest class in-era is Women's Featherweight (34 snapshot rows), so
    percentiles there are coarse by construction.
    """
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return float("nan")
    vals = reference.get((weight_class, attr))
    if vals is None or len(vals) == 0:
        return float("nan")
    return float(np.searchsorted(vals, value, side="right") / len(vals))


def build_snapshots(fights: pd.DataFrame, fight_stats: pd.DataFrame,
                    events: pd.DataFrame, fighters: pd.DataFrame,
                    per_bout: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    One row per fighter-at-fight, with features from PRIOR fights only.

    A row exists if the fighter has >= history.min_prior_fights priors, the fight
    is in-era, and they clear roster.min_ufc_bouts. Priors are strictly earlier by
    date. Every row carries prior_fight_ids (read by assert_no_leak) and the
    EVIDENCE_COLS that apply_shrinkage needs.
    """
    era_start = _as_date(config["roster"]["era_start"])
    min_prior = int(config["history"]["min_prior_fights"])
    min_bouts = int(config["roster"]["min_ufc_bouts"])
    strategy = config["history"]["strategy"]
    scope = config["roster"].get("bout_count_scope", "in_era")

    fb = _fighter_bouts(fights).sort_values(
        ["fighter_id", "date", "fight_id"], kind="stable").reset_index(drop=True)

    # --- roster gate ---------------------------------------------------------
    counted = fb[fb["date"] >= era_start] if scope == "in_era" else fb
    bout_counts = counted.groupby("fighter_id").size()
    roster = set(bout_counts[bout_counts >= min_bouts].index)
    print(f"Roster: {len(roster)} fighters with {min_bouts}+ bouts ({scope} count).")

    # --- per-fighter lookups, built once -------------------------------------
    pb = per_bout if "date" in per_bout.columns else per_bout.merge(
        fights[["fight_id", "date"]], on="fight_id", how="left")
    assert pb["date"].notna().all(), "per_bout rows reference fights with no date"

    pb_by_fighter = {fid: g for fid, g in pb.groupby("fighter_id", sort=False)}
    st_by_fighter = {fid: g for fid, g in fight_stats.groupby("fighter_id", sort=False)}
    fighter_rows = fighters.set_index("fighter_id")
    duration_by_fight = fights.set_index("fight_id")["duration_seconds"]
    phys_ref = _weight_class_reference(fighters, fb)

    rows = []
    key_signature = None

    for fighter_id, career in fb.groupby("fighter_id", sort=False):
        if fighter_id not in roster:
            continue

        career = career.reset_index(drop=True)
        pb_f = pb_by_fighter.get(fighter_id)
        st_f = st_by_fighter.get(fighter_id)
        fighter = fighter_rows.loc[fighter_id]

        for i, bout in enumerate(career.itertuples(index=False)):
            if bout.date < era_start:
                continue

            priors = career.iloc[:i]
            priors = priors[priors["date"] < bout.date]          # strict, see contract
            if strategy == "truncate_at_cutoff":
                priors = priors[priors["date"] >= era_start]
            if len(priors) < min_prior:
                continue

            # A catch-weight or open-weight bout has no division, so "percentile
            # within weight class" has no reference population. Fall back to the
            # fighter's most recent CLASSIFIED prior bout -- priors only, so the
            # fallback cannot leak. 73 in-era bouts need this.
            percentile_class = bout.weight_class
            if pd.isna(percentile_class):
                classified = priors["weight_class"].dropna()
                percentile_class = classified.iloc[-1] if len(classified) else None

            prior_ids = tuple(priors["fight_id"])
            prior_set = set(prior_ids)

            prior_bout_vectors = (pb_f[pb_f["fight_id"].isin(prior_set)]
                                  if pb_f is not None else pb.iloc[0:0])
            prior_stats = (st_f[st_f["fight_id"].isin(prior_set)]
                           if st_f is not None else fight_stats.iloc[0:0])

            evidence = {
                "n_prior_bouts": float(len(prior_ids)),
                "n_prior_sig_att": float(np.nansum(prior_stats["sig_str_att"])),
                "n_prior_td_att": float(np.nansum(prior_stats["td_att"])),
                "n_prior_minutes": float(np.nansum(duration_by_fight.loc[list(prior_ids)])) / 60.0,
            }

            features = build_feature_row(
                fighter_id=fighter_id,
                fight_id=bout.fight_id,
                date=bout.date,
                prior_fights=priors,
                prior_stats=prior_stats,
                per_bout=prior_bout_vectors,
                weight_class=bout.weight_class,
                percentile_class=percentile_class,
                fighter=fighter,
                phys_ref=phys_ref,
                config=config,
            )
            assert isinstance(features, dict), "build_feature_row must return a dict"

            # A key that appears conditionally becomes a NaN column for every
            # other fighter, which is indistinguishable from a real missing value.
            if key_signature is None:
                key_signature = set(features)
                stray = {k for k in key_signature
                         if not any(k.startswith(p) for p in BLOCK_PREFIXES.values())}
                assert not stray, (
                    f"feature keys must carry a block prefix {sorted(BLOCK_PREFIXES.values())}: "
                    f"{sorted(stray)}"
                )
                clash = key_signature & (set(META_COLS) | set(EVIDENCE_COLS))
                assert not clash, f"feature keys collide with reserved columns: {sorted(clash)}"
            else:
                assert set(features) == key_signature, (
                    f"feature keys differ for fighter {fighter_id} fight {bout.fight_id}: "
                    f"+{sorted(set(features) - key_signature)} "
                    f"-{sorted(key_signature - set(features))}"
                )

            # Provenance is the plumbing's to own. If the feature row reports its
            # own, it has to agree -- "the fights you used are the fights I gave you".
            reported = features.pop("prior_fight_ids", None)
            if reported is not None:
                assert tuple(reported) == prior_ids, (
                    f"build_feature_row used different prior fights than it was handed "
                    f"(fighter {fighter_id}, fight {bout.fight_id})"
                )

            rows.append({
                "fighter_id": fighter_id,
                "fight_id": bout.fight_id,
                "date": bout.date,
                "weight_class": bout.weight_class,
                "percentile_class": percentile_class,
                "opponent_id": bout.opponent_id,
                "prior_fight_ids": prior_ids,
                **evidence,
                **features,
            })

    snapshots_df = pd.DataFrame(rows)
    assert not snapshots_df.empty, "no snapshots produced -- check era_start and the roster gate"
    assert not snapshots_df.duplicated(subset=["fighter_id", "fight_id"]).any(), \
        "duplicate (fighter_id, fight_id) snapshots"

    print(f"Snapshots: {len(snapshots_df)} rows, "
          f"{snapshots_df['fighter_id'].nunique()} fighters, strategy={strategy}.")
    return snapshots_df


# Columns aggregated out of per_bout, by block. ctrl_share is style, not quality:
# the decision to grapple is what it mostly measures, and td_acc already covers
# whether he is good at it (DATA_NOTES).
PROP_FROM_PER_BOUT = [
    "head_share", "body_share", "leg_share",
    "distance_share", "clinch_share", "ground_share",
    "td_share", "sub_share", "strike_share", "sig_str_share", "ctrl_share",
    "sig_str_pace", "total_off_pace", "td_pace",
]
RATE_FROM_PER_BOUT = ["sig_str_acc", "td_acc", "kd_pace", "rev_pace"]

# Career average vs last-3-bouts, for the features that actually evolve. The gap
# between the two columns is the evolution signal; the model finds it itself.
RECENT_WINDOW = 3
RECENT_COLS = ["head_share", "body_share", "leg_share",
               "distance_share", "clinch_share", "ground_share", "sig_str_pace"]

# Spread across prior bouts. Shares are already on a common [0,1] scale so a raw
# std is comparable between them; pace is not, so it uses a coefficient of
# variation instead of being standardised against a population it cannot see.
DISPERSION_FAMILIES = {
    "target": ["head_share", "body_share", "leg_share"],
    "position": ["distance_share", "clinch_share", "ground_share"],
    "offense": ["td_share", "sub_share", "strike_share"],
}


def _wmean(x: pd.Series, w: pd.Series) -> float:
    """
    Duration-weighted mean, skipping rows where either side is missing.

    Masking both sides matters: (x * w).sum() treats a NaN x as 0 while w.sum()
    still counts that bout's weight, which drags the result toward zero.
    """
    m = x.notna() & w.notna()
    total = w[m].sum()
    if total <= 0:
        return float("nan")
    return float((x[m] * w[m]).sum() / total)


def build_feature_row(fighter_id: str, fight_id: str, date: pd.Timestamp,
                      prior_fights: pd.DataFrame, prior_stats: pd.DataFrame,
                      per_bout: pd.DataFrame, weight_class: str,
                      percentile_class: str, fighter: pd.Series, phys_ref: dict,
                      config: dict) -> dict:
    """
    Aggregate one fighter's PRIOR bouts into a single feature row.

    Everything handed in is already restricted to this fighter's prior fights:
      prior_fights  fighter-bout rows: fight_id, date, weight_class,
                    duration_seconds, title_bout, method, opponent_id, won
                    (won is NaN for no-contest/draw, not False)
      prior_stats   raw per-ROUND fight_stats for those fights
      per_bout      compute_per_bout_vectors output for those fights
      percentile_class  the division to take physical percentiles in -- use this
                    for phys_ lookups, not weight_class
      fighter       that fighter's row from fighters.csv
      phys_ref      reference populations for percentile_in_wc

    Nothing about the CURRENT fight may enter except its id, date and
    weight_class. Returns a flat dict of features, keys prefixed by block:

      prop_  proportions, from the prior per-bout vectors. The aggregation
             choice is real -- mean, attempt-weighted and recency-weighted make
             different claims about what a fighter's style is.
      rate_  per-minute and accuracy figures. Denominator is summed prior
             duration_seconds. Guard every one; an inf fails assert_feature_sanity.
      phys_  percentile_in_wc(...) only, never raw inches. Stance is a frozen
             probe and must not enter.
      disp_  spread of the prior per-bout vectors around their own centroid.
             Decide which subspace, and whether to standardise before taking
             distances. You always have >= 3 priors, so a NaN here is a bug.

    Respects config.features.blocks: a disabled block emits none of its keys.
    build_snapshots attaches prior_fight_ids and the n_prior_* counts itself.
    """
    blocks = config["features"]["blocks"]
    out = {}

    pb = per_bout.sort_values("date", kind="stable") if "date" in per_bout.columns else per_bout
    w = pb["duration_seconds"] if "duration_seconds" in pb.columns else pd.Series(dtype=float)

    # --- proportions: what he chooses to do ---------------------------------
    if blocks.get("proportions"):
        for col in PROP_FROM_PER_BOUT:
            out[f"prop_{col}"] = _wmean(pb[col], w) if col in pb.columns else float("nan")

        recent = pb.tail(RECENT_WINDOW)
        w_recent = recent["duration_seconds"] if len(recent) else pd.Series(dtype=float)
        for col in RECENT_COLS:
            out[f"prop_recent_{col}"] = (_wmean(recent[col], w_recent)
                                         if col in recent.columns and len(recent)
                                         else float("nan"))

    # --- rates: how well it works -------------------------------------------
    if blocks.get("rates"):
        for col in RATE_FROM_PER_BOUT:
            out[f"rate_{col}"] = _wmean(pb[col], w) if col in pb.columns else float("nan")

        won = prior_fights["won"]
        out["rate_win_rate"] = float(won.mean()) if won.notna().any() else float("nan")

        method = prior_fights["method"].astype(str)
        decided = won.notna()
        finishes = decided & ~method.str.startswith("Decision")
        out["rate_finish_rate"] = (float(finishes[decided].mean())
                                   if decided.any() else float("nan"))

        minutes = prior_fights["duration_seconds"].astype(float) / 60.0
        out["rate_avg_fight_minutes"] = float(minutes.mean()) if len(minutes) else float("nan")

        # Exact counts, not estimates -- see the note in DATA_NOTES about these
        # being shrunk along with the rest of the block.
        out["rate_n_prior_bouts"] = float(len(prior_fights))
        out["rate_career_minutes"] = float(minutes.sum())

    # --- physical: context ---------------------------------------------------
    if blocks.get("physical"):
        out["phys_reach_pct"] = percentile_in_wc(
            fighter.reach_in, percentile_class, "reach_in", phys_ref)
        out["phys_height_pct"] = percentile_in_wc(
            fighter.height_in, percentile_class, "height_in", phys_ref)

    # --- dispersion: how much he varies --------------------------------------
    if blocks.get("dispersion"):
        for name, cols in DISPERSION_FAMILIES.items():
            have = [c for c in cols if c in pb.columns]
            stds = [pb[c].std(ddof=0) for c in have if pb[c].notna().sum() >= 2]
            out[f"disp_{name}"] = float(np.mean(stds)) if stds else float("nan")

        pace = pb["sig_str_pace"] if "sig_str_pace" in pb.columns else pd.Series(dtype=float)
        mu = pace.mean()
        out["disp_pace_cv"] = (float(pace.std(ddof=0) / mu)
                               if pace.notna().sum() >= 2 and mu and mu > 0
                               else float("nan"))

    return out


# ============================================================================
# SECTION 4: SHRINKAGE
# ============================================================================

def _evidence_for(feature: str) -> str:
    """Which n counts as evidence for this feature. See EVIDENCE_RULES."""
    if feature in EVIDENCE_OVERRIDES:
        return EVIDENCE_OVERRIDES[feature]
    name = feature.lower()
    for needles, evidence_col in EVIDENCE_RULES:
        if any(needle in name for needle in needles):
            return evidence_col
    return "n_prior_bouts"


def apply_shrinkage(snapshots_df: pd.DataFrame, feature_cols: list,
                    config: dict) -> pd.DataFrame:
    """
    Shrink small-sample features toward the population mean.

        shrunk = n/(n+k) * observed + k/(n+k) * mean

    n is the fighter's evidence for that feature (attempts, minutes or bouts),
    k comes from config per evidence family. The mean is computed on train rows
    only. Proportions and rates only -- see SHRINKABLE_BLOCKS.
    """
    shrink_cfg = config["features"].get("shrinkage", {})
    if not shrink_cfg.get("enabled", False):
        print("Shrinkage disabled (features.shrinkage.enabled: false).")
        return snapshots_df

    assert "split" in snapshots_df.columns, "apply_shrinkage needs the split column"
    missing = [c for c in EVIDENCE_COLS if c not in snapshots_df.columns]
    assert not missing, f"evidence columns missing from snapshots: {missing}"

    k_default = float(shrink_cfg["k"])
    k_by_evidence = {**DEFAULT_K_BY_EVIDENCE, **(shrink_cfg.get("k_by_evidence") or {})}

    prefixes = tuple(BLOCK_PREFIXES[b] for b in SHRINKABLE_BLOCKS)
    targets = [c for c in feature_cols
               if c.startswith(prefixes) and c not in NOT_SHRINKABLE]
    skipped = [c for c in feature_cols if c not in targets]

    out = snapshots_df.copy()
    train = out["split"] == "train"
    assert train.any(), "no train rows to compute population means from"

    report = []
    for col in targets:
        evidence_col = _evidence_for(col)
        k = k_by_evidence.get(evidence_col) or k_default
        k = float(k)

        population_mean = float(out.loc[train, col].mean())   # train only, NaN-skipping
        n = out[evidence_col].fillna(0.0).astype(float)
        weight = n / (n + k)

        observed = out[col]
        shrunk = weight * observed + (1.0 - weight) * population_mean
        # n == 0 makes the formula's limit the population mean, but NaN * 0 is
        # NaN, so substitute it. This is what carries rate_td_acc for the 35% of
        # fighters who never shot a takedown. A NaN WITH evidence is a bug and
        # is left alone so nan_policy catches it.
        out[col] = shrunk.where(~(observed.isna() & (n <= 0)), population_mean)
        report.append((col, evidence_col, k, float(weight.median())))

    print(f"Shrinkage: {len(targets)} columns shrunk, {len(skipped)} left alone "
          f"({', '.join(sorted({c.split('_')[0] + '_' for c in skipped})) or 'none'}).")
    print(f"  {'feature':32s} {'evidence':18s} {'k':>6s} {'median w':>9s}")
    for col, evidence_col, k, median_w in report:
        print(f"  {col:32s} {evidence_col:18s} {k:6.1f} {median_w:9.3f}")
    print("  w -> 1 keeps the fighter's own number; w -> 0 replaces it with the "
          "population mean.")
    return out


# ============================================================================
# SECTION 5: SPLITS AND SCALING
# ============================================================================

def select_feature_cols(snapshots_df: pd.DataFrame, config: dict) -> list:
    """
    The explicit feature allowlist, in a stable order, honouring block toggles.

    Block order then alphabetical within block, so the order is reproducible
    across runs -- feature_cols.json is a contract with src/serve/handler.py and
    must not shuffle because a dict iterated differently.

    A disabled block contributes nothing; an ENABLED block that emitted no
    columns is an error, because that is how a silently-missing feature block
    turns into an ablation that "shows no effect".
    """
    blocks = config["features"]["blocks"]
    feature_cols = []
    for block, prefix in BLOCK_PREFIXES.items():
        present = sorted(c for c in snapshots_df.columns if c.startswith(prefix))
        if not blocks.get(block, False):
            if present:
                print(f"  {block}: {len(present)} columns present but block disabled, dropping")
            continue
        assert present, f"features.blocks.{block} is true but no {prefix}* columns were emitted"
        feature_cols.extend(present)

    reserved = set(META_COLS) | set(EVIDENCE_COLS)
    clash = reserved & set(feature_cols)
    assert not clash, f"feature columns collide with reserved names: {sorted(clash)}"

    print(f"Features: {len(feature_cols)} columns across "
          f"{sum(1 for b, on in blocks.items() if on)} enabled blocks.")
    return feature_cols


def apply_temporal_split(snapshots_df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Assign train/val/test by the current fight's date, never randomly.

    Every row lands in exactly one split; assert_split_covers_all_rows enforces
    it. A fighter appearing in two splits is fine -- the unit is the snapshot.
    """
    out = snapshots_df.copy()
    out["split"] = pd.NA

    for name in ("train", "val", "test"):
        start, end = (_as_date(b) for b in config["splits"][name]["date_range"])
        in_range = out["date"].between(start, end)
        overlap = in_range & out["split"].notna()
        assert not overlap.any(), f"{int(overlap.sum())} rows matched two split ranges"
        out.loc[in_range, "split"] = name

    unassigned = out["split"].isna()
    if unassigned.any():
        dates = out.loc[unassigned, "date"]
        raise AssertionError(
            f"{int(unassigned.sum())} snapshots fall outside every split range "
            f"({dates.min().date()} to {dates.max().date()}). Extend splits.test "
            f"rather than letting them disappear."
        )

    out["split"] = out["split"].astype(str)
    return out


def fit_and_save_scaler(snapshots_df: pd.DataFrame, feature_cols: list,
                        config: dict) -> StandardScaler:
    """
    Fit StandardScaler on train rows only and persist it.

    The parquet files hold RAW values; consumers apply the scaler at load time.
    feature_cols is an explicit allowlist and is saved beside the scaler --
    src/serve/handler.py needs that exact column order at inference.

    NaN handling follows features.nan_policy (default "error").
    """
    out_dir = Path(config["paths"]["snapshots"])
    out_dir.mkdir(parents=True, exist_ok=True)

    nan_policy = config["features"].get("nan_policy", "error")
    nan_counts = snapshots_df[feature_cols].isna().sum()
    offenders = nan_counts[nan_counts > 0]

    if not offenders.empty:
        if nan_policy == "error":
            raise AssertionError(
                f"NaN in feature columns, and features.nan_policy is 'error': "
                f"{offenders.to_dict()}. StandardScaler would pass them through and "
                f"PCA would fall over on day 5. Fix the feature or set "
                f"features.nan_policy: train_median."
            )
        medians = snapshots_df.loc[snapshots_df["split"] == "train", feature_cols].median()
        snapshots_df[feature_cols] = snapshots_df[feature_cols].fillna(medians)
        print(f"  imputed train medians into {len(offenders)} columns: {list(offenders.index)}")

    train = snapshots_df.loc[snapshots_df["split"] == "train", feature_cols]
    assert not train.empty, "train split is empty"

    scaler = StandardScaler().fit(train)   # DataFrame in, so feature_names_in_ is recorded

    with open(out_dir / "scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    with open(out_dir / "feature_cols.json", "w") as f:
        json.dump(list(feature_cols), f, indent=2)

    print(f"Scaler fitted on {len(train)} train rows, {len(feature_cols)} columns -> {out_dir}")
    return scaler


# ============================================================================
# SECTION 6: VALIDATION
# ============================================================================

def assert_no_leak(snapshots_df: pd.DataFrame, fights: pd.DataFrame) -> bool:
    """
    The assertion the whole project rests on.

    Contract:
      - Every fight_id in a row's prior_fight_ids is dated strictly before that
        row's own fight date.
      - A row's own fight_id never appears in its prior_fight_ids.
      - prior_fight_ids is non-empty and has at least min_prior_fights entries.

    This checks provenance, not sort order. The boilerplate version compared
    dates inside an already-sorted frame and was structurally incapable of
    failing, which is worse than having no assertion at all.
    """
    assert "prior_fight_ids" in snapshots_df.columns, \
        "prior_fight_ids missing -- build_feature_row must record which fights fed each row"

    fight_dates = fights.set_index("fight_id")["date"]
    violations = []

    for row in snapshots_df.itertuples(index=False):
        priors = list(row.prior_fight_ids)

        assert priors, f"empty prior_fight_ids: fighter {row.fighter_id} fight {row.fight_id}"
        assert row.fight_id not in priors, \
            f"SELF-LEAK: fight {row.fight_id} is listed among its own prior fights"

        unknown = set(priors) - set(fight_dates.index)
        assert not unknown, \
            f"prior_fight_ids references unknown fights: {sorted(unknown)[:3]}"

        latest_prior = fight_dates.loc[priors].max()
        if latest_prior >= row.date:
            violations.append((row.fighter_id, row.fight_id, row.date, latest_prior))

    assert not violations, (
        f"LEAK DETECTED in {len(violations)} rows. First: fighter {violations[0][0]} "
        f"fight {violations[0][1]} dated {violations[0][2].date()} used a prior fight "
        f"dated {violations[0][3].date()}"
    )

    print(f"✓ Leak-free: {len(snapshots_df)} rows, every prior fight strictly earlier.")
    return True


def assert_partitions_sum_to_one(per_bout: pd.DataFrame, tol: float = 1e-6) -> bool:
    """
    Per-bout share families are genuine partitions.

    Contract:
      - head + body + leg == 1 and distance + clinch + ground == 1, for rows
        where the partition is defined.
      - Every *_share column lies in [0, 1].
      - Unique on (fight_id, fighter_id).

    Checks whichever of the two families are present, so it stays useful while
    the proportions block is still being written.
    """
    assert not per_bout.duplicated(subset=["fight_id", "fighter_id"]).any(), \
        f"{per_bout.duplicated(subset=['fight_id', 'fighter_id']).sum()} duplicate (fight, fighter) rows"

    share_cols = [c for c in per_bout.columns if c.endswith("_share")]
    assert share_cols, "no *_share columns found -- is the proportions block written?"

    for col in share_cols:
        vals = per_bout[col].dropna()
        assert ((vals >= -tol) & (vals <= 1 + tol)).all(), \
            f"{col} outside [0, 1]: min {vals.min():.4f}, max {vals.max():.4f}"

    families = {
        "location": ["head_share", "body_share", "leg_share"],
        "position": ["distance_share", "clinch_share", "ground_share"],
    }
    for name, cols in families.items():
        if not all(c in per_bout.columns for c in cols):
            print(f"  ({name} partition not present yet, skipped)")
            continue
        sub = per_bout[cols].dropna()
        if sub.empty:
            continue
        total = sub.sum(axis=1)
        bad = (total - 1.0).abs() > 1e-3
        assert not bad.any(), \
            f"{name} partition does not sum to 1 in {bad.sum()} rows (worst: {total[bad].iloc[0]:.4f})"

    print(f"✓ Partitions valid: {len(share_cols)} share columns, {len(per_bout)} bouts.")
    return True


def assert_no_constant_features(snapshots_df: pd.DataFrame,
                                feature_cols: list) -> bool:
    """
    No feature column is constant.

    A constant column has zero variance and contributes literally nothing to PCA,
    an autoencoder, or a k-NN probe -- but it still occupies a d_in slot and looks
    like a real feature in every table you print.

    This exists because the boilerplate shipped reach_percentile_in_wc = 0.5,
    height_percentile_in_wc = 0.5 and style_dispersion = 0.0 as literal
    placeholders. Four of its ten features were constants. Catch that at build
    time rather than wondering on day 7 why the physical block ablation does
    nothing.
    """
    constant, near_constant = [], []
    for col in feature_cols:
        vals = snapshots_df[col].dropna()
        if vals.empty:
            constant.append(f"{col} (all NaN)")
        elif vals.nunique() == 1:
            constant.append(f"{col} (= {vals.iloc[0]})")
        elif float(vals.std()) < 1e-8:
            near_constant.append(col)

    assert not constant, f"constant feature columns: {constant}"
    assert not near_constant, f"near-zero-variance columns: {near_constant}"

    print(f"✓ All {len(feature_cols)} feature columns vary.")
    return True


def assert_feature_sanity(snapshots_df: pd.DataFrame, feature_cols: list) -> bool:
    """
    Features are finite and their NaN pattern is deliberate.

    Contract:
      - No inf / -inf anywhere. An inf means a zero denominator survived, which a
        later StandardScaler turns into NaN across the whole column.
      - Any column that is more than 5% NaN is reported. That is not automatically
        wrong (dispersion is undefined at one prior bout, 37 fighters have no
        reach), but it must be a decision you made, not one you discover.
    """
    numeric = snapshots_df[feature_cols].select_dtypes(include=[np.number])

    inf_counts = np.isinf(numeric).sum()
    offenders = inf_counts[inf_counts > 0]
    assert offenders.empty, \
        f"infinite values -- check for zero denominators: {offenders.to_dict()}"

    nan_frac = snapshots_df[feature_cols].isna().mean().sort_values(ascending=False)
    flagged = nan_frac[nan_frac > 0.05]
    if not flagged.empty:
        print("  NaN above 5% (confirm each is intentional):")
        for col, frac in flagged.items():
            print(f"    {col:32s} {100 * frac:5.1f}%")

    print(f"✓ Feature sanity: {len(feature_cols)} columns, no infinities.")
    return True


def assert_split_covers_all_rows(snapshots_df: pd.DataFrame) -> bool:
    """
    Every snapshot lands in exactly one split, and none is empty.

    The boilerplate initialised split to "unknown" and main() wrote out only
    train/val/test, so out-of-range rows disappeared silently and uncounted.
    """
    assert "split" in snapshots_df.columns, "split column missing"

    valid = {"train", "val", "test"}
    counts = snapshots_df["split"].value_counts()
    stray = set(counts.index) - valid

    assert not stray, (
        f"{snapshots_df['split'].isin(stray).sum()} rows in unexpected splits {stray} "
        f"-- these would be dropped without a trace"
    )
    for name in valid:
        assert counts.get(name, 0) > 0, f"{name} split is empty"

    print(f"✓ Split covers all {len(snapshots_df)} rows: "
          f"train {counts['train']}, val {counts['val']}, test {counts['test']}.")
    return True


def assert_scaler_roundtrip(scaler: StandardScaler, snapshots_df: pd.DataFrame,
                            feature_cols: list, config: dict) -> bool:
    """
    The persisted scaler can actually be reused at inference.

    Contract:
      - scaler.pkl and feature_cols.json both exist on disk.
      - The saved column list matches feature_cols exactly, in ORDER.
      - The scaler was fitted on that same width.
      - inverse_transform(transform(X)) recovers X, which catches a scaler fitted
        on a different column set than the one that got saved.
      - Train columns are standardised: mean ~0, std ~1.
    """
    out_dir = Path(config["paths"]["snapshots"])

    scaler_path, cols_path = out_dir / "scaler.pkl", out_dir / "feature_cols.json"
    assert scaler_path.exists(), f"{scaler_path} missing"
    assert cols_path.exists(), \
        f"{cols_path} missing -- serve/handler.py cannot rebuild the column order without it"

    with open(cols_path) as f:
        saved_cols = json.load(f)
    assert saved_cols == list(feature_cols), \
        "feature_cols.json does not match the columns actually scaled (order matters)"
    assert scaler.n_features_in_ == len(feature_cols), \
        f"scaler fitted on {scaler.n_features_in_} features, {len(feature_cols)} given"

    with open(scaler_path, "rb") as f:
        reloaded = pickle.load(f)

    sample = snapshots_df[feature_cols].head(100)
    recovered = reloaded.inverse_transform(reloaded.transform(sample))
    assert np.allclose(recovered, sample.to_numpy(), equal_nan=True, atol=1e-8), \
        "scaler round-trip failed -- fitted on different columns than were saved"

    train = snapshots_df.loc[snapshots_df["split"] == "train", feature_cols]
    scaled = reloaded.transform(train)
    assert np.abs(np.nanmean(scaled, axis=0)).max() < 1e-6, "train mean is not ~0 after scaling"
    assert np.abs(np.nanstd(scaled, axis=0) - 1).max() < 1e-3, "train std is not ~1 after scaling"

    print(f"✓ Scaler round-trips, {len(feature_cols)} columns persisted in order.")
    return True


def assert_row_counts(per_bout: pd.DataFrame, snapshots_df: pd.DataFrame) -> bool:
    """
    Output volume is in the range the day-2 scope analysis predicted.

    41,506 per-round stat rows collapse to roughly 17.6k fighter-bouts, and
    5+ / 2014 / extend was measured at ~969 fighters and ~7,751 snapshots. A
    large miss means a join dropped rows or a filter ran in the wrong order.
    """
    assert len(per_bout) > 15000, f"per_bout too small: {len(per_bout)} rows (expected ~17.6k)"
    assert len(snapshots_df) > 6000, f"snapshots too small: {len(snapshots_df)} rows (expected ~7.8k)"

    n_fighters = snapshots_df["fighter_id"].nunique()
    assert n_fighters > 800, f"only {n_fighters} fighters (expected ~969)"

    print(f"✓ Row counts plausible: {len(per_bout)} bouts, {len(snapshots_df)} snapshots, "
          f"{n_fighters} fighters, {snapshots_df['date'].min().date()} to "
          f"{snapshots_df['date'].max().date()}.")
    return True


# ============================================================================
# SECTION 7: MAIN
# ============================================================================

def write_outputs(per_bout: pd.DataFrame, snapshots_df: pd.DataFrame,
                  feature_cols: list, config: dict) -> None:
    """
    Persist per-bout vectors, the three splits, and a manifest.

    The manifest is what makes a result traceable: data/eval/*.json is a flat
    table keyed by model name, so without the vintage recorded next to the
    features, "contrastive beat AE by 0.03" cannot be separated from "we
    re-scraped in between".
    """
    per_bout_dir = Path(config["paths"]["per_bout"])
    snap_dir = Path(config["paths"]["snapshots"])
    per_bout_dir.mkdir(parents=True, exist_ok=True)
    snap_dir.mkdir(parents=True, exist_ok=True)

    per_bout.to_parquet(per_bout_dir / "per_bout_vectors.parquet", index=False)

    # parquet stores a tuple as a list; assert_no_leak does list(...) either way.
    writable = snapshots_df.copy()
    writable["prior_fight_ids"] = writable["prior_fight_ids"].apply(list)

    counts = {}
    for name in ("train", "val", "test"):
        part = writable[writable["split"] == name]
        part.to_parquet(snap_dir / f"{name}.parquet", index=False)
        counts[name] = len(part)

    manifest = {
        "snapshot_date": config["paths"].get("snapshot_date"),
        "era_start": str(config["roster"]["era_start"]),
        "min_ufc_bouts": config["roster"]["min_ufc_bouts"],
        "bout_count_scope": config["roster"].get("bout_count_scope", "in_era"),
        "min_prior_fights": config["history"]["min_prior_fights"],
        "history_strategy": config["history"]["strategy"],
        "blocks_enabled": {b: bool(v) for b, v in config["features"]["blocks"].items()},
        "shrinkage": config["features"].get("shrinkage", {}),
        "nan_policy": config["features"].get("nan_policy", "error"),
        "n_per_bout_rows": int(len(per_bout)),
        "n_snapshots": int(len(snapshots_df)),
        "n_fighters": int(snapshots_df["fighter_id"].nunique()),
        "split_counts": counts,
        "n_features": len(feature_cols),
        "feature_cols": list(feature_cols),
    }
    with open(snap_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nWrote {per_bout_dir / 'per_bout_vectors.parquet'} ({len(per_bout)} rows)")
    for name, n in counts.items():
        print(f"Wrote {snap_dir / (name + '.parquet')} ({n} rows)")
    print(f"Wrote {snap_dir / 'manifest.json'}")
    print(f"\nSet training.autoencoder.d_in and training.contrastive.d_in to "
          f"{len(feature_cols)} in configs/v1.yaml (28 is a placeholder).")


def main():
    """
    Order matters. Shrinkage needs the split (population means come from train
    only), and the scaler needs the shrunk values, so:

        config -> raw -> per_bout -> [validate] -> snapshots -> [assert_no_leak]
          -> split -> [validate] -> shrinkage -> scaler -> [validate] -> write

    Writes per_bout_vectors.parquet, {train,val,test}.parquet, scaler.pkl,
    feature_cols.json and manifest.json.
    """
    config = load_config()
    events, fights, fight_stats, fighters = load_raw_data(config)

    per_bout = compute_per_bout_vectors(fights, fight_stats)
    assert_partitions_sum_to_one(per_bout)

    snapshots_df = build_snapshots(fights, fight_stats, events, fighters, per_bout, config)
    assert_no_leak(snapshots_df, fights)

    snapshots_df = apply_temporal_split(snapshots_df, config)
    assert_split_covers_all_rows(snapshots_df)

    feature_cols = select_feature_cols(snapshots_df, config)
    assert_feature_sanity(snapshots_df, feature_cols)
    assert_no_constant_features(snapshots_df, feature_cols)

    snapshots_df = apply_shrinkage(snapshots_df, feature_cols, config)
    assert_feature_sanity(snapshots_df, feature_cols)   # shrinkage can reintroduce NaN

    scaler = fit_and_save_scaler(snapshots_df, feature_cols, config)
    assert_scaler_roundtrip(scaler, snapshots_df, feature_cols, config)
    assert_row_counts(per_bout, snapshots_df)

    write_outputs(per_bout, snapshots_df, feature_cols, config)


if __name__ == "__main__":
    main()
