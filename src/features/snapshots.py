"""
Build leak-free fighter snapshots.

For fighter F at fight N: every feature comes from fights 1 through N-1 only.
This module is where the data-quality contract lives.

Output:
  - data/per_bout/v1/per_bout_vectors.parquet
  - data/snapshots/v1/{train,val,test}.parquet
  - data/snapshots/v1/scaler.pkl        (fitted on train only)
  - data/snapshots/v1/feature_cols.json (column order, required by src/serve/handler.py)

Read DATA_NOTES.md before writing any feature in here. Column formats, known
defects, and the DWCS/TUF scope decisions are all documented there.

---------------------------------------------------------------------------
CARRIED-OVER DEFECTS FROM THE BOILERPLATE (fix while implementing)
---------------------------------------------------------------------------
1. events has no `date` column -- it is `DATE` (see fetch_events, which writes
   the standardised datetime back into DATE). The old build_snapshots read
   events["date"] and would KeyError immediately.

2. per_bout is keyed on (fight_id, fighter_id). The old build_feature_row
   sliced it on fight_id ALONE, so every proportion and dispersion feature
   averaged the fighter together with their opponent. Always filter on both.

3. history.strategy == "extend" was never implemented. The old code filtered
   fights to era_start BEFORE building each fighter's history, which is
   truncate. Under extend, the ERA bounds which fights get a snapshot ROW;
   it does not bound which prior fights feed that row. Worth +564 rows (8%).

4. total_fight_seconds was hardcoded (`ctrl_seconds + 1`, and `1` in the rates
   block), so every per-minute rate was meaningless. fights.duration_seconds
   is real and has zero nulls -- use it.

5. The old leak assertion compared dates within an already-sorted frame, so it
   could not fail. assert_no_leak below checks the fight_ids that actually fed
   each row instead, which is why build_feature_row must return prior_fight_ids.
"""

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.preprocessing import StandardScaler


# ============================================================================
# SECTION 1: CONFIG AND DATA LOADING
# ============================================================================

def load_config(config_path: str = "configs/v1.yaml") -> dict:
    """
    Load and validate the pipeline config.

    Contract:
      - Returns the parsed YAML as a dict.
      - Calls validate_config before returning. A config that cannot produce a
        coherent run should fail here, not nine days later inside a model.

    Args:
        config_path: path to the YAML config

    Returns:
        dict
    """
    raise NotImplementedError


def validate_config(config: dict) -> bool:
    """
    Assert the config is internally coherent before anything reads data.

    Contract:
      - Required keys present: roster.{min_ufc_bouts,era_start},
        history.{min_prior_fights,strategy}, features.blocks, splits.{train,val,test},
        paths.{raw,snapshots,per_bout}
      - era_start and every split bound parse as dates.
      - history.strategy is one of {"extend", "truncate_at_cutoff"}.
      - The three split ranges are ordered, non-overlapping, and leave no gap:
        train.end < val.start <= val.end < test.start, and each gap is one day.
      - train.start == roster.era_start. If they drift apart, snapshots exist in
        a window no split claims and get silently dropped downstream.

    Returns:
        bool: True if valid. Raises AssertionError with the offending value if not.
    """
    raise NotImplementedError


def load_raw_data(config: dict):
    """
    Load the four raw CSVs for the configured snapshot date.

    Contract:
      - Reads from paths.raw / <snapshot_date>. Pin the date in config rather
        than taking the newest directory: the day you re-scrape, every result in
        data/eval/*.json silently stops being comparable to the ones before it.
      - dtypes are declared, not inferred. Two that matter:
          * every *_id column is a 16-char hex string -- read as str, never let
            pandas guess (a hash of all digits would become an int and lose its
            leading zeros).
          * is_dwcs round-trips through CSV as the STRING "False", which is
            truthy. Parse it to real bool or the exclude filter inverts.
      - Dates are parsed here, once, not re-parsed at each call site.

    Args:
        config: validated config dict

    Returns:
        (events, fights, fight_stats, fighters) as DataFrames
    """
    raise NotImplementedError


# ============================================================================
# SECTION 2: PER-BOUT VECTORS
# ============================================================================

def compute_per_bout_vectors(fights: pd.DataFrame,
                             fight_stats: pd.DataFrame) -> pd.DataFrame:
    """
    Compute style proportions for each individual bout.

    This is the foundation. Before aggregating into a career vector, compute the
    same features per fight. That buys two things:
      1. Aggregation (mean / weighted) into the career vector.
      2. Dispersion -- spread across bouts, which is a feature in its own right.

    Contract:
      - Input fight_stats is one row per fighter per ROUND (41,506 rows). Sum to
        one row per (fight_id, fighter_id) first; expect ~17.6k rows out.
      - Unique on (fight_id, fighter_id). Assert it -- a duplicate here silently
        double-weights one bout in every downstream mean.
      - Every *_share column lies in [0, 1], and each partition family sums to
        1.0 within tolerance for rows with any strikes:
            head + body + leg == 1
            distance + clinch + ground == 1
        Rows with zero strikes landed have no defined share. Decide NaN vs 0.0
        and apply it consistently -- they mean different things to a mean().
      - Rate-shaped columns need a real denominator. Join fights.duration_seconds
        (zero nulls); do not reuse ctrl_seconds as a stand-in for fight length.
      - ctrl_seconds is NaN for 432 pre-2007 rows, and is per ROUND (0-300s), so
        it sums across rounds. NaN means unrecorded, not zero.

    Design decision to make and write down: LANDED or ATTEMPTED for each ratio.
    Attempts are the purer style signal -- they are what the fighter chose to do.
    Landed is contaminated by the opponent's defence, which is their quality, not
    this fighter's style. You may want both families.

    Args:
        fights: bout-level frame, supplies duration_seconds
        fight_stats: per-fighter-per-round stats

    Returns:
        DataFrame, one row per (fight_id, fighter_id), with the proportion block.
    """
    raise NotImplementedError


# ============================================================================
# SECTION 3: THE LEAK-FREE SNAPSHOT BUILDER
# ============================================================================

def build_snapshots(fights: pd.DataFrame, fight_stats: pd.DataFrame,
                    events: pd.DataFrame, fighters: pd.DataFrame,
                    config: dict) -> pd.DataFrame:
    """
    Build one row per fighter-at-fight, with features from PRIOR fights only.

    This is the critical function in the project. Every assertion downstream
    exists to protect this contract.

    Contract:
      - One row per (fighter_id, fight_id).
      - A row exists only if the fighter has >= history.min_prior_fights prior
        bouts AND the fight date >= roster.era_start AND the fighter clears
        roster.min_ufc_bouts career bouts.
      - Under strategy "extend", prior fights are the fighter's FULL career
        history, including bouts before era_start. Under "truncate_at_cutoff",
        history resets at era_start. The era gates which fights get a row; only
        truncate lets it gate which fights feed one.
      - Every row carries prior_fight_ids: a tuple of the fight_ids that actually
        fed its features. assert_no_leak reads this. A row whose features came
        from a fight not listed here is a lie the assertion cannot catch.
      - Expected scale at 5+ / 2014 / extend: ~969 fighters, ~7,751 rows.

    Args:
        fights, fight_stats, events, fighters: raw tables
        config: validated config

    Returns:
        DataFrame ready for shrinkage and splitting.
    """
    raise NotImplementedError


def build_feature_row(fighter_id: str, fight_id: str, date: pd.Timestamp,
                      prior_fights: pd.DataFrame, prior_stats: pd.DataFrame,
                      per_bout: pd.DataFrame, weight_class: str,
                      config: dict) -> dict:
    """
    Compute every feature for one fighter-at-fight snapshot.

    Contract:
      - Reads ONLY from prior_fights / prior_stats / the matching per_bout rows.
        Nothing about the current fight may enter except its id, date, and
        weight_class (which is known before the bout starts).
      - Filter per_bout on BOTH fight_id and fighter_id. See defect (2) above.
      - Returns a flat dict. Keys must be identical across every row -- a key
        that appears conditionally becomes a NaN column for every other fighter.
      - Includes prior_fight_ids (tuple) for the leak assertion.
      - Respects config.features.blocks: a disabled block emits none of its keys.

    The four blocks, and what each is for:

      PROPORTIONS (~20-25, STYLE). Ratios from the prior per-bout vectors.
        Normalise out quality: a wrestler and a boxer can both be accurate, the
        ratio is what separates them. The aggregation choice is a real decision,
        not a default -- mean, attempt-weighted, and recency-weighted encode
        three different claims about what a fighter's style "is".

      RATES (12-15, QUALITY). Per-minute and accuracy figures over prior bouts.
        Denominator is summed duration_seconds. These are the block that most
        directly measures "good", which is why they are ablated separately.

      PHYSICAL (6-8, CONTEXT). Percentile WITHIN weight class, not raw inches, so
        a heavyweight and a flyweight land on one scale. 37 of the 1,267-fighter
        roster have no reach on file -- decide impute vs drop and record it.
        Note stance is a frozen PROBE (see README), so it must not enter here.

      DISPERSION (3-4, ADAPTATION). Spread of the prior per-bout vectors around
        their own centroid. Needs >= 2 prior bouts. Two things to pin down:
        which subspace you measure in (proportions only, most likely), and
        whether you standardise before taking distances -- an unscaled Euclidean
        distance is dominated by whichever feature has the widest raw range.

    Returns:
        dict of identifiers + prior_fight_ids + features
    """
    raise NotImplementedError


# ============================================================================
# SECTION 4: SHRINKAGE
# ============================================================================

def apply_shrinkage(snapshots_df: pd.DataFrame, feature_cols: list,
                    config: dict) -> pd.DataFrame:
    """
    Shrink small-sample features toward the population mean (empirical Bayes).

    A fighter with 3 prior bouts and one with 20 both get a head_share, but the
    first is mostly noise. Shrink in proportion to scarcity:

        shrunk = (n / (n + k)) * observed + (k / (n + k)) * population_mean

    Contract:
      - k comes from config (features.shrinkage.k, currently 75). Do not inline it.
      - n is the fighter's evidence count for THAT feature -- attempts for an
        accuracy ratio, bouts for a per-bout mean. Using row count everywhere is
        the easy version and a weaker one.
      - The population mean must be computed on the TRAIN split only. Using all
        rows leaks test-period information into training features.
      - Applies to proportions and rates. Physical features are measurements, not
        estimates, and must not be shrunk.
      - Skip entirely when features.shrinkage.enabled is false, so the ablation
        (with vs without) is one config edit.

    Args:
        snapshots_df: snapshots with a split column already assigned
        feature_cols: the columns eligible for shrinkage
        config: validated config

    Returns:
        DataFrame with shrunk feature columns.
    """
    raise NotImplementedError


# ============================================================================
# SECTION 5: SPLITS AND SCALING
# ============================================================================

def apply_temporal_split(snapshots_df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Assign train/val/test by date. Temporal, never random.

    Contract:
      - Adds a `split` column with values in {train, val, test}.
      - EVERY row lands in exactly one split. The boilerplate defaulted rows to
        "unknown" and main() then wrote only the three named splits, so anything
        outside the ranges vanished without a count. assert_split_covers_all_rows
        enforces the opposite.
      - Split by the CURRENT fight's date. A fighter appearing in both train and
        test is expected and fine -- the unit is the snapshot, not the fighter.
        What must never happen is a train row whose features come from a fight
        dated after a test row's fight.

    Args:
        snapshots_df: all snapshots
        config: validated config

    Returns:
        Same DataFrame with the split column.
    """
    raise NotImplementedError


def fit_and_save_scaler(snapshots_df: pd.DataFrame, feature_cols: list,
                        config: dict) -> StandardScaler:
    """
    Fit StandardScaler on the train split and persist it with its column order.

    Contract:
      - Fit on train rows ONLY, then transform all splits with that fit. Fitting
        on everything leaks test distribution into training.
      - feature_cols is an explicit ALLOWLIST passed in, not "every column except
        these six". A denylist silently promotes any new metadata column into the
        feature matrix.
      - Persist feature_cols to feature_cols.json alongside scaler.pkl. The Lambda
        in src/serve/handler.py has to rebuild this exact column ORDER at
        inference; a scaler without its column list is a footgun with a nine-day
        fuse.
      - Decide the NaN policy before fitting. StandardScaler propagates NaN, and
        a NaN feature column poisons PCA and every torch loss downstream.

    Args:
        snapshots_df: snapshots with split assigned
        feature_cols: explicit list of feature columns, in order
        config: validated config

    Returns:
        The fitted StandardScaler.
    """
    raise NotImplementedError


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

def main():
    """
    Order matters. Shrinkage needs the split (population means come from train
    only), and the scaler needs the shrunk values, so:

        config -> raw -> per_bout -> [validate] -> snapshots -> [assert_no_leak]
          -> split -> [validate] -> shrinkage -> scaler -> [validate] -> write

    Writes per_bout_vectors.parquet, {train,val,test}.parquet, scaler.pkl and
    feature_cols.json. Remember to pip install pyarrow before the first write.
    """
    raise NotImplementedError


if __name__ == "__main__":
    main()
