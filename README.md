# UFC Style Embedding Engine

Provably separate style from quality in fighter statistics using leak-free snapshots, hand-labeled probes, and a complete eval harness.

## Sticky Note

**"Is this measuring style or is it measuring good?"** Every design decision must answer this.

## Quick Start

```bash
# 1. Configure
cp configs/v1.yaml.template configs/v1.yaml
# Edit: era_start, min_prior, feature selection

# 2. Scrape + scope
python src/scrape/fetch_ufcstats.py
python src/analysis/scope_count.py  # decides era cutoff

# 3. Features + splits
python src/features/build_snapshots.py
python src/features/split.py

# 4. Labels + eval harness
# Label background.csv (200 fighters) by hand
python src/eval/harness.py --baseline pca --name pca_8

# 5. Models
python src/models/autoencoder.py
python src/models/contrastive.py

# 6. Serve
terraform -chdir=infra apply
python src/serve/handler.py
```

## Data Flow

```
raw/YYYY-MM-DD/
  ├─ events.csv
  ├─ fights.csv
  ├─ fight_stats.csv
  └─ fighters.csv

snapshots/v1/
  ├─ train.parquet
  ├─ val.parquet
  ├─ test.parquet

per_bout/v1/
  └─ fighters_bouts.parquet

eval/
  ├─ pca_8.json
  ├─ ae_8.json
  └─ contrastive_8.json

models/
  ├─ ae_v1/
  │  ├─ encoder_weights.npz
  │  ├─ scaler.pkl
  │  └─ config.yaml
  └─ contrastive_v1/
     ├─ encoder_weights.npz
     ├─ scaler.pkl
     └─ config.yaml
```

## Key Concepts

### Leak-Free Snapshots

For fighter F at fight N: all features come from fights 1 through N-1 only.

```python
assert all(prior_fight.date < fight.date), "LEAK DETECTED"
```

### Per-Bout Style Vectors

Before aggregating into a career vector, compute the same proportions **per individual bout**. This enables:

1. Dispersion measurement (within-fighter consistency).
2. Matchup-adaptive embedding (future work).

### Probe Set (Frozen)

Never used as model inputs, only for validation:

- Martial-arts background (wrestler / striker / BJJ / hybrid)
- Guard type (high / low / hybrid)
- Stance (orthodox / southpaw / switch)
- Weight class
- Reach percentile

If your embedding recovers these without seeing them, the geometry is real.

### Three Feature Blocks

| Block           | Concept                                            | # Features | Why Separate                               |
| --------------- | -------------------------------------------------- | ---------- | ------------------------------------------ |
| **Proportions** | Target/position distribution, grappling preference | 20–25      | Style: ratios normalize out quality        |
| **Rates**       | Per-minute, accuracy, win rate, finish rate        | 12–15      | Quality: per-minute metrics encode skill   |
| **Physical**    | Height, reach, percentiles in weight class         | 6–8        | Context: a long fighter fights differently |
| **Dispersion**  | Within-fighter spread across bouts                 | 3–4        | Adaptation: do they adjust per matchup?    |

Train ablations to see which combinations separate style from quality.

### Eval Harness Contract

Every embedding, every model, one evaluation function:

```python
evaluate(embeddings: np.ndarray,   # (n_snapshots, d)
         meta: pd.DataFrame,        # fighter_id, fight_id, date, split, ...
         features: pd.DataFrame,    # for matchup baseline
         labels: pd.DataFrame,      # probes: background, guard, stance, wc
         name: str) -> dict
```

Runs four checks:

1. **Probe recovery** (k-NN classifier on embedding, balanced accuracy)
2. **Cycle non-transitivity** (evidence of style, not quality)
3. **Matchup AUC** (gap vs. raw stats baseline, with bootstrap CI)
4. **Dispersion correlation** (high-dispersion fighters in cycles?)

---

## Design Decisions

### Era Handling

- Never hardcode a year. Config only (`train_era_start`, `history_window`).
- On day 2: plot feature completeness and distributions by year. Let data decide the floor.
- Optional: era normalization (Z-score within year cohort) instead of truncation.

### Small-Sample Shrinkage

- Empirical Bayes: shrink toward population mean in proportion to data scarcity.
- k ∈ [50, 100] (number of attempts to shrink over). Tune by checking overfit reduction.

### Temporal Splits (no leakage)

- Train: era_start to 2021-12-31
- Val: 2022-01-01 to 2023-06-30
- Test: 2023-07-01 onward
- Fit `StandardScaler` on train only. Persist alongside weights.

---

## Deferred (v2+)

- Conditional embeddings `e(fighter | opponent_style)`
- Opponent adjustment (control for opponent quality in defensive stats)
- Recency weighting in aggregate vectors
- Weight-class conditioning beyond percentiles
- Sample-size weighting in loss

---

## Tools & Versions

- Python 3.11+
- PyTorch 2.0+
- scikit-learn (PCA, shrinkage, kNN)
- pandas, numpy
- Terraform 1.5+
- AWS: S3, DynamoDB, Lambda, ECR, API Gateway

---

## Reading Order

0. `DATA_NOTES.md` — what's actually in the CSV snapshot: scope, known defects, column formats
1. `configs/v1.yaml` — all tunable parameters
2. `src/features/snapshots.py` — leak-free builder
3. `src/eval/harness.py` — the source of truth
4. `src/models/autoencoder.py` — first model
5. `src/models/contrastive.py` — second model
6. `src/serve/handler.py` — inference contract

---

## Failure Modes

1. **Leaking future information into prior features.** This kills everything silently. Assert it in code.
2. **Hardcoding era cutoff.** Make it a config value. Future you will want to retune.
3. **Building the harness after training.** Build it first against random embeddings. If it scores well, it is broken.
4. **Ignoring shrinkage.** On 5k samples with 25 features, overfit is instant. Shrink or lose.
5. **Working 10 days straight.** Take day 5 off. Fatigue + day 7 model tuning = project death.

---

## Questions?

See `GUIDELINE.md` for labeling rules (background, guard type).
See individual module docstrings for detailed contracts.
