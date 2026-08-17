# Revised 10-Day Plan: Hands-On Coding Edition

**Core principle:** You hand-write every line of feature logic, model training, and evaluation. Boilerplate scaffolding below does the heavy lifting (configs, classes, contracts). You fill in the real work.

---

## **DAYS 1–2: Scraper + Scope Count**

### Day 1 (4 hours)

**File:** `src/scrape/fetch_ufcstats.py` (skeleton provided)

**What you write:**

- `fetch_events()`: Scrape UFC events from UFCStats or similar. You decide the source and parsing logic.
- `fetch_fights()`: Scrape individual bouts. Merge with events.
- `fetch_fight_stats()`: Scrape strike/grappling stats per fighter per fight.
- `fetch_fighters()`: Scrape fighter metadata (height, reach, stance, etc.).
- Data validation: Verify partition consistency (location + position = total strikes).
  3
  **You will hit friction:** Parsing HTML/JSON, handling missing data, understanding the data schema. This is intentional. You'll understand the data deeply.

**Output:** `data/raw/YYYY-MM-DD/` with four CSV files on S3.

---

### Day 2 (2 hours)

**File:** `src/analysis/scope_count.py` (skeleton provided)

**What you write:**

1. **Roster scope:** Load CSVs, count non-DWCS UFC bouts per fighter.
   - Print: fighters at 3+, fighters at 5+, the ratio.
2. **Bouts by year:** Break down fight volume 1994–2025.
   - Print: total bouts, % after 2014.
3. **Feature completeness:** For each year, % of fights with strike position data (target/distance).
   - Find inflection: when does data stabilize >80%?

4. **Interactive decision point:**
   - Roster: use 3+ or 5+ fighters?
   - Era: start at 2010? 2012? Let completeness decide.
   - History: extend or truncate at era_start?

**Output:** Three plots, one decision memo, `configs/v1.yaml` updated by hand.

**Timebox:** 2 hours total. If you can't decide, pick the default (5+, 2014, extend).

---

## **DAYS 3–4: Snapshot Table (the core, hands-on feature engineering)**

**File:** `src/features/snapshots.py` (structure provided, you code features)

### Critical: Leak-Free Assertion

First, write this in your code:

```python
for fighter_id, snap_row in snapshots.iterrows():
    prior_fights = snap_row["prior_fights"]
    current_fight_date = snap_row["fight_date"]
    assert all(f["date"] < current_fight_date for f in prior_fights), \
        f"LEAK: fighter {fighter_id} fight {snap_row['fight_id']}"
```

Run it on every snapshot. If it fails, you have a data bug—fix it immediately.

### You Hand-Code These Feature Groups

| Block                                   | Example Features                                                                                                 | Your Job                                                                                      | Why Separate                                                                                               |
| --------------------------------------- | ---------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------- |
| **Proportions** (style, 20–25 features) | `head_share = head_landed / sig_landed`, `distance_share`, `td_share`, `pace_per_min`, `sub_att_per_control_min` | Ratios from prior bouts. Aggregate (mean). Test on single fighter first.                      | Ratios normalize out quality. A wrestler and boxer can both be accurate; ratios capture _how_ they differ. |
| **Rates** (quality, 12–15 features)     | `sig_str_per_min`, `td_landed_per_15min`, `win_rate`, `finish_rate`                                              | Per-minute, per-bout aggregates. Min/max normalize within weight class.                       | Per-minute metrics encode skill. A 10-second finish != a 5-minute grind.                                   |
| **Physical** (6–8 features)             | `reach_percentile_in_wc`, `height_percentile_in_wc`, `weight_class_encoded`                                      | Percentilize within weight class (not raw). Both HW and FW on same scale.                     | Long fighter style ≠ short fighter style. Percentile controls for this.                                    |
| **Dispersion** (3–4 features)           | `style_dispersion = mean(dist(per_bout_i, centroid))`, `rate_dispersion`                                         | Build per-bout vectors first. Then distance from each to centroid. High = adapts per matchup. | Test hypothesis: high-dispersion fighters are overrepresented in non-transitive matchups.                  |

### Workflow per feature group

1. **Design the aggregation in pseudocode** (10 min)
   - "For fighter F at fight N, use only fights 1..N-1"
   - "Compute per-bout proportion, then mean + std across bouts"

2. **Write the code** (30 min per group)
   - Test on one fighter first: print their snapshots, eyeball the numbers.
   - Then run on full roster.

3. **Assert, assert, assert** (10 min)
   - No NaNs (except where allowed, e.g., first fight has no prior).
   - Distributions look reasonable (e.g., `head_share ∈ [0, 1]`).
   - Leak assertion passes.

4. **Shrinkage:** Pick k ∈ [50, 100] (number of attempts to shrink over).
   - Empirical Bayes: shrink toward population mean.
   - Run one ablation: train without shrinkage vs. with. See which overfits less.

5. **Splits:**
   - Temporal (not random): Train ≤2021, Val 2022–mid-2023, Test mid-2023+.
   - Fit `StandardScaler` on train only. Persist alongside model later.

### Day 4 Checkpoint

- [ ] `data/per_bout/v1/per_bout_vectors.parquet` exists (25k+ rows)
- [ ] `data/snapshots/v1/{train,val,test}.parquet` all exist, no NaNs
- [ ] Leak assertion passes on all rows
- [ ] You can explain every feature in one sentence

**Timebox:** Days 3-4 are brutal. If you're stuck on a feature, move on (v2 can add it). Keep a running list of features; tick them off as you finish.

---

## **DAY 5: Labels + Eval Harness + PCA Baseline (CHECKPOINT)**

### 5a. Hand-Label 200 Fighters (2–3 hours)

**File:** `src/labeling/GUIDELINE.md` (written before labeling)

**Categories:**

1. **Wrestler:** D1 wrestler, bases game on takedown + top control
2. **Striker (Boxing):** Professional boxer, bases on punches
3. **Striker (Kickboxing/Muay Thai):** Kicks and clinch, kickboxing/MT credentials
4. **BJJ/Grappler:** High-level BJJ (purple+), submission-heavy game
5. **Hybrid:** Two systems equally strong, opponent-adaptive
6. **Unclear:** Not enough data

**How to label one fighter (10 min):**

1. Read Wikipedia + Sherdog (amateur record, credentials)
2. Watch 30–60 seconds of two highlights (what do they choose to do?)
3. Assign one category
4. Write reasoning (source: Wikipedia, Sherdog, or video)

**Reliability check:**

- After ~50 labels, re-label 20 at random one week later (without looking at old labels).
- Compute Cohen's kappa. Target: κ > 0.60. If κ < 0.50, revise guideline and re-label.

**Output:** `data/labels/background.csv` with fighter_id, background, notes, confidence.

---

### 5b. Build Eval Harness (4–5 hours)

**File:** `src/eval/harness.py` (skeleton provided)

**The core function:** `evaluate(embeddings, meta, features, labels, name) -> dict`

**Every model—random, raw, PCA, AE, contrastive—runs through this once.**

**Four checks:**

1. **Probe Recovery (k-NN on embedding)**
   - Train k-NN classifier on embedding to predict background, stance, weight class.
   - Stratified cross-val on test split. Metric: balanced accuracy.
   - Compare to random embedding (should be near 1/n_classes).
   - Gap is evidence the embedding captures style.

2. **Cycles (Non-Transitivity)**
   - Build simple matchup model: P(A beats B) = sigmoid((e_a - e_b) @ w).
   - Count triples (A, B, C) where A > B AND B > C AND C > A.
   - Report cycle rate. Pure Elo (scalar rating) produces zero cycles.
   - Non-zero rate is evidence of style effects.

3. **Matchup AUC Delta**
   - Logistic regression on raw features: AUC baseline.
   - Logistic regression on embedding difference: AUC model.
   - Gap with 95% bootstrap CI.

4. **Dispersion Correlation**
   - Hypothesis: high-dispersion fighters are overrepresented in cycle triples.
   - Spearman correlation: dispersion vs. P(appears in cycle).

**Test harness on random embedding first.** If it scores well, the harness is broken.

**Output:** `data/eval/{name}.json` with all metrics, saved automatically.

---

### 5c. PCA Baseline (1 hour)

```python
pca = PCA(n_components=8)
pca.fit(X_train_scaled)
Z = pca.transform(X_all_scaled)
evaluate(Z, meta, features, labels, "pca_8")
```

Run five times:

- Random embedding
- Raw features (no PCA)
- PCA (style-only features)
- PCA (style + quality)
- PCA (style + physical + dispersion)

**Output:** Comparison table showing which feature blocks matter.

---

## **DAYS 6–7: Autoencoder (write it, tune it, learn regularization)**

**File:** `src/models/autoencoder.py` (skeleton provided)

### 6a. Architecture + Training Loop (6 hours)

**You write the model from scratch:**

```python
class StyleAE(nn.Module):
    def __init__(self, d_in, d_latent=8):
        # Encoder: d_in -> 32 -> 16 -> d_latent
        # Decoder: d_latent -> 16 -> 32 -> d_in
        # Activation: GELU, Dropout 0.15
        pass

    def forward(self, x):
        z = self.encode(x)
        x_hat = self.decode(z)
        return x_hat, z
```

**You write the training loop:**

```python
for epoch in range(max_epochs):
    # Train on train loader
    # Validate on val loader
    # Early stop if val loss plateaus (patience 20)
```

**Do not copy code.** You will hit PyTorch friction (device placement, gradient shapes, NaN losses). Learning to debug these is the point.

**Hyperparameters (from config):**

- Optimizer: Adam, lr=1e-3, weight_decay=1e-4
- Loss: MSE
- Early stopping: patience=20

### 6b. Diagnostics (day 7, 4–5 hours)

```python
# Check 1: Dead latent dims (var < 0.01)
print(Z.std(axis=0))

# Check 2: Effective rank (how many SVD singulars explain 95%?)
U, S, _ = np.linalg.svd(Z)
eff_rank = (S.cumsum() / S.sum() < 0.95).sum()

# Check 3: Per-feature reconstruction error
X_hat = model(X_test)[0]
recon_error = ((X_test - X_hat) ** 2).mean(axis=0)
# Which features are hardest to reconstruct?
```

### Ablations: Train 4 versions

| Version                   | Features                            | Probe Recovery | Cycle Rate | Matchup AUC |
| ------------------------- | ----------------------------------- | -------------- | ---------- | ----------- |
| Style-only                | Proportions only                    | ?              | ?          | ?           |
| Style+Quality             | Proportions + Rates                 | ?              | ?          | ?           |
| Style+Physical+Dispersion | Proportions + Physical + Dispersion | ?              | ?          | ?           |
| Everything                | All blocks                          | ?              | ?          | ?           |

Fill in the ? by running harness on each. **Be honest.** If AE loses to PCA, report it.

---

## **DAY 8: Contrastive (SimCLR-style, 6–8 hours)**

**File:** `src/models/contrastive.py` (skeleton provided)

### Design Choice (1 hour): How do you create positive pairs without trivial similarity?

**Option A: Gapped pairs**

```python
# Pair snapshot i with snapshot i+5 for the same fighter
# Enough has changed (5 fights) that the pair is non-trivial
```

**Option B: Round-subsample**

```python
# Split prior-fights into two random halves
# Build two snapshots from each half, use as positive pair
```

**Option C: Feature dropout**

```python
# Same snapshot, but randomly mask 20% of dimensions
# Simple augmentation, no data engineering
```

**Pick one and implement it.** Option B pairs naturally with your per-bout table.

### 8a. Encoder + Projector (2 hours)

```python
class StyleContrastive(nn.Module):
    def __init__(self, d_in, d_latent=8, d_proj=16):
        self.encoder = StyleEncoder(d_in, d_latent)  # reuse AE encoder
        self.projector = nn.Sequential(
            nn.Linear(d_latent, d_proj),
            nn.GELU(),
            nn.Linear(d_proj, 16)
        )

    def forward(self, x):
        z = self.encoder(x)
        p = self.projector(z)
        return z, p  # z for embedding, p for loss
```

### 8b. NT-Xent Loss + Train Loop (2 hours)

```python
def nt_xent_loss(z1, z2, tau=0.1):
    # Normalize
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)

    # Cosine similarity
    logits = (z1 @ z2.T) / tau
    labels = torch.arange(len(z1))
    loss = F.cross_entropy(logits, labels)
    return loss
```

### 8c. Harness (1 hour)

Extract embeddings (use encoder, not projector) and run harness.

**Output:** `data/eval/contrastive_8.json`

---

## **DAYS 9–10: Serving + Writeup (choose one, defer the other)**

**Files:** `src/serve/handler.py` (skeleton) + `infra/main.tf` (Terraform skeleton)

### Option A: Serving (if models are settled)

1. **Export weights to NumPy** (.npz, no PyTorch in Lambda)
2. **NumPy inference** (GELU, Linear layers, no dependencies)
3. **Lambda handler:** Takes fighter name, encodes, returns top-k similar
4. **DynamoDB:** Store embeddings + metadata
5. **Terraform:** S3, DynamoDB, Lambda, API Gateway, IAM policies

**Run `terraform destroy` then `terraform apply` once.** If it comes back identical, you learned IaC.

### Option B: Writeup (if models are still tuning)

Answer with numbers:

1. Did style embeddings (proportions-only) beat raw stats on matchup prediction? With CI?
2. Did martial-arts background emerge without being fed in?
3. Were there cycles? Real upsets in cycle triples?
4. Did high-dispersion fighters cluster in cycles? Spearman r?
5. Which model won: AE, contrastive, or PCA?

---

## **Failure Modes to Avoid**

1. **Leaking future data.** Assert in code. It will destroy results silently.
2. **Overfit punishes instantly.** 5k samples, 25 features → use shrinkage + dropout.
3. **Harness tests slow.** Run on test split only (200–500 rows). Test on random first.
4. **Days 3-4 drag.** Keep feature list tight. Tick off as you code them.
5. **Working 10 days straight.** Take day 5 off. Fatigue + day 7 tuning = death.

---

## **Sticky Note**

**"Is this measuring style or is it measuring good?"**

Every feature, every model decision, every ablation answers this one question. If you can't articulate the answer, redesign.

---

## **How to Get Started**

```bash
# 1. Create environment
conda create -n ufc python=3.11
conda activate ufc

# 2. Install dependencies
pip install -r requirements.txt

# 3. Day 1: Run scraper
python src/scrape/fetch_ufcstats.py --s3-bucket my-bucket

# 4. Day 2: Scope analysis
python src/analysis/scope_count.py

# 5. Manually update configs/v1.yaml with decisions

# 6. Days 3+: Run builders and models
python src/features/snapshots.py
python src/eval/harness.py --baseline all
python src/models/autoencoder.py
python src/models/contrastive.py
```

---

You have the boilerplate. You have the plan. You have the sticky note. Go build something honest.
