# Results — snapshot 2026-09-01

Every number comes from `data/eval/*.json` and is measured on the test split (fights from 2023-07-01 on).
There are 852 decided test fights and 567 test fighters.
Each model was trained once, with seed 42.

```bash
python src/eval/harness.py            # random, raw and PCA baselines
python src/models/autoencoder.py      # 4 feature-block versions
python src/models/contrastive.py      # 4 feature-block versions
python src/serve/export.py --model contrastive_style_8
```

## The table

Columns:
- **Probe columns:** balanced accuracy of a 10-nearest-neighbour guess, using folds that keep each fighter on one side.
- **Fight AUC:** the style model's score on test fights. The strength-only model, which uses just the 9 quality stats, scores **0.651**. A model on all 36 raw stats scores **0.648**.
- **vs strength:** how much the style model adds over strength-only. The brackets hold the 2.5–97.5% range from 1,000 resamples of test fights.
- **Confident cycles:** predicted A > B > C > A loops where every link is at least 55/45. They are counted out of 30.2M possible triples.

| Model | Background (chance 0.20) | Background, no hybrid (0.25) | Stance (0.33) | Weight class (0.08) | Fight AUC | vs strength | Confident cycles |
|---|---|---|---|---|---|---|---|
| random_8 | 0.191 | 0.255 | 0.334 | 0.089 | 0.648 | −0.003 [−0.008, +0.003] | 0 |
| raw_36 | 0.266 | 0.350 | 0.343 | 0.099 | 0.610 | −0.041 [−0.070, −0.014] | 29,481 |
| pca_style_8 | 0.255 | 0.326 | 0.347 | 0.089 | 0.651 | −0.000 [−0.001, +0.001] | 0 |
| pca_style_quality_8 | 0.247 | 0.307 | 0.348 | 0.088 | 0.619 | −0.032 [−0.058, −0.006] | 196,088 |
| pca_style_physical_dispersion_8 | 0.238 | 0.315 | 0.340 | 0.080 | 0.651 | pair term off | 0 |
| pca_all_8 | 0.252 | 0.306 | 0.348 | 0.091 | 0.651 | −0.000 [−0.009, +0.009] | 8 |
| ae_style_8 | 0.245 | 0.303 | 0.334 | 0.085 | 0.641 | −0.010 [−0.025, +0.004] | 13,342 |
| ae_style_quality_8 | 0.258 | 0.309 | 0.355 | 0.102 | 0.631 | −0.020 [−0.040, −0.002] | 30,356 |
| ae_style_physical_dispersion_8 | 0.256 | 0.320 | 0.341 | 0.083 | 0.643 | −0.008 [−0.017, +0.001] | 2,741 |
| ae_all_8 | 0.242 | 0.293 | 0.348 | 0.091 | 0.626 | −0.025 [−0.046, −0.004] | 22,801 |
| **contrastive_style_8** | **0.290** | **0.388** | **0.373** | 0.098 | 0.651 | pair term off | 0 |
| contrastive_style_quality_8 | 0.291 | 0.319 | 0.334 | 0.107 | 0.645 | −0.006 [−0.013, +0.001] | 684 |
| contrastive_style_physical_dispersion_8 | 0.261 | 0.351 | 0.348 | 0.087 | 0.651 | pair term off | 0 |
| contrastive_all_8 | 0.256 | 0.335 | 0.339 | 0.097 | 0.641 | −0.010 [−0.021, +0.001] | 1,660 |

"Pair term off" means val preferred no style term at all, so the fight model is the strength-only model.

## The five questions

**1. Did style embeddings beat raw stats at predicting fights?** No.
- No embedding beats the strength-only model.
- Whenever val switched the style term on, test AUC went down. For example, the autoencoder on style features scores −0.010 [−0.025, +0.004].
- On 852 fights, the 9 quality stats hold everything the fight model can use.

**2. Did martial-arts background come out without being fed in?** A little.
- The contrastive model on style features alone scores 0.290 against chance 0.20. Random scores 0.191, and five random seeds in the trust check average 0.200.
- With hybrids dropped, it scores 0.388 against 0.25.
- The raw 36 stats score 0.266 and 0.350. The best embedding beats them, but it still gets background wrong about 7 times in 10.

**3. Were there cycles?** The models predicted some, but they are not believable.
- The most came from PCA on style plus quality: 196k confident loops, 0.65% of triples.
- Every model that predicts loops also predicts test fights worse than strength-only. Nothing here shows the loops are real.
- Whether real upsets land inside those loops was not measured.

**4. Did high-dispersion fighters cluster in cycles?** Not shown.
- In models with loops, mean dispersion correlates with a fighter's share of possible loops at r = 0.17–0.25, p < 0.001. The random embedding gives 0.02.
- But dispersion also rises with the number of prior bouts (r = 0.24), which is the boring explanation.
- The loops also come from models that don't predict fights better.

**5. Which model won?** Contrastive, trained on style features only, and only on the probes.
- The autoencoder roughly ties PCA on the probes and loses to it on fights.
- Nothing wins on fights.

Sanity check on `contrastive_style_8`'s nearest neighbours:

| Fighter | Closest five |
|---|---|
| Alex Pereira | Shara Magomedov, Marcin Prachnio, Armen Petrosyan, Cesar Almeida, Hannah Cifers |
| Merab Dvalishvili | Derek Brunson, Jonathan Pearce, Dennis Bermudez, Movsar Evloev, Curtis Blaydes |
| Islam Makhachev | Jason Witt, Anthony Hernandez, Miesha Tate, Gregor Gillespie, Pat Sabatini |

Kickboxers land near Pereira and wrestlers near Merab, across weight classes and genders. That is what a style space should do.

## Decisions to challenge

**Fight model**
- Logistic regression with no intercept, so swapping A and B flips the prediction.
- Strength term: A minus B on the 9 `rate_` stats.
- Pair term: `za_i·zb_j − za_j·zb_i` on the embedding, after centering and scaling it on train-fight rows. Centering keeps a hidden strength rating out of the pair term.
- Val log loss (407 fights) picks the penalty from 1e-4 to 1 and the pair weight from {0, 0.1, 0.3, 1}. The strength-only model picks its own penalty.
- The earlier version shared one penalty between both terms, which made the style term lose by construction.

**Cycles**
- Counted exactly from the "who beats whom" matrix over each test fighter's latest snapshot, not sampled, so `cycle_sample_rate` was removed.
- An assertion checks that the strength-only model makes zero loops.

**Dispersion**
- Correlated against loops ÷ (wins × losses), because a fighter who beats everyone can't be in a loop. The raw count is saved too.

**Trust checks added**
- A random embedding must not beat strength-only on fights.
- The all-pairs matrix must match the per-fight predictions.
- The NumPy encoder must match PyTorch (max difference 5e-7).
- The served win probability must match the harness.

**Autoencoder**
- Uses the planned 32 → 16 → 8 layers with GELU and dropout 0.15.
- The best-val weights are restored at early stopping; the skeleton kept the last epoch.
- `epochs_max` went from 200 to 500 because 3 of 8 runs were still improving at 200.
- Effective rank uses squared singular values of the centered embedding.

**Contrastive**
- Positive pairs are the same fighter's snapshots 5 fights apart: 1,677 train pairs, and 531 val pairs whose later snapshot is in val.
- A fighter can appear twice in one batch and be counted as his own negative. That is not handled.
- The split-prior-fights-in-half option was not tried.

**Known weak spots**
- One seed per model.
- Only 407 val fights to tune on.
- The probe scores have no resampling range; the random seeds suggest about ±0.02 of noise.

## Serving bundle

`data/models/contrastive_style_8/serving/` holds:
- `encoder.npz`: weights `W0, b0, …` plus the scaler for the 21 style columns.
- `fight_model.npz`: `W`, `z_mean` and `z_std`. `W` is all zeros because the pair term is off.
- `fighters.json`: 969 records, each with the embedding, strength, name and background.

A fighter's record uses his latest snapshot. That snapshot was taken before his most recent fight, so it doesn't include that fight.

Serving code:
- `src/serve/inference.py`: NumPy-only math.
- `src/serve/handler.py`: the `/similar` and `/matchup` routes. Tested locally against a fake table.
- `src/serve/load_dynamodb.py`: fills the table.

## Deployment

This was deployed on AWS, following "Deploy it yourself" in the [README](README.md):
- **Serving:** NumPy-only inference, AWS Lambda (container image), Amazon API Gateway (HTTP API), Amazon DynamoDB, Amazon ECR, CloudWatch Logs, IAM
- **Infrastructure:** Terraform, Docker

It has since been torn down with `terraform destroy`, so there is no live endpoint.
