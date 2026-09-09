# Day 2 — Scope Decision Memo

Run: `.venv/bin/python src/analysis/scope_count.py` · Numbers: `data/scope_analysis.json`
Snapshot: `data/raw/2026-09-01` (8,832 bouts, 4,588 fighters)

## Decisions

| | Value | Why |
|---|---|---|
| `roster.min_ufc_bouts` | **3** | 1,459 fighters in-era vs 969 at 5+. The extra 490 fighters add only 425 snapshot rows — they sit near `min_prior_fights=3`, so they contribute breadth rather than volume. Taken deliberately: the Day-5 label sample is 200 fighters stratified by era tercile and weight class, and a wider roster stratifies better. Shrinkage (`k=75`) is what absorbs the thinner per-fighter aggregates. |
| `roster.era_start` | **2014-01-01** | Keeps 71% of all bouts (6,287) at 90.4% reach coverage. 2016 buys 5 points of reach for 1,100 fewer rows. |
| `history.strategy` | **extend** | A fighter's 2012 bouts inform their 2015 style. Worth +733 rows (10%) at this cutoff. |

## Decision table

```
 era  bouts  fighters   extend  truncate     gain
2010    3+      1689     9632      9170   +462 (5%)
2012    3+      1588     9000      8411   +589 (7%)
2014    3+      1459     8176      7443   +733 (10%)   <-- chosen
2016    3+      1258     7071      6160   +911 (15%)
2017    3+      1185     6441      5474   +967 (18%)

2014    5+       969     7751      7187   +564 (8%)    (the 5+ alternative)
```

The extend gain grows with a later cutoff — the later you cut, the more history sits
behind the wall. That is the argument for extend stated as a number.

## Negative result: completeness does not decide the era

The plan assumed strike-position completeness would reveal an era knee. It does not.
`fetch_ufcstats.py` already dropped the 42 statless rows, the partition contract passes
on everything remaining, and the only bouts with zero stats are 1994–1998. Fight
statistics are complete from **1999** onward, so the check cannot discriminate between
2010 and 2017.

The era call therefore rests on bout volume and **reach coverage**, the only completeness
signal that varies inside the candidate range: 92.3% (2010) → 90.4% (2014) → 95.3% (2016)
→ 98.1% (2017). Height and stance are ~100% throughout.

## Two things to carry into Day 3

**1. Snapshot yield is ~8.2k rows, not 25k.** The plan's target came from the raw
fighter-bout count *before* `min_prior_fights=3` removes every fighter's first three
bouts. At ~8.2k rows against 25–28 features, shrinkage and dropout are load-bearing,
not optional. `training.autoencoder.d_in: 28` is still a placeholder — set it from the
real feature count once `snapshots.py` produces columns.

**2. Physical coverage falls at the recent end.** 2025 is 98.6% reach / 98.6% height /
98.7% stance; 2026 falls to 91.0 / 91.3 / 93.0. This is scrape freshness — recent
debutants are not in `fighter_tott.csv` yet — not a data-quality era effect. It lands in
the **test split** (2023-07-01 onward), so the physical block is thinner at eval time
than at train time. That is a train/test distribution shift, not a scope question.
Decide on imputation in Day 3 and re-check after any refresh.

## Reconciled: the 3+ fighter count

`DATA_NOTES.md` recorded 1,875 fighters at 3+; the adapter output gives **1,882**. The 5+
figure (1,267) reproduces exactly, so the discrepancy is specific to the 3+ row. Attempts
to reproduce 1,875 from the raw scraper CSVs give 1,884 (raw `BOUT` names), 1,881
(deduped on `URL`) and 1,880 (restricted to names present in `fighter_tott`) — none of
them 1,875. The original figure could not be reproduced by any method, so `DATA_NOTES.md`
has been corrected to the adapter-derived numbers, which are what the code actually uses.

This matters more now than it did: 3+ is the chosen threshold, so it is the number the
roster is built from.
