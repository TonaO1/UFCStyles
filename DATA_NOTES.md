# Data Notes — UFCStats CSV Snapshot

Findings from profiling `data/scrape_ufc_stats-main/`. Everything here was verified
against the committed CSVs, not assumed. Re-verify after any refresh.

**Snapshot:** 2026-08-16 · latest event UFC 330 (2026-08-15) · 784 events · 8,859 bout
rows (8,834 unique) · 41,672 fighter-round stat rows · 4,588 fighters

**Provenance:** vendored copy of [Greco1899/scrape_ufc_stats](https://github.com/Greco1899/scrape_ufc_stats)
(GPL-3.0). Entry point is `statistics/events/completed?page=all`, which lists only
UFC-branded events. Everything below follows from that traversal.

---

## Scope: what is and isn't in here

### DWCS is absent — no filter needed

Zero hits for `contender|dana white|dwcs` across all six CSVs, including a raw byte-level
grep. Confirmed in `event_details`, `fight_details`, `fight_results`, and `fight_stats`.

DWCS exists on ufcstats.com but is not reachable from the completed-events index, so the
event → fight → stats traversal structurally cannot reach it.

**Consequence:** `exclude_dwcs: true` in `configs/v1.yaml` is a no-op, and
`flag_dwcs_events()` filters nothing. Keep the function anyway as an assertion that the
count stays zero — if upstream ever changes the index URL, silent contamination is exactly
the failure this would catch.

### The Ultimate Fighter — 28 events, all Finale cards. DO NOT FILTER.

All 28 TUF events end in "Finale". No in-house tournament events are present.

- **TUF house fights** — exhibition, unpaid, don't count on a pro record. Not in this
  dataset, same as DWCS.
- **TUF Finale cards** — live, commission-sanctioned, count on official pro records.
  **290 bouts, 481 distinct fighters, 3.3% of the dataset.** These are professional UFC
  bouts and belong in the roster.

Filtering them would delete 4 genuine UFC world title fights (Demetrious Johnson vs. Tim
Elliott; Jędrzejczyk vs. Gadelha; Esparza vs. Namajunas; Montaño vs. Modafferi),
Griffin vs. Bonnar, and real bouts from Usman, Diaz, Bisping, Evans, Dillashaw, Ferguson.

The naming misleads: `Ultimate Fighter 14 Bantamweight Tournament Title Bout` reads like a
house fight, but tournament finals were always contested live on the Finale card. Only the
earlier bracket rounds happened in the house, and those aren't here.

### Event-name families (784 events)

| Family | Count |
|---|---|
| UFC Fight Night | 372 |
| UFC (numbered) | 358 |
| The Ultimate Fighter | 28 |
| UFC on FUEL TV | 9 |
| UFC on FX | 8 |
| UFC Live | 6 |
| Noche UFC | 2 |
| `Ortiz vs Shamrock 3: The Final Chapter` | 1 |

`Ortiz vs Shamrock 3` is a legitimate 2006 UFC event that does **not** start with "UFC" —
a naive `startswith("UFC")` filter silently drops it. It is the only such case.

`UFC - Road to UFC 4.6` (2 bouts) appears in `fight_details` but has no row in
`event_details`, so it carries a `NaT` date and drops out naturally once a successful
event join is required.

---

## Data integrity

### Trailing space on every `fight_results.EVENT`

```
fight_details:  'UFC 330: Makhachev vs. Machado Garry'
fight_results:  'UFC 330: Makhachev vs. Machado Garry '   ← trailing space
```

Cause: `parse_fight_results()` reads `.text` without `.strip()`, then cleans with
`.replace('  ', '')`, which collapses only *pairs* of spaces. `parse_fight_details()` and
`combine_fighter_stats_dfs()` both use `.text.strip()`.

**Consequence:** merging `fight_results` to anything on `EVENT` matches **zero rows**,
silently. `BOUT` is unaffected.

**Fix:** join `fight_details` ↔ `fight_results` on `URL` — both tables carry it. Only
`fight_stats` lacks a URL and needs the `EVENT`+`BOUT` string join; both sides there are
already stripped by the library.

### 25 duplicate bouts from two renamed events

UFCStats renamed two cards after the fact:

- `UFC Fight Night: Grasso vs. Shevchenko 2` → `Noche UFC: Grasso vs. Shevchenko 2` (11 bouts)
- `UFC Fight Night: Lopes vs. Silva` → `Noche UFC: Lopes vs. Silva` (14 bouts)

The refresh script overwrites `event_details` but *appends* to `fight_details`,
`fight_results`, and `fight_stats`, so both names survive. Result: 25 duplicate fight URLs
in details and results, 124 duplicate rows in stats, 50 fighters affected.

The orphan-named copies have no `event_details` row, so they carry **no date**.

**This is the dangerous one.** A `NaT` passes the leak-free assertion in `snapshots.py`
without tripping it. Dedupe on `fight_id` (the URL hash) at the adapter boundary, and
assert zero `NaT` dates after the event join.

Expect this to recur on every refresh where UFCStats renames a card.

### Fighter identity — no clean solution

`fight_stats` identifies fighters by **name string only**, no URL. Mapping 2,723 distinct
names to `fighter_id` hits:

- **8 names shared by 16 different fighters** — Bruno Silva, Jean Silva, Mike Davis,
  Michael McDonald, Tony Johnson, Joey Gomez, Anthony Figueroa, Victor Valenzuela
- **15 names absent from `fighter_tott.csv`** — Patricio Pitbull, Kai Kamaka III,
  Levi Rodrigues Jr., Michael Aswell Jr., Muhammad Saidov, …

Pick a policy and **assert the unresolved count** so it cannot grow silently on refresh.
For the collisions, event date plus `WEIGHTCLASS` separates most pairs.

---

## Column formats

| Field | Format | Notes |
|---|---|---|
| `event_details.DATE` | `August 15, 2026` | `format="%B %d, %Y"` — all 784 parse |
| `fighter_tott.DOB` | `Feb 01, 1994` | `format="%b %d, %Y"` — abbreviated month; use `errors="coerce"` |
| `HEIGHT` | `5' 8"` | note the space after the apostrophe |
| `REACH` | `66"` | |
| `WEIGHT` | `155 lbs.` | |
| `STANCE` | `Orthodox` | also `Southpaw`, `Switch`, `Open Stance`, `Sideways` |
| Strike columns | `3 of 5` | `r"^\s*(\d+)\s+of\s+(\d+)"` → landed / attempted |
| `CTRL` | `3:38` or `--` | zero malformed values |
| `ROUND` | `Round 1`..`Round 5` | no summary row — library discards it |
| `OUTCOME` | `W/L`, `L/W`, `NC/NC`, `D/D` | 158 fights have no winner |
| `TIME FORMAT` | `3 Rnd (5-5-5)` | needed for per-minute denominators; don't assume 3×5 |

Missing sentinel is `"--"` for HEIGHT / REACH / WEIGHT / DOB; `STANCE` uses real `NaN`.

`WEIGHTCLASS` has **120 distinct values**, 104 of them title-bout variants
(`UFC Light Heavyweight Title Bout`). Strip the `UFC ` prefix and ` Title`/` Bout`
suffixes for the base class; derive `title_bout` from the presence of `Title`.

---

## Verified good

- **The partition contract holds exactly.** `head + body + leg == sig_str` and
  `distance + clinch + ground == sig_str`, **0 violations, max error 0.0** across all
  41,630 parseable rows. `validate_fights()` passes clean.
- **`CTRL` is uniformly well-formed** — 0 malformed values, 432 `"--"`.
- 42 of 41,672 stat rows are unparseable — genuinely statless old fights, expected NaN.

### Two judgment calls to make deliberately

- **`CTRL == "--"` means *unrecorded*, not zero.** The skeleton's `to_seconds()` maps it to
  `0.0`, which reads downstream as "held zero control time." 432 rows, mostly old fights.
  Return `NaN` or keep `0.0` knowingly.
- **`validate_row_counts()` asserts `len(events) > 1000`; there are 784.** Lower the
  threshold to ~700. The other three thresholds pass.

---

## Scope numbers (pre-computed for Day 2)

| Threshold | Fighters | REACH complete | HEIGHT | STANCE |
|---|---|---|---|---|
| 3+ UFC bouts | 1,875 | 94.0% | 99.7% | 99.6% |
| 5+ UFC bouts | 1,267 | 97.0% | 99.9% | 99.8% |

Reach is 45.7% missing across all 4,588 fighters but only **3% missing on the 5+ roster** —
the physical feature block is viable. Don't drop the column based on the roster-wide number.

**Bouts per year:** steady at ~1,000/year from 2014 onward (2014: 1,006 · 2019: 1,032 ·
2023: 1,040 · 2025: 1,040). `era_start: 2014` yields roughly 12k bouts / 24k fighter-bout
rows, matching the plan's 25k target.

---

## Refresh behaviour

`scrape_ufc_stats_unparsed_data.py` diffs live events against `event_details`, scrapes only
new or incomplete events, then sweeps 26 alphabetical pages for new fighters. It rewrites
all six CSVs in place, **prepending** new rows — so every refresh is a whole-file diff in git.

It cannot correct stats for events already present; it only adds missing ones.

Before running, check locally whether anything is actually missing — 0 incomplete events and
0 bouts without stats as of this snapshot.

**Version drift:** the scraper pins `pandas==2.2.3` / `numpy==1.26.4`; the project venv has
pandas 3.0.5 / numpy 2.5.2. The library relies on `pd.concat` with empty frames,
`df.loc[len(df)] = list`, and `merge(how='inner')` with no explicit `on`. First place to look
if a refresh errors.
