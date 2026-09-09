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

**Implemented** slightly differently: `fetch_fights` doesn't need `fight_details` at all —
it takes `fight_id` straight from `fight_results.URL` and strips `EVENT` on both frames
before the events join. `fetch_fight_stats` uses the EVENT+BOUT join to `fight_details`
as described above.

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

**Implemented:** `fetch_fights` requires a successful event join (rows left with `NaT`
dates are dropped) and asserts `fight_id` uniqueness; `fetch_fight_stats` scopes to
`fights_df` with an inner join and dedupes on `(fight_id, FIGHTER, ROUND)`.

### Sakuraba vs. Silveira — two fights, one card, identical EVENT+BOUT

At `UFC - Ultimate Japan` (1997-12-21), Kazushi Sakuraba and Marcus Silveira fought
**twice on the same card**: their tournament bout was ruled a no contest, and they were
rematched later that night in the final. UFCStats records them as two distinct fights
(URLs `ec1bda9a4c2aab42` and `2750ac5854e8b28b`) with byte-identical `EVENT` and `BOUT`
strings. It is the only (EVENT, BOUT) pair in the snapshot mapping to more than one URL.

**Consequence:** `fight_stats` has no URL, so the EVENT+BOUT join to `fight_details`
cannot tell the two fights apart — each of the 4 stat rows matches both URLs and fans out
to 8. Deduping on `(fight_id, FIGHTER, ROUND)` with `keep="first"` leaves both fight_ids
carrying copies of the first-listed fight's stats; the second fight's actual stats
(`0 of 1` / `4 of 10`) are dropped entirely. The two fights remain correctly separate in
`fight_details`/`fight_results` (URL-keyed) — only the stats attribution is ambiguous.

**Accepted as-is:** pre-era 1997 data, structurally unreachable from the 2014+ scope.
If it ever matters, the fix is disambiguation by source row order, not the join.

### Fighter identity — no clean solution

`fight_stats` identifies fighters by **name string only**, no URL. Mapping 2,723 distinct
names to `fighter_id` hits:

- **8 names shared by 16 different fighters** — Bruno Silva, Jean Silva, Mike Davis,
  Michael McDonald, Tony Johnson, Joey Gomez, Anthony Figueroa, Victor Valenzuela
- **15 names absent from `fighter_tott.csv`** — Patricio Pitbull, Kai Kamaka III,
  Levi Rodrigues Jr., Michael Aswell Jr., Muhammad Saidov, …

Pick a policy and **assert the unresolved count** so it cannot grow silently on refresh.
For the collisions, event date plus `WEIGHTCLASS` separates most pairs.

**Policy implemented (2026-09-01):** `fetch_fights` joins bout names to `fighter_details`
and breaks the 8 collisions on listed weight vs. the bout's weight class
(`_attach_fighter_id`); five broken bout-name forms are hand-mapped in `BOUT_NAME_FIXES`.
`fetch_fight_stats` never matches names globally — it joins each stats row to its bout via
EVENT+BOUT → `fight_id`, then matches `FIGHTER` against only that bout's two names.
Unresolved counts are asserted at zero in both functions.

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

Both still open as of 2026-09-01: `to_seconds()` still returns `0.0` for `"--"` and the
events threshold is still 1000 (so `validate_row_counts` currently fails, by design).

---

## Scope numbers (recomputed from adapter output, 2026-09-09)

Counted from `data/raw/2026-09-01/fights.csv`, per fighter, all years:

| Threshold | Fighters | REACH complete | HEIGHT | STANCE |
|---|---|---|---|---|
| 3+ UFC bouts | 1,882 | 93.7% | 99.7% | 99.8% |
| 5+ UFC bouts | 1,267 | 97.1% | 100.0% | 100.0% |

**Corrected.** This table previously read 1,875 / 94.0 / 99.7 / 99.6 at the 3+ threshold.
The 5+ fighter count (1,267) reproduces exactly, but 1,875 could not be reproduced by any
method: raw `BOUT` names give 1,884, deduped on `URL` 1,881, restricted to names present in
`fighter_tott` 1,880. The adapter output — what the code actually consumes — gives **1,882**.
Numbers above are now adapter-derived, so `scope_count.py` and this file agree.

Reach is 45.7% missing across all 4,588 fighters but only **6% missing on the 3+ roster** and
3% on 5+ — the physical feature block is viable. Don't drop the column based on the
roster-wide number.

**Bouts per year:** steady at ~1,000/year from 2014 onward (2014: 1,006 · 2019: 1,032 ·
2023: 1,040 · 2025: 1,040). `era_start: 2014` retains 6,287 bouts / 12,574 fighter-bout rows.

**The plan's 25k target does not survive contact.** That figure counts fighter-bout rows
*before* `history.min_prior_fights: 3` removes every fighter's first three bouts. Actual
snapshot yield at the chosen config (2014 / 3+ / extend) is **8,176 rows** — roughly a third
of the assumed training set, and squarely in the "5k samples, 25 features → overfit is
instant" regime. Shrinkage and dropout are load-bearing, not optional.

### Per-fighter-bout physical coverage by year

The table above is per *fighter*. Per *fighter-bout* — the shape the snapshot table takes —
coverage varies by year, and reach is the only completeness signal that moves inside the
era-candidate range:

| Year | 2010 | 2012 | 2014 | 2016 | 2017 | 2018 | 2020–24 | 2025 | 2026 |
|---|---|---|---|---|---|---|---|---|---|
| reach % | 92.3 | 90.9 | 90.4 | 95.3 | 98.1 | 99.5 | ~100 | 98.6 | 91.0 |

Height and stance sit at ~100% throughout until the same recent drop.

**The recent decline is scrape freshness, not a data-quality era effect** — recent debutants
are not in `fighter_tott.csv` yet. 2026 falls to 91.0% reach / 91.3% height / 93.0% stance.
This lands inside the **test split** (2023-07-01 onward), so the physical block is thinner at
eval time than at train time: a train/test distribution shift, not a scope question. Decide
imputation in Day 3 and re-check after every refresh.

### Stat completeness does not constrain the era

The Day-2 plan assumed strike-position completeness would reveal an era knee. It does not.
The 42 statless rows are already dropped by `fetch_ufcstats.py`, the partition contract passes
on everything remaining, and the only bouts with **zero** stat rows are the 21 in 1994–1998.
Fight statistics are complete from **1999** onward, so the check cannot discriminate between
2010 and 2017. Likewise `CTRL == "--"` is entirely pre-2000. Recorded as an explicit negative
so the question stays answered rather than re-opened.

---

## Adapter output (verified 2026-09-01)

`fetch_ufcstats.py` run end-to-end against this snapshot produces:

- **`fights`: 8,832 rows**, unique on `fight_id`, no `NaT` dates, no unresolved fighter ids
  (27 rows dropped by the required event join: 25 renamed-event orphans + 2 Road to UFC).
- **`fight_stats`: 41,506 rows**, unique on `(fight_id, fighter_id, round)`. Dropped along
  the way: 42 statless placeholders (frozen assert), 8 Road-to-UFC orphans (inner join),
  120 duplicate rows (116 renamed-event copies + 4 Sakuraba fan-out). Partition contract
  passes on all rows; `ctrl_time` parses 0–300s.
- **`fighters`: all 4,588 rows kept** — completeness filtering deliberately rejected; scope
  is Day-2 config's job. Missing after parse: 477 height, 2,099 reach, 777 DOB. Every
  `fighter_id` referenced in `fights` exists here. Height spans 55"–89"; the extremes are
  probably source errors on obscure fighters — harmless under within-weight-class
  percentiles, unaudited.

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
