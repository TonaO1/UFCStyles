"""
Pick the fighters to hand-label and write a blank data/labels/background.csv.

Samples only fighters with test-split snapshots (the probe scores test rows only),
proportional by each fighter's most common test weight class.
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.features.snapshots import load_config

HEX_ID = r"^[0-9a-f]{16}$"


def sample_fighters(test: pd.DataFrame, fighters: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    wc = test.groupby("fighter_id")["weight_class"].agg(lambda s: s.mode().iloc[0])
    pool = wc.rename("weight_class").reset_index()
    frac = n / len(pool)
    picked = pd.concat(g.sample(max(1, round(len(g) * frac)), random_state=seed)
                       for _, g in pool.groupby("weight_class"))
    picked = picked.merge(fighters[["fighter_id", "fighter_name"]], on="fighter_id", how="left")
    assert picked["fighter_name"].notna().all(), "sampled fighter missing from fighters.csv"
    assert picked["fighter_id"].is_unique
    return picked.sort_values(["weight_class", "fighter_name"])


def main():
    config = load_config()
    paths = config["paths"]
    out = Path(paths["labels"]) / "background.csv"

    if out.exists():
        existing = pd.read_csv(out)
        assert not existing["fighter_id"].astype(str).str.match(HEX_ID).any(), \
            f"{out} already holds real labels; refusing to overwrite"

    test = pd.read_parquet(Path(paths["snapshots"]) / "test.parquet")
    fighters = pd.read_csv(Path(paths["raw"]) / paths["snapshot_date"] / "fighters.csv")
    picked = sample_fighters(test, fighters, config["labels"]["background"]["n_samples"],
                             config["training"]["seed"])

    # weight_class stays out of the file: the harness would read it as a probe column.
    sheet = picked[["fighter_id", "fighter_name"]].assign(background="", confidence="", notes="")
    sheet.to_csv(out, index=False)
    print(f"wrote {len(sheet)} fighters to {out}")
    print(picked["weight_class"].value_counts().to_string())


if __name__ == "__main__":
    main()
