"""
Pull each sampled fighter's Wikipedia intro and background/early-life text.

Reads data/labels/background.csv, writes data/labels/wiki_background.csv
(fighter_id, fighter_name, wiki_title, intro, background_text). Source text for labeling.
"""

import json
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.features.snapshots import load_config

API = "https://en.wikipedia.org/w/api.php"
UA = "UFCStyles-labeling/0.1 (tonaotoro@gmail.com)"
SECTION_KEYS = ("background", "early life", "early years", "personal life", "amateur", "career")
MAX_CHARS = 2500


def _get(params: dict) -> dict:
    url = API + "?" + urllib.parse.urlencode({**params, "format": "json"})
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    for attempt in range(5):
        try:
            return json.load(urllib.request.urlopen(req, timeout=20))
        except Exception:
            time.sleep(5 * (attempt + 1))
    return {}


def _extract(title: str) -> tuple[str, str]:
    """Return (resolved title, plain text) or ("", "") if missing or not an MMA page."""
    data = _get({"action": "query", "prop": "extracts", "explaintext": 1,
                 "exsectionformat": "wiki", "redirects": 1, "titles": title})
    page = next(iter(data.get("query", {}).get("pages", {}).values()), {})
    text = page.get("extract", "")
    if not text or "mixed martial" not in text.lower():
        return "", ""
    return page.get("title", title), text


def split_text(text: str) -> tuple[str, str]:
    parts = re.split(r"\n(==+\s*[^=]+?\s*==+)\n", text)
    intro = parts[0].strip()
    kept = []
    for heading, body in zip(parts[1::2], parts[2::2]):
        name = heading.strip("= ").lower()
        if any(k in name for k in SECTION_KEYS) and "mixed martial" not in name and body.strip():
            kept.append(f"[{name}] {body.strip()}")
    return intro[:800], " ".join(kept)[:MAX_CHARS]


def main():
    labels_dir = Path(load_config()["paths"]["labels"])
    sheet = pd.read_csv(labels_dir / "background.csv")
    out = labels_dir / "wiki_background.csv"
    done = pd.read_csv(out) if out.exists() else pd.DataFrame(columns=["fighter_id"])

    rows = done.to_dict("records")
    for r in sheet.itertuples():
        if r.fighter_id in set(done["fighter_id"]):
            continue
        title, text = "", ""
        for candidate in (r.fighter_name, f"{r.fighter_name} (fighter)"):
            title, text = _extract(candidate)
            time.sleep(1.0)
            if text:
                break
        intro, background = split_text(text) if text else ("", "")
        rows.append({"fighter_id": r.fighter_id, "fighter_name": r.fighter_name,
                     "wiki_title": title, "intro": intro, "background_text": background})
        pd.DataFrame(rows).to_csv(out, index=False)

    result = pd.DataFrame(rows)
    assert result["fighter_id"].is_unique
    print(f"{(result['wiki_title'].fillna('') != '').sum()} / {len(result)} fighters have an MMA Wikipedia page")
    print(f"{(result['background_text'].fillna('') != '').sum()} have background/early-life text")


if __name__ == "__main__":
    main()
