"""
Audit how the is_dead label is produced, per franchise.

The scraper labels characters in two different ways depending on the wiki, and
the two are not comparable:

  Regime A ("status"):   the label comes from the infobox `status` field. If the
                         field is missing, or its text matches neither the
                         wiki's dead_statuses nor its alive_statuses, is_dead is
                         None -- and build_pipeline.py then drops the row. On
                         Fandom, "Status: Deceased" tends to be filled in while
                         living characters are left blank, so the dropped rows
                         skew alive and the surviving denominator is too small.

  Regime B ("dod_only"): any recorded date of death means dead, no death field
                         means alive. Nothing is dropped, but the label now
                         means "is dead somewhere in the fictional chronology"
                         rather than "dies during the story", which counts
                         ancestors and historical figures as deaths.

This script measures both effects: how many pages each franchise loses to the
drop, and what its mortality rate looks like before and after those rows are put
back. It writes data/clean/label_audit.csv and prints the table.

Run:  python -m src.labels.audit_labels
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

from src.paths import CLEAN_DIR, RAW_DIR, REPO_ROOT

# Matches   "wiki_key": {  ...  "dod_only": True   within one config block.
_CONFIG_BLOCK_RE = re.compile(r'\n    "(\w+)": \{')


def load_regimes() -> dict[str, str]:
    """
    Map each *franchise* to its labeling regime by reading wiki_configs.py.

    Keyed on the config's "franchise" string rather than its dict key, because
    the scraped records carry the franchise name ("Star Wars"), not the config
    key ("star_wars"), and matching on the key silently misfiles wikis whose
    short name differs -- which would hide exactly the effect being measured.

    Parsed from source rather than imported because the config is a plain dict
    literal and importing it would pull in the scraper's dependencies.
    """
    src = (REPO_ROOT / "scraper" / "wiki_configs.py").read_text(encoding="utf-8")
    blocks = _CONFIG_BLOCK_RE.split(src)
    # blocks = [preamble, key1, body1, key2, body2, ...]
    regimes = {}
    for key, body in zip(blocks[1::2], blocks[2::2]):
        regime = "dod_only" if '"dod_only": True' in body else "status"
        match = re.search(r'"franchise":\s*"([^"]+)"', body)
        regimes[match.group(1) if match else key] = regime
    return regimes


def scan_raw() -> pd.DataFrame:
    """One row per scraped character page, with just the fields the audit needs."""
    rows = []
    files = sorted(RAW_DIR.glob("*.jsonl"))
    if not files:
        raise SystemExit(
            f"No JSONL files in {RAW_DIR}. data/raw is gitignored -- regenerate it "
            f"with `python -m scraper.run_scraper --wiki all`."
        )

    for path in files:
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if rec.get("is_redirect"):
                    continue  # redirects are not characters
                infobox = rec.get("infobox") or {}
                rows.append(
                    {
                        "wiki": rec.get("wiki", ""),
                        "franchise": rec.get("franchise", ""),
                        "is_dead": rec.get("is_dead"),
                        # Presence of an actor field is the on-screen proxy we
                        # validate against the gold set in step 1c.
                        "has_actor": int(
                            any(
                                k.lower() in {"actor", "portrayed", "portrayed by", "played by"}
                                for k in infobox
                            )
                        ),
                    }
                )
    return pd.DataFrame(rows)


def build_audit(df: pd.DataFrame, regimes: dict[str, str]) -> pd.DataFrame:
    """Per-franchise page counts, drop rate, and mortality under two conventions."""
    unmatched = set()
    out = []
    for franchise, grp in df.groupby("franchise"):
        n_pages = len(grp)
        n_dead = int((grp["is_dead"] == 1).sum())
        n_alive = int((grp["is_dead"] == 0).sum())
        n_dropped = int(grp["is_dead"].isna().sum())
        n_labeled = n_dead + n_alive

        regime = regimes.get(franchise)
        if regime is None:
            unmatched.add(franchise)
            regime = "unknown"

        out.append(
            {
                "franchise": franchise,
                "regime": regime,
                "pages": n_pages,
                "labeled": n_labeled,
                "dropped": n_dropped,
                "drop_pct": 100 * n_dropped / n_pages if n_pages else 0.0,
                # What the milestone reported: deaths over surviving rows only.
                "mortality_as_reported": 100 * n_dead / n_labeled if n_labeled else float("nan"),
                # Dropped rows put back and counted as alive.
                "mortality_dropped_as_alive": 100 * n_dead / n_pages if n_pages else float("nan"),
                "pct_with_actor": 100 * grp["has_actor"].mean(),
            }
        )

    if unmatched:
        # Loud, because a silent fallback here is what produced a wrong regime
        # split the first time around.
        raise SystemExit(
            "These franchises have no matching wiki_configs entry, so their "
            f"labeling regime is unknown: {sorted(unmatched)}"
        )

    audit = pd.DataFrame(out).sort_values("mortality_as_reported", ascending=False)
    audit["mortality_inflation_pp"] = (
        audit["mortality_as_reported"] - audit["mortality_dropped_as_alive"]
    )
    return audit.reset_index(drop=True)


def main() -> None:
    regimes = load_regimes()
    print(f"Read {len(regimes)} wiki configs "
          f"({sum(r == 'dod_only' for r in regimes.values())} dod_only, "
          f"{sum(r == 'status' for r in regimes.values())} status-based)\n")

    df = scan_raw()
    print(f"Scanned {len(df):,} character pages (redirects excluded)\n")

    audit = build_audit(df, regimes)

    with pd.option_context("display.width", 200, "display.max_rows", 60):
        print(
            audit[
                [
                    "franchise", "regime", "pages", "labeled", "dropped",
                    "drop_pct", "mortality_as_reported",
                    "mortality_dropped_as_alive", "mortality_inflation_pp",
                ]
            ].to_string(
                index=False,
                float_format=lambda v: f"{v:.1f}",
            )
        )

    status = audit[audit["regime"] == "status"]
    print(f"\nStatus-based franchises: {len(status)}")
    if len(status) > 2:
        rho = status["drop_pct"].corr(status["mortality_as_reported"], method="spearman")
        print(f"  Spearman(drop_pct, reported mortality) = {rho:+.3f}")
        print("  A positive value means the franchises that lost the most rows are "
              "the ones reporting the highest mortality.")

    dod = audit[audit["regime"] == "dod_only"]
    print(f"\nMean reported mortality by regime:")
    print(f"  dod_only ({len(dod):2d} franchises): {dod['mortality_as_reported'].mean():.1f}%")
    print(f"  status   ({len(status):2d} franchises): {status['mortality_as_reported'].mean():.1f}%")

    CLEAN_DIR.mkdir(parents=True, exist_ok=True)
    out_path = CLEAN_DIR / "label_audit.csv"
    audit.to_csv(out_path, index=False)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
