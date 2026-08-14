import json
import re

import pandas as pd

from src.constants import ACTOR_INFOBOX_KEYS, REGIME_DOD, REGIME_STATUS
from src.paths import CLEAN_DIR, RAW_DIR, REPO_ROOT

CONFIG_BLOCK_RE = re.compile(r'\n    "(\w+)": \{')
FRANCHISE_RE = re.compile(r'"franchise":\s*"([^"]+)"')


class LabelAuditor:
    """Mortality per franchise under each of the two infobox conventions."""

    @staticmethod
    def load_regimes() -> dict[str, str]:
        """Franchise name -> regime. Keyed on the franchise string, not the config key."""
        source = (REPO_ROOT / "scraper" / "wiki_configs.py").read_text(encoding="utf-8")
        blocks = CONFIG_BLOCK_RE.split(source)

        regimes = {}
        for key, body in zip(blocks[1::2], blocks[2::2]):
            regime = REGIME_DOD if '"dod_only": True' in body else REGIME_STATUS
            match = FRANCHISE_RE.search(body)
            regimes[match.group(1) if match else key] = regime
        return regimes

    @staticmethod
    def scan_raw() -> pd.DataFrame:
        """One row per scraped character page."""
        files = sorted(RAW_DIR.glob("*.jsonl"))
        if not files:
            raise SystemExit(
                f"No JSONL files in {RAW_DIR}. Regenerate with "
                f"`python -m scraper.run_scraper --wiki all`."
            )

        rows = []
        for path in files:
            with path.open(encoding="utf-8") as handle:
                for line in handle:
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if record.get("is_redirect"):
                        continue
                    infobox = record.get("infobox") or {}
                    rows.append({
                        "franchise": record.get("franchise", ""),
                        "is_dead": record.get("is_dead"),
                        "has_actor": int(
                            any(k.lower() in ACTOR_INFOBOX_KEYS for k in infobox)
                        ),
                    })
        return pd.DataFrame(rows)

    @staticmethod
    def build(df: pd.DataFrame, regimes: dict[str, str]) -> pd.DataFrame:
        """Per-franchise counts and mortality under both conventions."""
        rows, unmatched = [], set()

        for franchise, group in df.groupby("franchise"):
            regime = regimes.get(franchise)
            if regime is None:
                unmatched.add(franchise)
                continue

            pages = len(group)
            dead = int((group["is_dead"] == 1).sum())
            alive = int((group["is_dead"] == 0).sum())
            unlabelled = int(group["is_dead"].isna().sum())
            labelled = dead + alive

            rows.append({
                "franchise": franchise,
                "regime": regime,
                "pages": pages,
                "labeled": labelled,
                "dropped": unlabelled,
                "drop_pct": 100 * unlabelled / pages if pages else 0.0,
                "mortality_as_reported": 100 * dead / labelled if labelled else float("nan"),
                "mortality_dropped_as_alive": 100 * dead / pages if pages else float("nan"),
                "pct_with_actor": 100 * group["has_actor"].mean(),
            })

        if unmatched:
            raise SystemExit(
                f"No wiki_configs entry for these franchises: {sorted(unmatched)}"
            )

        audit = pd.DataFrame(rows).sort_values("mortality_as_reported", ascending=False)
        audit["mortality_inflation_pp"] = (
            audit["mortality_as_reported"] - audit["mortality_dropped_as_alive"]
        )
        return audit.reset_index(drop=True)

    @staticmethod
    def report(audit: pd.DataFrame) -> None:
        columns = [
            "franchise", "regime", "pages", "labeled", "dropped", "drop_pct",
            "mortality_as_reported", "mortality_dropped_as_alive",
            "mortality_inflation_pp",
        ]
        with pd.option_context("display.width", 200, "display.max_rows", 60):
            print(audit[columns].to_string(
                index=False, float_format=lambda v: f"{v:.1f}"
            ))

        status = audit[audit["regime"] == REGIME_STATUS]
        dod = audit[audit["regime"] == REGIME_DOD]

        if len(status) > 2:
            rho = status["drop_pct"].corr(
                status["mortality_as_reported"], method="spearman"
            )
            print(f"\nSpearman(drop_pct, reported mortality) over status wikis = {rho:+.3f}")

        print("\nMean reported mortality by regime:")
        print(f"  {REGIME_DOD:9s} ({len(dod):2d} franchises): "
              f"{dod['mortality_as_reported'].mean():.1f}%")
        print(f"  {REGIME_STATUS:9s} ({len(status):2d} franchises): "
              f"{status['mortality_as_reported'].mean():.1f}%")


def main() -> None:
    regimes = LabelAuditor.load_regimes()
    print(f"Read {len(regimes)} wiki configs "
          f"({sum(r == REGIME_DOD for r in regimes.values())} {REGIME_DOD}, "
          f"{sum(r == REGIME_STATUS for r in regimes.values())} {REGIME_STATUS})\n")

    df = LabelAuditor.scan_raw()
    print(f"Scanned {len(df):,} character pages (redirects excluded)\n")

    audit = LabelAuditor.build(df, regimes)
    LabelAuditor.report(audit)

    CLEAN_DIR.mkdir(parents=True, exist_ok=True)
    out_path = CLEAN_DIR / "label_audit.csv"
    audit.to_csv(out_path, index=False)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
