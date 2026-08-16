"""Inter-rater reliability for the annotated sample.

Two independent coders labelled the same 600 characters against the scheme in
src/labels/ANNOTATION_RUBRIC.md, writing to data/gold/annotations/pass1/ and
pass2/. The second coder never saw the first coder's output.

Reported per label:

  raw agreement    the share of rows the two coders assigned the same code
  Cohen's kappa    the same quantity corrected for the agreement two coders
                   would reach by chance given their marginal rates

The evaluation lecture's worked example is the reason both appear. Raw
agreement rewards a coder for following the majority category, so on a label
where 92% of rows fall in one class two coders can agree 92% of the time while
carrying almost no shared information. Our labels are more skewed than that:
`is_character` runs above 80% in one class. Kappa subtracts the chance
agreement implied by each coder's own marginals, which is what makes it
readable across labels with different skews. Values are read against the
Landis and Koch bands the lecture gives.

Also reported: kappa between the automatic label rule and coder 1. Agreement
between two coders measures whether the coding scheme is reproducible;
agreement between the rule and a coder measures whether the rule is right.
The two questions are separate and the numbers should not be pooled.

  python -m src.labels.reliability
"""

import json

import pandas as pd

from src.constants import ANNOTATION_LABELS, ANNOTATOR_MODEL, UNDECIDABLE
from src.paths import CLEAN_DIR, GOLD_DIR

ANNOT_DIR = GOLD_DIR / "annotations"

# Landis and Koch (1977), as given in the evaluation lecture.
LANDIS_KOCH = [
    (0.00, "no agreement"),
    (0.20, "slight"),
    (0.40, "fair"),
    (0.60, "moderate"),
    (0.80, "substantial"),
    (1.01, "almost perfect"),
]


class Reliability:

    @staticmethod
    def band(kappa: float) -> str:
        if pd.isna(kappa):
            return "undefined"
        if kappa < 0:
            return "no agreement"
        for ceiling, name in LANDIS_KOCH:
            if kappa < ceiling:
                return name
        return "almost perfect"

    @staticmethod
    def cohen_kappa(a: pd.Series, b: pd.Series) -> tuple[float, float, int]:
        """Cohen's kappa, plus the raw agreement and n it was computed on.

        Expected agreement is the sum over categories of the product of the two
        coders' marginal rates, which is the chance-agreement term in the
        lecture's formula. Kappa is undefined when that term reaches 1, which
        happens when both coders used a single category throughout; a constant
        label carries no information to agree about, and returning NaN says so
        rather than reporting a spurious 0 or 1.
        """
        both = pd.concat([a, b], axis=1).dropna()
        if both.empty:
            return float("nan"), float("nan"), 0

        x, y = both.iloc[:, 0], both.iloc[:, 1]
        n = len(both)
        categories = sorted(set(x) | set(y))

        observed = float((x == y).mean())
        expected = sum(
            ((x == c).sum() / n) * ((y == c).sum() / n) for c in categories
        )
        if expected >= 1.0:
            return float("nan"), observed, n
        return (observed - expected) / (1 - expected), observed, n

    @staticmethod
    def load_pass(pass_id: int) -> pd.DataFrame | None:
        source = ANNOT_DIR / f"pass{pass_id}"
        files = sorted(source.glob("*.jsonl"))
        if not files:
            return None

        records, malformed = [], 0
        for path in files:
            for line in path.read_text(encoding="utf-8-sig").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    malformed += 1
        if malformed:
            print(f"  pass {pass_id}: skipped {malformed} malformed lines")

        return pd.DataFrame(records).drop_duplicates(subset="char_id", keep="first")

    @staticmethod
    def inter_coder(first: pd.DataFrame, second: pd.DataFrame,
                    name: str) -> pd.DataFrame:
        merged = first.merge(second, on="char_id", suffixes=("_1", "_2"))

        rows = []
        for label in ANNOTATION_LABELS:
            left, right = merged[f"{label}_1"], merged[f"{label}_2"]
            kappa, agreement, n = Reliability.cohen_kappa(left, right)
            rows.append({
                "comparison": name,
                "label": label,
                "n": n,
                "raw_agreement": agreement,
                "kappa": kappa,
                "band": Reliability.band(kappa),
                "coder1_positive_rate": float((left == 1).mean()),
                "coder2_positive_rate": float((right == 1).mean()),
            })

        # dies_in_narrative carries a third code for undecidable rows, and a
        # coder who declines to judge is a different kind of disagreement from
        # one who judges the other way. Scored again with those rows dropped so
        # the binary kappa is comparable to the other two labels.
        decided = merged[
            (merged["dies_in_narrative_1"] != UNDECIDABLE)
            & (merged["dies_in_narrative_2"] != UNDECIDABLE)
        ]
        if len(decided):
            kappa, agreement, n = Reliability.cohen_kappa(
                decided["dies_in_narrative_1"], decided["dies_in_narrative_2"]
            )
            rows.append({
                "comparison": name,
                "label": "dies_in_narrative (both decided)",
                "n": n,
                "raw_agreement": agreement,
                "kappa": kappa,
                "band": Reliability.band(kappa),
                "coder1_positive_rate": float((decided["dies_in_narrative_1"] == 1).mean()),
                "coder2_positive_rate": float((decided["dies_in_narrative_2"] == 1).mean()),
            })

        return pd.DataFrame(rows)

    @staticmethod
    def rule_vs_coder(sample: pd.DataFrame, first: pd.DataFrame,
                      second: pd.DataFrame | None) -> pd.DataFrame:
        """Chance-corrected agreement between the automatic label and a coder.

        Restricted to on-screen characters the coder could decide, which is the
        population the automatic label claims to describe. Outside it the rule
        is not wrong so much as inapplicable.
        """
        rows = []
        for coder_id, coder in [(1, first), (2, second)]:
            if coder is None:
                continue
            merged = sample.merge(
                coder[["char_id", *ANNOTATION_LABELS]], on="char_id", how="inner"
            )
            scored = merged[
                (merged["is_character"] == 1)
                & (merged["appears_onscreen"] == 1)
                & (merged["dies_in_narrative"].isin([0, 1]))
            ]
            if scored.empty:
                continue

            auto = scored["auto_is_dead"].fillna(0).astype(int)
            truth = scored["dies_in_narrative"].astype(int)
            kappa, agreement, n = Reliability.cohen_kappa(auto, truth)
            rows.append({
                "comparison": f"automatic label vs coder {coder_id}",
                "label": "is_dead",
                "n": n,
                "raw_agreement": agreement,
                "kappa": kappa,
                "band": Reliability.band(kappa),
                "coder1_positive_rate": float(auto.mean()),
                "coder2_positive_rate": float(truth.mean()),
            })
        return pd.DataFrame(rows)

    @staticmethod
    def disagreements(first: pd.DataFrame, second: pd.DataFrame,
                      sample: pd.DataFrame) -> pd.DataFrame:
        """Rows the coders coded differently, with both rationales.

        Kept because a kappa on its own cannot distinguish a hard sample from a
        rubric that two readers can read two ways, and reading the disagreements
        is the only way to tell which one happened.
        """
        merged = first.merge(second, on="char_id", suffixes=("_1", "_2"))
        differs = merged[
            (merged["is_character_1"] != merged["is_character_2"])
            | (merged["appears_onscreen_1"] != merged["appears_onscreen_2"])
            | (merged["dies_in_narrative_1"] != merged["dies_in_narrative_2"])
        ]
        keep = ["char_id", "name", "franchise"]
        out = differs.merge(
            sample[[c for c in keep if c in sample.columns]],
            on="char_id", how="left",
        )
        columns = keep + [
            f"{label}_{i}" for label in ANNOTATION_LABELS for i in (1, 2)
        ] + ["rationale_1", "rationale_2"]
        return out[[c for c in columns if c in out.columns]]


# Each entry is (pass A, pass B, what the comparison measures).
COMPARISONS = [
    (1, 2, "pass1 vs pass2 (v1 rubric)"),
    (3, 4, "pass3 vs pass4 (v2 rubric)"),
]


def main() -> None:
    sample = pd.read_csv(GOLD_DIR / "gold_sample.csv")
    passes = {i: Reliability.load_pass(i) for i in range(1, 5)}

    if passes[1] is None:
        raise SystemExit("No annotations in pass1")

    print("Coding scheme: src/labels/ANNOTATION_RUBRIC.md")
    print(f"Every coder is an independent run of {ANNOTATOR_MODEL}.")
    print("Agreement between two coders measures whether the scheme is")
    print("reproducible, not whether the labels are correct. Correctness against")
    print("an external source is measured in src/labels/validate_registry.py.\n")

    for pass_id, frame in passes.items():
        state = f"{len(frame)} rows" if frame is not None else "absent"
        print(f"  pass {pass_id}: {state}")
    print()

    reports = []
    for left, right, name in COMPARISONS:
        if passes[left] is None or passes[right] is None:
            print(f"Skipping {name}: one of the passes is missing.\n")
            continue
        reports.append(Reliability.inter_coder(passes[left], passes[right], name))

    reports.append(Reliability.rule_vs_coder(sample, passes[1], passes[3]))
    report = pd.concat(reports, ignore_index=True)

    shown = ["comparison", "label", "n", "raw_agreement", "kappa", "band"]
    print(report[shown].to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    print("\nRaw agreement exceeds kappa on every label by the amount the two")
    print("coders' marginal rates would have produced on their own:")
    for _, row in report.iterrows():
        if pd.notna(row["kappa"]):
            gap = row["raw_agreement"] - row["kappa"]
            print(f"  {row['comparison']:28s} {row['label']:34s} "
                  f"agreement {row['raw_agreement']:.3f}  "
                  f"kappa {row['kappa']:+.3f}  (gap {gap:.3f})")

    out_path = CLEAN_DIR / "label_reliability.csv"
    report.to_csv(out_path, index=False, encoding="utf-8")

    # Disagreements from the v1 pair, which is where the rubric gaps were found.
    if passes[2] is not None:
        differs = Reliability.disagreements(passes[1], passes[2], sample)
        differs_path = CLEAN_DIR / "label_disagreements.csv"
        differs.to_csv(differs_path, index=False, encoding="utf-8")
        print(f"\n{len(differs)} of {len(passes[1])} rows coded differently on "
              f"at least one label under v1 -> {differs_path.name}")

    print(f"Wrote {out_path}")


if __name__ == "__main__":
    from src import setup_run_log
    setup_run_log(__spec__)
    main()
