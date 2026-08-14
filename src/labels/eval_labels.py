import argparse
import json

import pandas as pd

from src.constants import ANNOTATION_LABELS, ANNOTATOR_MODEL, REGIME_DOD, REGIME_STATUS
from src.paths import CLEAN_DIR, GOLD_DIR

ANNOT_DIR = GOLD_DIR / "annotations"


class LabelEvaluator:
    """Score the automatic is_dead label against the annotated sample."""

    @staticmethod
    def cohen_kappa(a: pd.Series, b: pd.Series) -> float:
        both = pd.concat([a, b], axis=1).dropna()
        if both.empty:
            return float("nan")

        x, y = both.iloc[:, 0], both.iloc[:, 1]
        n = len(both)
        categories = sorted(set(x) | set(y))

        observed = (x == y).mean()
        expected = sum(((x == c).sum() / n) * ((y == c).sum() / n) for c in categories)
        if expected == 1.0:
            return float("nan")
        return (observed - expected) / (1 - expected)

    @staticmethod
    def load_pass(pass_id: int) -> pd.DataFrame | None:
        source = ANNOT_DIR / f"pass{pass_id}"
        files = sorted(source.glob("*.jsonl"))
        if not files:
            return None

        records, malformed = [], 0
        for path in files:
            # utf-8-sig: some chunks come back with a BOM
            for line in path.read_text(encoding="utf-8-sig").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    malformed += 1
        if malformed:
            print(f"  skipped {malformed} malformed annotation lines")

        return pd.DataFrame(records).drop_duplicates(subset="char_id", keep="first")

    @staticmethod
    def stratum_populations() -> dict[str, int]:
        """How many corpus characters each stratum represents."""
        audit = pd.read_csv(CLEAN_DIR / "label_audit.csv")
        audit["dead"] = (audit["labeled"] * audit["mortality_as_reported"] / 100).round()
        audit["alive"] = audit["labeled"] - audit["dead"]

        populations: dict[str, int] = {}
        for regime, group in audit.groupby("regime"):
            populations[f"{regime}:dead"] = int(group["dead"].sum())
            populations[f"{regime}:alive"] = int(group["alive"].sum())
            dropped = int(group["dropped"].sum())
            if dropped:
                populations[f"{regime}:dropped"] = dropped
        return populations

    @staticmethod
    def print_provenance() -> None:
        print(f"Annotations produced by {ANNOTATOR_MODEL}.")
        print("  Inter-pass agreement measures reliability, not correctness.")
        print("  For accuracy see src/labels/registry_check.py.\n")

    @staticmethod
    def report_agreement(first: pd.DataFrame, second: pd.DataFrame | None) -> None:
        if second is None:
            print("Only one annotation pass present.\n")
            return

        merged = first.merge(second, on="char_id", suffixes=("_1", "_2"))
        print(f"Inter-pass agreement on {len(merged)} shared characters")
        for label in ANNOTATION_LABELS:
            kappa = LabelEvaluator.cohen_kappa(merged[f"{label}_1"], merged[f"{label}_2"])
            agreement = (merged[f"{label}_1"] == merged[f"{label}_2"]).mean()
            print(f"  {label:18s} raw agreement {agreement:5.1%}   kappa {kappa:+.3f}")
        print()

    @staticmethod
    def report_composition(df: pd.DataFrame) -> None:
        populations = LabelEvaluator.stratum_populations()
        total = sum(populations.get(s, 0) for s in df["stratum"].unique())

        categories = {
            "not an individual character":
                lambda g: (g.is_character == 0).mean(),
            "character, never on screen":
                lambda g: ((g.is_character == 1) & (g.appears_onscreen == 0)).mean(),
            "on-screen character":
                lambda g: ((g.is_character == 1) & (g.appears_onscreen == 1)).mean(),
        }

        print("Corpus composition, reweighted by stratum size")
        rows = []
        for name, fn in categories.items():
            weighted = sum(
                fn(group) * populations.get(stratum, 0)
                for stratum, group in df.groupby("stratum")
            ) / total
            print(f"  {name:28s} {weighted:5.1%}   (unweighted sample: {fn(df):5.1%})")
            rows.append({
                "category": name,
                "share_weighted": round(100 * weighted, 1),
                "share_unweighted": round(100 * fn(df), 1),
                "n_annotated": len(df),
                "n_corpus_represented": total,
            })
        print(f"  corpus characters represented: {total:,}\n")

        # Written out so the corpus figure reads these shares rather than
        # carrying its own hardcoded copy of them.
        pd.DataFrame(rows).to_csv(CLEAN_DIR / "corpus_composition.csv", index=False)

    @staticmethod
    def report_unlabelled(df: pd.DataFrame) -> None:
        rows = df[df.stratum == f"{REGIME_STATUS}:dropped"]
        if rows.empty:
            return

        onscreen = rows[(rows.is_character == 1) & (rows.appears_onscreen == 1)]
        print(f"Rows with no parsed status (n={len(rows)})")
        print(f"  not a character or not on screen : {1 - len(onscreen) / len(rows):5.1%}")
        if len(onscreen):
            dies = onscreen.dies_in_narrative
            print(f"  of the on-screen ones (n={len(onscreen)}):")
            print(f"    die in narrative   {(dies == 1).mean():5.1%}")
            print(f"    survive            {(dies == 0).mean():5.1%}")
            print(f"    undecidable        {(dies == -1).mean():5.1%}")
        print()

    @staticmethod
    def report_dod(df: pd.DataFrame) -> None:
        rows = df[df.stratum == f"{REGIME_DOD}:dead"]
        if rows.empty:
            return

        onscreen = rows[(rows.is_character == 1) & (rows.appears_onscreen == 1)]
        print(f"{REGIME_DOD} characters labelled dead (n={len(rows)})")
        print(f"  never appear on screen : {1 - len(onscreen) / len(rows):5.1%}")
        if len(onscreen):
            print(f"  of the on-screen ones (n={len(onscreen)}): "
                  f"{(onscreen.dies_in_narrative == 1).mean():5.1%} die in the story")
        print()

    @staticmethod
    def score(df: pd.DataFrame) -> pd.DataFrame:
        """Precision and recall per stratum. Not pooled: the sample is stratified."""
        scored = df[
            (df.is_character == 1)
            & (df.appears_onscreen == 1)
            & (df.dies_in_narrative.isin([0, 1]))
        ]

        rows = []
        for stratum, group in scored.groupby("stratum"):
            auto = group.auto_is_dead.fillna(0).astype(int)
            truth = group.dies_in_narrative.astype(int)

            tp = int(((auto == 1) & (truth == 1)).sum())
            fp = int(((auto == 1) & (truth == 0)).sum())
            fn = int(((auto == 0) & (truth == 1)).sum())
            tn = int(((auto == 0) & (truth == 0)).sum())

            rows.append({
                "stratum": stratum,
                "n": len(group),
                "TP": tp, "FP": fp, "FN": fn, "TN": tn,
                "precision": tp / (tp + fp) if tp + fp else float("nan"),
                "recall": tp / (tp + fn) if tp + fn else float("nan"),
                "accuracy": (tp + tn) / len(group),
            })
        return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pass-id", type=int, default=1)
    args = parser.parse_args()

    sample = pd.read_csv(GOLD_DIR / "gold_sample.csv")
    first = LabelEvaluator.load_pass(args.pass_id)
    if first is None:
        raise SystemExit(f"No annotations in {ANNOT_DIR / f'pass{args.pass_id}'}")
    second = LabelEvaluator.load_pass(2) if args.pass_id == 1 else None

    columns = ["char_id", *ANNOTATION_LABELS, "confidence"]
    df = sample.merge(first[columns], on="char_id", how="inner")
    print(f"{len(df)} of {len(sample)} sampled characters annotated\n")

    LabelEvaluator.print_provenance()
    LabelEvaluator.report_agreement(first, second)
    LabelEvaluator.report_composition(df)
    LabelEvaluator.report_unlabelled(df)
    LabelEvaluator.report_dod(df)

    report = LabelEvaluator.score(df)
    print("Automatic label vs annotation, on-screen characters only")
    print(report.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    CLEAN_DIR.mkdir(parents=True, exist_ok=True)
    out_path = CLEAN_DIR / "label_eval.csv"
    report.to_csv(out_path, index=False)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    from src import setup_run_log
    setup_run_log(__spec__)
    main()
