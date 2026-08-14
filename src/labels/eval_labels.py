"""
Compare the pipeline's automatic is_dead label against the annotated sample.

Four questions, in the order they matter:

  1. The pipeline deletes every character whose infobox status it cannot parse
     (6,887 pages, and 79-89% of some franchises). Are those characters
     actually alive? If they are, the fix is to count them as alive instead of
     deleting them, and franchise mortality drops sharply.

  2. On dod_only wikis, is_dead means "has a date of death somewhere in the
     lore". How many of those characters actually die on screen during the
     story? Lord of the Rings reports 72% mortality, which is what a corpus
     looks like when ancestors count.

  3. How much of the corpus is not even a character, or never appears on
     screen? Those rows belong in neither numerator nor denominator.

  4. Given all that, what are precision and recall of the current label?

Run:  python -m src.labels.eval_labels
"""

from __future__ import annotations

import argparse
import json

import pandas as pd

from src.paths import CLEAN_DIR, GOLD_DIR

ANNOT_DIR = GOLD_DIR / "annotations"
LABELS = ("is_character", "appears_onscreen", "dies_in_narrative")


def cohen_kappa(a: pd.Series, b: pd.Series) -> float:
    """
    Cohen's kappa for two aligned label series.

    Written out rather than pulled from sklearn so the agreement number in the
    writeup is traceable to an expression someone can check.
    """
    both = pd.concat([a, b], axis=1).dropna()
    if both.empty:
        return float("nan")
    x, y = both.iloc[:, 0], both.iloc[:, 1]
    categories = sorted(set(x) | set(y))
    n = len(both)

    observed = (x == y).mean()
    expected = sum(
        ((x == c).sum() / n) * ((y == c).sum() / n) for c in categories
    )
    if expected == 1.0:
        return float("nan")  # undefined when one category is unanimous
    return (observed - expected) / (1 - expected)


def load_pass(pass_id: int) -> pd.DataFrame | None:
    """
    Read every JSONL file for a pass.

    Parsed line by line rather than with pd.read_json because the annotator
    writes files with inconsistent conventions -- some carry a UTF-8 BOM, and a
    single malformed line should not lose the other 99 in the file.
    """
    src = ANNOT_DIR / f"pass{pass_id}"
    files = sorted(src.glob("*.jsonl"))
    if not files:
        return None

    records, bad = [], 0
    for path in files:
        # utf-8-sig strips a BOM if present and is a no-op otherwise.
        for line in path.read_text(encoding="utf-8-sig").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                bad += 1
    if bad:
        print(f"  skipped {bad} malformed annotation lines")

    df = pd.DataFrame(records)
    return df.drop_duplicates(subset="char_id", keep="first")


def report_agreement(p1: pd.DataFrame, p2: pd.DataFrame | None) -> None:
    if p2 is None:
        print("Pass 2 not present, so no inter-pass agreement to report.\n")
        return
    merged = p1.merge(p2, on="char_id", suffixes=("_1", "_2"))
    print(f"Inter-pass agreement on {len(merged)} shared characters")
    for label in LABELS:
        k = cohen_kappa(merged[f"{label}_1"], merged[f"{label}_2"])
        agree = (merged[f"{label}_1"] == merged[f"{label}_2"]).mean()
        print(f"  {label:18s} raw agreement {agree:5.1%}   kappa {k:+.3f}")
    print()


def print_provenance() -> None:
    """
    State plainly where these labels come from, because it bounds every claim
    built on them.

    The annotations are model output, not human judgement. Inter-pass agreement
    below measures reliability -- whether two runs say the same thing -- which
    is not the same as being right; two passes can agree and both be wrong.
    Accuracy is measured separately in registry_check.py against death lists
    scraped from listofdeaths.fandom.com, which no model touched.
    """
    print("Label provenance: annotations are Haiku 4.5 output, not human labels.")
    print("  Inter-pass agreement measures reliability, not correctness.")
    print("  For accuracy, see src/labels/registry_check.py, which scores these")
    print("  labels against independently scraped episode death lists.\n")


def stratum_populations() -> dict[str, int]:
    """
    How many characters in the whole corpus each sampled stratum stands for.

    Derived from data/clean/label_audit.csv, which counts pages, labelled rows
    and dropped rows per franchise alongside that franchise's regime.
    """
    audit = pd.read_csv(CLEAN_DIR / "label_audit.csv")
    audit["dead"] = (audit["labeled"] * audit["mortality_as_reported"] / 100).round()
    audit["alive"] = audit["labeled"] - audit["dead"]

    pops: dict[str, int] = {}
    for regime, grp in audit.groupby("regime"):
        pops[f"{regime}:dead"] = int(grp["dead"].sum())
        pops[f"{regime}:alive"] = int(grp["alive"].sum())
        dropped = int(grp["dropped"].sum())
        if dropped:
            pops[f"{regime}:dropped"] = dropped
    return pops


def report_composition(df: pd.DataFrame) -> None:
    """Corpus-level composition estimate, reweighting the stratified sample."""
    pops = stratum_populations()
    total = sum(pops.get(s, 0) for s in df["stratum"].unique())

    categories = {
        "not an individual character": lambda g: (g.is_character == 0).mean(),
        "character, never on screen": lambda g: (
            (g.is_character == 1) & (g.appears_onscreen == 0)
        ).mean(),
        "on-screen character": lambda g: (
            (g.is_character == 1) & (g.appears_onscreen == 1)
        ).mean(),
    }

    print("Corpus composition, reweighted by stratum size")
    print(f"  (the sample is stratified, so raw sample proportions would be misleading)")
    for name, fn in categories.items():
        weighted = sum(
            fn(grp) * pops.get(stratum, 0)
            for stratum, grp in df.groupby("stratum")
        ) / total
        raw = fn(df)
        print(f"  {name:28s} {weighted:5.1%}   (unweighted sample: {raw:5.1%})")
    print(f"  corpus characters represented: {total:,}")
    print()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pass-id", type=int, default=1)
    args = ap.parse_args()

    sample = pd.read_csv(GOLD_DIR / "gold_sample.csv")
    p1 = load_pass(args.pass_id)
    if p1 is None:
        raise SystemExit(f"No annotations in {ANNOT_DIR / f'pass{args.pass_id}'}")
    p2 = load_pass(2) if args.pass_id == 1 else None

    df = sample.merge(p1[["char_id", *LABELS, "confidence"]], on="char_id", how="inner")
    print(f"{len(df)} of {len(sample)} sampled characters annotated\n")

    print_provenance()
    report_agreement(p1, p2)

    # ---- Q3 first: what fraction of the corpus is not usable at all --------
    # The sample is stratified, so the raw proportions describe the sample, not
    # the corpus. Reweighting each stratum by how many characters it actually
    # represents turns them into a corpus estimate.
    report_composition(df)

    # ---- Q1: are the dropped rows alive? -----------------------------------
    dropped = df[df.stratum == "status:dropped"]
    if len(dropped):
        onscreen = dropped[(dropped.is_character == 1) & (dropped.appears_onscreen == 1)]
        print(f"Rows the pipeline currently DELETES (n={len(dropped)})")
        print(f"  not a character or not on screen : "
              f"{1 - len(onscreen) / len(dropped):5.1%}")
        if len(onscreen):
            dies = onscreen.dies_in_narrative
            print(f"  of the on-screen ones (n={len(onscreen)}):")
            print(f"    die in narrative   {(dies == 1).mean():5.1%}")
            print(f"    survive            {(dies == 0).mean():5.1%}")
            print(f"    undecidable        {(dies == -1).mean():5.1%}")
            print("  Treating every dropped row as alive is therefore "
                  f"{'defensible' if (dies == 1).mean() < 0.25 else 'NOT safe'}.")
        print()

    # ---- Q2: does a date of death mean an on-screen death? -----------------
    dod_dead = df[df.stratum == "dod_only:dead"]
    if len(dod_dead):
        onscreen = dod_dead[(dod_dead.is_character == 1) & (dod_dead.appears_onscreen == 1)]
        print(f"dod_only characters labelled DEAD (n={len(dod_dead)})")
        print(f"  never appear on screen : "
              f"{1 - len(onscreen) / len(dod_dead):5.1%}")
        if len(onscreen):
            print(f"  of the on-screen ones (n={len(onscreen)}): "
                  f"{(onscreen.dies_in_narrative == 1).mean():5.1%} actually die in the story")
        print()

    # ---- Q4: precision and recall of the current label ---------------------
    print("Automatic label vs annotation, on-screen characters only")
    scored = df[(df.is_character == 1) & (df.appears_onscreen == 1)
                & (df.dies_in_narrative.isin([0, 1]))].copy()
    if scored.empty:
        print("  nothing to score")
        return

    # The sample is stratified, not representative, so these are per-stratum
    # rates. Pooling them would describe a corpus that does not exist.
    rows = []
    for stratum, grp in scored.groupby("stratum"):
        auto = grp.auto_is_dead.fillna(0).astype(int)  # dropped rows count as alive
        truth = grp.dies_in_narrative.astype(int)
        tp = int(((auto == 1) & (truth == 1)).sum())
        fp = int(((auto == 1) & (truth == 0)).sum())
        fn = int(((auto == 0) & (truth == 1)).sum())
        tn = int(((auto == 0) & (truth == 0)).sum())
        prec = tp / (tp + fp) if tp + fp else float("nan")
        rec = tp / (tp + fn) if tp + fn else float("nan")
        rows.append({
            "stratum": stratum, "n": len(grp), "TP": tp, "FP": fp,
            "FN": fn, "TN": tn, "precision": prec, "recall": rec,
            "accuracy": (tp + tn) / len(grp),
        })
    report = pd.DataFrame(rows)
    print(report.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    CLEAN_DIR.mkdir(parents=True, exist_ok=True)
    out = CLEAN_DIR / "label_eval.csv"
    report.to_csv(out, index=False)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
