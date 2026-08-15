"""Check the death label against human-curated death registries.

What it measures is recall: a character in a death registry should be labelled
dead. Matching is exact on a normalised name and restricted to names that occur 
once on each side.

  python -m src.labels.validate_registry
"""

import re

import pandas as pd

from src.constants import analysis_population
from src.paths import CLEAN_DIR, DATASETS_DIR

APOSTROPHE = "’"


class RegistryValidation:

    @staticmethod
    def normalise(name: object) -> str:
        text = str(name or "").lower().replace(APOSTROPHE, "'")
        text = re.sub(r"[^a-z0-9 ()'-]", " ", text)
        return re.sub(r"\s+", " ", text).strip()

    @staticmethod
    def got_registry() -> list[str]:
        path = DATASETS_DIR / "Game Of Thrones" / "game-of-thones-deaths.csv"
        if not path.exists():
            return []
        registry = pd.read_csv(path)
        return [RegistryValidation.normalise(n) for n in registry["Name"].dropna()]

    @staticmethod
    def the_100_registry() -> list[str]:
        """Victims are one cell per episode: "Name - how they died; Name - ...". """
        path = DATASETS_DIR / "The 100" / "the_100_deaths.csv"
        if not path.exists():
            return []
        names = []
        for cell in pd.read_csv(path)["Victims"].dropna():
            for victim in str(cell).split(";"):
                names.append(RegistryValidation.normalise(victim.split(" - ")[0]))
        return names

    @staticmethod
    def compare(population: pd.DataFrame, franchise: str,
                registry_names: list[str]) -> dict:
        ours = population[population["franchise"] == franchise].copy()
        ours["key"] = ours["name"].map(RegistryValidation.normalise)
        # Only names that identify exactly one character on our side.
        ours = ours[~ours["key"].duplicated(keep=False)]

        registry = {n for n in registry_names if n}
        matched = ours[ours["key"].isin(registry)]

        if matched.empty:
            return {
                "franchise": franchise,
                "registry_names": len(registry),
                "our_characters": len(ours),
                "matched": 0,
                "agree_dead": 0,
                "recall": float("nan"),
            }

        return {
            "franchise": franchise,
            "registry_names": len(registry),
            "our_characters": len(ours),
            "matched": len(matched),
            "agree_dead": int(matched["is_dead_v2"].sum()),
            "recall": float(matched["is_dead_v2"].mean()),
            "disagreements": "; ".join(
                matched.loc[matched["is_dead_v2"] == 0, "name"].astype(str).tolist()
            ),
        }


def main() -> None:
    characters = pd.read_csv(CLEAN_DIR / "characters_model.csv", low_memory=False)
    population = analysis_population(characters)

    rows = [
        RegistryValidation.compare(
            population, "Game of Thrones", RegistryValidation.got_registry()
        ),
        RegistryValidation.compare(
            population, "The 100", RegistryValidation.the_100_registry()
        ),
    ]
    results = pd.DataFrame(rows)

    print("Death label vs human-curated registries")
    for row in rows:
        if not row["matched"]:
            print(f"  {row['franchise']:18s} no 1:1 name matches "
                  f"({row['registry_names']} registry names against "
                  f"{row['our_characters']} of ours) -- too few characters "
                  f"scraped to validate")
            continue
        print(f"  {row['franchise']:18s} {row['agree_dead']:4d}/{row['matched']:<4d} "
              f"= {row['recall']:.1%} of registry deaths labelled dead")
        if row.get("disagreements"):
            print(f"  {'':18s} we say alive: {row['disagreements']}")

    usable = results[results["matched"] > 0]
    if not usable.empty:
        total_matched = int(usable["matched"].sum())
        total_agree = int(usable["agree_dead"].sum())
        print(f"\n  pooled: {total_agree}/{total_matched} = "
              f"{total_agree / total_matched:.1%} across "
              f"{len(usable)} franchise(s)")

    out_path = CLEAN_DIR / "label_registry_validation.csv"
    results.to_csv(out_path, index=False, encoding="utf-8")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    from src import setup_run_log
    setup_run_log(__spec__)
    main()
