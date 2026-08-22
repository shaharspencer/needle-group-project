"""Build the Q1 feature matrix from the scraped pages.

Extracts category, infobox, relation, network, community, and text features.
Death-related text and label fields are excluded to reduce label leakage.
Communities use Louvain, which scales better than Girvan-Newman for our largest
graphs. The label is not read while constructing these features.

  python -m src.q1_character.features
"""

import json
import re
import sys
from collections import Counter, defaultdict

import networkx as nx
import numpy as np
import pandas as pd

from src.constants import is_non_character
from src.paths import CLEAN_DIR, RAW_DIR

CATEGORY_RE = re.compile(r"Category:([^\n]+)")
TOKEN_RE = re.compile(r"[a-z0-9]+")

# Any category or infobox field matching these restates the outcome. Survival
# words matter as much as death words: a category named "Alive" is the label
# with the sign flipped.
DEATH_TERMS = re.compile(
    r"death|deceas|died|dead|\bdie\b|\bdies\b|\bdying\b|"
    r"kill|murder|victim|casualt|fatal|corpse|posthum|grave|funeral|obitu|"
    r"execut|assassinat|suicide|martyr|sacrific|slain|slay|perish|massacre|"
    r"memorial|remembr|undead|ghost|revenant|resurrect|alive|living|surviv|"
    r"fallen|fate|legacy|avenge|avenging|mourn|buried|burial|\btomb\b|"
    r"\blate\b|\bbody\b|\bbodies\b|\bremains\b",
    re.IGNORECASE,
)

# Fields the label itself is derived from. Short names that no substring rule
# would catch: the scraper reads is_dead out of `status` and `dod`.
LABEL_SOURCE_FIELDS = frozenset({
    "status", "dod", "dob", "death", "died", "fate", "cause", "deathep",
    "causeofdeath", "cause of death", "manner of death",
})

FAMILY_KEYS = frozenset({
    "children", "siblings", "parents", "family", "spouse", "husband", "wife",
    "brother", "sister", "mother", "father", "son", "daughter", "relatives",
    "issue", "partners",
})

NAME_STOPWORDS = frozenset({
    "the", "and", "of", "for", "with", "his", "her", "their", "was", "were",
    "who", "that", "this", "from", "had", "has", "him", "she", "they",
})

TOP_CATEGORIES = 300
TOP_INFOBOX_FIELDS = 80
MIN_NAME_TOKEN_LEN = 4
INTRO_CHARS = 700

# Louvain moves nodes in an arbitrary order, so the partition depends on the
# seed. Fixed here so the feature matrix is reproducible, and checked across
# seeds by `python -m src.q1_character.features --seed-check`.
LOUVAIN_SEED = 0

# The community detection lecture's rule of thumb: Q in the 0.3-0.7 band
# indicates community structure beyond what the configuration model produces.
MODULARITY_FLOOR = 0.3

# Seeds compared by `--seed-check` to confirm the partitions are not an artefact
# of the order Louvain happens to visit nodes in.
SEED_CHECK_SEEDS = 5


class FeatureBuilder:

    @staticmethod
    def strip_death_text(text: str) -> str:
        """Drop every sentence that mentions death or survival.

        DEATH_TERMS covers survival words as well as death words, for the same
        reason it does everywhere else in this file: "still alive" is the label
        with the sign flipped.

        This used to run a second pass over individual words, on the theory that
        a stray token could survive a sentence that was itself clean. It cannot:
        a regex search over a sentence is a superset of the same search over
        that sentence's words, so any word the token pass would catch lives in a
        sentence the first pass already dropped. The second pass was dead code
        and is gone. What it was really compensating for was gaps in
        DEATH_TERMS, which is where the fix belongs.
        """
        kept = [
            s for s in re.split(r"(?<=[.!?])\s+", text or "")
            if not DEATH_TERMS.search(s)
        ]
        return re.sub(r"\s+", " ", " ".join(kept)).strip()

    @staticmethod
    def page_tokens(text: str) -> set[str]:
        return {
            t for t in TOKEN_RE.findall(text.lower())
            if len(t) >= MIN_NAME_TOKEN_LEN and t not in NAME_STOPWORDS
        }

    @staticmethod
    def name_tokens(name: str) -> frozenset[str]:
        return frozenset(
            t for t in TOKEN_RE.findall(name.lower())
            if len(t) >= MIN_NAME_TOKEN_LEN and t not in NAME_STOPWORDS
        )

    @staticmethod
    def read_franchise(path) -> list[dict]:
        """Parse one wiki file into the per-page fields the features need."""
        pages = []
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if record.get("is_redirect") or is_non_character(record):
                    continue

                text = record.get("page_text") or ""
                infobox = record.get("infobox") or {}
                categories = [
                    c.strip() for c in CATEGORY_RE.findall(text)
                    if not DEATH_TERMS.search(c)
                ]
                fields = [
                    k.strip().lower() for k in infobox
                    if not DEATH_TERMS.search(k)
                    and k.strip().lower() not in LABEL_SOURCE_FIELDS
                ]

                relatives = 0
                for key, value in infobox.items():
                    if key.lower() in FAMILY_KEYS and value:
                        relatives += len(
                            [p for p in re.split(r"[,;\n]|\{|\}", str(value)) if len(p.strip()) > 2]
                        )

                pages.append({
                    "page_url": record.get("page_url", ""),
                    "franchise": record.get("franchise", ""),
                    "name": record.get("name", ""),
                    "categories": categories,
                    "infobox_fields": fields,
                    "n_relatives": relatives,
                    "intro": re.sub(r"\s+", " ", text[:INTRO_CHARS]),
                    # Same text with every death- or survival-mentioning
                    # sentence dropped, then any surviving death token
                    # removed. A dead character's opening paragraph says "was
                    # killed by", so TF-IDF over the raw intro would mostly
                    # rediscover the label -- the same trap the violence
                    # lexicon fell into earlier in this project.
                    "intro_clean": FeatureBuilder.strip_death_text(text[:INTRO_CHARS]),
                    "tokens": FeatureBuilder.page_tokens(text[:20000]),
                })
        return pages

    @staticmethod
    def build_graph(pages: list[dict]) -> nx.DiGraph:
        """
        Co-mention graph for one franchise.

        Built through an inverted token index so each page is scanned once
        rather than compared against every character name.
        """
        graph = nx.DiGraph()
        index: dict[str, set[int]] = defaultdict(set)
        name_sets = []

        for i, page in enumerate(pages):
            tokens = FeatureBuilder.name_tokens(page["name"])
            name_sets.append(tokens)
            graph.add_node(page["page_url"])
            for token in tokens:
                index[token].add(i)

        for i, page in enumerate(pages):
            candidates: set[int] = set()
            for token in page["tokens"]:
                candidates |= index.get(token, set())
            for j in candidates:
                if i == j or not name_sets[j]:
                    continue
                if name_sets[j] <= page["tokens"]:
                    graph.add_edge(page["page_url"], pages[j]["page_url"])
        return graph

    @staticmethod
    def centrality(graph: nx.DiGraph) -> dict[str, dict]:
        """PageRank, HITS and degree for every node."""
        n = graph.number_of_nodes()
        if n == 0:
            return {}

        pagerank = nx.pagerank(graph, alpha=0.85) if graph.number_of_edges() else {
            node: 1 / n for node in graph
        }
        try:
            hubs, authorities = nx.hits(graph, max_iter=200, normalized=True)
        except (nx.PowerIterationFailedConvergence, ZeroDivisionError):
            hubs = authorities = {node: 0.0 for node in graph}

        undirected = graph.to_undirected()
        components = {
            node: len(component)
            for component in nx.connected_components(undirected)
            for node in component
        }

        return {
            node: {
                "pagerank": pagerank.get(node, 0.0),
                "hub": hubs.get(node, 0.0),
                "authority": authorities.get(node, 0.0),
                "in_degree": graph.in_degree(node),
                "out_degree": graph.out_degree(node),
                "component_size": components.get(node, 1),
                # Normalised so franchises of different sizes are comparable.
                "pagerank_rank": 0.0,
            }
            for node in graph
        }

    @staticmethod
    def communities(graph: nx.DiGraph, seed: int = LOUVAIN_SEED) -> tuple[dict[str, dict], dict]:
        """Partition one franchise's co-mention graph with the Louvain method.

        The graph is collapsed to an undirected weighted one first, because
        modularity is defined against the configuration model on undirected
        degrees. A reciprocated co-mention (each page names the other) is
        stronger evidence that two characters share scenes than a one-way
        mention, so reciprocity becomes the edge weight rather than being
        discarded.

        Returns per-node community columns and a per-franchise summary. Both
        are computed from topology alone; `is_dead_v2` is not read here, and
        whether communities predict fate is asked separately in
        src/q1_character/community.py.
        """
        undirected = nx.Graph()
        undirected.add_nodes_from(graph.nodes)
        for source, target in graph.edges:
            if undirected.has_edge(source, target):
                undirected[source][target]["weight"] = 2.0
            else:
                undirected.add_edge(source, target, weight=1.0)

        n = undirected.number_of_nodes()
        if n == 0:
            return {}, {}
        if undirected.number_of_edges() == 0:
            # No edges means every node is its own community and modularity is
            # undefined rather than zero. Reported as such instead of imputed.
            parts = [{node} for node in undirected]
            modularity = float("nan")
        else:
            parts = nx.community.louvain_communities(
                undirected, weight="weight", seed=seed
            )
            modularity = nx.community.modularity(undirected, parts, weight="weight")

        membership = {
            node: index for index, part in enumerate(parts) for node in part
        }
        degree = dict(undirected.degree(weight="weight"))

        columns = {}
        for index, part in enumerate(parts):
            # Degree rank inside the community: 0 for the most-connected member,
            # 1 for the least. Distinguishes the character a narrative cluster
            # is built around from the rest of the cluster.
            ordered = sorted(part, key=lambda node: -degree[node])
            for position, node in enumerate(ordered):
                internal = sum(
                    data.get("weight", 1.0)
                    for neighbour, data in undirected[node].items()
                    if membership[neighbour] == index
                )
                total = degree[node]
                columns[node] = {
                    "community_id": index,
                    "community_size": len(part),
                    "community_share": len(part) / n,
                    # Share of a character's co-mention weight that stays inside
                    # their own community. High means the character belongs to
                    # one storyline; low means they cross between storylines.
                    "embeddedness": internal / total if total else 0.0,
                    "community_rank": position / max(len(part) - 1, 1),
                }

        summary = {
            "n_nodes": n,
            "n_edges": undirected.number_of_edges(),
            "n_communities": len(parts),
            "modularity": modularity,
            "largest_community_share": max(len(p) for p in parts) / n,
        }
        return columns, summary

    @staticmethod
    def build() -> tuple[pd.DataFrame, pd.DataFrame]:
        rows = []
        structure = []
        category_counts: Counter = Counter()
        field_counts: Counter = Counter()

        for path in sorted(RAW_DIR.glob("*.jsonl")):
            pages = FeatureBuilder.read_franchise(path)
            if not pages:
                continue

            graph = FeatureBuilder.build_graph(pages)
            scores = FeatureBuilder.centrality(graph)
            community_columns, community_summary = FeatureBuilder.communities(graph)
            if community_summary:
                structure.append({
                    "franchise": pages[0]["franchise"], **community_summary
                })

            # Rank pagerank within the franchise, so the value means "how
            # central relative to this cast" rather than "how big this wiki is".
            ordered = sorted(scores, key=lambda k: -scores[k]["pagerank"])
            for position, node in enumerate(ordered):
                scores[node]["pagerank_rank"] = position / max(len(ordered) - 1, 1)

            for page in pages:
                category_counts.update(page["categories"])
                field_counts.update(page["infobox_fields"])
                rows.append({
                    "page_url": page["page_url"],
                    "franchise": page["franchise"],
                    "n_relatives": page["n_relatives"],
                    "n_categories": len(page["categories"]),
                    "intro": page["intro"],
                    "intro_clean": page["intro_clean"],
                    "categories": page["categories"],
                    "infobox_fields": page["infobox_fields"],
                    **scores.get(page["page_url"], {}),
                    **community_columns.get(page["page_url"], {}),
                })
            q = community_summary.get("modularity", float("nan"))
            print(f"  {path.stem:38s} {len(pages):6d} pages  "
                  f"{graph.number_of_edges():7d} edges  "
                  f"{community_summary.get('n_communities', 0):5d} communities  "
                  f"Q={q:.3f}")

        # The scrape lists some characters under more than one category, so the
        # same page can appear twice.
        df = pd.DataFrame(rows).drop_duplicates(subset="page_url", keep="first")

        top_categories = [c for c, _ in category_counts.most_common(TOP_CATEGORIES)]
        top_fields = [f for f, _ in field_counts.most_common(TOP_INFOBOX_FIELDS)]

        for category in top_categories:
            column = "cat_" + re.sub(r"\W+", "_", category.lower())[:40]
            df[column] = df["categories"].apply(lambda cs: int(category in cs))
        for field in top_fields:
            column = "ib_" + re.sub(r"\W+", "_", field)[:30]
            df[column] = df["infobox_fields"].apply(lambda fs: int(field in fs))

        return (
            df.drop(columns=["categories", "infobox_fields"]),
            pd.DataFrame(structure),
        )


def seed_check(seeds: int = 5) -> pd.DataFrame:
    """How much does the Louvain partition depend on the seed?

    Louvain moves nodes in an arbitrary order, so the partition it returns is
    not unique. We run it at several seeds per franchise and report how often
    two characters are placed together consistently, using the pairwise
    community-membership measure from the community detection lecture.

    A franchise whose partition is stable across seeds can have its community
    columns read as narrative structure. One that is not is reporting an
    artefact of node ordering.
    """
    rows = []
    for path in sorted(RAW_DIR.glob("*.jsonl")):
        pages = FeatureBuilder.read_franchise(path)
        if not pages:
            continue

        graph = FeatureBuilder.build_graph(pages)
        labellings, modularities = [], []
        for seed in range(seeds):
            columns, summary = FeatureBuilder.communities(graph, seed=seed)
            if not columns:
                break
            order = sorted(columns)
            labellings.append(
                np.array([columns[node]["community_id"] for node in order])
            )
            modularities.append(summary["modularity"])
        if len(labellings) < 2:
            continue

        agreements = []
        for a in range(len(labellings)):
            for b in range(a + 1, len(labellings)):
                same_a = labellings[a][:, None] == labellings[a][None, :]
                same_b = labellings[b][:, None] == labellings[b][None, :]
                agreements.append(float((same_a == same_b).mean()))

        rows.append({
            "franchise": pages[0]["franchise"],
            "n_nodes": graph.number_of_nodes(),
            "pair_agreement": float(np.mean(agreements)),
            "modularity_mean": float(np.mean(modularities)),
            "modularity_sd": float(np.std(modularities)),
        })
        print(f"  {path.stem:38s} agreement {rows[-1]['pair_agreement']:.4f}  "
              f"Q {rows[-1]['modularity_mean']:.3f} "
              f"+/- {rows[-1]['modularity_sd']:.3f}")

    return pd.DataFrame(rows)


def main() -> None:
    if "--seed-check" in sys.argv:
        print(f"Louvain seed sensitivity over {SEED_CHECK_SEEDS} seeds:")
        stability = seed_check(SEED_CHECK_SEEDS)
        out_path = CLEAN_DIR / "q1_louvain_stability.csv"
        stability.to_csv(out_path, index=False, encoding="utf-8")
        print(f"\nmean pairwise agreement across seeds "
              f"{stability['pair_agreement'].mean():.4f}")
        print(f"lowest: {stability.nsmallest(3, 'pair_agreement')[['franchise', 'pair_agreement']].to_string(index=False)}")
        print(f"modularity sd across seeds, max {stability['modularity_sd'].max():.4f}")
        print(f"Wrote {out_path}")
        return

    print("Building features per franchise:")
    df, structure = FeatureBuilder.build()

    out_path = CLEAN_DIR / "character_features.csv"
    df.to_csv(out_path, index=False, encoding="utf-8")

    structure_path = CLEAN_DIR / "q1_community_structure.csv"
    structure.to_csv(structure_path, index=False, encoding="utf-8")

    generated = [c for c in df.columns if c.startswith(("cat_", "ib_"))]
    print(f"\n{len(df):,} pages, {len(df.columns)} columns "
          f"({len(generated)} one-hot category/infobox features)")

    # The community detection lecture puts significant structure at Q > 0.3.
    # Franchises below that get a partition the algorithm will always return,
    # but one no better than the configuration-model null would give, so their
    # community columns should not be read as narrative clusters.
    quality = structure["modularity"].dropna()
    strong = int((quality > MODULARITY_FLOOR).sum())
    print(f"\nLouvain: {structure['n_communities'].sum():,} communities across "
          f"{len(structure)} franchises")
    print(f"  median modularity Q {quality.median():.3f}   "
          f"range {quality.min():.3f}-{quality.max():.3f}")
    print(f"  {strong} of {len(quality)} franchises above the Q > "
          f"{MODULARITY_FLOOR} threshold for interpretable community structure")
    print(f"\nWrote {out_path}")
    print(f"Wrote {structure_path}")


if __name__ == "__main__":
    from src import setup_run_log
    setup_run_log(__spec__)
    main()
