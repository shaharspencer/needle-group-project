"""Build the wide feature matrix for Q1 from the raw scrape.

Everything here comes from data already on disk. Five families:

  categories   Fandom category lines, one-hot over the most common ones.
               99.3% of pages carry them and they encode species, occupation,
               affiliation and status.
  infobox      which infobox fields exist, rather than only how many.
  relations    counts of named family members.
  network      per-franchise co-mention graph. An edge runs from a character
               to every other character named on their page. PageRank, HITS
               and degree measure how central a character is to the story,
               which is the link-analysis question.
  intro        the opening paragraph, kept as text for TF-IDF downstream.

Anything naming death is excluded throughout. A "Deceased" category or a
cause-of-death field restates the label rather than predicting it.

  python -m src.q1_character.features
"""

import json
import re
from collections import Counter, defaultdict

import networkx as nx
import pandas as pd

from src.paths import CLEAN_DIR, RAW_DIR

CATEGORY_RE = re.compile(r"Category:([^\n]+)")
TOKEN_RE = re.compile(r"[a-z0-9]+")

# Any category or infobox field matching these restates the outcome. Survival
# words matter as much as death words: a category named "Alive" is the label
# with the sign flipped.
DEATH_TERMS = re.compile(
    r"death|deceas|died|dead|kill|murder|victim|casualt|fatal|corpse|"
    r"posthum|grave|funeral|obitu|execut|assassinat|suicide|martyr|sacrific|"
    r"slain|slay|perish|massacre|memorial|remembr|undead|ghost|revenant|"
    r"resurrect|alive|living|surviv|fallen|fate",
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


class FeatureBuilder:

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
                if record.get("is_redirect"):
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
    def build() -> pd.DataFrame:
        rows = []
        category_counts: Counter = Counter()
        field_counts: Counter = Counter()

        for path in sorted(RAW_DIR.glob("*.jsonl")):
            pages = FeatureBuilder.read_franchise(path)
            if not pages:
                continue

            graph = FeatureBuilder.build_graph(pages)
            scores = FeatureBuilder.centrality(graph)

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
                    "categories": page["categories"],
                    "infobox_fields": page["infobox_fields"],
                    **scores.get(page["page_url"], {}),
                })
            print(f"  {path.stem:38s} {len(pages):6d} pages  "
                  f"{graph.number_of_edges():7d} edges")

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

        return df.drop(columns=["categories", "infobox_fields"])


def main() -> None:
    print("Building features per franchise:")
    df = FeatureBuilder.build()

    out_path = CLEAN_DIR / "character_features.csv"
    df.to_csv(out_path, index=False, encoding="utf-8")

    generated = [c for c in df.columns if c.startswith(("cat_", "ib_"))]
    print(f"\n{len(df):,} pages, {len(df.columns)} columns "
          f"({len(generated)} one-hot category/infobox features)")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
