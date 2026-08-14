# Spoiler Alert: Who Makes It to the Finale?

Group #56 — 67978 A Needle in a Data Haystack, Hebrew University of Jerusalem.

Which fictional characters die, and why? We scraped 38 Fandom wikis, enriched them
with TMDB cast data, and asked three questions: whether a character's own
attributes or their franchise better predicts their death, what franchise
properties explain deadliness, and how script text relates to on-screen deaths.

Interactive demo: `docs/index.html` (see [Demo](#demo)).
Run order and file-by-file detail: **[PIPELINE.md](PIPELINE.md)**.

## Data

| source | size | what it gives |
|---|---|---|
| 38 Fandom wikis | 75,521 pages, 262 MB | infoboxes, full page text, categories |
| TMDB | 280 titles, 28,151 cast credits | billing order, episode counts, genre, ratings |
| Kaggle / listofdeaths | 4 shows | per-episode death registries |
| Transcripts | 1,136 episodes, 11 shows | dialogue, mostly without speaker labels |

Not every scraped page is a character. From 600 annotated pages reweighted to the
full corpus, **40% are on-screen characters**, 46% are characters who never appear
in a film or episode (Star Wars Legends novels, Harry Potter reference books), and
12% are not characters at all — creature types, vehicles, unnamed roles, and real
historical figures picked up from the Young Indiana Jones wiki. All analysis is
restricted to the first group: **15,633 characters, 35.1% of whom die**.

### How the label is built

The wikis follow two infobox conventions, so `is_dead` is derived two ways. Thirty
wikis record a `status` string; eight record a date of death, from which death is
inferred. The two are not directly comparable, and rows whose status cannot be
parsed were previously dropped rather than counted. `is_dead_v2` restricts to
characters who appear on screen and keeps unparsed rows in the denominator.

`is_onscreen` is `has_actor OR tmdb_match`. Measured against the annotated sample:
**precision 0.79, recall 0.77**. Neither signal is sufficient alone (0.61 and 0.58
recall respectively).

Annotations were produced by Claude Haiku 4.5, not by people. Inter-pass agreement
would measure reliability rather than correctness, and the label has no external
non-LLM validation — see [Limitations](#limitations).

## Findings

### Q1 — the character, and the franchise, and neither alone

Seven feature sets, three models, two cross-validation regimes, two baselines.

| features | random split | held-out franchise |
|---|---|---|
| all features | **0.822** | 0.557 |
| character + franchise | 0.730 | 0.532 |
| character + network | 0.721 | 0.552 |
| character | 0.684 | 0.518 |
| network centrality only | 0.655 | 0.480 |
| franchise only | 0.562 | 0.365 |
| baseline | 0.351 | 0.351 |

PR-AUC, best model per set. Character attributes beat franchise identity, and the
two are complementary. Stripped of wiki-metadata proxies, character-only falls to
0.596 — still above franchise-only, but the margin narrows.

The two splits disagree in a useful way. Fandom category features add +0.09 on a
random split and +0.005 when whole franchises are held out: they encode
franchise-specific role vocabulary that does not transfer. **Network centrality is
the opposite** — PageRank and HITS over the per-franchise co-mention graph add
about +0.035 in *both* regimes, so how central a character is to the story
generalises to franchises the model has never seen.

### Gender

Being female roughly halves the odds of dying: **OR 0.55 (95% CI 0.50–0.60)**,
or 11.4 percentage points lower death probability, after adjusting for prominence,
role, alignment, species and franchise. The raw gap is 14.9 points, so about a
quarter of it is explained by men holding more disposable roles.

Of 19 franchises where the effect could be estimated, **9 reach p<0.05 and none
point the other way**. The weak cases (Grey's Anatomy, Avatar, The Sopranos) are
null effects, not reversals.

### Q3 — franchise properties explain very little

Weighted least squares on logit(mortality), franchise as the unit, n=36. Only run
span (p=0.010) and first release year (p=0.027) are significant; medium, number of
installments and audience rating are not. **R² = 0.22, adjusted R² = 0.09.**

k-means over franchise properties does not recover mortality groupings. The best
silhouette is at k=2, and that split separates large multi-title film franchises
from everything else — the two clusters have near-identical mortality (35.0% vs
37.8%).

## Demo

`docs/` is a self-contained page: look up any of 3,336 characters to see their
predicted risk, which attributes drove it, whether they actually died, and
comparable characters — or build a character from scratch and watch the estimate
move.

The model runs in the browser from exported logistic-regression coefficients. It
uses only attributes a person can choose, so it scores lower (ROC-AUC 0.741) than
the project's full model; the page says so.

Locally:

```bash
cd docs && python -m http.server 8765     # then open http://localhost:8765
```

On GitHub Pages: repository **Settings → Pages → Source: Deploy from a branch**,
branch `main`, folder `/docs`.

## Layout

```
src/
  constants.py     shared constants
  paths.py         canonical data locations
  labels/          label audit, gold-set sampling, annotation, evaluation, rebuild
  enrich/          TMDB titles, cast, and name matching
  q1_character/    feature building and the character-vs-franchise models
  q3_franchise/    franchise mortality, regression, clustering
  viz/             figure rendering and shared chart style
  app/             demo model export
scraper/           Fandom scraper (38 wiki configs)
data/
  raw/             scraped JSONL (gitignored, 262 MB)
  external/        third-party dumps and the TMDB cache (gitignored)
  clean/           built tables and figures
  gold/            annotation sample and labels
datasets/          transcripts, death registries, per-show character features
docs/              the demo, served by GitHub Pages
tests/             pytest
```

## Running it

See **[PIPELINE.md](PIPELINE.md)** for the full order. Short version:

```bash
python -m src.labels.audit_labels        # label audit
python -m src.enrich.tmdb_enrich         # needs TMDB_API_KEY in .env
python -m src.labels.rebuild_label       # corrected label
python -m src.q1_character.features      # ~1,000 features
python -m src.q1_character.models        # Q1
python -m src.q1_character.effects       # gender effect sizes
python -m src.q3_franchise.mortality     # Q3
python -m src.viz.figures                # figures
python -m src.app.export_demo            # demo data
python -m pytest tests -q
```

On Windows, prefix the legacy top-level scripts (`build_pipeline.py`, `tfidf.py`)
with `PYTHONUTF8=1` — they print non-ASCII and the console defaults to cp1252.

## Limitations

- **The label rests on LLM annotations.** 600 pages labelled by Haiku 4.5, with no
  human-labelled or otherwise independent validation set. The Fandom `Deceased`
  category looked like external validation but is written by the same editors from
  the same source, and using it as a feature would be leakage.
- **`is_onscreen` is noisy** (P 0.79 / R 0.77). Roughly a fifth of on-screen
  characters are excluded and a fifth of those included do not belong. That error
  propagates into every number here, including the gender effect sizes.
- **The two strongest Q1 features are wiki metadata.** `infobox_field_count` and
  `page_text_length` measure how much fans wrote, not anything about the character.
  Dropping them costs 0.088 PR-AUC. `prominence_tier` is derived from page length
  upstream, so even the "no wiki metadata" ablation is not fully clean.
- **Franchises with sparse wikis carry wide label uncertainty.** Boardwalk Empire's
  mortality is somewhere between 20% and 100% depending on how 440 characters with
  no recorded status resolve. Figures show that range rather than a point estimate.
- **Star Wars dominates the corpus** (56% of scraped pages, 18% after the on-screen
  filter). Grouped cross-validation and per-franchise reporting mitigate this but
  do not remove it.
- **Two leakage bugs were found and fixed** by inspecting model coefficients rather
  than trusting scores: a violence lexicon containing death words, and the
  `status`/`dod` infobox fields the label is derived from. Both inflated results
  before removal.
