# Pipeline

Run order, what each stage reads, and what it writes. Every stage is
idempotent: re-running overwrites its outputs and nothing else.

On Windows, prefix legacy top-level scripts with `PYTHONUTF8=1`. Anything under
`src/` sets UTF-8 itself.

## Stage 0 — collection

| step | command | writes |
|---|---|---|
| Scrape 38 Fandom wikis | `python -m scraper.run_scraper --wiki all` | `data/raw/*.jsonl` (75,521 pages, 262 MB, gitignored) |
| Build feature table | `python build_pipeline.py` | `data/clean/characters_raw.csv` (67,327 x 119), `characters_keys.csv` |

`characters_keys.csv` is row-aligned with `characters_raw.csv` and is the only
way back to character names and URLs.

## Stage 1 — labels

| step | command | writes |
|---|---|---|
| Audit label derivation | `python -m src.labels.audit_labels` | `data/clean/label_audit.csv` |
| Draw annotation sample | `python -m src.labels.build_gold_set` | `data/gold/gold_sample.csv` (600 chars) |
| Split into chunks | `python -m src.labels.annotate --split` | `data/gold/chunks/chunk_NN.csv` |
| Annotate | Haiku 4.5 subagents, one per chunk | `data/gold/annotations/passN/chunk_NN.jsonl` |
| Score the label | `python -m src.labels.eval_labels` | `data/clean/label_eval.csv` |
| Build modelling table | `python -m src.labels.rebuild_label` | `data/clean/characters_model.csv` |

### What the label means

The wikis use two infobox conventions, so `is_dead` is derived two ways:

- **status** (30 wikis) — read from the infobox `status` field. Rows whose
  status is absent or unrecognised are left unlabelled and dropped upstream.
- **dod_only** (8 wikis) — a recorded date of death means dead. Nothing is
  dropped, but the label then means "dead somewhere in the fictional
  chronology" rather than "dies during the story".

`is_dead_v2` restricts the analysis to characters who appear on screen.
`is_onscreen` is `has_actor OR tmdb_match`; measured against the annotated
sample it scores precision 0.79, recall 0.77. Neither signal is enough alone
(0.61 and 0.58 recall respectively).

Annotations come from Haiku 4.5, not from people. Inter-pass agreement measures
reliability, not correctness.

## Stage 2 — enrichment

| step | command | writes |
|---|---|---|
| TMDB titles and cast | `python -m src.enrich.tmdb_enrich` | `data/clean/tmdb_titles.csv` (280), `tmdb_cast.csv` (28,151) |

Needs `TMDB_API_KEY` in `.env` (gitignored). Responses cache to
`data/external/tmdb/`, so re-runs cost no requests.

Supplies `billing_order` and `episode_count` as prominence measures, replacing
the hardcoded page-length cutoffs, plus per-title genre, rating and season
counts for the franchise-level analysis.

## Stage 3 — features

| step | command | writes |
|---|---|---|
| Wide feature matrix | `python -m src.q1_character.features` | `data/clean/character_features.csv` (369 one-hot + network) |

Five families, all from data already on disk: Fandom categories, infobox field
names, family-relation counts, per-franchise co-mention graphs (PageRank, HITS,
degree, component size), and the opening paragraph kept as text.

Anything naming death **or survival** is excluded, as are the `status` and `dod`
fields the label is derived from. Both were caught by reading model coefficients
after the fact; a category named `Alive` is the label with the sign flipped.

## Stage 4 — research questions

| question | command | writes |
|---|---|---|
| Q1 character vs franchise | `python -m src.q1_character.models` | `q1_model_comparison.csv`, `q1_feature_importance.csv` |
| Q1 gender effect sizes | `python -m src.q1_character.effects` | `q1_gender_pooled.csv`, `q1_gender_by_franchise.csv` |
| Q3 franchise mortality | `python -m src.q3_franchise.mortality` | `q3_franchise_mortality.csv`, `q3_kmeans_diagnostics.csv`, `q3_franchise_clusters.csv` |
| Q2 script text | not built yet | |

Q2 is blocked on death registries. Transcripts exist for 11 shows but episode
death lists only for four, so the largest transcript set (464 Grey's Anatomy
episodes) is currently unusable.

## Stage 5 — outputs

| step | command | writes |
|---|---|---|
| Figures | `python -m src.viz.figures` | `data/clean/figures/*.png` |
| Demo data | `python -m src.app.export_demo` | `data/clean/demo_data.json` → copy to `docs/` |

The demo model is a logistic regression over user-selectable attributes only, so
it can be evaluated in JavaScript. Its coefficients, category levels and scaler
parameters are exported; the browser reproduces the Python prediction exactly
(verified across all 3,336 exported characters).

## Tests

```
python -m pytest tests -q
```

## Layout

```
src/
  constants.py       shared constants, no magic numbers in the modules
  paths.py           canonical data locations
  labels/            label audit, annotation, evaluation, rebuild
  enrich/            TMDB titles and cast
  q1_character/      character vs franchise models
  q2_text/           script text (empty)
  q3_franchise/      franchise mortality (empty)
  viz/               figures (empty)
  app/               demo (empty)
data/
  raw/               scraped JSONL (gitignored)
  external/          third-party dumps and TMDB cache (gitignored)
  clean/             built tables and figures (tracked)
  gold/              annotation sample and labels (tracked)
datasets/            per-show transcripts, death lists, character features
scraper/             Fandom scraper
tests/
```
