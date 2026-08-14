# Pipeline

Run order, what each stage reads, and what it writes. Every stage is
idempotent: re-running overwrites its outputs and nothing else.

On Windows, prefix legacy top-level scripts with `PYTHONUTF8=1`. Anything under
`src/` sets UTF-8 itself.

## Stage 0 — collection

| step | command | writes |
|---|---|---|
| Scrape 38 Fandom wikis | `python -m scraper.run_scraper --wiki all` | `data/raw/*.jsonl` (gitignored) |
| Build feature table | `python build_pipeline.py` | `data/clean/characters_raw.csv`, `characters_keys.csv` |

`build_pipeline.py` checkpoints each stage to `data/clean/.ckpt/`, so an
interrupted run resumes rather than starting over. Delete that directory to
force a clean build.

**The TVmaze stage is off by default.** It cost about 35 minutes — roughly two
minutes per TV franchise — and matched only 1,851 of 67,159 rows (2.8%). The
actor age it exists to provide came out 0.9% non-null, which is too sparse to
model on, and nothing under `src/` reads any of its four columns. Re-enable with:

```bash
WITH_TVMAZE=1 python build_pipeline.py
```

`characters_keys.csv` is row-aligned with `characters_raw.csv` and is the only
way back to character names and URLs.

Pages are discovered by asking each wiki which pages transclude the templates in
`scraper/wiki_configs.py`. Two things about that are worth knowing before
re-scraping:

- Transclusion follows template redirects but the parser matches the template
  name as written in the wikitext, so the scraper now also fetches each
  template's redirects and lets the parser accept those names. Without this, a
  wiki whose pages use a redirect gets found and then silently dropped.
- The busiest character-looking template on a wiki is frequently not characters.
  `Infobox Cast` on the Peaky Blinders wiki is the real actors; Fast & Furious's
  busiest is `Infobox Car`. Check sample pages before adding a template.

`build_pipeline.py` also drops pages that are objects rather than people
(talking portraits, background paintings) via `is_non_character` in
`src/constants.py`. Rows with no parseable `is_dead` are dropped here too, which
is why a better scrape does not help the franchises whose infoboxes carry no
status field.

## Stage 1 — labels

| step | command | writes |
|---|---|---|
| Audit label derivation | `python -m src.labels.audit_labels` | `data/clean/label_audit.csv` |
| Draw annotation sample | `python -m src.labels.build_gold_set` | `data/gold/gold_sample.csv` (600 chars) |
| Split into chunks | `python -m src.labels.annotate --split` | `data/gold/chunks/chunk_NN.csv` |
| Annotate | Haiku 4.5 subagents, one per chunk | `data/gold/annotations/passN/chunk_NN.jsonl` |
| Score the label | `python -m src.labels.eval_labels` | `data/clean/label_eval.csv`, `corpus_composition.csv` |
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

### The analysis population

Every question restricts rows the same way, through `analysis_population()` in
`src/constants.py`: on screen, and named. "Named" excludes pages titled by role
rather than by a character — "Unidentified Ugnaught worker", "Unnamed pilot" —
which are real on-screen extras but not what this project measures, and which
were 7.2% of the on-screen set, concentrated in Star Wars and Harry Potter.

To reproduce the numbers with those rows kept in:

```bash
INCLUDE_UNNAMED_ROLES=1 python -m src.q1_character.models
```

The filter only catches explicit `Unidentified`/`Unnamed`/`Unknown` prefixes, so
role-titled pages like Dexter's "Acupuncturist" still get through — see Known
issues in the README.

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
| Wide feature matrix | `python -m src.q1_character.features` | `data/clean/character_features.csv` (369 one-hot + network + text) |

Five families, all from data already on disk: Fandom categories, infobox field
names, family-relation counts, per-franchise co-mention graphs (PageRank, HITS,
degree, component size), and the opening paragraph as text. Every feature's
definition and computation is in **[FEATURES.md](FEATURES.md)**.

Anything naming death **or survival** is excluded, as are the `status` and `dod`
fields the label is derived from. Both were caught by reading model coefficients
after the fact; a category named `Alive` is the label with the sign flipped.

The text column needs the same care and gets it twice over. `intro_clean` is the
opening paragraph with every death- or survival-mentioning sentence dropped and
then any surviving death token removed, because a dead character's first
paragraph says "was killed by" and TF-IDF over the raw text would mostly
rediscover the label. The vectoriser itself is fit **inside** the
cross-validation pipeline (`make_pipeline` in `src/q1_character/models.py`), so
the vocabulary and IDF weights never see a test document. Checked afterwards:
the 300-term vocabulary contains no death or survival token, and no tense
marker either — `is` and `was` are English stopwords and already dropped.

## Stage 4 — research questions

| question | command | writes |
|---|---|---|
| Q1 character vs franchise | `python -m src.q1_character.models` | `q1_model_comparison.csv`, `q1_feature_importance.csv` |
| Q1 train/test assignment | `python -m src.q1_character.splits` | `split_assignment.csv` |
| Q1 honest held-out score | `python -m src.q1_character.evaluate` | `q1_holdout_results.csv`, `q1_dev_selection.csv`, `q1_dev_errors.csv` |
| Q1 gender effect sizes | `python -m src.q1_character.effects` | `q1_gender_pooled.csv`, `q1_gender_by_franchise.csv` |
| Q2 text (partial) | folded into `src.q1_character.models` as the `text` and `character+text` sets | — |
| Q3 franchise mortality | `python -m src.q3_franchise.mortality` | `q3_franchise_mortality.csv`, `q3_glm_summary.csv`, `q3_kmeans_diagnostics.csv`, `q3_franchise_clusters.csv` |

`models.py` compares every feature set against every model under cross
validation — the right tool for "which features carry the signal". `evaluate.py`
answers the different question "how good is this actually", by selecting with
k-fold *inside* trainval and then scoring `test` exactly once, because the best
of 27 cross-validated cells is optimistic if you report it as a result.

Q2's text half is answered through the `text` and `character+text` feature sets
above. The dialogue half is blocked on death registries rather than transcripts:
transcripts exist for 11 shows but usable episode-level death lists for only
two, so the largest transcript set (464 Grey's Anatomy episodes) has nothing to
align against, and most transcripts lack speaker labels anyway. Registries on
disk: Game of Thrones, The 100, The Walking Dead (`datasets/*/`), plus the
FiveThirtyEight GoT table in `data/external/kaggle/`. The
listofdeaths.fandom.com scrapers that produced them are in `legacy/scrapers/`.

## Stage 5 — outputs

| step | command | writes |
|---|---|---|
| Figures | `python -m src.viz.figures` | `data/clean/figures/*.png` |
| Demo data | `python -m src.app.export_demo` (needs `data/clean/split_assignment.csv` from Stage 4) | `data/clean/demo_data.json` and `docs/demo_data.json` |

The demo model is a logistic regression over user-selectable attributes only, so
it can be evaluated in JavaScript. Its coefficients, category levels and scaler
parameters are exported; the browser reproduces the Python prediction exactly.

It's fit on `split_random == "trainval"` and only ever shows `split_random ==
"test"` characters — fitting and scoring on the same rows was an earlier bug
(see README, Known issues). Run `src.q1_character.splits` first if
`split_assignment.csv` doesn't exist yet.

`export_demo.py` writes the same JSON to `data/clean/` and to `docs/`. Copying
it by hand was a step easy to forget, which left the served page one model
behind the one that had just been fit.

Figures read their numbers from the tables rather than restating them: the
gender forest counts its own significant franchises, the run-span caption reads
`q3_glm_summary.csv`, and the corpus bar reads `corpus_composition.csv`. Three
captions had gone stale against the data before this.

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
  q1_character/      character vs franchise models and gender effects
  q2_text/           script text (empty — see Stage 4)
  q3_franchise/      franchise mortality, regression, clustering
  viz/               figure rendering and shared chart style
  app/               demo data export
data/
  raw/               scraped JSONL (gitignored)
  external/          third-party dumps and TMDB cache (gitignored)
  clean/             built tables and figures (tracked)
  gold/              annotation sample and labels (tracked)
datasets/            per-show transcripts, death lists, character features
scraper/             Fandom scraper
legacy/              pre-restructure scripts, kept for the record, not run
tests/
```
