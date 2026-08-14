# Spoiler Alert: Who Makes It to the Finale?

Group #56 — 67978 A Needle in a Data Haystack, Hebrew University of Jerusalem.

Which fictional characters die, and why? We scraped 38 Fandom wikis, enriched
them with TMDB cast data, and asked whether a character's own attributes or the
franchise they are in better predicts their death.

**This file is the operational one: how to set the project up, what to run, and
where the outputs land.** For what we found and what it means, read
[WRITEUP.md](WRITEUP.md). For how each feature is computed, read
[FEATURES.md](FEATURES.md). For a stage-by-stage description of the pipeline,
read [PIPELINE.md](PIPELINE.md).

| document | answers |
|---|---|
| README.md (this file) | how do I run it, and where does the output go |
| [WRITEUP.md](WRITEUP.md) | what did we find |
| [PIPELINE.md](PIPELINE.md) | what does each stage read and write |
| [FEATURES.md](FEATURES.md) | how is each feature computed |

## Setup

Python 3.12. From the repository root:

```bash
pip install pandas numpy scikit-learn statsmodels matplotlib networkx requests
```

**Install `git-lfs` before cloning.** `data/clean/character_features.csv` is
about 140 MB and is stored in Git LFS. Without git-lfs you get a 134-byte pointer
in its place and every stage that reads it fails with a parse error rather than
a missing-file error, which is a confusing way to find out.

```bash
git lfs install && git lfs pull      # if you already cloned without it
```

Two more things are needed before a full run and neither is in the repository:

- **`.env`** with `TMDB_API_KEY=<key>`. Gitignored. Only the enrichment stage
  reads it, and TMDB responses cache to `data/external/tmdb/`, so a second run
  costs no requests.
- **`data/raw/`** — 268 MB of scraped Fandom JSONL, gitignored because it is
  reproducible. Regenerate with `python -m scraper.run_scraper --wiki all`,
  which takes hours. Everything downstream of it is tracked, so you only need
  this if you are changing the scrape or rebuilding the feature table.

Everything under `data/clean/` and `data/gold/` **is** tracked, including the
built feature matrix. From a fresh clone you can therefore reproduce every number
in the writeup without re-scraping, starting at Stage 4 — Stages 0 and 3 are the
only ones that need `data/raw/`.

On Windows, prefix `build_pipeline.py` with `PYTHONUTF8=1`. It prints non-ASCII
and the console defaults to cp1252, which otherwise kills the run.

## Running it

[PIPELINE.md](PIPELINE.md) has the full order with inputs and outputs per stage.
The short version, in dependency order:

```bash
# Stage 0 — collection (only if data/raw changed; slow)
python -m scraper.run_scraper --wiki all
PYTHONUTF8=1 python build_pipeline.py     # -> characters_raw.csv

# Stage 1 — labels
python -m src.labels.audit_labels
python -m src.labels.rebuild_label        # -> characters_model.csv
python -m src.labels.eval_labels          # -> label_eval.csv, corpus_composition.csv
python -m src.labels.validate_registry    # check the label against outside sources
python -m src.labels.fill_unlabelled --collect   # then Haiku, then --merge

# Stage 2 — enrichment (needs TMDB_API_KEY)
python -m src.enrich.tmdb_enrich

# Stage 3 — features
python -m src.q1_character.features       # -> character_features.csv

# Stage 4 — the questions
python -m src.q1_character.splits         # train/test assignment; run before models
python -m src.q1_character.models         # 9 feature sets x 3 models x 2 CV regimes
python -m src.q1_character.evaluate       # the held-out test score
python -m src.q1_character.effects        # gender effect sizes
python -m src.q2_text.coefficients        # -> q2_text_coefficients.csv
python -m src.q3_franchise.mortality      # franchise mortality, WLS, k-means

# Stage 5 — outputs
python -m src.viz.figures                 # -> data/clean/figures/*.png
python -m src.app.export_demo             # -> data/clean/ and docs/demo_data.json

python -m pytest tests -q
```

Stage order matters in three places, and nowhere else:

- `splits` before `models`, `evaluate` and `export_demo`, which all read
  `split_assignment.csv`.
- `eval_labels`, `q2_text.coefficients` and `q3_franchise.mortality` before
  `figures`. Figures read their numbers from those CSVs rather than restating
  them, so a missing one is a crash rather than a stale caption.
- `features` before anything in Stage 4.

Every stage is idempotent: re-running overwrites its own outputs and nothing
else. `build_pipeline.py` checkpoints to `data/clean/.ckpt/` and resumes from
there, so an interrupted build does not start over — delete that directory to
force a clean run.

Each `python -m src.X.Y` mirrors its console output to `logs/X.Y.log`, appended
with a timestamp header. Nothing reads those logs; they exist so you can watch a
long stage from a second shell.

## Where things live

```
build_pipeline.py  scrape -> characters_raw.csv (structural features, enrichment)
scraper/           Fandom scraper, 38 wiki configs in wiki_configs.py
src/
  constants.py     shared constants, and the rule that keeps data cleaning
                    from leaking across a train/test boundary
  paths.py         canonical data locations — import these, don't hardcode
  labels/          label audit, gold-set sampling, annotation, evaluation,
                    registry validation, Haiku fill for unlabelled rows
  enrich/          TMDB titles, cast, name matching
  q1_character/    features, model comparison, splits, held-out evaluation,
                    gender effect sizes
  q2_text/         coefficients.py — the fitted TF-IDF weights behind Q2
  q3_franchise/    franchise mortality, WLS regression, k-means
  viz/             figures.py renders every writeup figure, style.py is shared
  app/             demo model export
data/
  raw/             scraped JSONL (gitignored, 268 MB)
  external/        third-party dumps and the TMDB cache (gitignored)
  clean/           built tables, figures/ (tracked — these are our outputs)
  gold/            600-page annotation sample and its labels, plus the Haiku
                    fill for unlabelled rows (tracked)
datasets/          transcripts, death registries, per-show character features
docs/              the demo, served by GitHub Pages
legacy/            pre-restructure scripts, kept for the record, never run
logs/              per-run console output (gitignored)
tests/             pytest
```

Three conventions worth keeping to:

- **Import paths from `src/paths.py`.** No stage should hardcode a data
  location.
- **Labels produced by a model stay in their own files.** The 600-page
  annotation sample and the fill for unlabelled rows were both labelled by
  Claude Haiku 4.5, and neither is merged into `is_dead`. The fill feeds exactly
  one number, the Q3 mortality point estimates. Keep that separation if you add
  to it.
- **Constants go in `src/constants.py`.** `analysis_population()` lives there
  and is how every question restricts rows to the same population — on screen,
  and named. If you filter rows yourself somewhere else, the tables stop
  agreeing with each other, which has already happened once.

## Demo

**Live: https://shaharspencer.github.io/needle-group-project/**

`docs/` is a self-contained page. Look up any of ~2,000 characters to see their
predicted risk, which attributes drove it, whether they actually died, and
comparable characters — or build a character from scratch and watch the estimate
move.

To run it locally instead:

```bash
cd docs && python -m http.server 8765     # http://localhost:8765
```

The live link is served by GitHub Pages from this repository. If it 404s, the
setting is **Settings → Pages → Source: Deploy from a branch**, branch `final`,
folder `/docs` — `final` rather than `main`, since that is the branch this work
is on. `docs/.nojekyll` is required and already committed; without it Pages
runs the page through Jekyll and drops files it does not recognise.

The model runs in the browser from exported logistic-regression coefficients.
`src/app/export_demo.py` writes `demo_data.json` to both `data/clean/` and
`docs/`, so the served page cannot lag the model that was just fit.

Two constraints on it, both deliberate and both documented at the top of
`export_demo.py`:

- It is fit on `trainval` and only ever displays `test` rows, so every risk
  score and every "the model called this right" verdict is a held-out
  prediction.
- It exposes only attributes a person could actually choose for an invented
  character. That rules out page length and infobox field count — our two
  strongest features, and the two that measure how much fans wrote rather than
  anything about the character — along with anything under 2% populated. It
  therefore scores below the project's full model, and the page says so.

## Status

| question | state |
|---|---|
| Q1 — character vs franchise | done: 9 feature sets × 3 models × 2 CV regimes, plus a held-out test score |
| Q1 — gender effect sizes | done: pooled and per-franchise |
| Q2 — does text predict death? | done for descriptive text: TF-IDF over each character's wiki description, the best-transferring feature family we have. Dialogue is out of scope — see below |
| Q3 — franchise properties | done: WLS regression, k-means, Haiku-filled mortality for the four worst franchises |
| demo | done, deployed from `docs/` |
| tests | 17 passing, all on `src/labels/` |

Q2 is answered for text *about* a character. Dialogue is a different question and
is blocked on death registries rather than on transcripts: there are transcripts
for 11 shows but a usable per-episode death list for only two, so the largest
transcript set (464 Grey's Anatomy episodes) has nothing to align against, and
most transcripts carry no speaker labels, which makes attributing a line to a
character a prerequisite problem in its own right.

The text features themselves are built in `src/q1_character/features.py`
(`intro_clean`) and scored in `models.py`, since that is where the comparison
lives. `src/q2_text/coefficients.py` refits the text-only model on trainval and
writes the fitted weights, which is what Figure 5 in the writeup is drawn from.

## Known issues

Open, roughly in the order they would bite someone picking this up:

- **Some features are absent for whole franchises, not at random.**
  `affiliation_alignment` is missing for 12 franchises (20% of characters)
  because those wikis have no affiliation field, and `n_relatives` is 0 for
  seven franchises (37% of characters, including the MCU) because those wikis
  do not use the family infobox keys. Under grouped cross-validation, a
  held-out franchise can therefore be missing a feature the model learned to
  lean on. Treat per-franchise feature coverage as something to check before
  adding a feature, not after.
- **Several wikis are under-collected** and we have not confirmed the right
  template: The Walking Dead (49 rows), Stranger Things (78 of 221 candidate
  pages, apparently a per-character template the scraper design does not
  handle), Vikings, Spartacus, Westworld. Read the next item first.
- **The busiest character-shaped template on a wiki is often not characters.**
  Peaky Blinders' is its cast of real actors; Fast & Furious's is
  `Infobox Car`. Check a template against real pages before trusting it.
- **No minimum-n rule anywhere.** The Matrix contributes 4 characters and Peaky
  Blinders 11 to grouped CV, where each franchise is one fold weighted equally
  with Star Wars.
- **The unnamed-role filter only catches explicit markers** —
  `Unidentified`/`Unnamed`/`Unknown` prefixes — so Star Wars's "Unidentified
  Ugnaught worker" is dropped while Dexter's "Acupuncturist" stays.
  Inconsistent by construction.
- **414 Matrix pages cannot be labelled from source at all.** Their infoboxes
  have no status and no date-of-death field, 0 of 414, checked directly. Not a
  parsing bug and nothing to fix.
- **Species-based mortality is probably not clean.** A walker is a dead human;
  a vampire is dead by species but alive in the story, so `is_dead` does not
  mean the same thing for them. The explicit `undead` bucket is 12 rows and
  clearly is not catching these cases. Twilight sitting at 25.5% rather than
  near 100% suggests it is not distorting much in practice, but check before
  leaning on per-species numbers.
- **Fuller data for two thin franchises sits unused in `datasets/`** — 623
  Walking Dead and 811 Prison Break rows from an earlier per-show scrape. Not
  merged because those files have no page text or categories, so their rows
  could not get the network and category features every other row has.
- **There is only one annotation pass.** `src/labels/eval_labels.py` computes
  inter-pass agreement and Cohen's kappa if a second pass exists, and
  `data/gold/annotations/pass2/` is an empty directory, so it prints "Only one
  annotation pass present" and no reliability figure is produced. Re-running
  `src.labels.annotate` into `pass2` would fill it.
- **Test coverage is narrow.** 17 tests, all on `src/labels/`. Nothing tests
  feature building, the models, the Q3 regression, the figures or the demo
  export — which is where a silent bug does the most damage to the writeup.
- **`character_categories` in `scraper/wiki_configs.py` is dead config.**
  It reads like it controls collection. Nothing has ever read it.

Fixed, recorded because the numbers moved:

- `run_span` was computed as the spread of TMDB *release* years, so any
  franchise that is a single long-running series scored zero — Grey's Anatomy
  ran 2005–2026 and was recorded as spanning no time at all, as were Lost, The
  Wire and thirteen others. TMDB's `last_air_date` is now captured and used.
  This removed the only significant predictor in Q3.
- Gender inference from pronouns only ran when the infobox field was missing,
  not when it was present but unparseable. The Dexter wiki writes an
  unparseable value on 1,058 of its 1,077 scraped pages — three quarters of
  every `Other/Unknown` in the corpus — so an entire franchise was effectively
  ungendered while its pages carried plenty of pronouns.
- Talking portraits and unnamed background roles were being counted as
  characters. A portrait is definitionally dead, which made it a free win for a
  death classifier.
- The portrait filter was applied in one of the three places that read
  `data/raw/*.jsonl`, so different tables disagreed. All three now share one
  predicate in `src/constants.py`.
- The infobox field count was partly counting the death label. Fixed with
  `clean_infobox_field_count`; 55% of that feature's separating power was the
  outcome written back into the input.
- Page discovery follows template redirects but the parser matched template
  names literally, so any page using a redirect was found and then silently
  discarded. That cost 477 Matrix pages by itself; three other wikis had the
  wrong template configured outright.
- The `intro` text column was computed for every character and then dropped
  before use — a whole feature family sitting unused on disk. It is now
  `intro_clean`, death- and survival-stripped, TF-IDF'd inside the CV pipeline,
  and it turned out to be the best-transferring family we have.
- The demo was fitting on every character and then scoring those same
  characters. It now fits on `trainval` and shows only `test` rows.
- Figure captions restated numbers by hand and two had gone stale against the
  data: the gender forest claimed nine of nineteen franchises significant
  against an actual 11 of 22, and the run-span caption cited the pre-fix
  p-value. Captions now read from `q3_glm_summary.csv`,
  `corpus_composition.csv` and the tables themselves, so they cannot drift
  again.
