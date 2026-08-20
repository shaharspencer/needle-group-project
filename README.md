# Spoiler Alert: Who Makes It to the Finale?

Group #56 — 67978 A Needle in a Data Haystack, Hebrew University of Jerusalem.

Which fictional characters die, and why? We scraped 38 Fandom wikis, enriched
them with TMDB cast data, and asked whether a character's own attributes or the
franchise they are in better predicts their death.

**This file is the operational one: how to set the project up, what to run, and
where the outputs land.** For what we found and what it means, read
[noimages_group56.md](noimages_group56.md). For how each feature is computed,
read [meta/FEATURES.md](meta/FEATURES.md). For a stage-by-stage description of
the pipeline, read [meta/PIPELINE.md](meta/PIPELINE.md).

| document | answers |
|---|---|
| README.md (this file) | how do I run it, and where does the output go |
| [noimages_group56.md](noimages_group56.md) | what did we find |
| [meta/PIPELINE.md](meta/PIPELINE.md) | what does each stage read and write |
| [meta/FEATURES.md](meta/FEATURES.md) | how is each feature computed |

## Where the project stands

Everything is built, run and committed. All three questions are answered, the
demo is live, and every figure in the writeup is generated from the tables in
`data/clean/` rather than drawn by hand.

**Two split rules, used the same way here and in the writeup.** *Random CV* is
stratified over characters; *grouped CV* is `GroupKFold` holding out whole
franchises. Each is applied both to the cross-validated feature comparison and
to the once-scored test estimate, so a score needs both labels to be unambiguous
— "grouped test" and "grouped CV" are different numbers.

**The results, in one place.** Population is 15,686 on-screen, named characters
across 38 franchises, 35.6% of whom die.

| question | answer |
|---|---|
| Q1 — character or franchise? | Both, mostly the character. Character attributes reach 0.633 PR-AUC against 0.556 for franchise identity and 0.743 for everything combined, on a 0.356 base rate (all cross-validated, random split). On the test set the full model gets **0.752 PR-AUC / 0.850 ROC-AUC** under a random split, dropping to **0.664 / 0.776** under a grouped split that holds out whole franchises. |
| Q1 — gender | Women are about half as likely to die: **adjusted OR 0.564** (95% CI 0.516–0.616), roughly 11 percentage points. 11 of 22 franchises reach p<0.05 and none point the other way. |
| Q2 — does text predict death? | Yes, for descriptive text. TF-IDF over each character's wiki description is our best-transferring feature family: +0.066 PR-AUC under random CV, +0.058 under grouped CV. Dialogue is a separate question we did not answer. |
| Q3 — are some franchises deadlier? | They differ enormously (Grey's Anatomy 9.3%, The 100 72.7%) but **no franchise property predicts which**. Nothing in the weighted least-squares model reaches significance and adjusted R² is negative. A clean null. |

**If you worked on this before, two things changed the numbers.** Both were
fields that were computed correctly for one kind of row and silently wrong for
another, so neither showed up as missing data:

- `run_span` measured the spread of TMDB *release* years. A TV series is one
  title, so 15 shows were recorded as running for zero years — Grey's Anatomy
  aired 2005–2026 and scored 0, as did Lost and The Wire. Fixed by reading
  `last_air_date`. **This removed the only significant result Q3 had**: run span
  went from p=0.022 to p=0.185.
- Gender was inferred from pronouns only when the infobox field was *missing*,
  never when it was present but unparseable. The Dexter wiki writes an
  unparseable value on 1,058 of its 1,077 pages, so an entire franchise had no
  usable gender. Fixed; recovered 1,024 characters, 826 of them Dexter, and
  Dexter now appears in the per-franchise gender estimates for the first time.

Every table, figure and demo file was regenerated after those fixes. **Old
figures and numbers from before this are stale — take them from `data/clean/`,
not from an earlier copy.**

**What else was added:** `src/q2_text/coefficients.py` (Q2's package was empty,
and the coefficients the writeup quoted weren't saved anywhere), a seventh figure
for Q2, `in_degree` in the demo, and figure captions that read from the CSVs
instead of restating numbers by hand — two of them had gone stale against the
data.

Scope and measurement constraints are set out under
[Limitations and scope](#limitations-and-scope) below.

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

[meta/PIPELINE.md](meta/PIPELINE.md) has the full order with inputs and outputs per stage.
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
python -m src.q1_character.models         # 9 feature sets x 3 models x 2 split rules
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
                    registry validation, Haiku fill for unlabelled rows,
                    reliability.py (Cohen's kappa across coding passes),
                    ANNOTATION_RUBRIC.md (the coding scheme itself)
  enrich/          TMDB titles, cast, name matching
  q1_character/    features (incl. co-mention graph, centrality, Louvain
                    communities), model comparison, splits, held-out
                    evaluation, gender effect sizes, community.py (do
                    communities share fates?), significance.py (paired per-fold
                    tests), costs.py (weighted-loss thresholds)
  q2_text/         coefficients.py — the fitted TF-IDF weights behind Q2
  q3_franchise/    franchise mortality, WLS regression, k-means with internal
                    and external validation, Ward cross-check
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
tests/             pytest (labels, community statistic, cost sweep)
meta/              non-code, non-data project material: PIPELINE.md,
                   FEATURES.md, project_guidelines/ (assignment spec),
                   example_projects/ (other groups' reference writeups),
                   milestone/ (earlier milestone submission and feedback)
noimages_group56.md   the writeup (images stripped, per the assignment spec)
```

Three conventions worth keeping to:

- **Import paths from `src/paths.py`.** No stage should hardcode a data
  location.
- **Labels produced by a model stay in their own files.** The 600-page
  annotation sample and the fill for unlabelled rows were both labelled by
  Claude Haiku 4.5, and neither is merged into `is_dead`. The fill feeds exactly
  one number, the Q3 mortality point estimates. Keep that separation if you add
  to it.
- **Two populations, deliberately.** `analysis_population()` in
  `src/constants.py` restricts to on-screen, named rows, but Q1/Q2 apply it to
  `characters_model.csv` (15,686 rows, since a row needs a usable label to be
  modelled) while Q3 applies it to `onscreen_flags.csv` (17,584 rows, keeping
  characters whose status never parsed so they stay in the mortality
  denominator). That gap is intended and the writeup states it; do not
  "fix" the two to agree without reading the Q3 bounds discussion first.
- **Constants go in `src/constants.py`.** `analysis_population()` is how every
  question restricts rows — on screen, and named. If you filter rows yourself somewhere else, the tables stop
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
| Q1 — character vs franchise | 12 feature sets × 3 models × 2 split rules (random and grouped), paired per-fold comparison, plus a once-scored test estimate |
| Q1 — gender effect sizes | pooled and per-franchise |
| Q1 — narrative communities | Louvain partitions per franchise, modularity reported, fate concentration tested against a label-shuffling null |
| Q1 — cost-sensitive thresholds | weighted loss swept over thresholds and cost ratios, globally and per franchise |
| Q2 — does text predict death? | answered for descriptive text: TF-IDF over each character's wiki description, the best-transferring feature family in the project. Dialogue is out of scope, see below |
| Q3 — franchise properties | WLS regression, k-means with internal and external validation, Ward cross-check, Haiku-filled mortality for the four least-labelled franchises |
| demo | done, deployed from `docs/` |
| labels | four independent coding passes, Cohen's kappa per label under two rubric versions |
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

## Limitations and scope

Constraints that bear on how the results should be read.

- **Some features are missing for whole franchises rather than at random.**
  `n_relatives` is 0 across six franchises (33.7% of characters, including the
  MCU), because those wikis do not use the family infobox keys. Under grouped
  cross-validation a held-out franchise can therefore lack a feature the model
  relies on, which is one reason grouped scores sit below random ones. Check
  per-franchise coverage before adding a feature.
- **Franchise sizes are very unequal.** Prison Break contributes 28 characters
  and the MCU 3,489, and grouped folds are franchises, so folds differ in size
  by two orders of magnitude. The per-fold spreads in `q1_fold_scores.csv` are
  reported for this reason.
- **414 Matrix pages carry no status and no date-of-death field.** Their
  infoboxes have neither, in 0 of 414 cases, so those characters cannot be
  labelled from the source at all and are excluded from Q1 and Q2.
- **Coverage varies by wiki.** The Walking Dead (49 rows), Stranger Things,
  Vikings, Spartacus and Westworld are collected more thinly than the rest,
  since their wikis use per-character templates the scraper handles less well.
  Franchise-level results for these carry correspondingly wider intervals.
- **`is_dead` does not mean the same thing for every species.** A walker is a
  dead human; a vampire is dead by species and alive in the story. Twilight
  sitting at 25.5% rather than near 100% suggests the effect on aggregates is
  small, but per-species mortality should not be read directly.
- **The unnamed-role filter matches explicit markers only.** It catches
  `Unidentified`, `Unnamed` and `Unknown` prefixes, so role-titled pages
  without one of those markers remain in the population. Results with the
  filter disabled are reproducible with `INCLUDE_UNNAMED_ROLES=1`.
- **Dialogue is out of scope.** Transcripts exist for 11 shows, but a usable
  per-episode death list exists for two, and most transcripts carry no speaker
  labels, so attributing a line to a character is a prerequisite problem.

## Measurement decisions that changed the results

Recorded because each moved a number in the writeup.

- `run_span` was computed from TMDB *release* years, so any franchise that is a
  single long-running series scored zero. Grey's Anatomy ran 2005-2026 and was
  recorded as spanning no time, as were Lost, The Wire and thirteen others.
  Using `last_air_date` removed the only significant predictor in Q3.
- Gender inference from pronouns ran only when the infobox field was absent,
  not when it was present but unparseable. The Dexter wiki writes an
  unparseable value on 1,058 of its 1,077 pages, three quarters of every
  `Other/Unknown` in the corpus, so a whole franchise was effectively
  ungendered while its pages carried plenty of pronouns.
- Talking portraits and unnamed background roles were counted as characters. A
  portrait is dead by definition, which made it a free win for a death
  classifier.
- The portrait filter ran in one of the three places that read
  `data/raw/*.jsonl`, so the tables disagreed. All three now share one
  predicate in `src/constants.py`.
- `infobox_field_count` was partly counting the label, since a character who
  dies acquires "Cause of death" and "Date of death" fields. 55% of that
  feature's separating power was the outcome written back into the input;
  `clean_infobox_field_count` removes it.
- Page discovery followed template redirects while the parser matched template
  names literally, so pages reached through a redirect were found and then
  discarded. That cost 477 Matrix pages alone.
- The `intro` text column was computed and then dropped before use. It is now
  `intro_clean`, death- and survival-stripped and vectorised inside the CV
  pipeline, and it is the best-transferring feature family in the project.
- The demo fitted on every character and scored the same characters. It now
  fits on `trainval` and displays `test` rows only.
- Figure captions restated numbers by hand and two had drifted from the data.
  Captions now read from `q3_glm_summary.csv`, `corpus_composition.csv` and the
  result tables directly.
