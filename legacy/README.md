# Legacy

Scripts from before the project moved to `src/` (commit `810f4a4`). None of
these run as part of the current pipeline — nothing under `src/`,
`build_pipeline.py`, or `scraper/` imports them. Kept for the record rather
than deleted, since they're where most of the early exploratory work
happened.

Also here, moved out of the project root so the top level holds only what the
current pipeline uses:

- `tfidf.py` — standalone TF-IDF experiment. The text features it prototyped
  now live in `src/q1_character/features.py` (`intro_clean`) and are fit inside
  the CV pipeline in `src/q1_character/models.py`.
- `milestone.md` — the milestone write-up, superseded by `WRITEUP.md`. Its
  figures were rendered by the ad hoc plotting scripts listed below and no
  longer exist.
- `original_research_questions.txt` — the three questions as first proposed,
  in Hebrew. Kept because the writeup reports how each one changed on contact
  with the data.

- `feature_scan.py`, `text_leakage_check.py`, `text_features_clean.py`,
  `social_scan.py`, `social_scan2.py`, `sanity_checks.py` — the scripts
  `analysis_findings.md` was written from. Their findings (the violence-word
  leakage bug, within-franchise normalization, gender gaps) are now folded
  into README.md's Findings and Limitations sections.
- `add_nlp_features.py` — added lexicon features to `characters_full.csv`,
  a table nothing downstream reads anymore.
- `archetype_plot.py`, `gender_plot.py`, `make_plots.py`, `prominence_plot.py`,
  `sanity_plots.py`, `q2_alt_dotplot.py` — ad hoc matplotlib scripts for the
  milestone figures. Superseded by `src/viz/figures.py` and `src/viz/style.py`,
  which render the current writeup's figures in one consistent style.
- `analysis_findings.md` — the prose write-up this scan work fed into. Its
  content is now in README.md.
- `scrapers/` — an earlier, per-show scraping attempt (one script per show:
  Battlestar Galactica, Breaking Bad transcripts, The 100 deaths, etc.),
  fully replaced by the unified, config-driven `scraper/` package that scrapes
  all 38 wikis through one code path.

`build_pipeline.py` and `tfidf.py` are *not* here — both are still live and
run as documented in `README.md` / `PIPELINE.md`. `datasets/` isn't legacy
either — it's the current (partial) input Q2 is blocked on.
