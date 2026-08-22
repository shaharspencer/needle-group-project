# Pipeline workspace: datasets and generated outputs

This is the pipeline's working directory, so it contains intermediate tables
and generated outputs as well as data. For a clean folder containing only the
four final datasets, use [`../dataset/`](../dataset/). The sections below
document the pipeline files; they are not the Data link used in the writeup.

## Final datasets used in the analyses

These are the clearest files for inspecting the observations that were
actually analysed.

| Dataset | Rows | What one row represents |
| --- | ---: | --- |
| [Q1 character dataset](clean/q1_analysis_data.csv) | 15,686 | Internal pipeline copy of `dataset/q1_characters.csv`: one named, on-screen character with 2 identifiers, the death label, and 402 candidate predictors used across the 12 Q1 feature sets. |
| [Q2 franchise dataset](clean/q2_franchise_mortality.csv) | 38 | One franchise. It contains mortality, population counts, medium, genre, release years, run span, titles, and rating used for Q2. |
| [Annotation sample](gold/gold_sample.csv) | 600 | One wiki page sampled for the label-quality checks. |
| [Labels for unclear cases](gold/fill_unlabelled/haiku_fill.csv) | 1,858 | One previously unclear case and its rubric-based outcome: dead, alive, or undecided. |

The Q1 dataset contains 5,579 deaths (35.57%) across 38 franchises. The Q2
dataset summarizes 17,584 named, on-screen characters. The label-fill outcomes
match the writeup: 123 dead, 1,223 alive, and 512 undecided.

## Supporting and intermediate datasets

These files show how the final datasets were assembled, but they are not the
final model-ready tables.

| File | Purpose |
| --- | --- |
| [Compiled character table](clean/characters_model.csv) | 68,463 wiki-page rows with cleaned labels and population flags. |
| [Feature-building table](clean/character_features.csv) | Intermediate table with 73,632 wiki-page rows and 387 columns: 2 identifiers, 291 category indicators, 78 infobox indicators, 9 graph measures, 5 community measures, and 2 text fields. It has no death label; it is merged with the character table and filtered before modelling. |
| [On-screen flags](clean/onscreen_flags.csv) | Naming and on-screen decisions used to define the Q2 population. |
| [Split assignments](clean/split_assignment.csv) | Fixed train, validation, and test membership for Q1. |
| [TMDB titles](clean/tmdb_titles.csv) | 280 film and television title records used for franchise attributes. |
| [TMDB cast](clean/tmdb_cast.csv) | 28,151 cast credits used mainly for billing order and appearance information. |
| [`gold/`](gold/) | Annotation chunks and checking files used to build and evaluate the supplied labels. |

## Generated outputs — not datasets

The remaining files under `clean/` are products of the analysis code:

- `q1_model_comparison.csv`, `q1_holdout_results.csv`, and
  `q1_fold_scores.csv` contain model evaluation results.
- The other `q1_*.csv` files contain feature-importance, cost, gender,
  community, pruning, and diagnostic results.
- `q2_regression_summary.csv`, the other `q2_*.csv` files, and the cluster
  assignments contain regression or clustering results.
- `figures/` contains plots generated from those result tables.
- `label_*.csv`, `corpus_composition.csv`, and `demo_data.json` are generated
  checks or application outputs.

These outputs are included for reproducibility, but they should not be read as
additional observations or source datasets.

The three files submitted for grading are in [`../submission/`](../submission/).
The code ZIP contains the four final datasets above, the supporting processed
tables, and the generated outputs. Raw scraped pages and third-party source
dumps are omitted because they are not needed to inspect or rerun the reported
analyses from the supplied processed inputs.
