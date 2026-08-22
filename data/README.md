# Data guide

This folder contains the data used for the final report. If you want to inspect
the analysis without rebuilding the scrape, start with the files in the table
below. CSV files open directly on GitHub; large files stored with Git LFS have a
download button on their GitHub file page.

## Main analysis files

| File | What it contains |
| --- | --- |
| [`clean/q1_analysis_data.csv`](clean/q1_analysis_data.csv) | Exact Q1 analysis table: the 15,686 modelled characters, death label, and the 401 candidate feature columns used across the 12 feature sets. |
| [`clean/character_features.csv`](clean/character_features.csv) | Broader feature-building table that is joined to the cleaned characters before the Q1 population filter is applied. |
| [`clean/characters_model.csv`](clean/characters_model.csv) | Compiled 68,463-row character table containing the cleaned labels and population flags. |
| [`clean/onscreen_flags.csv`](clean/onscreen_flags.csv) | On-screen and naming filters used to define the franchise-level population. |
| [`clean/split_assignment.csv`](clean/split_assignment.csv) | Fixed train, validation, and test assignments used by the models. |
| [`clean/q1_model_comparison.csv`](clean/q1_model_comparison.csv) | Cross-validation results for the Q1 feature and model comparison. |
| [`clean/q1_holdout_results.csv`](clean/q1_holdout_results.csv) | Final held-out Q1 scores. |
| [`clean/q1_fold_scores.csv`](clean/q1_fold_scores.csv) | Per-fold scores behind the fold-spread comparison. |
| [`clean/q1_feature_importance.csv`](clean/q1_feature_importance.csv) | Character-feature importance values shown in the report. |
| [`clean/q1_cost_curves.csv`](clean/q1_cost_curves.csv) | Weighted error costs across classification thresholds. |
| [`clean/q2_franchise_mortality.csv`](clean/q2_franchise_mortality.csv) | One row per franchise with mortality and franchise attributes. |
| [`clean/q2_regression_summary.csv`](clean/q2_regression_summary.csv) | Ordinary linear-regression output for franchise mortality. |
| [`clean/q2_franchise_clusters.csv`](clean/q2_franchise_clusters.csv) | Final k-means cluster assignment for each franchise. |
| [`gold/gold_sample.csv`](gold/gold_sample.csv) | Stratified annotation sample used for label checks. |

The other `q1_*.csv` and `q2_*.csv` files contain the supporting values used to
produce the remaining tables and figures. Images in `clean/figures/` are
generated from these CSVs.

## Folder layout

| Folder | Purpose |
| --- | --- |
| `gold/` | Annotation samples, model-assisted labels, and their checking files. |
| `clean/` | Processed tables, model results, and report figures. |

The exact Q1 file contains 15,686 characters from 38 franchises, including
5,579 deaths (35.57%). The Q2 mortality table contains all 38 franchises and is
built from 17,584 named on-screen characters. The label-fill file contains the
1,858 cases reported in the writeup: 123 dead, 1,223 alive, and 512 undecided.

The three submission files are in [`../submission/`](../submission/). The code
ZIP includes these final processed tables and the annotation data. Raw scraped
pages and third-party source dumps are omitted from the grader-facing branch;
they are not needed to inspect or rerun the reported analyses from the supplied
processed inputs.
