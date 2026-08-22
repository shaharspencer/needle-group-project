# Spoiler Alert: Who Makes It to the Finale?

Group 56 final project for *A Needle in a Data Haystack* (67978), Hebrew
University of Jerusalem.

We study fictional-character deaths across 38 film and television franchises.
The project compares character and franchise features for predicting death,
then examines how mortality rates differ between franchises.

## Final submission

- [Illustrated writeup](submission/writeup_group56.pdf)
- [Writeup without images](submission/noimages_group56.pdf)
- [Code and analysis data](submission/code_group56.zip)
- [Data guide and downloadable files](data/README.md)
- [Interactive demo](https://shaharspencer.github.io/needle-group-project/)

## Repository contents

```text
build_pipeline.py   builds the main character tables
scraper/            collects character pages from the 38 Fandom wikis
src/labels/         constructs and checks the death labels
src/enrich/         adds TMDB cast and title information
src/q1_character/   character-level models and comparisons
src/q2_franchise/   franchise-level regression and clustering
src/viz/            creates the report figures
data/               raw, supporting, processed, and annotation data
docs/               browser-based demo
submission/         the three files submitted for grading
```

## Running the code

The project was run with Python 3.12. Install the packages from the repository
root:

```bash
pip install -r requirements.txt
```

The final processed inputs are already in `data/clean/`, so the analysis can be
run without scraping again. For example:

```bash
python -m src.q1_character.models
python -m src.q2_franchise.mortality
python -m src.viz.figures
```

`build_pipeline.py` rebuilds the character tables. Re-running the enrichment
stage requires a personal TMDB API key in an untracked `.env` file. Large CSV,
JSON, JSONL, TSV, Parquet, and text files use Git LFS; after cloning, run
`git lfs pull` if they were not downloaded automatically.

## Data and labels

The [data guide](data/README.md) identifies the main analysis tables and
explains each data folder. The submitted ZIP and the grader-facing repository
contain the final processed tables and annotation data used for the report.
The much larger raw scrape and third-party source dumps are not required to
inspect or rerun the reported analyses from these processed inputs.

Most death labels were parsed from wiki infobox fields. We used Claude Haiku
4.5 with a fixed written rubric for cases where the infobox did not give a
usable answer. The details and reliability checks are reported in the writeup.

## Team

- Stav Abukarat
- Shahar Spencer
- Bar Dalal
- Ido Oron
