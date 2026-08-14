"""Canonical project paths, so no script has to guess where the data lives."""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = REPO_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"           # scraped Fandom JSONL (gitignored, 262MB)
CLEAN_DIR = DATA_DIR / "clean"       # our built tables and figures (tracked)
GOLD_DIR = DATA_DIR / "gold"         # annotated ground-truth labels (tracked)
EXTERNAL_DIR = DATA_DIR / "external" # third-party Kaggle/CMU dumps (gitignored)
DATASETS_DIR = REPO_ROOT / "datasets"  # per-show transcripts, death lists, features
FIG_DIR = CLEAN_DIR                  # figures currently live alongside the tables
