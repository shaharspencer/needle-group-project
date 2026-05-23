"""
utils.py
--------
Logging setup, path helpers, and checkpoint utilities.
"""

import json
import logging
import os
from pathlib import Path


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging(level: str = "INFO") -> None:
    """Configure root logger to print timestamped messages."""
    logging.basicConfig(
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
        level=getattr(logging, level.upper(), logging.INFO),
    )


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

# Absolute path to group_project/
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR  = DATA_DIR / "raw"
CHECKPOINT_DIR = DATA_DIR / "checkpoints"


def ensure_dirs() -> None:
    """Create output directories if they don't exist."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)


def raw_path(wiki_key: str) -> Path:
    """Return the canonical output JSONL path for a wiki."""
    return RAW_DIR / f"{wiki_key}_characters.jsonl"


def checkpoint_path(wiki_key: str) -> Path:
    """Return the checkpoint file path for a wiki (stores scraped page titles)."""
    return CHECKPOINT_DIR / f"{wiki_key}_checkpoint.json"


# ---------------------------------------------------------------------------
# Checkpointing — lets us resume interrupted scrapes
# ---------------------------------------------------------------------------

def load_checkpoint(wiki_key: str) -> set:
    """Return the set of page titles already scraped for this wiki."""
    path = checkpoint_path(wiki_key)
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return set(json.load(f))
    return set()


def save_checkpoint(wiki_key: str, scraped_titles: set) -> None:
    """Persist the set of scraped titles to disk."""
    path = checkpoint_path(wiki_key)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(sorted(scraped_titles), f, ensure_ascii=False, indent=2)


def clear_checkpoint(wiki_key: str) -> None:
    """Delete checkpoint (use when starting fresh)."""
    path = checkpoint_path(wiki_key)
    if path.exists():
        os.remove(path)
