"""
build_dataset.py
----------------
Master cleaning pipeline: reads all raw JSONL files, normalises records
into a flat schema, deduplicates, filters, and outputs a clean CSV.

Output: data/clean/characters_cleaned.csv

Schema columns:
    wiki, franchise, content_rating,
    name, gender, species, nationality,
    dob, dod, affiliation, actor,
    is_dead,          # 0 = alive, 1 = dead
    has_dod,          # 1 if a date-of-death field was populated
    infobox_field_count,   # how many infobox fields the page had
    page_text_length       # length of stripped page text (proxy for character prominence)

Usage:
    python -m group_project.scraper.build_dataset
    # or, from group_project/:
    python -m scraper.build_dataset
"""

import json
import re
import sys
from pathlib import Path
from collections import defaultdict

import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
try:
    # Running as group_project.scraper.build_dataset
    RAW_DIR = HERE.parent / "data" / "raw"
    CLEAN_DIR = HERE.parent / "data" / "clean"
    sys.path.insert(0, str(HERE.parent.parent))
    from group_project.scraper.wiki_configs import WIKI_CONFIGS
except ImportError:
    # Running as scraper.build_dataset from group_project/
    RAW_DIR = HERE.parent / "data" / "raw"
    CLEAN_DIR = HERE.parent / "data" / "clean"
    from scraper.wiki_configs import WIKI_CONFIGS

CLEAN_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = CLEAN_DIR / "characters_cleaned.csv"

# ---------------------------------------------------------------------------
# Field mappings — canonical names we want in the final CSV
# ---------------------------------------------------------------------------
CANONICAL_FIELDS = {
    # infobox key (lowercase) -> output column
    "real_name":    "real_name",
    "name":         "real_name",
    "real name":    "real_name",
    "fullname":     "real_name",
    "full name":    "real_name",
    "gender":       "gender",
    "sex":          "gender",
    "species":      "species",
    "race":         "species",
    "nationality":  "nationality",
    "citizenship":  "nationality",
    "culture":      "nationality",
    "origin":       "nationality",
    "homeworld":    "homeworld",
    "born":         "dob",
    "birth":        "dob",
    "dob":          "dob",
    "birth date":   "dob",
    "birthdate":    "dob",
    "birth_date":   "dob",
    "died":         "dod",
    "death":        "dod",
    "dod":          "dod",
    "deathdate":    "dod",
    "death date":   "dod",
    "actor":        "actor",
    "portrayed":    "actor",
    "portrayer":    "actor",
    "portrayed by": "actor",
    "voice":        "actor",
    "voiced by":    "actor",
    "affiliation":  "affiliation",
    "loyalty":      "affiliation",
    "house":        "affiliation",
    "allegiance":   "affiliation",
    "faction":      "affiliation",
    "clan":         "affiliation",
    "side":         "affiliation",
    # title / occupation
    "title":        "title",
    "titles":       "title",
    "occupation":   "title",
    "profession":   "title",
    "job":          "title",
    "rank":         "title",
    "role":         "title",
    "livelihood":   "title",
    "class":        "title",
}

_STAR_MARKUP_RE = re.compile(r"\*\d+px\|link=[^\s*]*")  # e.g. *20px|link=...
_JUNK_VALUES = {"unknown", "n/a", "none", "null", "--", "--,", "?", "tbd", "amortal"}

def first_bullet_value(text: str):
    """Strip wiki *-bullet list markup and return the first non-empty item, or None."""
    if not isinstance(text, str):
        return text
    text = _STAR_MARKUP_RE.sub("", text)
    parts = [p.strip() for p in text.split("*") if p.strip()]
    return parts[0] if parts else None

def extract_primary_color(text):
    if not isinstance(text, str):
        return None
    text = text.lower()
    colors = ['blonde', 'blond', 'brown', 'black', 'white', 'red', 'blue', 'green', 
              'grey', 'gray', 'hazel', 'silver', 'gold', 'yellow', 'purple', 'pink', 'pale', 'fair']
    for color in colors:
        if color in text:
            if color == 'blond': return 'blonde'
            if color == 'gray': return 'grey'
            return color
    return 'other'

def extract_flat(record: dict, config: dict) -> dict:
    """
    Flatten a JSONL record into a single-level dict with canonical column names.
    """
    infobox = record.get("infobox", {}) or {}
    field_map = config.get("field_map", {})

    flat = {
        "wiki":              record.get("wiki", ""),
        "franchise":         record.get("franchise", ""),
        "content_rating":    record.get("content_rating", ""),
        "name":              record.get("name", ""),
        "page_url":          record.get("page_url", ""),
        "is_dead":           record.get("is_dead"),
        "real_name":         None,
        "gender":            None,
        "species":           None,
        "nationality":       None,
        "homeworld":         None,
        "dob":               None,
        "dod":               None,
        "actor":             None,
        "affiliation":       None,
        "title":             None,
        "infobox_field_count": len(infobox),
        "page_text_length":    len(record.get("page_text", "") or ""),
        "has_dob":             0,
        "has_dod":             0,
        "has_causeofdeath":    0,
        "has_family":          0,
        "has_image":           0,
        "has_alias":           0,
        "appearance_count":    0,
        "has_pronouns":        0,
        "hair_color":          None,
        "eye_color":           None,
    }

    # Walk infobox keys -> map to canonical columns
    for raw_key, raw_val in infobox.items():
        lk = raw_key.lower().strip()
        # First try the wiki-level field_map
        canonical = field_map.get(lk) or field_map.get(raw_key.strip())
        # Then try our global CANONICAL_FIELDS map
        if not canonical:
            canonical = CANONICAL_FIELDS.get(lk)
        if canonical and canonical in flat and flat[canonical] is None:
            flat[canonical] = raw_val[:300]  # cap length

    # Strip * wiki-list markup from multi-value text fields
    for col in ("real_name", "species", "affiliation", "title", "nationality", "actor"):
        if flat.get(col):
            flat[col] = first_bullet_value(flat[col])

    # Null out junk dob/dod values (Unknown, --, N/A, etc.)
    for col in ("dob", "dod"):
        val = flat.get(col)
        if val and val.strip().lower() in _JUNK_VALUES:
            flat[col] = None

    # Mark has_dob and has_dod
    if flat.get("dob"):
        flat["has_dob"] = 1
    if flat.get("dod"):
        flat["has_dod"] = 1

    # Mark has_causeofdeath (separate from dod — this is the manner/cause, not the date)
    cod_keys = {"causeofdeath", "cause of death", "cause", "death cause", "manner of death",
                "deathby", "death by", "killedby", "killed by", "death reason", "deathreason"}
    for raw_key in infobox.keys():
        if raw_key.lower().strip() in cod_keys and infobox[raw_key]:
            flat["has_causeofdeath"] = 1
            break

    # Normalise gender
    g = (flat.get("gender") or "").lower().strip()
    if g in ("male", "m", "masculine"):
        flat["gender"] = "Male"
    elif g in ("female", "f", "feminine"):
        flat["gender"] = "Female"
    elif g:
        flat["gender"] = "Other/Unknown"
    else:
        flat["gender"] = None

    # Extract physical traits
    hair_raw = None
    eye_raw = None
    for raw_key, val in infobox.items():
        lk = raw_key.lower().strip()
        if lk in ("hair", "hair color", "hair colour", "haircolor"):
            hair_raw = val
        elif lk in ("eyes", "eye color", "eye colour", "eyecolor", "eye"):
            eye_raw = val

    if hair_raw: flat["hair_color"] = extract_primary_color(hair_raw)
    if eye_raw: flat["eye_color"] = extract_primary_color(eye_raw)

    # Check for family fields
    family_keys = {
        "children", "siblings", "families", "parents", "partners",
        "family", "spouse", "husband", "wife", "brother", "sister",
        "mother", "father", "son", "daughter", "relatives", "relationships",
    }
    alias_keys = {"alias", "aliases", "nickname", "nicknames", "aka", "other names", "also known as"}
    appearance_keys = {
        "tv series", "movie", "game", "book", "comic", "first", "last",
        "appearances", "seasons", "episodes", "films", "series", "appears in",
    }
    
    app_count = 0
    for raw_key in infobox.keys():
        lk = raw_key.lower().strip()
        val = infobox[raw_key]
        if lk in family_keys and val:
            flat["has_family"] = 1
        if lk in alias_keys and val:
            flat["has_alias"] = 1
        if lk == "image" and val:
            flat["has_image"] = 1
        if lk == "pronouns" and val:
            flat["has_pronouns"] = 1
        if lk in appearance_keys and val:
            app_count += 1
            
    flat["appearance_count"] = app_count

    return flat


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    all_records = []
    per_wiki_stats = {}

    for jsonl_path in sorted(RAW_DIR.glob("*_characters.jsonl")):
        wiki_key = jsonl_path.stem.replace("_characters", "")
        config = WIKI_CONFIGS.get(wiki_key, {})

        rows = []
        try:
            with open(jsonl_path, encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        rows.append(json.loads(line))
        except Exception as e:
            print(f"  [{wiki_key}] Read error: {e}")
            continue

        flat_rows = [extract_flat(r, config) for r in rows]
        per_wiki_stats[wiki_key] = {
            "raw": len(rows),
            "flat": len(flat_rows),
        }
        all_records.extend(flat_rows)
        print(f"  [{wiki_key:20s}] raw={len(rows):6d}  flat={len(flat_rows):6d}")

    print(f"\nTotal records before dedup/filter: {len(all_records):,}")

    df = pd.DataFrame(all_records)

    # ---------------------------------------------------------------------------
    # Deduplication: same page_url is the strongest dedup key
    # ---------------------------------------------------------------------------
    before_dedup = len(df)
    df = df.drop_duplicates(subset=["page_url"])
    print(f"After URL dedup:          {len(df):,}  (removed {before_dedup - len(df):,})")

    # Secondary dedup: same franchise + name
    before_dedup2 = len(df)
    df = df.drop_duplicates(subset=["franchise", "name"])
    print(f"After franchise+name dedup: {len(df):,}  (removed {before_dedup2 - len(df):,})")

    # ---------------------------------------------------------------------------
    # Filter: keep only records with a definitive label
    # ---------------------------------------------------------------------------
    df_labelled = df[df["is_dead"].notna()].copy()
    df_labelled["is_dead"] = df_labelled["is_dead"].astype(int)
    print(f"\nLabelled (is_dead != None): {len(df_labelled):,}")
    print(f"  Dead  (is_dead=1):  {(df_labelled['is_dead'] == 1).sum():,}")
    print(f"  Alive (is_dead=0):  {(df_labelled['is_dead'] == 0).sum():,}")
    pct = (df_labelled['is_dead'] == 1).mean() * 100
    print(f"  Class balance: {pct:.1f}% dead")

    # ---------------------------------------------------------------------------
    # Per-franchise breakdown
    # ---------------------------------------------------------------------------
    print("\nPer-franchise summary (labelled records):")
    summary = (
        df_labelled.groupby("franchise")["is_dead"]
        .agg(total="count", dead="sum")
        .assign(alive=lambda x: x["total"] - x["dead"],
                pct_dead=lambda x: (x["dead"] / x["total"] * 100).round(1))
        .sort_values("total", ascending=False)
    )
    print(summary.to_string())

    # ---------------------------------------------------------------------------
    # Save
    # ---------------------------------------------------------------------------
    col_order = [
        "wiki", "franchise", "content_rating",
        "name", "real_name", "gender", "species",
        "nationality", "homeworld", "dob", "dod",
        "actor", "affiliation", "title",
        "is_dead", "has_dob", "has_dod", "has_causeofdeath", "has_family",
        "has_image", "has_alias", "has_pronouns",
        "appearance_count", "infobox_field_count",
        "hair_color", "eye_color",
        "page_text_length", "page_url",
    ]
    # Only include columns that exist
    col_order = [c for c in col_order if c in df_labelled.columns]
    df_labelled[col_order].to_csv(OUT_PATH, index=False, encoding="utf-8")
    print(f"\nSaved {len(df_labelled):,} labelled records to {OUT_PATH}")

    # Also save the full (unlabelled included) dataset for analysis
    full_path = CLEAN_DIR / "characters_all.csv"
    df[col_order].to_csv(full_path, index=False, encoding="utf-8")
    print(f"Saved {len(df):,} total records to {full_path}")


if __name__ == "__main__":
    main()
