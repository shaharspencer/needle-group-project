"""
infobox_parser.py
-----------------
Parses MediaWiki wikitext using mwparserfromhell to extract character
infobox data and map it to the canonical schema.
"""

import re
import logging
from typing import Optional

import mwparserfromhell

logger = logging.getLogger(__name__)

# Regex to strip wiki markup from a plain-text value
_WIKILINK_RE = re.compile(r"\[\[(?:[^|\]]*\|)?([^\]]+)\]\]")
_REF_RE = re.compile(r"<ref[^>]*>.*?</ref>", re.DOTALL | re.IGNORECASE)
_REF_SELF_RE = re.compile(r"<ref[^/]*/?>", re.IGNORECASE)
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_TEMPLATE_RE = re.compile(r"\{\{[^}]*\}\}")
_MULTI_SPACE_RE = re.compile(r"\s+")


def clean_value(raw: str) -> str:
    """
    Strip wiki markup from an infobox parameter value, returning plain text.
    Order matters: strip refs first, then links, then templates, then HTML.
    """
    s = str(raw)
    s = _REF_RE.sub("", s)
    s = _REF_SELF_RE.sub("", s)
    # Replace [[Target|Label]] or [[Label]] with Label
    s = _WIKILINK_RE.sub(r"\1", s)
    # Drop remaining templates (e.g. {{Ref|IM}})
    s = _TEMPLATE_RE.sub("", s)
    s = _HTML_TAG_RE.sub("", s)
    # Normalise whitespace
    s = s.replace("\n", " ").replace("'", "'")
    s = _MULTI_SPACE_RE.sub(" ", s).strip()
    return s


def normalize_status(raw_status: str, dead_statuses: list[str],
                     alive_statuses: list[str]) -> Optional[int]:
    """
    Convert a raw status string to a binary label:
        1  = dead
        0  = alive
        None = unknown / ambiguous
    """
    lowered = clean_value(raw_status).lower()
    if not lowered:
        return None
    for kw in dead_statuses:
        if kw in lowered:
            return 1
    for kw in alive_statuses:
        if kw in lowered:
            return 0
    return None  # Couldn't determine — caller will log as unknown


def find_infobox_template(wikicode, template_names: list[str]):
    """
    Return the first Template object in *wikicode* whose name matches
    any of *template_names* (case-insensitive).
    Returns None if no match found.
    """
    for template in wikicode.filter_templates():
        tname = template.name.strip().lower()
        for candidate in template_names:
            if candidate.lower() == tname:
                return template
    return None


def parse_character_page(
    wikitext: str,
    wiki_config: dict,
) -> dict:
    """
    Parse a single character page's wikitext.

    Args:
        wikitext:    Raw wikitext string from the API.
        wiki_config: The config dict for this wiki from wiki_configs.py.

    Returns:
        A dict with canonical schema keys populated as best as possible.
        Fields that cannot be found are set to None.
    """
    # Follow REDIRECT if present
    redirect_match = re.match(r"#REDIRECT\s*\[\[([^\]]+)\]\]", wikitext, re.IGNORECASE)
    is_redirect = redirect_match is not None

    record: dict = {
        "is_redirect": is_redirect,
        "is_dead": None,
        "infobox": {},
        "page_text": "",
    }

    if is_redirect:
        # Can't extract infobox data from a redirect page
        return record

    try:
        wikicode = mwparserfromhell.parse(wikitext)
        # Extract full plain text
        record["page_text"] = wikicode.strip_code().strip()
    except Exception as exc:
        logger.warning("mwparserfromhell parse error: %s", exc)
        return record

    template = find_infobox_template(wikicode, wiki_config["infobox_templates"])
    if template is None:
        logger.debug("No matching infobox template found.")
        return record

    field_map: dict = wiki_config["field_map"]
    dead_statuses: list = wiki_config["dead_statuses"]
    alive_statuses: list = wiki_config["alive_statuses"]

    # Walk every parameter in the infobox template
    for param in template.params:
        raw_key = str(param.name).strip().lower()
        raw_val = str(param.value).strip()
        cleaned_val = clean_value(raw_val)

        # Always save to the dynamic infobox dict
        if cleaned_val:
            # use original param name stripped as the key
            clean_key = str(param.name).strip()
            record["infobox"][clean_key] = cleaned_val

    # Derive is_dead from the status or dod field if mapped
    mapped_status = None
    mapped_dod = None
    for raw_key, canonical in field_map.items():
        val = record["infobox"].get(raw_key) or record["infobox"].get(raw_key.title())
        if not val:
            # Try case-insensitive lookup
            for k, v in record["infobox"].items():
                if k.lower() == raw_key:
                    val = v
                    break
        if val:
            if canonical == "status":
                mapped_status = val
            elif canonical == "dod":
                mapped_dod = val

    if mapped_status:
        record["is_dead"] = normalize_status(
            mapped_status, dead_statuses, alive_statuses
        )
    elif mapped_dod:
        # Explicit date-of-death → dead
        record["is_dead"] = 1
    elif wiki_config.get("dod_only"):
        # DOD-only wiki: no DOD field found → character is alive
        record["is_dead"] = 0

    return record
