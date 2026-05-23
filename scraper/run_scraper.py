"""
run_scraper.py
--------------
CLI entry point for the Fandom character death scraper.

Usage examples:
    # Scrape a single wiki
    python -m scraper.run_scraper --wiki mcu

    # Scrape multiple wikis
    python -m scraper.run_scraper --wiki mcu got

    # Scrape all configured wikis
    python -m scraper.run_scraper --wiki all

    # Fresh start (ignore checkpoint and overwrite output)
    python -m scraper.run_scraper --wiki mcu --fresh

    # Limit to N characters (useful for quick testing)
    python -m scraper.run_scraper --wiki mcu --limit 50

Options:
    --wiki      One or more wiki keys (e.g. mcu got walking_dead star_wars harry_potter)
                Use 'all' to scrape every configured wiki.
    --fresh     Ignore existing checkpoints and re-scrape from scratch.
    --limit     Stop after scraping N characters per wiki (for testing).
    --delay     Override min/max request delay in seconds (e.g. --delay 1.5 3.0).
    --log       Logging level: DEBUG, INFO, WARNING (default: INFO).
"""

import argparse
import logging
import math
from itertools import islice

import json
import pandas as pd
from tqdm import tqdm

from scraper.fandom_api import FandomAPIClient
from scraper.infobox_parser import parse_character_page
from scraper.wiki_configs import WIKI_CONFIGS
from scraper.utils import (
    setup_logging, ensure_dirs, raw_path,
    load_checkpoint, save_checkpoint, clear_checkpoint,
)

logger = logging.getLogger(__name__)

# How many pages to fetch in one batched API call
BATCH_SIZE = 50

# How often to flush rows to disk (in number of characters processed)
FLUSH_EVERY = 200


def scrape_wiki(
    wiki_key: str,
    fresh: bool = False,
    limit: int = None,
    min_delay: float = 1.0,
    max_delay: float = 2.5,
) -> pd.DataFrame:
    """
    Scrape one wiki and return a DataFrame with canonical columns.
    Saves progress incrementally to CSV and maintains a checkpoint file.
    """
    config = WIKI_CONFIGS[wiki_key]
    logger.info("=" * 60)
    logger.info("Starting scrape: %s (%s)", config["name"], config["franchise"])
    logger.info("Content rating : %s", config["content_rating"])
    logger.info("=" * 60)

    if fresh:
        clear_checkpoint(wiki_key)

    already_scraped: set = load_checkpoint(wiki_key)
    if already_scraped:
        logger.info("Resuming — %d pages already in checkpoint.", len(already_scraped))

    client = FandomAPIClient(config["api_url"], min_delay=min_delay, max_delay=max_delay)
    out_path = raw_path(wiki_key)

    # Load existing rows from disk if resuming
    existing_rows: list[dict] = []
    if out_path.exists() and not fresh:
        with open(out_path, "r", encoding="utf-8") as f:
            for line in f:
                existing_rows.append(json.loads(line))
        logger.info("Loaded %d existing rows from %s.", len(existing_rows), out_path)

    new_rows: list[dict] = []
    page_buffer: list[dict] = []  # Accumulate titles for batched wikitext fetch

    # -----------------------------------------------------------------------
    # Step 1: Collect all pages that embed the designated infobox templates
    # -----------------------------------------------------------------------
    all_pages: dict[int, str] = {}  # pageid → title
    for template in config["infobox_templates"]:
        logger.info("Finding pages embedding template: '%s' ...", template)
        try:
            for member in client.iter_embeddedin_template(template):
                pid, title = member["pageid"], member["title"]
                if pid not in all_pages:
                    all_pages[pid] = title
        except Exception as exc:
            logger.error("Failed to find pages for template '%s': %s", template, exc)

    logger.info("Total unique pages found: %d", len(all_pages))

    # Filter out non-character pages (list/aggregate/disambiguation pages)
    def _is_character_page(title: str) -> bool:
        tl = title.lower()
        skip_prefixes = ("list of ", "category:", "template:", "file:",
                         "talk:", "user:", "help:", "portal:")
        skip_keywords = ("disambiguation",)
        if any(tl.startswith(p) for p in skip_prefixes):
            return False
        if any(kw in tl for kw in skip_keywords):
            return False
        # Pages with "/" are usually sub-list pages (e.g. "List of Minor Characters/Phase One")
        if "/" in title:
            return False
        return True

    all_pages_filtered = {
        pid: title for pid, title in all_pages.items()
        if _is_character_page(title)
    }
    logger.info("After filtering non-character pages: %d", len(all_pages_filtered))

    # Apply limit (for testing) and exclude already-scraped pages
    pages_to_scrape = [
        {"pageid": pid, "title": title}
        for pid, title in all_pages_filtered.items()
        if title not in already_scraped
    ]
    if limit:
        pages_to_scrape = pages_to_scrape[:limit]

    logger.info("Pages to scrape this run: %d", len(pages_to_scrape))

    # -----------------------------------------------------------------------
    # Step 2: Fetch wikitext in batches of BATCH_SIZE and parse each page
    # -----------------------------------------------------------------------
    total = len(pages_to_scrape)
    num_batches = math.ceil(total / BATCH_SIZE)

    with tqdm(total=total, desc=config["name"], unit="char") as pbar:
        for batch_idx in range(num_batches):
            batch = pages_to_scrape[batch_idx * BATCH_SIZE:(batch_idx + 1) * BATCH_SIZE]
            titles = [p["title"] for p in batch]
            pid_map = {p["title"]: p["pageid"] for p in batch}

            try:
                wikitext_map = client.fetch_pages_wikitext(titles)
            except Exception as exc:
                logger.error("Batch fetch failed: %s — skipping batch.", exc)
                pbar.update(len(batch))
                continue

            for page in batch:
                title = page["title"]
                pid   = page["pageid"]
                wikitext = wikitext_map.get(title, "")

                record: dict = {
                    "wiki": config["name"],
                    "franchise": config["franchise"],
                    "name": title,
                    "page_id": pid,
                    "page_url": f"{config['base_url']}/wiki/{title.replace(' ', '_')}",
                    "content_rating": config["content_rating"],
                }

                if wikitext:
                    parsed = parse_character_page(wikitext, config)
                    record.update(parsed)
                else:
                    logger.debug("No wikitext for '%s' — possibly a redirect or empty.", title)

                # Only keep non-redirect pages that have an infobox dict
                is_redirect = wikitext and wikitext.lstrip().upper().startswith("#REDIRECT")
                has_infobox = bool(record.get("infobox"))
                if not is_redirect and has_infobox:
                    new_rows.append(record)
                elif not is_redirect and not has_infobox:
                    logger.debug("Skipping '%s' — no character infobox found.", title)

                already_scraped.add(title)
                pbar.update(1)

            # Flush to disk periodically
            if len(new_rows) >= FLUSH_EVERY:
                _flush(wiki_key, existing_rows, new_rows, out_path)
                existing_rows.extend(new_rows)
                new_rows.clear()
                save_checkpoint(wiki_key, already_scraped)

    # Final flush
    if new_rows:
        _flush(wiki_key, existing_rows, new_rows, out_path)
        existing_rows.extend(new_rows)
    save_checkpoint(wiki_key, already_scraped)

    logger.info("Finished %s. Total rows saved: %d", config["name"], len(existing_rows))
    return existing_rows


def _flush(wiki_key: str, existing: list, new: list, out_path) -> None:
    """Append new rows to the output JSONL."""
    with open(out_path, "a" if existing else "w", encoding="utf-8") as f:
        for row in new:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    logger.debug("Flushed %d new rows → %s", len(new), out_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Scrape character death data from Fandom wikis.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--wiki", nargs="+", required=True,
        help="Wiki key(s) to scrape (e.g. mcu got) or 'all'.",
    )
    parser.add_argument(
        "--fresh", action="store_true",
        help="Ignore checkpoint and re-scrape from scratch.",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Max characters to scrape per wiki (for testing).",
    )
    parser.add_argument(
        "--delay", nargs=2, type=float, default=[1.0, 2.5],
        metavar=("MIN", "MAX"),
        help="Request delay range in seconds (default: 1.0 2.5).",
    )
    parser.add_argument(
        "--log", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: INFO).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    setup_logging(args.log)
    ensure_dirs()

    # Resolve wiki keys
    if "all" in args.wiki:
        wiki_keys = list(WIKI_CONFIGS.keys())
    else:
        wiki_keys = []
        for key in args.wiki:
            if key not in WIKI_CONFIGS:
                logger.error("Unknown wiki key '%s'. Available: %s", key,
                             ", ".join(WIKI_CONFIGS.keys()))
            else:
                wiki_keys.append(key)

    if not wiki_keys:
        logger.error("No valid wiki keys provided. Exiting.")
        return

    min_delay, max_delay = args.delay

    summary = []
    for key in wiki_keys:
        rows = scrape_wiki(
            wiki_key=key,
            fresh=args.fresh,
            limit=args.limit,
            min_delay=min_delay,
            max_delay=max_delay,
        )
        n_total = len(rows)
        n_dead  = sum(1 for r in rows if r.get("is_dead") == 1)
        n_alive = sum(1 for r in rows if r.get("is_dead") == 0)
        n_unknown = n_total - n_dead - n_alive
        summary.append({
            "wiki":    key,
            "total":   n_total,
            "dead":    n_dead,
            "alive":   n_alive,
            "unknown": n_unknown,
        })

    # Print summary table
    print("\n" + "=" * 50)
    print(f"{'Wiki':<15} {'Total':>7} {'Dead':>7} {'Alive':>7} {'Unknown':>9}")
    print("-" * 50)
    for row in summary:
        print(f"{row['wiki']:<15} {row['total']:>7} {row['dead']:>7} "
              f"{row['alive']:>7} {row['unknown']:>9}")
    print("=" * 50)
    print(f"Output saved to: {raw_path('*').parent}")


if __name__ == "__main__":
    main()
