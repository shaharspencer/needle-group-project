"""
fandom_api.py
-------------
Rate-limited, anti-blocking MediaWiki API client for Fandom wikis.

Anti-blocking measures:
  - Randomized delay between requests (default 1.0–2.5 s)
  - Realistic browser-like User-Agent header
  - Persistent requests.Session (connection reuse, keep-alive)
  - Exponential backoff with jitter on 429 / 503 / connection errors
  - Batch fetching: up to 50 pages per API call to minimise request count
  - Respects cmcontinue pagination tokens
"""

import time
import random
import logging
from typing import Generator, Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)

# Realistic User-Agent – identifies us as a research scraper, not a bot
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36 "
    "FandomCharacterResearchBot/1.0 (+academic-research)"
)

# How long to wait between requests (seconds) — randomised to appear human
MIN_DELAY = 1.0
MAX_DELAY = 2.5

# Back-off settings when the server pushes back
MAX_RETRIES = 5
BACKOFF_BASE = 2.0  # seconds — doubles each retry


def _make_session() -> requests.Session:
    """Create a session with retry logic baked in at the transport level."""
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})

    retry = Retry(
        total=MAX_RETRIES,
        backoff_factor=BACKOFF_BASE,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


class FandomAPIClient:
    """Thin wrapper around the MediaWiki action API for a single Fandom wiki."""

    def __init__(self, api_url: str, min_delay: float = MIN_DELAY,
                 max_delay: float = MAX_DELAY):
        self.api_url = api_url
        self.min_delay = min_delay
        self.max_delay = max_delay
        self._session = _make_session()
        self._last_request_time: float = 0.0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _throttle(self) -> None:
        """Sleep enough to keep a randomised gap between requests."""
        elapsed = time.time() - self._last_request_time
        delay = random.uniform(self.min_delay, self.max_delay)
        if elapsed < delay:
            time.sleep(delay - elapsed)
        self._last_request_time = time.time()

    def _get(self, params: dict) -> dict:
        """
        Make a single GET request to the API, with throttling and error handling.
        Returns the parsed JSON dict, or raises on unrecoverable errors.
        """
        self._throttle()
        params.setdefault("format", "json")
        params.setdefault("formatversion", "2")

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                resp = self._session.get(self.api_url, params=params, timeout=30)

                if resp.status_code == 429:
                    wait = BACKOFF_BASE ** attempt + random.uniform(0, 1)
                    logger.warning("Rate-limited (429). Waiting %.1fs (attempt %d/%d).",
                                   wait, attempt, MAX_RETRIES)
                    time.sleep(wait)
                    continue

                resp.raise_for_status()
                return resp.json()

            except requests.exceptions.ConnectionError as exc:
                wait = BACKOFF_BASE ** attempt + random.uniform(0, 1)
                logger.warning("Connection error: %s. Retrying in %.1fs.", exc, wait)
                time.sleep(wait)

            except requests.exceptions.Timeout:
                logger.warning("Request timed out (attempt %d/%d).", attempt, MAX_RETRIES)
                time.sleep(BACKOFF_BASE ** attempt)

        raise RuntimeError(f"Failed to fetch {self.api_url} after {MAX_RETRIES} attempts.")

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    def iter_category_members(
        self,
        category: str,
        namespace: int = 0,
        batch_size: int = 500,
    ) -> Generator[dict, None, None]:
        """
        Yield all pages in *category* (namespace=0 = article pages only).
        Handles pagination automatically via cmcontinue.

        Each yielded item: {"pageid": int, "title": str}
        """
        params = {
            "action": "query",
            "list": "categorymembers",
            "cmtitle": f"Category:{category}",
            "cmnamespace": namespace,
            "cmlimit": batch_size,
            "cmtype": "page",
        }
        continue_token: Optional[str] = None

        while True:
            if continue_token:
                params["cmcontinue"] = continue_token

            data = self._get(params)

            members = data.get("query", {}).get("categorymembers", [])
            for member in members:
                yield member

            cont = data.get("continue", {})
            if "cmcontinue" in cont:
                continue_token = cont["cmcontinue"]
                logger.debug("Paginating: cmcontinue=%s", continue_token)
            else:
                break

    def iter_embeddedin_template(
        self,
        template: str,
        namespace: int = 0,
        batch_size: int = 500,
    ) -> Generator[dict, None, None]:
        """
        Yield all pages that embed the specified template.
        Handles pagination automatically via eicontinue.
        """
        params = {
            "action": "query",
            "list": "embeddedin",
            "eititle": f"Template:{template}",
            "einamespace": namespace,
            "eilimit": batch_size,
        }
        continue_token: Optional[str] = None

        while True:
            if continue_token:
                params["eicontinue"] = continue_token

            data = self._get(params)

            members = data.get("query", {}).get("embeddedin", [])
            for member in members:
                yield member

            cont = data.get("continue", {})
            if "eicontinue" in cont:
                continue_token = cont["eicontinue"]
                logger.debug("Paginating: eicontinue=%s", continue_token)
            else:
                break

    def fetch_pages_wikitext(
        self,
        titles: list[str],
    ) -> dict[str, str]:
        """
        Fetch raw wikitext for up to 50 page titles in a single API call.
        Returns {title: wikitext_string}.
        """
        assert len(titles) <= 50, "MediaWiki API allows max 50 titles per batch."

        params = {
            "action": "query",
            "prop": "revisions",
            "titles": "|".join(titles),
            "rvprop": "content",
            "rvslots": "main",
        }

        data = self._get(params)
        pages = data.get("query", {}).get("pages", [])

        result: dict[str, str] = {}
        for page in pages:
            title = page.get("title", "")
            revisions = page.get("revisions", [])
            if not revisions:
                continue
            wikitext = revisions[0].get("slots", {}).get("main", {}).get("content", "")
            if wikitext:
                result[title] = wikitext

        return result

    def fetch_page_wikitext(self, title: str) -> Optional[str]:
        """Convenience wrapper for a single page."""
        result = self.fetch_pages_wikitext([title])
        return result.get(title)
