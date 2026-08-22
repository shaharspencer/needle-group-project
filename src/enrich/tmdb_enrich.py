import argparse
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request

import pandas as pd

from src.constants import (
    TMDB_API_ROOT,
    TMDB_MAX_DISCOVER_PAGES,
    TMDB_REQUEST_SLEEP,
    TMDB_RETRIES,
)
from src.enrich.tmdb_titles import FRANCHISE_TITLES
from src.paths import CLEAN_DIR, EXTERNAL_DIR, REPO_ROOT

CACHE_DIR = EXTERNAL_DIR / "tmdb"


class TmdbEnricher:
    """
    Pull cast and title metadata from TMDB for each franchise.

    Gives billing order and episode counts as a prominence measure, actor
    birthdays for character age, and per-title genre, rating and season counts.
    Responses are cached to disk so re-runs cost no requests.
    """

    @staticmethod
    def api_key() -> str:
        key = os.environ.get("TMDB_API_KEY")
        if key:
            return key

        env = REPO_ROOT / ".env"
        if env.exists():
            for line in env.read_text(encoding="utf-8").splitlines():
                if line.startswith("TMDB_API_KEY="):
                    return line.split("=", 1)[1].strip()

        raise SystemExit("No TMDB API key. Set TMDB_API_KEY in .env or the environment.")

    @staticmethod
    def get(path: str, key: str, params: dict | None = None) -> dict:
        params = dict(params or {})
        query = urllib.parse.urlencode(sorted(params.items()))
        cache_path = CACHE_DIR / (path.replace("/", "_") + (f"_{query}" if query else "") + ".json")

        if cache_path.exists():
            try:
                return json.loads(cache_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                cache_path.unlink()

        params["api_key"] = key
        url = TMDB_API_ROOT + path + "?" + urllib.parse.urlencode(params)

        for attempt in range(TMDB_RETRIES):
            try:
                with urllib.request.urlopen(url, timeout=30) as response:
                    data = json.load(response)
                break
            except urllib.error.HTTPError as exc:
                if exc.code == 404:
                    data = {"_missing": True}
                    break
                if exc.code == 429:
                    time.sleep(2 * (attempt + 1))
                    continue
                raise
            except (urllib.error.URLError, TimeoutError):
                time.sleep(1.5 * (attempt + 1))
        else:
            raise SystemExit(f"TMDB request failed repeatedly: {path}")

        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(data), encoding="utf-8")
        time.sleep(TMDB_REQUEST_SLEEP)
        return data

    @staticmethod
    def resolve_titles(spec: dict, key: str) -> list[dict]:
        """Expand a franchise spec into a flat list of titles."""
        titles: list[dict] = []

        titles += [{"media_type": "tv", "id": i} for i in spec.get("tv", [])]
        titles += [{"media_type": "movie", "id": i} for i in spec.get("movie", [])]

        for collection_id in spec.get("collection", []):
            data = TmdbEnricher.get(f"collection/{collection_id}", key)
            titles += [{"media_type": "movie", "id": p["id"]} for p in data.get("parts", [])]

        for company_id in spec.get("company", []):
            page = 1
            while page <= TMDB_MAX_DISCOVER_PAGES:
                data = TmdbEnricher.get("discover/movie", key, {
                    "with_companies": company_id,
                    "sort_by": "release_date.asc",
                    "page": page,
                })
                titles += [{"media_type": "movie", "id": r["id"]} for r in data.get("results", [])]
                if page >= data.get("total_pages", 1):
                    break
                page += 1

        # A film can belong to two collections, e.g. Alien and Predator.
        seen, unique = set(), []
        for title in titles:
            pair = (title["media_type"], title["id"])
            if pair not in seen:
                seen.add(pair)
                unique.append(title)
        return unique

    @staticmethod
    def fetch_title(franchise: str, title: dict, key: str) -> tuple[dict, list[dict]]:
        media_type, tmdb_id = title["media_type"], title["id"]
        detail = TmdbEnricher.get(f"{media_type}/{tmdb_id}", key)
        if detail.get("_missing"):
            return {}, []

        if media_type == "tv":
            credits = TmdbEnricher.get(f"tv/{tmdb_id}/aggregate_credits", key)
            name = detail.get("name", "")
            release = detail.get("first_air_date", "")
            # A series is one title, so a franchise like Grey's Anatomy would
            # otherwise have first_year == last_year and a run span of zero
            # despite running since 2005. Q2 regresses mortality on run span,
            # so without this every single-series franchise looked instantaneous.
            end = detail.get("last_air_date", "") or ""
        else:
            credits = TmdbEnricher.get(f"movie/{tmdb_id}/credits", key)
            name = detail.get("title", "")
            release = detail.get("release_date", "")
            # A film ends when it is released.
            end = release

        title_row = {
            "franchise": franchise,
            "media_type": media_type,
            "tmdb_id": tmdb_id,
            "title": name,
            "release_date": release,
            "release_year": int(release[:4]) if release[:4].isdigit() else None,
            "end_date": end,
            "end_year": int(end[:4]) if end[:4].isdigit() else None,
            "genres": "|".join(g["name"] for g in detail.get("genres", [])),
            "vote_average": detail.get("vote_average"),
            "vote_count": detail.get("vote_count"),
            "popularity": detail.get("popularity"),
            "runtime": detail.get("runtime"),
            "n_seasons": detail.get("number_of_seasons"),
            "n_episodes": detail.get("number_of_episodes"),
            "cast_size": len(credits.get("cast", [])),
        }

        cast_rows = []
        for member in credits.get("cast", []):
            if media_type == "tv":
                roles = member.get("roles") or [{}]
                character = roles[0].get("character", "")
                episodes = member.get("total_episode_count")
            else:
                character = member.get("character", "")
                episodes = None

            cast_rows.append({
                "franchise": franchise,
                "media_type": media_type,
                "tmdb_id": tmdb_id,
                "title": name,
                "release_year": title_row["release_year"],
                "character": character,
                "actor": member.get("name", ""),
                "actor_tmdb_id": member.get("id"),
                "billing_order": member.get("order"),
                "episode_count": episodes,
                "actor_gender_code": member.get("gender"),  # 1 female, 2 male, 0 unknown
                "actor_popularity": member.get("popularity"),
            })
        return title_row, cast_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--franchise", action="append")
    args = parser.parse_args()

    key = TmdbEnricher.api_key()
    wanted = args.franchise or sorted(FRANCHISE_TITLES)

    all_titles, all_cast = [], []
    for franchise in wanted:
        spec = FRANCHISE_TITLES.get(franchise)
        if spec is None:
            print(f"  unknown franchise: {franchise}")
            continue

        titles = TmdbEnricher.resolve_titles(spec, key)
        credits = 0
        for title in titles:
            title_row, cast_rows = TmdbEnricher.fetch_title(franchise, title, key)
            if title_row:
                all_titles.append(title_row)
                all_cast.extend(cast_rows)
                credits += len(cast_rows)
        print(f"  {franchise:34s} {len(titles):4d} titles  {credits:6d} cast credits")

    titles_df = pd.DataFrame(all_titles)
    cast_df = pd.DataFrame(all_cast)

    CLEAN_DIR.mkdir(parents=True, exist_ok=True)
    titles_df.to_csv(CLEAN_DIR / "tmdb_titles.csv", index=False, encoding="utf-8")
    cast_df.to_csv(CLEAN_DIR / "tmdb_cast.csv", index=False, encoding="utf-8")

    distinct = cast_df["character"].nunique() if len(cast_df) else 0
    print(f"\n{len(titles_df)} titles, {len(cast_df)} cast credits, {distinct} character names")


if __name__ == "__main__":
    from src import setup_run_log
    setup_run_log(__spec__)
    main()
