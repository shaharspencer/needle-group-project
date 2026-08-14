"""
Pull cast and title metadata from TMDB for every franchise.

Two things the character table badly needs and does not have:

  prominence  got_screentime_min is 99.7% missing, so the pipeline currently
              tiers characters by wiki page length with hardcoded cutoffs
              (<500 / <2000 / <10000 chars). TMDB gives billing order for every
              title, and for TV also episode count per actor, which measures
              the same thing without inventing thresholds.

  age         actor_age_at_release is 98.9% missing. Actor birthday plus title
              release date gives a usable proxy for character age.

Per-title metadata (genre, release date, rating, season/episode counts) is
collected at the same time, which is what turns the franchise-level regression
in Q3 from n=38 into n in the hundreds.

Everything is cached under data/external/tmdb/ as raw JSON, one file per API
call, so re-runs cost no requests and an interrupted run resumes where it
stopped.

Run:  python -m src.enrich.tmdb_enrich            # all franchises
      python -m src.enrich.tmdb_enrich --franchise "Game of Thrones"
"""

from __future__ import annotations

import argparse
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

from src.enrich.tmdb_titles import FRANCHISE_TITLES
from src.paths import CLEAN_DIR, EXTERNAL_DIR, REPO_ROOT

API = "https://api.themoviedb.org/3/"
CACHE_DIR = EXTERNAL_DIR / "tmdb"
# TMDB allows ~50 req/s. This is far below that; the run is not rate-limited in
# practice and the sleep just keeps us clearly inside their limits.
SLEEP = 0.05


def api_key() -> str:
    """Read TMDB_API_KEY from the environment or from a gitignored .env."""
    key = os.environ.get("TMDB_API_KEY")
    if key:
        return key
    env = REPO_ROOT / ".env"
    if env.exists():
        for line in env.read_text(encoding="utf-8").splitlines():
            if line.startswith("TMDB_API_KEY="):
                return line.split("=", 1)[1].strip()
    raise SystemExit(
        "No TMDB API key. Put TMDB_API_KEY=... in .env (which is gitignored) "
        "or set it in the environment."
    )


def get(path: str, key: str, params: dict | None = None) -> dict:
    """GET an API path, caching the response on disk keyed by the full query."""
    params = dict(params or {})
    query = urllib.parse.urlencode(sorted(params.items()))
    cache_name = (path.replace("/", "_") + ("_" + query if query else "")) + ".json"
    cache_path = CACHE_DIR / cache_name

    if cache_path.exists():
        try:
            return json.loads(cache_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            cache_path.unlink()  # truncated by an interrupted run; refetch

    params["api_key"] = key
    url = API + path + "?" + urllib.parse.urlencode(params)
    for attempt in range(4):
        try:
            with urllib.request.urlopen(url, timeout=30) as resp:
                data = json.load(resp)
            break
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                data = {"_missing": True}
                break
            if exc.code == 429:  # rate limited
                time.sleep(2 * (attempt + 1))
                continue
            raise
        except (urllib.error.URLError, TimeoutError):
            time.sleep(1.5 * (attempt + 1))
    else:
        raise SystemExit(f"TMDB request failed repeatedly: {path}")

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(data), encoding="utf-8")
    time.sleep(SLEEP)
    return data


def resolve_titles(franchise: str, spec: dict, key: str) -> list[dict]:
    """Expand a franchise spec into a flat list of {media_type, id} titles."""
    titles: list[dict] = []

    for tv_id in spec.get("tv", []):
        titles.append({"media_type": "tv", "id": tv_id})

    for movie_id in spec.get("movie", []):
        titles.append({"media_type": "movie", "id": movie_id})

    for coll_id in spec.get("collection", []):
        data = get(f"collection/{coll_id}", key)
        for part in data.get("parts", []):
            titles.append({"media_type": "movie", "id": part["id"]})

    for company_id in spec.get("company", []):
        page = 1
        while True:
            data = get("discover/movie", key, {
                "with_companies": company_id,
                "sort_by": "release_date.asc",
                "page": page,
            })
            for r in data.get("results", []):
                titles.append({"media_type": "movie", "id": r["id"]})
            if page >= data.get("total_pages", 1) or page >= 10:
                break
            page += 1

    # A film can sit in two collections (Alien vs Predator pulls both).
    seen, unique = set(), []
    for t in titles:
        pair = (t["media_type"], t["id"])
        if pair not in seen:
            seen.add(pair)
            unique.append(t)
    return unique


def fetch_title(franchise: str, title: dict, key: str) -> tuple[dict, list[dict]]:
    """Return (title_row, cast_rows) for one film or TV series."""
    mt, tid = title["media_type"], title["id"]
    detail = get(f"{mt}/{tid}", key)
    if detail.get("_missing"):
        return {}, []

    if mt == "tv":
        credits = get(f"tv/{tid}/aggregate_credits", key)
        name = detail.get("name", "")
        release = detail.get("first_air_date", "")
    else:
        credits = get(f"movie/{tid}/credits", key)
        name = detail.get("title", "")
        release = detail.get("release_date", "")

    title_row = {
        "franchise": franchise,
        "media_type": mt,
        "tmdb_id": tid,
        "title": name,
        "release_date": release,
        "release_year": int(release[:4]) if release[:4].isdigit() else None,
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
        if mt == "tv":
            roles = member.get("roles") or [{}]
            character = roles[0].get("character", "")
            episodes = member.get("total_episode_count")
        else:
            character = member.get("character", "")
            episodes = None
        cast_rows.append({
            "franchise": franchise,
            "media_type": mt,
            "tmdb_id": tid,
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--franchise", action="append",
                    help="limit to one franchise (repeatable)")
    args = ap.parse_args()

    key = api_key()
    wanted = args.franchise or sorted(FRANCHISE_TITLES)

    all_titles, all_cast = [], []
    for franchise in wanted:
        spec = FRANCHISE_TITLES.get(franchise)
        if spec is None:
            print(f"  skip unknown franchise: {franchise}")
            continue

        titles = resolve_titles(franchise, spec, key)
        n_cast = 0
        for title in titles:
            title_row, cast_rows = fetch_title(franchise, title, key)
            if title_row:
                all_titles.append(title_row)
                all_cast.extend(cast_rows)
                n_cast += len(cast_rows)
        print(f"  {franchise:34s} {len(titles):4d} titles  {n_cast:6d} cast credits")

    titles_df = pd.DataFrame(all_titles)
    cast_df = pd.DataFrame(all_cast)

    CLEAN_DIR.mkdir(parents=True, exist_ok=True)
    titles_df.to_csv(CLEAN_DIR / "tmdb_titles.csv", index=False, encoding="utf-8")
    cast_df.to_csv(CLEAN_DIR / "tmdb_cast.csv", index=False, encoding="utf-8")

    print(f"\n{len(titles_df)} titles, {len(cast_df)} cast credits, "
          f"{cast_df['character'].nunique() if len(cast_df) else 0} distinct character names")
    print(f"Wrote {CLEAN_DIR / 'tmdb_titles.csv'} and {CLEAN_DIR / 'tmdb_cast.csv'}")


if __name__ == "__main__":
    main()
