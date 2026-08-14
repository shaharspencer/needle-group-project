import os
import re
import time
from urllib.parse import urljoin
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright

BASE_URL = "https://www.springfieldspringfield.co.uk/"
SHOW_SLUG = "greys-anatomy"
SHOW_INDEX_URL = f"{BASE_URL}episode_scripts.php?tv-show={SHOW_SLUG}"
OUTPUT_DIR = "Greys_Anatomy_Springfield"

def parse_season_episode(href_or_text):
    """
    Extracts season and episode numbers from href strings like:
    'view_episode_scripts.php?tv-show=greys-anatomy&episode=s02e05'
    """
    match = re.search(r'episode=s(\d+)e(\d+)', href_or_text, re.IGNORECASE)
    if match:
        return int(match.group(1)), int(match.group(2))
    return None, None

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    total_saved = 0

    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=False,
            args=["--disable-blink-features=AutomationControlled"]
        )
        context = browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
        )
        page = context.new_page()

        print(f"Opening show index: {SHOW_INDEX_URL}")
        page.goto(SHOW_INDEX_URL)
        time.sleep(3)

        # Detect total available seasons (Season 1 to Season N)
        soup = BeautifulSoup(page.content(), "html.parser")
        season_links = soup.find_all("a", href=re.compile(r"season=\d+", re.IGNORECASE))
        
        max_seasons = 20  # Fallback if season navigation links are hidden
        if season_links:
            seasons_found = [int(re.search(r'season=(\d+)', a['href']).group(1)) for a in season_links if re.search(r'season=(\d+)', a['href'])]
            if seasons_found:
                max_seasons = max(seasons_found)

        print(f"[✔] Detected up to Season {max_seasons}\n")

        visited_episodes = set()

        for season in range(1, max_seasons + 1):
            season_url = f"{SHOW_INDEX_URL}&season={season}"
            print(f"Scanning Season {season} page: {season_url}")
            
            try:
                page.goto(season_url)
                time.sleep(2)
            except Exception as e:
                print(f"  [!] Could not load Season {season}: {e}")
                continue

            season_soup = BeautifulSoup(page.content(), "html.parser")
            
            # Match 'view_episode_scripts.php' (plural 'scripts')
            ep_anchors = season_soup.find_all("a", href=re.compile(r"view_episode_scripts\.php.*episode=s\d+e\d+", re.IGNORECASE))

            if not ep_anchors:
                print(f"  └─ No episodes found for Season {season}.")
                continue

            print(f"  └─ Found {len(ep_anchors)} episode links in Season {season}.")

            for a in ep_anchors:
                href = a["href"]
                full_url = urljoin(BASE_URL, href)
                season_num, ep_num = parse_season_episode(href)

                if season_num is None or ep_num is None or full_url in visited_episodes:
                    continue

                visited_episodes.add(full_url)

                season_dir = os.path.join(OUTPUT_DIR, f"Season_{season_num:02d}")
                os.makedirs(season_dir, exist_ok=True)

                filename = f"S{season_num:02d}E{ep_num:02d}.txt"
                filepath = os.path.join(season_dir, filename)

                if os.path.exists(filepath):
                    print(f"     Skipping existing: S{season_num:02d}E{ep_num:02d}")
                    continue

                print(f"     Downloading S{season_num:02d}E{ep_num:02d}...")

                try:
                    page.goto(full_url)
                    time.sleep(1.5)

                    ep_soup = BeautifulSoup(page.content(), "html.parser")
                    script_div = (
                        ep_soup.find("div", class_="scrolling-script-container") or
                        ep_soup.find("div", class_="episode_script") or
                        ep_soup.find("div", class_="scrolling-script")
                    )

                    if not script_div:
                        print(f"     [!] Container missing for S{season_num:02d}E{ep_num:02d}")
                        continue

                    for br in script_div.find_all("br"):
                        br.replace_with("\n")

                    lines = [line.strip() for line in script_div.get_text().split("\n") if line.strip()]
                    content = "\n".join(lines)

                    if content and len(content) > 100:
                        with open(filepath, "w", encoding="utf-8") as f:
                            f.write(f"Show: Grey's Anatomy\n")
                            f.write(f"Season {season_num}, Episode {ep_num}\n")
                            f.write(f"Source: {full_url}\n")
                            f.write("=" * 60 + "\n\n")
                            f.write(content)
                        total_saved += 1
                except Exception as e:
                    print(f"     [!] Failed downloading {full_url}: {e}")

        browser.close()

    print(f"\n[✔] Extraction complete! Saved {total_saved} transcripts in '{OUTPUT_DIR}'.")

if __name__ == "__main__":
    main()