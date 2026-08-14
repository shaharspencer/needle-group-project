import os
import re
import time
import cloudscraper
from bs4 import BeautifulSoup

BASE_URL = "https://subslikescript.com"
SHOW_URL = f"{BASE_URL}/series/The_Walking_Dead-1520211"
OUTPUT_DIR = "The_Walking_Dead_Transcripts"

# Setup scraper with realistic browser spoofing
scraper = cloudscraper.create_scraper(
    browser={
        'browser': 'chrome',
        'platform': 'windows',
        'desktop': True
    }
)

def sanitize_filename(name):
    """Clean string to create safe filenames."""
    clean = re.sub(r'[\\/*?:"<>|]', "", name).strip()
    return clean.replace(" ", "_")

def get_transcript_text(url):
    """Fetch an episode page and extract transcript text."""
    try:
        resp = scraper.get(url, timeout=15)
        if resp.status_code != 200:
            return ""

        soup = BeautifulSoup(resp.text, "html.parser")
        script_div = soup.find("div", class_="full-script")
        
        if not script_div:
            return ""

        # Preserve line breaks
        for br in script_div.find_all("br"):
            br.replace_with("\n")

        return script_div.get_text().strip()
    except Exception as e:
        print(f"      [!] Error fetching {url}: {e}")
        return ""

def main():
    print(f"Fetching main page: {SHOW_URL}")
    resp = scraper.get(SHOW_URL)
    
    if resp.status_code != 200:
        print(f"[!] Access blocked by Cloudflare (HTTP {resp.status_code}).")
        return

    soup = BeautifulSoup(resp.text, "html.parser")

    # Find all episode links on the page using subslikescript's URL pattern
    # URL pattern: /series/The_Walking_Dead-1520211/season-1/episode-1-Days_Gone_Bye
    all_links = soup.find_all("a", href=re.compile(r"/season-\d+/episode-\d+"))

    if not all_links:
        print("[!] Still no links found. Printing page title to diagnose:")
        title = soup.find("title")
        print(f"    Page Title: {title.get_text() if title else 'No title'}")
        return

    print(f"[✔] Found {len(all_links)} total episode links across all seasons!")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Group links by season
    seasons = {}
    for a in all_links:
        href = a["href"]
        title = a.get_text(strip=True)
        
        match = re.search(r"/season-(\d+)/episode-(\d+)", href)
        if match:
            season_num = int(match.group(1))
            ep_num = int(match.group(2))
            full_url = BASE_URL + href if href.startswith("/") else href
            
            if season_num not in seasons:
                seasons[season_num] = []
            
            seasons[season_num].append((ep_num, title, full_url))

    total_saved = 0

    # Process each season
    for season_num in sorted(seasons.keys()):
        season_dir = os.path.join(OUTPUT_DIR, f"Season_{season_num:02d}")
        os.makedirs(season_dir, exist_ok=True)
        episodes = seasons[season_num]
        
        print(f"\n📁 Season {season_num:02d} ({len(episodes)} episodes) -> {season_dir}")

        for ep_num, ep_title, ep_url in episodes:
            clean_title = sanitize_filename(ep_title)
            filename = f"S{season_num:02d}E{ep_num:02d}_{clean_title}.txt"
            filepath = os.path.join(season_dir, filename)

            print(f"  └─ Fetching S{season_num:02d}E{ep_num:02d}: {ep_title}...")
            content = get_transcript_text(ep_url)

            if content and len(content) > 100:
                with open(filepath, "w", encoding="utf-8") as f:
                    f.write(f"Show: The Walking Dead\n")
                    f.write(f"Season {season_num}, Episode {ep_num}: {ep_title}\n")
                    f.write(f"Source: {ep_url}\n")
                    f.write("=" * 60 + "\n\n")
                    f.write(content)
                total_saved += 1
            else:
                print(f"     [!] Warning: Empty or short transcript for {ep_title}")

            time.sleep(0.5)

    print(f"\n[✔] Completed! Saved {total_saved} transcripts into '{OUTPUT_DIR}'.")

if __name__ == "__main__":
    main()