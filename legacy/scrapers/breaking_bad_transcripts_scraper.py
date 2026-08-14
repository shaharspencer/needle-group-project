import os
import re
import time
from urllib.parse import urljoin, urlparse
import cloudscraper
from bs4 import BeautifulSoup

SHOW_URL = "https://subslikescript.com/series/Breaking_Bad-903747"

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

def get_show_title(soup, url):
    """Extract show title from page or URL slug."""
    h1 = soup.find("h1")
    if h1:
        return h1.get_text(strip=True)
    parsed = urlparse(url)
    slug = parsed.path.split('/')[-1]
    title_part = slug.split('-')[0] if '-' in slug else slug
    return title_part.replace('_', ' ')

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

        for br in script_div.find_all("br"):
            br.replace_with("\n")

        lines = [line.strip() for line in script_div.get_text().split("\n") if line.strip()]
        return "\n".join(lines)
    except Exception as e:
        print(f"      [!] Error fetching {url}: {e}")
        return ""

def main():
    print(f"Fetching main page: {SHOW_URL}")
    resp = scraper.get(SHOW_URL)
    
    if resp.status_code != 200:
        print(f"[!] Access blocked or page not found (HTTP {resp.status_code}).")
        return

    soup = BeautifulSoup(resp.text, "html.parser")
    show_name = get_show_title(soup, SHOW_URL)
    output_dir = f"{sanitize_filename(show_name)}_Transcripts"

    all_links = soup.find_all("a", href=re.compile(r"season-\d+/episode-\d+", re.IGNORECASE))

    if not all_links:
        print("[!] No episode links found. Checking page title...")
        title_tag = soup.find("title")
        print(f"    Page Title: {title_tag.get_text() if title_tag else 'No title'}")
        return

    print(f"[✔] Show identified: '{show_name}'")
    print(f"[✔] Found {len(all_links)} total episode links across all seasons!")
    os.makedirs(output_dir, exist_ok=True)

    seasons = {}
    visited_urls = set()

    for a in all_links:
        href = a["href"]
        title = a.get_text(strip=True)
        full_url = urljoin(SHOW_URL, href)

        if full_url in visited_urls:
            continue
        visited_urls.add(full_url)
        
        match = re.search(r"season-(\d+)/episode-(\d+)", href, re.IGNORECASE)
        if match:
            season_num = int(match.group(1))
            ep_num = int(match.group(2))
            
            if season_num not in seasons:
                seasons[season_num] = []
            
            seasons[season_num].append((ep_num, title, full_url))

    total_saved = 0

    for season_num in sorted(seasons.keys()):
        season_dir = os.path.join(output_dir, f"Season_{season_num:02d}")
        os.makedirs(season_dir, exist_ok=True)
        episodes = seasons[season_num]
        
        print(f"\n📁 Season {season_num:02d} ({len(episodes)} episodes) -> {season_dir}")

        for ep_num, ep_title, ep_url in episodes:
            clean_title = sanitize_filename(ep_title)
            filename = f"S{season_num:02d}E{ep_num:02d}_{clean_title}.txt"
            filepath = os.path.join(season_dir, filename)

            if os.path.exists(filepath):
                print(f"  └─ Skipping S{season_num:02d}E{ep_num:02d} (Already exists)")
                continue

            print(f"  └─ Fetching S{season_num:02d}E{ep_num:02d}: {ep_title}...")
            content = get_transcript_text(ep_url)

            if content and len(content) > 100:
                with open(filepath, "w", encoding="utf-8") as f:
                    f.write(f"Show: {show_name}\n")
                    f.write(f"Season {season_num}, Episode {ep_num}: {ep_title}\n")
                    f.write(f"Source: {ep_url}\n")
                    f.write("=" * 60 + "\n\n")
                    f.write(content)
                total_saved += 1
            else:
                print(f"     [!] Warning: Empty or short transcript for {ep_title}")

            time.sleep(0.5)

    print(f"\n[✔] Completed! Saved {total_saved} transcripts into '{output_dir}'.")

if __name__ == "__main__":
    main()