import os
import re
import time
from urllib.parse import urljoin
from bs4 import BeautifulSoup
from seleniumbase import Driver

BASE_URL = "https://transcripts.foreverdreaming.org/"
FORUM_URL = f"{BASE_URL}viewforum.php?f=192"
OUTPUT_DIR = "Vikings_Transcripts"

def parse_episode_title(title):
    """Parses season, episode number, and episode title."""
    patterns = [
        r'(?:S|s)?(\d{1,2})[xXEe\.](\d{1,2})\s*[-:]?\s*(.+)',
        r'^(\d{1,2})(\d{2})\s*[-:]?\s*(.+)'
    ]
    for pattern in patterns:
        match = re.search(pattern, title)
        if match:
            return int(match.group(1)), int(match.group(2)), match.group(3).strip()
    return None, None, title

def sanitize_filename(name):
    """Remove unsafe characters for file naming."""
    clean = re.sub(r'[\\/*?:"<>|]', "", name).strip()
    return clean.replace(" ", "_")

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("Launching Undetected Browser (SeleniumBase UC)...")
    driver = Driver(uc=True, headless=False)
    
    try:
        print(f"Connecting to forum: {FORUM_URL}")
        # Automatically handles Cloudflare Turnstile handshake
        driver.uc_open_with_reconnect(FORUM_URL, reconnect_time=6)
        
        # Step 1: Collect all episode URLs across all forum index pages
        topic_list = []
        visited_urls = set()
        start_idx = 0
        
        while True:
            page_url = f"{FORUM_URL}&start={start_idx}"
            print(f"Scanning forum index (start={start_idx})...")
            
            driver.get(page_url)
            time.sleep(2)
            
            # Re-verify if Cloudflare triggers mid-scan
            if "Just a moment" in driver.title or "Verify you are human" in driver.page_source:
                driver.uc_open_with_reconnect(page_url, reconnect_time=6)
            
            soup = BeautifulSoup(driver.page_source, "html.parser")
            raw_links = soup.find_all("a", href=re.compile(r"viewtopic\.php\?.*t=\d+"))
            
            page_topics = 0
            for a in raw_links:
                href = a["href"]
                title = a.get_text().strip()
                full_url = urljoin(BASE_URL, href)
                
                if full_url not in visited_urls and title and not title.startswith("Re:"):
                    visited_urls.add(full_url)
                    topic_list.append((title, full_url))
                    page_topics += 1
                    
            if page_topics == 0:
                print("No more new topics found. Reached end of forum index.")
                break
                
            start_idx += 25  # phpBB forum pagination step
            
        print(f"\n[✔] Found {len(topic_list)} total episode links. Starting downloads...")
        
        # Step 2: Download each transcript
        saved_count = 0
        for raw_title, topic_url in topic_list:
            season_num, ep_num, ep_title = parse_episode_title(raw_title)
            
            if season_num is None or ep_num is None:
                continue
                
            season_dir = os.path.join(OUTPUT_DIR, f"Season_{season_num:02d}")
            os.makedirs(season_dir, exist_ok=True)
            
            filename = f"S{season_num:02d}E{ep_num:02d}_{sanitize_filename(ep_title)}.txt"
            filepath = os.path.join(season_dir, filename)
            
            print(f"  └─ Fetching S{season_num:02d}E{ep_num:02d}: {ep_title}...")
            driver.get(topic_url)
            time.sleep(1.5)
            
            if "Just a moment" in driver.title:
                driver.uc_open_with_reconnect(topic_url, reconnect_time=5)
                
            soup = BeautifulSoup(driver.page_source, "html.parser")
            post_div = soup.find("div", class_="postbody") or soup.find("div", class_="content")
            
            if not post_div:
                print(f"     [!] Skipping: Content element not found for {ep_title}")
                continue
                
            # Clean unwanted signature/ad nodes
            for elem in post_div.find_all(["div", "script", "style"], class_=["postprofile", "signature", "notice", "inline-ad"]):
                elem.decompose()
                
            for br in post_div.find_all("br"):
                br.replace_with("\n")
                
            lines = [line.strip() for line in post_div.get_text().split("\n") if line.strip()]
            content = "\n".join(lines)
            
            if content and len(content) > 100:
                with open(filepath, "w", encoding="utf-8") as f:
                    f.write(f"Show: Vikings\n")
                    f.write(f"Season {season_num}, Episode {ep_num}: {ep_title}\n")
                    f.write(f"Source: {topic_url}\n")
                    f.write("=" * 60 + "\n\n")
                    f.write(content)
                saved_count += 1
            else:
                print(f"     [!] Warning: Minimal text found for {ep_title}")
                
        print(f"\n[✔] Extraction complete! Saved {saved_count} transcripts in '{OUTPUT_DIR}'.")
        
    finally:
        driver.quit()

if __name__ == "__main__":
    main()