import os
import json
import csv
import time
import re
from urllib.parse import urljoin
import cloudscraper
from bs4 import BeautifulSoup

BASE_URL = "https://walkingdead.fandom.com"
START_URL = f"{BASE_URL}/wiki/TV_Series_Characters"
OUTPUT_JSON = "twd_characters_features.json"
OUTPUT_CSV = "twd_characters_features.csv"

# Initialize cloudscraper with browser emulation
scraper = cloudscraper.create_scraper(
    browser={
        'browser': 'chrome',
        'platform': 'windows',
        'desktop': True
    }
)

def get_character_links(start_url):
    """Scrapes the index page for individual character page URLs."""
    print(f"[+] Connecting to Fandom index: {start_url}")
    resp = scraper.get(start_url, timeout=15)
    
    if resp.status_code != 200:
        print(f"[!] Blocked or failed to load index (HTTP {resp.status_code})")
        return []

    soup = BeautifulSoup(resp.text, "html.parser")
    content_div = soup.find("div", class_="mw-parser-output")
    
    if not content_div:
        print("[!] Could not locate main content block.")
        return []

    links = set()
    for a in content_div.find_all("a", href=True):
        href = a["href"]
        if (
            href.startswith("/wiki/") and
            not any(ignored in href for ignored in [
                ":", "#", "TV_Series_Characters", "The_Walking_Dead_(TV_Series)",
                "Season_", "Episode_", "Gallery", "Comic"
            ])
        ):
            full_url = urljoin(BASE_URL, href)
            links.add(full_url)

    print(f"[✔] Found {len(links)} unique character links.")
    return sorted(list(links))

def extract_character_features(url):
    """Fetches a character page and extracts infobox fields."""
    try:
        resp = scraper.get(url, timeout=15)
        if resp.status_code != 200:
            print(f"      [!] HTTP {resp.status_code} for {url}")
            return None

        soup = BeautifulSoup(resp.text, "html.parser")
        
        title_heading = soup.find("h1", id="firstHeading")
        char_name = title_heading.get_text(strip=True) if title_heading else url.split("/")[-1].replace("_", " ")

        character_data = {
            "Name": char_name,
            "URL": url
        }

        # Locate Fandom Portable Infobox
        infobox = soup.find("aside", class_=re.compile(r"portable-infobox")) or soup.find("div", class_=re.compile(r"portable-infobox"))

        if infobox:
            img_tag = infobox.find("img", class_=re.compile(r"pi-image-thumbnail"))
            if img_tag and img_tag.get("src"):
                character_data["Image_URL"] = img_tag["src"]

            data_items = infobox.find_all("div", class_=re.compile(r"pi-data"))
            for item in data_items:
                label_elem = item.find("h3", class_=re.compile(r"pi-data-label"))
                val_elem = item.find("div", class_=re.compile(r"pi-data-value"))

                if label_elem and val_elem:
                    key = label_elem.get_text(strip=True)
                    val_text = val_elem.get_text(" ", strip=True)
                    val_text = re.sub(r"\[\d+\]", "", val_text).strip()
                    character_data[key] = val_text

        return character_data

    except Exception as e:
        print(f"      [!] Error scraping {url}: {e}")
        return None

def main():
    char_urls = get_character_links(START_URL)

    if not char_urls:
        print("[!] No character links found. Exiting.")
        return

    all_characters = []
    all_feature_keys = set(["Name", "URL", "Image_URL"])

    print("\n[+] Extracting character attributes...\n")

    for idx, url in enumerate(char_urls, 1):
        print(f"[{idx}/{len(char_urls)}] Scraping: {url.split('/')[-1]}...")
        data = extract_character_features(url)

        if data:
            all_characters.append(data)
            all_feature_keys.update(data.keys())

        time.sleep(0.5)

    # Export to JSON
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(all_characters, f, indent=4, ensure_ascii=False)
    print(f"\n[✔] Saved JSON to '{OUTPUT_JSON}'")

    # Export to CSV
    fieldnames = list(all_feature_keys)
    for priority_col in ["Image_URL", "URL", "Name"]:
        if priority_col in fieldnames:
            fieldnames.remove(priority_col)
            fieldnames.insert(0, priority_col)

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_characters)
    print(f"[✔] Saved CSV to '{OUTPUT_CSV}'")

if __name__ == "__main__":
    main()