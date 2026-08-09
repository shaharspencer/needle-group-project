import csv
import json
import re
import time
from urllib.parse import urljoin
import cloudscraper
from bs4 import BeautifulSoup

BASE_URL = "https://thehungergames.fandom.com"
START_URL = f"{BASE_URL}/wiki/List_of_characters_in_The_Hunger_Games_series"
OUTPUT_JSON = "hungergames_characters_features.json"
OUTPUT_CSV = "hungergames_characters_features.csv"

# Initialize cloudscraper to handle Cloudflare anti-bot checks
scraper = cloudscraper.create_scraper(
    browser={
        'browser': 'chrome',
        'platform': 'windows',
        'desktop': True
    }
)

def get_character_links(start_url):
    """Parses the main characters list page to find links to all individual character pages."""
    print(f"[+] Fetching character list from: {start_url}")
    resp = scraper.get(start_url, timeout=15)
    
    if resp.status_code != 200:
        print(f"[!] Failed to fetch index page (HTTP {resp.status_code})")
        return []

    soup = BeautifulSoup(resp.text, "html.parser")
    content_div = soup.find("div", class_="mw-parser-output")
    
    if not content_div:
        print("[!] Could not locate main page content.")
        return []

    links = set()
    ignored_keywords = [
        "Category:", "Special:", "File:", "Talk:", "Template:", "Help:",
        "List_of_characters", "The_Hunger_Games_(series)", "The_Hunger_Games_wiki",
        "74th_Hunger_Games", "75th_Hunger_Games", "District_", "Capitol", "Panem"
    ]

    for a in content_div.find_all("a", href=True):
        href = a["href"]
        
        # Filter for character wiki links while ignoring utility/meta links
        if href.startswith("/wiki/") and not any(kw in href for kw in ignored_keywords) and "#" not in href:
            full_url = urljoin(BASE_URL, href)
            links.add(full_url)

    print(f"[✔] Found {len(links)} unique character page links.")
    return sorted(list(links))

def extract_character_features(url):
    """Scrapes individual character page and extracts all Portable Infobox key-value pairs."""
    try:
        resp = scraper.get(url, timeout=15)
        if resp.status_code != 200:
            print(f"      [!] HTTP {resp.status_code} for {url}")
            return None

        soup = BeautifulSoup(resp.text, "html.parser")

        heading = soup.find("h1", id="firstHeading")
        name = heading.get_text(strip=True) if heading else url.split("/")[-1].replace("_", " ")

        character_data = {
            "Name": name,
            "URL": url
        }

        # Locate Fandom Portable Infobox
        infobox = soup.find("aside", class_=re.compile(r"portable-infobox")) or \
                  soup.find("div", class_=re.compile(r"portable-infobox"))

        if infobox:
            # Extract main thumbnail image URL
            img = infobox.find("img", class_=re.compile(r"pi-image-thumbnail"))
            if img and img.get("src"):
                character_data["Image_URL"] = img["src"]

            # Extract dynamic key-value attribute rows
            items = infobox.find_all("div", class_=re.compile(r"pi-data"))
            for item in items:
                label = item.find("h3", class_=re.compile(r"pi-data-label"))
                val = item.find("div", class_=re.compile(r"pi-data-value"))

                if label and val:
                    key = label.get_text(strip=True)
                    text_val = val.get_text(" ", strip=True)
                    # Remove footnote citation markers like [1], [2]
                    text_val = re.sub(r"\[\d+\]", "", text_val).strip()
                    character_data[key] = text_val

        return character_data

    except Exception as e:
        print(f"      [!] Error scraping {url}: {e}")
        return None

def main():
    urls = get_character_links(START_URL)
    
    if not urls:
        print("[!] No character links found. Exiting.")
        return

    all_characters = []
    all_keys = set(["Name", "URL", "Image_URL"])

    print("\n[+] Extracting features from individual character pages...\n")

    for idx, url in enumerate(urls, 1):
        char_slug = url.split("/")[-1]
        print(f"[{idx}/{len(urls)}] Scraping: {char_slug}")
        
        data = extract_character_features(url)
        if data:
            all_characters.append(data)
            all_keys.update(data.keys())

        time.sleep(0.4)  # Respectful delay

    # 1. Export JSON
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(all_characters, f, indent=4, ensure_ascii=False)
    print(f"\n[✔] Saved JSON dataset to '{OUTPUT_JSON}'")

    # 2. Export CSV
    fieldnames = list(all_keys)
    for col in ["Image_URL", "URL", "Name"]:
        if col in fieldnames:
            fieldnames.remove(col)
            fieldnames.insert(0, col)

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_characters)
    print(f"[✔] Saved CSV dataset to '{OUTPUT_CSV}'")

if __name__ == "__main__":
    main()