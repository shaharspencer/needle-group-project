import csv
import json
import re
import time
from urllib.parse import urljoin
import cloudscraper
from bs4 import BeautifulSoup

BASE_URL = "https://the100.fandom.com"
CATEGORY_URL = f"{BASE_URL}/wiki/Category:Characters"
OUTPUT_JSON = "the100_characters_features.json"
OUTPUT_CSV = "the100_characters_features.csv"

# Initialize cloudscraper to bypass Cloudflare protection
scraper = cloudscraper.create_scraper(
    browser={
        'browser': 'chrome',
        'platform': 'windows',
        'desktop': True
    }
)

def get_all_character_urls(start_url):
    """Crawls through Category:Characters and handles pagination to collect all character URLs."""
    urls = set()
    next_url = start_url

    print("[+] Collecting character URLs from category pages...")

    while next_url:
        print(f"  └─ Fetching page: {next_url}")
        resp = scraper.get(next_url, timeout=15)
        
        if resp.status_code != 200:
            print(f"[!] Failed to fetch category page (HTTP {resp.status_code})")
            break

        soup = BeautifulSoup(resp.text, "html.parser")

        # Extract character links from the category grid/list
        members = soup.find_all("a", class_="category-page__member-link")
        for a in members:
            href = a.get("href", "")
            # Ensure it is a wiki page link and not a special category/namespace page
            if href.startswith("/wiki/") and ":" not in href[6:]:
                urls.add(urljoin(BASE_URL, href))

        # Check for pagination (Next page link)
        next_btn = soup.find("a", class_="category-page__pagination-next")
        if next_btn and next_btn.get("href"):
            next_url = next_btn["href"]
        else:
            next_url = None

        time.sleep(0.5)

    print(f"[✔] Found {len(urls)} total unique character page URLs.")
    return sorted(list(urls))

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
            # Extract main image URL
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
                    text_val = re.sub(r"\[\d+\]", "", text_val).strip()  # Clean footnote references like [1], [2]
                    character_data[key] = text_val

        return character_data

    except Exception as e:
        print(f"      [!] Error scraping {url}: {e}")
        return None

def main():
    urls = get_all_character_urls(CATEGORY_URL)
    
    if not urls:
        print("[!] No character links discovered. Exiting.")
        return

    all_characters = []
    all_keys = set(["Name", "URL", "Image_URL"])

    print("\n[+] Scraping features from individual character pages...\n")

    for idx, url in enumerate(urls, 1):
        char_slug = url.split("/")[-1]
        print(f"[{idx}/{len(urls)}] Scraping: {char_slug}")
        
        data = extract_character_features(url)
        if data:
            all_characters.append(data)
            all_keys.update(data.keys())

        time.sleep(0.4)  # Politeness delay

    # 1. Export JSON
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(all_characters, f, indent=4, ensure_ascii=False)
    print(f"\n[✔] Saved JSON to '{OUTPUT_JSON}'")

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
    print(f"[✔] Saved CSV to '{OUTPUT_CSV}'")

if __name__ == "__main__":
    main()