import csv
import json
import re
import time
from urllib.parse import urljoin
import cloudscraper
from bs4 import BeautifulSoup

BASE_URL = "https://harrypotter.fandom.com"
START_CATEGORY_URL = f"{BASE_URL}/wiki/Category:Individuals"
OUTPUT_JSON = "harrypotter_individuals_features.json"
OUTPUT_CSV = "harrypotter_individuals_features.csv"

# Initialize cloudscraper to bypass Cloudflare protection
scraper = cloudscraper.create_scraper(
    browser={
        'browser': 'chrome',
        'platform': 'windows',
        'desktop': True
    }
)

def crawl_category(category_url, visited_categories, character_urls):
    """Recursively crawls category pages and subcategories to gather all character URLs."""
    if category_url in visited_categories:
        return
    
    visited_categories.add(category_url)
    print(f"  └─ Crawling Category: {category_url.split('/')[-1]}")
    
    next_url = category_url

    while next_url:
        try:
            resp = scraper.get(next_url, timeout=15)
            if resp.status_code != 200:
                print(f"      [!] Failed to load {next_url} (HTTP {resp.status_code})")
                break

            soup = BeautifulSoup(resp.text, "html.parser")

            # 1. Collect individual character links from the member list
            members = soup.find_all("a", class_="category-page__member-link")
            for a in members:
                href = a.get("href", "")
                full_url = urljoin(BASE_URL, href)
                
                if href.startswith("/wiki/Category:"):
                    # Subcategory found - queue for recursive traversal
                    if full_url not in visited_categories:
                        crawl_category(full_url, visited_categories, character_urls)
                elif href.startswith("/wiki/") and ":" not in href[6:]:
                    # Regular page - add to character dataset
                    character_urls.add(full_url)

            # 2. Check for pagination on the current category page
            next_btn = soup.find("a", class_="category-page__pagination-next")
            if next_btn and next_btn.get("href"):
                next_url = next_btn["href"]
                time.sleep(0.3)
            else:
                next_url = None

        except Exception as e:
            print(f"      [!] Error crawling category {next_url}: {e}")
            break

def extract_character_features(url):
    """Scrapes an individual character page and extracts all Portable Infobox features."""
    try:
        resp = scraper.get(url, timeout=15)
        if resp.status_code != 200:
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

            # Extract dynamic key-value attributes
            items = infobox.find_all("div", class_=re.compile(r"pi-data"))
            for item in items:
                label = item.find("h3", class_=re.compile(r"pi-data-label"))
                val = item.find("div", class_=re.compile(r"pi-data-value"))

                if label and val:
                    key = label.get_text(strip=True)
                    text_val = val.get_text(" ", strip=True)
                    # Strip citation references like [1], [2]
                    text_val = re.sub(r"\[\d+\]", "", text_val).strip()
                    character_data[key] = text_val

        return character_data

    except Exception as e:
        print(f"      [!] Error scraping {url}: {e}")
        return None

def main():
    visited_categories = set()
    character_urls = set()

    print("[+] Starting recursive category search under Category:Individuals...")
    crawl_category(START_CATEGORY_URL, visited_categories, character_urls)

    sorted_urls = sorted(list(character_urls))
    print(f"\n[✔] Category crawl complete. Found {len(sorted_urls)} unique character pages.\n")

    if not sorted_urls:
        print("[!] No character links found. Exiting.")
        return

    all_characters = []
    all_keys = set(["Name", "URL", "Image_URL"])

    print("[+] Extracting features from individual character pages...\n")

    for idx, url in enumerate(sorted_urls, 1):
        char_slug = url.split("/")[-1]
        print(f"[{idx}/{len(sorted_urls)}] Scraping: {char_slug}")
        
        data = extract_character_features(url)
        if data:
            all_characters.append(data)
            all_keys.update(data.keys())

        time.sleep(0.3)  # Politeness delay

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