import csv
import re
import cloudscraper
from bs4 import BeautifulSoup

URL = "https://listofdeaths.fandom.com/wiki/The_100"
OUTPUT_CSV = "the_100_deaths.csv"

scraper = cloudscraper.create_scraper(
    browser={
        'browser': 'chrome',
        'platform': 'windows',
        'desktop': True
    }
)

def scrape_the100_deaths():
    print(f"[+] Fetching page: {URL}")
    response = scraper.get(URL, timeout=15)

    if response.status_code != 200:
        print(f"[!] HTTP Error {response.status_code}")
        return

    soup = BeautifulSoup(response.text, "html.parser")
    content = soup.find("div", class_="mw-parser-output")

    if not content:
        print("[!] Could not locate main body content.")
        return

    records = []
    current_season = "Season 1"

    # Traverse all section headings
    for heading in content.find_all(["h2", "h3", "h4"]):
        headline = heading.find("span", class_="mw-headline")
        title_text = headline.get_text(strip=True) if headline else heading.get_text(strip=True)
        title_text = re.sub(r"\[edit\]|\[\d+\]", "", title_text).strip()

        # Update current season
        if heading.name == "h2" and "season" in title_text.lower():
            current_season = title_text
            continue

        # Process episode headings (h3 / h4)
        if heading.name in ["h3", "h4"]:
            if any(skip in title_text.lower() for skip in ["contents", "navigation", "references", "see also"]):
                continue

            episode_title = title_text
            victims = []
            total_deaths = 0
            found_total_line = False

            # Collect siblings until the next heading tag
            curr = heading.find_next_sibling()
            while curr and curr.name not in ["h2", "h3", "h4"]:
                # Inspect list items and paragraph text
                elements = curr.find_all("li") if curr.name in ["ul", "ol"] else [curr]
                
                for el in elements:
                    text = el.get_text(" ", strip=True)
                    clean_text = re.sub(r"\[\d+\]", "", text).strip()

                    # Direct regex capture for "Total - X", "Total – X", or "Total: X"
                    total_match = re.search(r'\bTotal\s*[\-–—:]\s*([\d,]+)', clean_text, re.IGNORECASE)
                    
                    if total_match:
                        total_deaths = int(total_match.group(1).replace(",", ""))
                        found_total_line = True
                    elif curr.name in ["ul", "ol"] and clean_text:
                        # Append as victim entry if it isn't the Total summary line
                        victims.append(clean_text)

                curr = curr.find_next_sibling()

            # If no explicit "Total - X" line existed, default to victim list length
            if not found_total_line:
                total_deaths = len(victims)

            # Extract episode identifier if available (e.g., "1x01", "Episode 1")
            ep_num_match = re.search(r"(\d+x\d+|S\d+E\d+|Episode\s*\d+)", episode_title, re.IGNORECASE)
            ep_num = ep_num_match.group(1) if ep_num_match else "N/A"

            records.append({
                "Season": current_season,
                "Episode Number": ep_num,
                "Episode Title": episode_title,
                "Total Deaths": total_deaths,
                "Victims": "; ".join(victims) if victims else "None listed"
            })

    # Save output to CSV
    if records:
        fieldnames = ["Season", "Episode Number", "Episode Title", "Total Deaths", "Victims"]
        with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(records)

        print(f"[✔] Successfully exported {len(records)} episode records to '{OUTPUT_CSV}'.")

if __name__ == "__main__":
    scrape_the100_deaths()