import os
import re
import time
import requests
from bs4 import BeautifulSoup

API_URL = "https://the100.fandom.com/api.php"
OUTPUT_DIR = "The_100_Transcripts"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) The100TranscriptBot/2.0"
}

def sanitize_filename(name):
    """Remove invalid filename characters."""
    clean = re.sub(r'[\\/*?:"<>|]', "", name).strip()
    return clean.replace(" ", "_")

def get_season_episodes():
    """Parse the main Transcripts page to map episode titles & links by season."""
    params = {
        "action": "parse",
        "page": "Transcripts",
        "prop": "text",
        "format": "json"
    }
    response = requests.get(API_URL, headers=HEADERS, params=params)
    response.raise_for_status()
    html_content = response.json()["parse"]["text"]["*"]
    soup = BeautifulSoup(html_content, "html.parser")

    seasons_data = {}
    
    # Fandom transcript nav table or content tables
    tables = soup.find_all("table", class_=["wikitable", "article-table"])
    
    # Season link mapping fallback
    season_idx = 1
    for table in tables:
        links = []
        for a in table.find_all("a", href=True):
            href = a["href"]
            title = a.get_text(strip=True)
            if "/wiki/" in href and not any(x in href for x in ["Special:", "Category:", "File:", "Transcripts"]):
                # Construct proper wiki page title
                page_name = href.split("/wiki/")[-1]
                if not page_name.endswith("/Transcript"):
                    page_name += "/Transcript"
                if (page_name, title) not in links:
                    links.append((page_name, title))
        
        if links:
            seasons_data[season_idx] = links
            season_idx += 1

    return seasons_data

def fetch_transcript_content(page_title):
    """Retrieve full transcript content handling both table and paragraph formats."""
    params = {
        "action": "parse",
        "page": page_title,
        "prop": "text",
        "format": "json"
    }
    response = requests.get(API_URL, headers=HEADERS, params=params)
    response.raise_for_status()
    data = response.json()

    if "parse" not in data or "text" not in data["parse"]:
        return ""

    raw_html = data["parse"]["text"]["*"]
    soup = BeautifulSoup(raw_html, "html.parser")
    content_div = soup.find("div", {"class": "mw-parser-output"})

    if not content_div:
        return ""

    # Remove non-transcript UI noise (keep transcript tables intact)
    for elem in content_div.find_all(["script", "style", "nav"]):
        elem.decompose()
    for elem in content_div.find_all("div", class_=["navbox", "toc", "mw-editsection", "aside", "portable-infobox"]):
        elem.decompose()

    lines = []

    # Iterate through transcript content blocks (Headings, Tables, Paragraphs)
    for elem in content_div.find_all(["h2", "h3", "h4", "table", "p", "dl", "ul"]):
        # Section headings (e.g., "Scene 1", "Act I")
        if elem.name in ["h2", "h3", "h4"]:
            heading_text = elem.get_text().replace("[edit]", "").strip()
            if heading_text and not heading_text.lower().startswith("see also"):
                lines.append(f"\n--- {heading_text} ---")

        # Dialogue formatted inside HTML tables (Speaker | Dialogue)
        elif elem.name == "table":
            for row in elem.find_all("tr"):
                cols = row.find_all(["td", "th"])
                col_texts = [c.get_text(strip=True) for c in cols]
                if not col_texts:
                    continue
                # Ignore table header row
                if len(col_texts) >= 2 and col_texts[0].lower() in ["sp.", "sp", "speaker"] and col_texts[1].lower() in ["dialogue", "line"]:
                    continue
                
                if len(col_texts) >= 2:
                    speaker, dialogue = col_texts[0], col_texts[1]
                    if speaker and dialogue:
                        lines.append(f"{speaker}: {dialogue}")
                    elif dialogue:
                        lines.append(dialogue)
                elif len(col_texts) == 1:
                    lines.append(col_texts[0])

        # Paragraph / inline dialogue format
        else:
            text = elem.get_text().strip()
            if text and not text.startswith("Categories:"):
                lines.append(text)

    return "\n\n".join(lines)

def main():
    print("Extracting season structure from 'The 100' main transcript index...")
    seasons_data = get_season_episodes()

    if not seasons_data:
        print("[!] Failed to map season structure automatically.")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    total_saved = 0

    for season_num, episodes in seasons_data.items():
        season_dir = os.path.join(OUTPUT_DIR, f"Season_{season_num:02d}")
        os.makedirs(season_dir, exist_ok=True)
        print(f"\n📁 Season {season_num:02d} ({len(episodes)} episodes) -> {season_dir}")

        for ep_idx, (page_title, ep_name) in enumerate(episodes, start=1):
            clean_title = sanitize_filename(ep_name if ep_name else f"Episode_{ep_idx}")
            filename = f"S{season_num:02d}E{ep_idx:02d}_{clean_title}.txt"
            filepath = os.path.join(season_dir, filename)

            print(f"  └─ Downloading S{season_num:02d}E{ep_idx:02d}: {page_title}...")
            content = fetch_transcript_content(page_title)

            if content and len(content) > 100:
                with open(filepath, "w", encoding="utf-8") as f:
                    f.write(f"Show: The 100\n")
                    f.write(f"Season {season_num}, Episode {ep_idx}: {ep_name}\n")
                    f.write(f"Page Title: {page_title}\n")
                    f.write("=" * 60 + "\n\n")
                    f.write(content)
                total_saved += 1
            else:
                print(f"     [!] Warning: Empty or minimal content for {page_title}")

            time.sleep(0.3)

    print(f"\n[✔] Done! Saved {total_saved} non-empty transcript files into '{OUTPUT_DIR}'.")

if __name__ == "__main__":
    main()