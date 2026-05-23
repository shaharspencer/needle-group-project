# Needle in a Haystack — Character Mortality Dataset

**67,978: A Needle in a Data Haystack | Hebrew University of Jerusalem**

Binary classification dataset predicting fictional character deaths across 38 franchises and ~60k characters.

---

## Quick Start

```bash
# 1. Install dependencies
pip install requests pandas textblob kaggle

# 2. Set Kaggle API token (get one at kaggle.com/settings)
export KAGGLE_API_TOKEN=<your_token>        # Mac/Linux
set KAGGLE_API_TOKEN=<your_token>           # Windows

# 3. Build the base dataset (~35–40 min)
python build_pipeline.py

# 4. Optionally add NLP text features (~15 min)
python add_nlp_features.py
```

**Outputs:**
- `data/clean/characters_raw.csv` — ~60k rows, ~100 structural features
- `data/clean/characters_full.csv` — same + NLP text-analysis features (~130 cols)
- `data/clean/characters_keys.csv` — name/URL index for external joins

**Only required input:** `data/raw/*.jsonl` (38 Fandom scrape files, one per franchise)

---

## Data Sources

### 1. Fandom Wiki Scraper (raw input)
The raw JSONL files in `data/raw/` were scraped from [Fandom](https://www.fandom.com) wikis using the MediaWiki API. Each record contains the character's infobox fields, page text, and a pre-computed `is_dead` label derived from status keywords per franchise.

**38 franchises scraped:** Star Wars, Harry Potter, MCU, Game of Thrones, Indiana Jones, James Bond, Alien vs Predator, DC Extended Universe, Jurassic Park, Pirates of the Caribbean, Grey's Anatomy, Lord of the Rings, Lost, Breaking Bad / Better Call Saul, Dune, Avatar, Transformers, Sons of Anarchy, Supernatural, The Hunger Games, The Sopranos, The Twilight Saga, Spartacus, Dexter, The 100, Vikings, Westworld, Ozark, Fast & Furious, Mission: Impossible, Stranger Things, The Boys, The Walking Dead, Prison Break, Peaky Blinders, The Matrix, Boardwalk Empire, The Wire.

---

### 2. API Sources (fetched at runtime, no auth required)

| API | URL | What we use | Rate limit |
|-----|-----|-------------|------------|
| **An API of Ice and Fire** | `anapioficeandfire.com/api/characters` | TV seasons, book count, POV count, allegiances, title count for GoT characters | 0.3s/page |
| **Harry Potter API** | `hp-api.onrender.com/api/characters` | House, ancestry, Hogwarts student/staff flags | Single call |
| **SWAPI** | `swapi.py4e.com/api/people/` | Film count, height, mass, birth year for SW characters | 0.5s/page |
| **PotterDB** | `api.potterdb.com/v1/characters` | Blood status, house, species, nationality, animagus, patronus, titles, jobs for 5,387 HP characters | 0.5s/page, 54 pages |
| **SW Akabab** | `akabab.github.io/starwars-api/api/all.json` | Species, affiliation count, masters/apprentices/cybernetics for 87 SW characters | Single call |
| **TVmaze** | `api.tvmaze.com` | Episode counts, main vs guest cast, actor birthday → age at premiere for 21 TV franchises | 0.3s between episode calls (20 req/10s limit) |
| **Fandom MediaWiki API** | `{wiki}.fandom.com/api.php` | Category count, stub flag, featured/recurring flags for all 60k characters | 0.3s/batch of 50 pages |

**TVmaze endpoints used:**
- `GET /singlesearch/shows?q={name}` — find show ID and premiere date
- `GET /shows/{id}/cast` — main cast with actor birthdays
- `GET /shows/{id}/episodes` — all episode IDs
- `GET /episodes/{id}/guestcast` — guest characters per episode

**Fandom MediaWiki endpoint:**
```
GET https://{wiki}.fandom.com/api.php
  ?action=query
  &titles={title1|title2|...|title50}
  &prop=info|categories
  &cllimit=100
  &format=json
```

---

### 3. CMU Movie Summary Corpus
- **URL:** `http://www.cs.cmu.edu/~ark/personas/data/MovieSummaries.tar.gz`
- **Auto-downloaded** if not present in `data/external/cmu/`
- **What we use:** `character.metadata.tsv` — actor age at film release (proxy for character apparent age)
- **Files used:** `character.metadata.tsv` (450k character entries), `movie.metadata.tsv` (81k films)
- **Citation:** Bamman, O'Connor & Smith (ACL 2013)

---

### 4. Kaggle Datasets
Requires `KAGGLE_API_TOKEN` environment variable. Auto-downloaded to `data/external/kaggle/`.

| Kaggle Ref | Key File | Features | Leaky columns dropped |
|-----------|----------|----------|-----------------------|
| `mylesoneill/game-of-thrones` | `character-deaths.csv` | GoT nobility, book appearances (5 books) | Death Year, Book of Death, Death Chapter |
| `mylesoneill/game-of-thrones` | `character-predictions.csv` | GoT popularity, age, dead relations, married, noble | actual, pred, alive, plod, isAlive, DateoFdeath |
| `thedevastator/uncover-the-mystery-behind-got-characters-screen` | `GOT_screentimes.csv` | GoT screen time (minutes), episode count from IMDb | — |
| `claudiodavi/superhero-set` | `heroes_information.csv` | Superhero alignment (good/bad/neutral), race | — |
| `claudiodavi/superhero-set` | `super_hero_powers.csv` | 10 key powers (Immortality, Magic, Resurrection, The Force, etc.) | — |
| `paultimothymooney/lord-of-the-rings-data` | `lotr_characters.csv` | LOTR race (Elf/Man/Hobbit/Dwarf), realm, has\_spouse | birth, death |
| `paultimothymooney/lord-of-the-rings-data` | `WordsByCharacter.csv` | LOTR total words spoken per character | — |
| `mexwell/the-lord-of-the-rings` | `lotr_scripts.csv` | LOTR dialogue word count per character | — |
| `saguit03/the-hunger-games-characters` | `HungerGames_Characters_Dataset_ALL.csv` | HG district, tribute/winner/mentor flags, profession | — |
| `thedevastator/the-hunger-games-dataset-a-survival-analysis` | `Hunger Games survival analysis data set.csv` | HG tribute age, career, training rating | — |
| `bac3917/frank-herberts-dune-characters` | `duneCharacters_kaggle.csv` | Dune house allegiance, culture | Born, Died |
| `kyleakepanidtaworn/marvel-characters-dataset` | `marvel_characters_dataset.csv` | Marvel identity (public/secret), teams, origin, universe | Alive |

---

## Pipeline Architecture

```
data/raw/*.jsonl (38 files)
        │
        ▼
[Step 1] Parse JSONL → flat schema
         Deduplicate (URL, then franchise+name)
         Filter to labelled records (is_dead not None)
        │
        ▼
[Step 2] Structural features
         prominence_tier, archetype (from title/text)
         gender (infobox + pronoun inference)
         is_human, affiliation_alignment
         franchise metadata (release_year, decade, era, medium, genre)
         franchise aggregate stats (mortality_rate, size, female_ratio)
        │
        ├──[Step 3]── Franchise APIs (GoT, HP, SWAPI, PotterDB, SW Akabab)
        ├──[Step 4]── TVmaze (episode counts + actor age)
        ├──[Step 5]── CMU Movie Summary Corpus (actor age for film chars)
        ├──[Step 6]── Kaggle datasets (GoT, Superhero, LOTR, HG, Dune, Marvel)
        ├──[Step 7]── Fandom MediaWiki API (category counts, stub flag)
        └──[Step 8]── Species category (computed from species fields)
        │
        ▼
data/clean/characters_raw.csv   ← ~100 features, no NLP
        │
        ▼  (optional: python add_nlp_features.py)
[NLP]   Page text analysis: word densities, sentiment, centrality
        │
        ▼
data/clean/characters_full.csv  ← ~130 features
```

---

## Feature Reference

### Core (100% coverage)
| Feature | Type | Description |
|---------|------|-------------|
| `is_dead` | int (0/1) | **Target variable** — 1=dead, 0=alive |
| `franchise` | str | Franchise name (38 values) |
| `content_rating` | str | PG-13, TV-MA, R, TV-14 |
| `medium` | str | "Film" or "TV" |
| `genre` | str | Action, Fantasy, Sci-Fi, Crime, etc. |
| `franchise_release_year` | int | Year of first major release |
| `franchise_decade` | str | "1970s", "2000s", etc. |
| `franchise_era` | str | Pre-1980 / 1980s-90s / 2000s / 2010s+ |
| `franchise_mortality_rate` | float | Fraction of characters dead in franchise |
| `franchise_size` | int | Number of labelled characters in franchise |
| `franchise_female_ratio` | float | Fraction female in franchise |
| `infobox_field_count` | int | Number of infobox fields (prominence proxy) |
| `page_text_length` | int | Length of Fandom wiki page text |
| `prominence_tier` | str | Minor / Supporting / Major / Lead |
| `archetype` | str | Military / Royalty / Criminal / Force User / Law/Order / Medical / Academic / Political / Religious / Worker / Entertainer / Other / Unknown |
| `has_dob` | int (0/1) | Has a date-of-birth in infobox |
| `has_family` | int (0/1) | Has family member fields in infobox |
| `has_image` | int (0/1) | Has an image on the wiki page |
| `has_alias` | int (0/1) | Has known aliases |
| `has_pronouns` | int (0/1) | Has explicit pronoun field |
| `appearance_count` | int | Number of appearance-related infobox fields |
| `fandom_category_count` | int | Number of non-leaky wiki categories |
| `fandom_is_stub` | int (0/1) | Page is a stub (minor character proxy) |
| `fandom_is_featured` | int (0/1) | Featured or good article |
| `fandom_is_recurring` | int (0/1) | In a "recurring character" category |

### Partially populated
| Feature | Coverage | Description |
|---------|----------|-------------|
| `gender` | ~89% | Male / Female / Other/Unknown |
| `is_human` | ~85% | 1=human, 0=non-human |
| `affiliation_alignment` | ~60% | Good / Evil / Ambiguous / Neutral |
| `actor_age_at_release` | ~1% | Actor age at series/film premiere |
| `age_group` | ~1% | child / young_adult / adult / senior |
| `tvmaze_episode_count` | TV only | Number of episodes appeared in |
| `tvmaze_is_main_cast` | TV only | 1 if main cast member |

### Game of Thrones specific
`got_tv_seasons`, `got_book_count`, `got_pov_count`, `got_allegiances`, `got_titles_count`, `got_nobility`, `book1_appears`–`book5_appears`, `got_books_total`, `got_popularity`, `got_age`, `got_dead_relations`, `got_married`, `got_noble`, `got_screentime_min`, `got_episode_count_imdb`

### Harry Potter specific
`hp_house`, `hp_ancestry`, `hp_hogwarts_student`, `hp_hogwarts_staff`, `pdb_house`, `pdb_blood_status`, `pdb_species`, `pdb_nationality`, `pdb_animagus`, `pdb_patronus`, `pdb_titles_count`, `pdb_jobs_count`

### Star Wars specific
`sw_film_count`, `sw_height`, `sw_mass`, `sw_birth_year`, `sw_species_detail`, `sw_affiliations_n`, `sw_has_masters`, `sw_has_apprentices`, `sw_has_cybernetics`

### Other franchise features
`lotr_race`, `lotr_realm`, `lotr_has_spouse`, `lotr_total_words`, `lotr_dialogue_words`, `hg_district`, `hg_is_tribute`, `hg_winner`, `hg_is_mentor`, `hg_profession`, `hg_age`, `hg_career`, `hg_rating`, `dune_house`, `dune_culture`, `hero_alignment`, `hero_race`, `marvel_identity`, `marvel_marital`, `marvel_has_teams`, `marvel_origin`, `marvel_universe`, `power_immortality`, `power_magic`, `power_resurrection`, `power_the_force`, `power_telepathy`, `power_mind_control`, `power_invulnerability`, `power_regeneration`, `power_flight`, `power_super_strength`

### NLP features (characters_full.csv only)
`word_count`, `sentence_count`, `paragraph_count`, `section_count`, `avg_sentence_length`, `unique_word_ratio`, `violence/conflict/leadership/family/hero/villain/power/vulnerability _word_count + _density`, `dialogue_count`, `relationship_mentions`, `named_characters_mentioned`, `social_connectedness`, `moral_alignment`, `power_vulnerability_ratio`, `has_biography_section`, `has_dialogue`, `is_described_young`, `is_described_old`, `age_mentioned`, `sentiment_score`, `name_word_count`, `has_title_in_name`, `name_centrality`, `page_mention_rank`

---

## Leaky Columns Excluded

These columns encode death information and are explicitly excluded from all outputs:

| Column | Why it's leaky |
|--------|---------------|
| `has_dod` | 0.85 correlation with `is_dead` — having a date-of-death field means the character died |
| `dod` | Raw date-of-death string — directly encodes the label |
| `has_causeofdeath` | 0.18 correlation — cause-of-death fields only exist if the character died |
| `has_death_section` | Page section titled "Death" only exists for dead characters |
| `info_completeness` | Partially leaky — dead characters have death fields populated, boosting completeness |

From Kaggle GoT predictions: `actual`, `pred`, `alive`, `plod`, `isAlive`, `DateoFdeath` — all encode survival labels from another model/dataset.

From LOTR data: `birth`, `death` — raw in-universe dates.

From Marvel 92k: `Alive` — direct label.

---

## Known Limitations

1. **Star Wars dominance**: 42,271 / ~60k rows (63%) are Star Wars characters. Any global model will be heavily influenced by SW patterns. Use stratified sampling or franchise-specific submodels.

2. **Sparse franchise features**: GoT/HP/LOTR/HG/Dune features are only populated for their respective franchises (~1–3% of the full dataset). Tree-based models (XGBoost, LightGBM) handle NaN natively.

3. **Name matching**: External datasets are merged on lowercased character names. Spelling variants (e.g., "Daenerys" vs "Daenerys Targaryen") can cause mismatches.

4. **Actor age ≠ character age**: We use actor age at the time of production as a proxy for character apparent age. This is a rough approximation and is missing for ~99% of rows.

5. **Label source**: `is_dead` is derived from Fandom wiki status fields using keyword matching. Some characters may be mislabelled (e.g., resurrected characters, fictional deaths).
