# Feature dictionary

Every feature the models use, what it means, and exactly how it's computed.
Source is `build_pipeline.py` (structural features) and
`src/q1_character/features.py` (network and category features) unless noted.

A note on what's *not* here: anything naming death or survival is excluded
before modelling by `DEATH_TERM_RE` in `src/q1_character/models.py`, including
the `status` and date-of-death fields the label is derived from and any Fandom
category matching `deceas|dead|killed|...`. A category named "Alive" is the
label with the sign flipped, so survival words are excluded as well as death
words.

## Character attributes

| feature | type | how it's computed |
|---|---|---|
| `gender` | categorical | Infobox `gender` field, normalised to Male / Female / Other-Unknown. For the ~13% of pages with no gender field, inferred from pronoun counts in the first 500 words: at least two of "he/him/his" and a clear majority gives Male, same for "she/her" gives Female, otherwise Other/Unknown. |
| `archetype` | categorical | Keyword rule, because no wiki has a usable occupation column. First pass matches the infobox title/occupation field against per-role word lists (Military = soldier, sergeant, captain, admiral…; Royalty = king, queen, prince…; Criminal = thief, gangster…). If the title yields nothing, a second pass scans the first ~2,000 characters of page text with the same patterns and takes the role with the most hits. About 28% land in Unknown. |
| `affiliation_alignment` | categorical | Regex over the infobox `affiliation` field against two lexicons. Matching the good list only → Good (rebel, jedi, gryffindor, avengers, order of the phoenix, night's watch…); the evil list only → Evil (empire, sith, death eater, hydra, decepticon, white walker…); both → Ambiguous; a non-empty affiliation matching neither → Neutral; empty → missing. |
| `prominence_tier` | categorical | Fixed thresholds on `page_text_length`: <500 chars Minor, <2,000 Supporting, <10,000 Major, ≥10,000 Lead. Deliberately hardcoded rather than quantiles — a data-derived cutoff would carry information about other rows across a train/test boundary. |
| `is_human` | binary | 1 if the infobox `species` is in a human-synonym list, or if the franchise is human-only by construction (e.g. The Sopranos); 0 if species is present and not human; missing if species is absent. |
| `billing_order` | numeric | Cast position from TMDB for the matched credit — 0 is top billing. Missing means no TMDB match; the demo model fills those with 40 (bottom) rather than a median, since "uncredited" is what missing actually means. |
| `episode_count` | numeric | Number of episodes the matched TMDB credit appears in. |
| `page_text_length` | numeric | Characters of plain page text after stripping wiki markup. |
| `infobox_field_count_clean` | numeric | Number of filled infobox fields, **excluding** any field name matching `death\|died\|dod\|deceas\|killed\|cause\|fate\|posthum`. The raw count leaks: a character who dies acquires "Cause of death" and "Date of death" fields, and 55% of the raw count's separation between dead and alive was that. `status` is deliberately kept, since living and dead characters both have it. See `clean_infobox_field_count` in `src/constants.py`. |
| `appearance_count` | numeric | How many of a fixed set of appearance-ish infobox keys are filled: `tv series, movie, game, book, comic, first, last, appearances, seasons, episodes, films, series, appears in`. A rough proxy for how many works the character shows up in. |
| `has_dob`, `has_family`, `has_image`, `has_alias` | binary | Whether the infobox has a filled date-of-birth, any family field (children/siblings/parents/spouse/…), an image, or an alias field. These measure how completely fans documented the page, not anything about the character. |

## Franchise attributes

| feature | type | how it's computed |
|---|---|---|
| `franchise` | categorical | Which of the 38 wikis the page came from. |
| `genre`, `medium` | categorical | Hand-assigned per franchise (Film vs TV; Sci-Fi, Fantasy, Crime, …), cross-checked against TMDB. |
| `content_rating` | categorical | Per-franchise rating (PG-13, TV-MA, …), assigned in the scraper config. |
| `franchise_release_year` | numeric | Year the franchise's first installment came out, 1954 (Lord of the Rings) to 2019 (The Boys). |

Deliberately excluded as leaky: `franchise_mortality_rate`, `franchise_size`,
`franchise_female_ratio`. Each is computed from the outcome column across the
whole corpus, so any of them hands the model the answer for its own franchise.

## Network centrality

All computed per franchise on a **co-mention graph**, built from page text: an
edge runs from page A to page B when every name token of B appears in A's text.
So an edge means "A's page talks about B." This is the one feature family that
describes the story rather than the wiki's documentation of it, and it's the
family that transfers best to franchises the model hasn't seen.

| feature | type | how it's computed |
|---|---|---|
| `pagerank` | numeric | PageRank over the franchise's co-mention graph, damping α=0.85. Uniform 1/n if the franchise graph has no edges. |
| `pagerank_rank` | numeric | The same score turned into a within-franchise rank, scaled 0 (most central) to 1 (least). This is what the models mostly use, because raw PageRank isn't comparable between a 42,000-page wiki and a 14-page one. |
| `hub`, `authority` | numeric | HITS scores over the same graph (max 200 iterations, normalised). |
| `in_degree`, `out_degree` | numeric | How many pages mention this character, and how many it mentions. |
| `component_size` | numeric | Size of the weakly connected component the character sits in. |
| `n_relatives` | numeric | Count of comma/semicolon-separated entries across the infobox family fields. |
| `n_categories` | numeric | Number of Fandom category tags on the page, after death/survival categories are removed. |

## Wide category features (the `rich` set only)

369 one-hot columns: the most common Fandom **category tags** and the most
common **infobox field names** across the corpus, each as a 0/1 column. These
are the biggest single contributor on a random split and add much less when
whole franchises are held out — they encode franchise-specific role vocabulary
("Jedi", "Death Eater") that doesn't transfer to an unseen show.

## Reading these honestly

Four of these — `page_text_length`, `infobox_field_count_clean`,
`appearance_count`, and the `has_*` flags — measure how much fans wrote about a
character, not a property of the character. Death causes fan attention, so the
arrow points the wrong way. They're the strongest features in the model and
dropping them costs real accuracy, which is why the project's results are
framed as descriptive rather than predictive. The
`character, no wiki metadata` feature set exists to quantify that: it scores
0.591 on a random split against 0.642 with them, but *higher* held-out (0.506
vs 0.496), which is exactly what you'd expect from features that help within a
known franchise and hurt generalisation to a new one.
