# Spoiler Alert: Who Makes It to the Finale?

**A Needle in a Data Haystack (67978) — Final Project, Group #56**

| Name | Email | CS id |
|---|---|---|
| Stav Abukarat | stav.abukarat@mail.huji.ac.il | stava |
| Shahar Spencer | shahar.spencer@mail.huji.ac.il | shahar.spencer |
| Bar Dalal | bar.dalal@mail.huji.ac.il | bar.dalal |
| Ido Oron | ido.oron@mail.huji.ac.il | ido_oron |

Interactive demo: **https://shaharspencer.github.io/needle-group-project/**

---

## Domain

We study whether a fictional character dies on-screen, based on their attributes (prominence, role, gender), the franchise they appear in (genre, medium, rating), and how they're described on fan wikis. Our unit of observation is one character in one franchise; the label is binary (dies or survives across the full story as aired). The data comes from Fandom wikis (character descriptions, infobox fields) and TMDB (cast credits, billing order, production metadata).

---

## Code Structure

| folder | purpose |
|---|---|
| `scraper/` | Wiki configuration per franchise (`wiki_configs.py`): URLs, infobox templates, field mappings, status keywords. |
| `build_pipeline.py` | Main data-collection script: scrapes wikis, parses infoboxes, joins TMDB credits, writes raw + clean CSVs. |
| `src/labels/` | Label construction: `rebuild_label.py` builds `is_dead_v2` from infobox/registry; `fill_unlabelled.py` fills gaps via Haiku for Q3 only; `validate_registry.py` checks against external death registries; `eval_labels.py` scores the on-screen filter. |
| `src/constants.py` | Shared filter rules (non-character detection, unnamed-role regex, analysis population definition). |
| `src/q1_character/` | Q1 models: `features.py` builds the feature matrix; `models.py` runs 3 models × 9 feature sets under CV; `splits.py` assigns characters to train/test; `evaluate.py` scores the winner on held-out test; `effects.py` estimates the gender odds ratio. |
| `src/q2_text/` | Q2 text analysis: `coefficients.py` extracts and displays the strongest TF-IDF terms. |
| `src/q3_franchise/` | Q3 franchise analysis: `mortality.py` computes franchise mortality rates, runs the WLS regression and k-means clustering. |
| `src/enrich/` | Feature enrichment: TMDB join, network graph construction, PageRank/HITS computation. |
| `src/viz/` | All figure generation (`figures.py`). |
| `src/app/` | `export_demo.py` fits a reduced logistic regression and exports JSON for the browser demo. |
| `docs/` | GitHub Pages demo (HTML/JS/CSS). |
| `tests/` | pytest suite covering label logic, filter consistency, and pipeline smoke tests. |
| `legacy/` | Earlier exploration notebooks (not used in final pipeline). |
| `FEATURES.md` | Exact definitions for every model feature. |
| `PIPELINE.md` | Stage-by-stage run order with commands. |

---

## Data

### Sources

| source | provides | amount |
|---|---|---|
| Fandom wikis (38 wikis: GoT, MCU, Star Wars, HP, Breaking Bad, +33 more) | infobox fields, page text, category tags | 68,463 pages, ~262 MB raw |
| TMDB API (280 titles) | cast credits, billing order, episode counts, genre, rating | 28,151 cast credits |
| Kaggle death registries | ground-truth death per episode | 2 of 38 shows |
| listofdeaths.fandom.com | episode-by-episode death catalogue | external validation only |

Per-wiki scraping configuration is in `scraper/wiki_configs.py`; the collection logic is in `build_pipeline.py`. No single source answers the question alone — wikis lack billing data, TMDB lacks death outcomes, registries cover 2/38 shows — so the dataset is a join across all four, following the course's data-integration principle. Final modelling table: 68,463 rows × 129 columns.

### Labels

The primary label (`is_dead_v2`) is built from wiki infobox fields in `src/labels/rebuild_label.py`: 30 wikis supply a status string ("Deceased"/"Alive"); 8 wikis use a date-of-death field. Characters with unparseable status are excluded from the Q1/Q2 modelling population entirely.

For Q3 only, 1,858 characters with unparseable status (concentrated in The Wire, Prison Break, Boardwalk Empire, Lost) were labelled by Claude Haiku 4.5 (`src/labels/fill_unlabelled.py`), with "can't tell" allowed: 123 dead / 1,223 alive / 512 undecided. These labels are never used in Q1/Q2 — no model trains or scores against them. They narrow the franchise-mortality estimate in Q3 from a wide range to a point estimate for 4 franchises, reported alongside bounds rather than replacing them. Keeping noisier labels out of any trained model follows the evaluations lecture's point about label quality: a weaker source should not contaminate the evaluation of a cleaner one.

### Data quality

**Not every page is a character.** We annotated a 600-page stratified sample and found the corpus is 41% on-screen characters, 46% novel/reference-only, 12% non-characters (creatures, vehicles, real people). Our inclusion rule (`is_onscreen = has_actor OR tmdb_match`, implemented in `src/constants.py`) scores precision 0.79 / recall 0.77 against this sample — the OR of two weak signals deliberately trades precision for recall, following the course's precision-recall trade-off framework.

- **Figure 1** (`corpus_composition.png`) — what the scraped corpus actually contains. Three in five scraped pages are not on-screen characters; all analysis is restricted to the leftmost segment.

We also removed 168 object pages (portraits/paintings from HP/Star Wars) that are dead by definition and 1,122 unnamed background roles that match our `UNNAMED_RE` regex in `src/constants.py`. Neither filter uses the label or cross-row statistics, so they are safe to apply before any train/test split — a filter built from label statistics would leak, as covered in the supervised learning lecture.

**Final population: 15,686 named, on-screen characters, 35.6% die.** Star Wars is 56% of the raw scrape but only 12% after filtering, which is why we use grouped CV throughout Q1/Q2.

### External validation

We matched our labels against `listofdeaths.fandom.com` (different wiki, different editors, organized by episode) for GoT and The 100 (`src/labels/validate_registry.py`):

| franchise | matched | agree |
|---|---|---|
| Game of Thrones | 134 | 132 (98.5%) |
| The 100 | 68 | 65 (95.6%) |

197/202 agree (97.5%). This is recall against the registry, not chance-corrected agreement: the registry lists only deaths, so the matched sample has no negatives. As the evaluations lecture showed, percent agreement overstates reliability under uneven category assignment — exactly this case. Two of the 5 disagreements look like registry errors (Jaqen H'ghar changes face; Ellaria Sand is imprisoned).

---

## Q1: Is it who you are, or what show you're in?

### Problem

Can character attributes predict death? Does franchise identity add signal beyond the character, or substitute for it?

### Solution

We trained logistic regression (C=1.0), decision tree (no depth limit, to show overfitting), and random forest (300 trees, max depth 12) on 9 feature sets (`src/q1_character/models.py`): franchise only; character only; character without wiki-metadata proxies; character + franchise; franchise + network; character + network; TF-IDF text only; character + text; everything (including 369 one-hot Fandom category columns). Feature definitions are in `FEATURES.md`; the feature matrix is built by `src/q1_character/features.py`.

The model lineup follows the supervised learning lecture: logistic regression as a linear floor, a single decision tree for interpretable splits, and a random forest to guard against the overfitting a single deep tree is prone to.

**Network features:** we built a per-franchise co-mention graph from wiki page text — two characters share an edge if one's name appears on the other's page, weighted by mention count (`src/enrich/`). From this graph we extracted PageRank, HITS hub/authority, and in/out degree per character. These describe structural position in the story, following the course's graph centrality material.

**Text leakage control:** a dead character's page says "was killed by" — raw TF-IDF rediscovers the label. We strip every sentence mentioning death or survival, remove surviving death tokens, and verify the resulting 300-term vocabulary contains none (`src/q2_text/coefficients.py` asserts this in code). The vectoriser is fit inside the CV pipeline so no test vocabulary leaks into training — the same leakage-prevention principle from the supervised learning lecture, harder to spot when it hides in near-synonyms of the label.

### Evaluation

**Metric:** PR-AUC (primary), ROC-AUC (secondary). At a 35.6% base rate, a majority-class classifier scores 64.4% "accuracy" while being useless — the same class-imbalance problem from the evaluations lecture. PR-AUC's chance level equals the base rate, making it harder to game. No human subjects were used in this project.

**Setup — two experiments:**

*Experiment A (feature-set comparison):* 5-fold CV over all 15,686 characters, under two grouping rules. Random CV lets franchise characters appear in both train and validation folds, so the model can memorize per-show base rates. Grouped CV (`GroupKFold` on franchise) puts all of a franchise's characters into exactly one fold — measuring transfer to a franchise the model has never seen. The course's cross-validation lecture treats what you group by as the substantive modelling decision, not a technical detail; we run both to see what franchise memorization is worth.

*Experiment B (held-out test):* The winning config from Experiment A is re-tuned by 5-fold CV inside an 80% trainval split, then scored once on the untouched 20% test split (`src/q1_character/evaluate.py`). Split assignment is in `src/q1_character/splits.py`. For the grouped rule, franchises are assigned largest-first to whichever side is furthest below its quota, keeping the realized split close to 80/20 despite franchise sizes spanning 11 to 3,489.

### Results

**Experiment A — CV comparison:**

| features | random CV | grouped CV |
|---|---|---|
| everything | **0.743** | **0.545** |
| character + text | 0.699 | **0.545** |
| character + franchise | 0.692 | 0.485 |
| character + network | 0.680 | 0.519 |
| franchise + network | 0.650 | 0.486 |
| text only | 0.650 | 0.509 |
| character only | 0.633 | 0.487 |
| character, no wiki metadata | 0.590 | 0.504 |
| franchise only | 0.556 | 0.398 |
| baseline | 0.356 | 0.356 |

Character attributes beat franchise identity (0.633 vs. 0.556); combined beats either. The full model drops 0.198 from random → grouped CV — much of the random-CV signal is per-show memorization. Network centrality transfers well (+0.094 random, +0.088 grouped added to franchise). Fandom category features show the opposite: large random-CV gain, weak grouped-CV gain — franchise-specific vocabulary ("Jedi", "Death Eater") that means nothing elsewhere.

**Experiment B — held-out test:**

| split rule | test PR-AUC | test ROC-AUC |
|---|---|---|
| random | **0.752** | 0.850 |
| grouped | **0.664** | 0.776 |

Test exceeded CV under both rules — the selection procedure didn't overfit. The grouped gap (0.562 CV → 0.664 test) shouldn't be over-read: only 38 franchises, result depends on the draw.

**Feature importance** (permutation, character-only forest): page length 0.167, archetype 0.111, gender 0.107, de-leaked field count 0.101, prominence tier 0.101, billing order 0.092. The two strongest features are wiki-page measurements, not character attributes — this is the project's central caveat. We used permutation importance rather than raw tree-split importance because it measures actual predictive contribution, following the supervised learning lecture.

**Gender effect** (`src/q1_character/effects.py`): Female → roughly half the odds of dying: OR 0.564, 95% CI 0.516–0.616 (≈11.1pp lower death probability), adjusted for prominence, role, alignment, species, and franchise fixed effects. Franchise fixed effects are included so the gender effect isn't just "franchises with more female leads happen to be less deadly." The unadjusted gap is 14.5pp — less than a quarter of the raw difference is explained by men holding more disposable roles. Per-franchise: 22 estimable, 11 reach p<0.05, none reverse direction.

### Visualization

- **Figure 2** (`q1_model_comparison.png`) — PR-AUC by feature set, one panel per split rule. Bars that barely shrink left-to-right (network, text) carry transferable signal; bars that collapse (franchise, categories) were memorizing.
- **Figure 3** (`q1_feature_importance.png`) — permutation importance, colour-coded by character attribute vs. wiki-page measurement. The colour split is the takeaway: our two tallest bars are page measurements.
- **Figure 4** (`q1_gender_forest.png`) — per-franchise gender odds ratios with 95% CIs. Every marker sits left of 1; interval width proxies sample size.

### Impediments

We ranked out-of-fold trainval errors and found two franchises overrepresented — failing in opposite directions. **Spartacus** (14% of worst-100 errors, only 1.4% of data): mostly false positives. Base mortality 71%; centrality doesn't help — Agron and Nasir are both central, one dies, one doesn't. **Grey's Anatomy** (2.5× overrepresented): all false negatives. Base mortality 9.3%. These two need the threshold moved in opposite directions — no single global threshold fixes both, a direct application of confusion-matrix diagnosis from the evaluations lecture. Left uncorrected: franchise-specific patches would hide a real limit.

---

## Q2: Does the text predict death?

### Problem

Does unstructured wiki text carry signal beyond structured attributes?

### Solution

TF-IDF (300-term vocabulary cap) over each character's wiki description, same death/survival-stripping as Q1. Vectoriser fit inside each CV fold, never on the full corpus. `src/q2_text/coefficients.py` explicitly asserts no death/survival token survived stripping — the check is code, not a claim.

### Evaluation

Same metric (PR-AUC), baselines (majority-class, prior-probability), and ground truth (`is_dead_v2`) as Q1. Same CV folds and franchise grouping, so Q1 and Q2 results are directly comparable row-for-row. Success criterion: text improves PR-AUC over character-only under **both** split rules — grouped improvement specifically tests transfer to an unseen franchise.

### Results

`character + text` (0.545 grouped CV) ties the everything-model (369 more columns) as the best grouped-CV configuration. Text alone (0.650) beats character attributes alone (0.633). Adding text to character: +0.066 (random), +0.058 (grouped). Text is the only feature family that transfers to an unseen franchise as well as to a familiar one.

Under grouped CV, **logistic regression beats random forest** on character+text (0.545 vs. 0.533), despite losing to it under random CV (0.645 vs. 0.699). The forest drops 0.166 going from random to grouped; logistic drops only 0.100. The forest can memorize franchise-specific term combinations (e.g., "hydra" + "command" together means MCU villain), while logistic regression treats each term independently and can't overfit to those combos — which turns out to be an advantage when the test franchise is entirely new. This is a concrete instance of the capacity/overfitting trade-off from the supervised learning lecture: the more expressive model wins when it can rely on franchise identity, and loses when it can't.

Strongest coefficient: `antagonist` (+3.81). Other portable terms: `command` (+1.74), `background` (+1.43), `recurring` (+1.27), `military` (+0.82), `criminal` (+0.81). On the survival side, terms are less portable: `relationships` (−2.77), `category` (−2.69), `characters` (−2.66) are wiki section headings, while `hospital` (−2.33, Grey's Anatomy), `awakening` (−2.44, Star Wars), `debra` (−1.92, Dexter) are franchise fingerprints. Of the top 40 terms (20 each direction), roughly a third are portable narrative terms, a third are franchise fingerprints, and the rest are wiki structural. The death side is more portable than the survival side — which is why adding text helps transfer but doesn't help as much as its random-CV score would suggest.

### Visualization

- **Figure 5** (`q2_text_terms.png`) — strongest TF-IDF coefficients in each direction. Portable narrative terms (antagonist, command, background) cluster on the death side; franchise fingerprints and wiki headings dominate the survival side.
- **Figure 6** (`q2_text_transfer.png`) — how much each model loses going from random to grouped CV on text features. The forest's taller faded bar shows it relies more on franchise-specific patterns.

### Impediments

Answers the question for descriptive wiki text, not dialogue. 1,136 transcripts across 11 shows exist, but episode-level death registry covers only 2 and most transcripts lack speaker labels — attribution is a prerequisite problem in itself.

---

## Q3: Are some franchises deadlier?

### Problem

Do mortality rates differ by franchise, and do franchise properties (genre, medium, rating, run length) explain the difference?

### Solution

Code: `src/q3_franchise/mortality.py`. First we tried a binomial GLM treating every character as an independent trial. It returned p<1e-10 on every term — but a franchise's 3,647 MCU characters are not independent tests. The model's own Pearson χ²/df = 36.6 (should be ≈1) confirms severe pseudo-replication, exactly the problem the supervised learning lecture warns about.

**Fix:** collapse to one row per franchise (38 rows, not 17,584), regress `logit(mortality)` on franchise properties (release year, run span, title count, medium, rating), weighted by cast size. Also ran k-means (k chosen by silhouette score, checked across 10 seeds) over standardized franchise property vectors — standardized because raw Euclidean distance is meaningless when cast size runs to thousands and rating sits between 6–9. Treated as exploratory per the clustering lecture's framing that 38 points in 8 dimensions admit several defensible groupings.

### Evaluation

Population: 17,584 on-screen named characters — wider than Q1/Q2's 15,686, because excluding unlabelled characters from a *rate* would bias it. For the 4 mostly-unlabelled franchises, we report lower bound (unparsed → alive), upper bound (unparsed → dead), and a point estimate using Haiku-resolved labels where available. Every mortality estimate carries a Wilson interval alongside the label bounds.

### Results

Mortality varies widely: Grey's Anatomy 9.3% to The 100 72.7%. But **no franchise property is significant.** Run span closest at p=0.185; R² = 0.079, adjusted R² = −0.065 — worse than predicting the mean. k-means: best k=2 (silhouette 0.43, stable across 10 seeds) splits 6 large multi-title film franchises (MCU, Star Wars, HP, LOTR, Fast & Furious, Transformers) from the other 32, at 35.4% vs. 37.1% mortality — not a meaningful gap.

For the 4 mostly-unlabelled franchises:

| franchise | bounds | point estimate |
|---|---|---|
| The Wire | 14.3–99.2% | 16.3% |
| Prison Break | 6.6–91.3% | 21.3% |
| Boardwalk Empire | 19.7–99.8% | 23.2% |
| Lost | 23.2–69.0% | 26.6% |

**Measurement check that changed the answer:** run span defined on release years alone reaches p=0.022 — but that just separates multi-film franchises from TV shows, not duration. Defined correctly (first-to-last air date for TV): both p and R² vanish. A direct example of the supervised learning lecture's point that a significant term can be measuring the wrong construct.

### Visualization

- **Figure 6** (`q3_franchise_mortality.png`) — mortality per franchise with Wilson CI and label-bounds band layered on each bar. The four wide-banded franchises are the low-confidence ones.
- **Figure 7** (`q3_run_span.png`) — mortality vs. run span, marker size ∝ cast size. The flat fit line visually confirms the null result.

---

## Interactive Demo

**"Plot Armor Index"** — https://shaharspencer.github.io/needle-group-project/

A browser-only logistic regression over user-selectable attributes (gender, role, alignment, franchise, billing order, centrality, relatives, species), exported by `src/app/export_demo.py`. It uses fewer features than the full Q1 model and says so on the page.

Two tabs: **Look up a character** lets you search held-out test characters, see a per-feature contribution breakdown ("why did the model score this character this way?") and the 5 nearest-risk neighbours. **Build a character** lets you set every feature by hand and watch the predicted death risk update live — a what-if tool for exploring what the model learned. Every displayed character is from the held-out test set, never a training row, so "got it right / wrong" reflects real generalization.

---

## Future Work

- **Community detection** on the co-mention graph — only centrality is extracted currently; community structure would test whether characters in the same narrative cluster share fates (following the community detection lecture's Louvain/modularity framework).
- **Significance tests** on the Q1 feature-set comparison — PR-AUCs currently ranked by point estimate; a cross-fold test would settle whether e.g. 0.699 vs. 0.692 is real.
- **Cost-sensitive loss** — Spartacus and Grey's Anatomy fail in opposite directions; a weighted loss would make the relative cost explicit.
- **Inter-rater reliability** — the 600-page sample was labelled by a single coder (Haiku); `src/labels/eval_labels.py` already computes Cohen's kappa if a second pass exists. Following the evaluations lecture, kappa is appropriate here given the uneven category split (41% on-screen / 12% non-character).
- **Prospective features** — rebuild features from what was knowable partway through a series and predict forward, rather than from post-hoc wiki text.

---

## Conclusion

Character attributes carry more signal than franchise identity: 0.633 PR-AUC (character alone) vs. 0.556 (franchise alone) vs. 0.743 (everything), on a 0.356 base rate. Held-out test: 0.752 PR-AUC / 0.850 ROC-AUC, falling to 0.664 / 0.776 with whole franchises held out. Text is the single most useful addition and the only feature family that transfers as well to an unseen franchise. Women are roughly half as likely to die (adjusted, 11/22 franchises significant, no reversals). Franchise properties explain none of the variation in franchise-level mortality.

**Main limitation:** wikis are written after the story ends, by fans who write more about characters they cared about, and our two strongest features (page length, infobox field count) measure fan attention, which death itself attracts. The two families built from the story's own structure — network centrality, narrative-role text terms — are also the two that survive the move to an unseen franchise.
