# Spoiler Alert: Who Makes It to the Finale?

**A Needle in a Data Haystack (67978) — Final Project, Group #56**

| Name | Email | CS id |
|---|---|---|
| Stav Abukarat | stav.abukarat@mail.huji.ac.il | stava |
| Shahar Spencer | shahar.spencer@mail.huji.ac.il | shahar.spencer |
| Bar Dalal | bar.dalal@mail.huji.ac.il | bar.dalal |
| Ido Oron | ido.oron@mail.huji.ac.il | ido_oron |

Interactive demo: **https://shaharspencer.github.io/needle-group-project/**

Code: see `README.md` for the full run order.

---

## Domain

- **Object of study:** one character in one franchise.
- **Outcome (label):** binary — does this character die on screen over the course of the story as aired.
- **Hypothesis space:**
  - Character-level attributes (prominence, role, gender) predict death.
  - Franchise identity (what show, what genre) predicts death independently of the character.
- **Data origin:** fan wikis (character description) + TMDB (cast/production metadata).

---

## Data

### Sources

| source | URL / access | provides | amount |
|---|---|---|---|
| Fandom wikis | MediaWiki API, 38 wikis (GoT, MCU, Star Wars, Harry Potter, Breaking Bad, +33 more) | infobox fields, full page text, category tags | 68,463 pages, ~262 MB raw |
| TMDB | TMDB API, 280 titles | cast credits, billing order, episode counts, genre, audience rating | 28,151 cast credits |
| Kaggle death registries | episode-level death records | ground-truth death per episode | 2 of 38 shows only |
| listofdeaths.fandom.com | scrape, GoT + The 100 | episode-by-episode death catalogue | external validation only, not a feature source |

- Final modelling table: 68,463 rows × 129 columns.
- **[Course: data integration]** — no single source answers the question alone (wikis lack billing data, TMDB lacks death outcomes, registries cover 2/38 shows), so the dataset is a join across all four.

### Labels

- Primary label from the wiki infobox:
  - 30 wikis → `status` string (`"Deceased"` / `"Alive"`)
  - 8 wikis → date-of-death field (filled → dead)
  - Collapsed into binary `is_dead_v2`
- `is_dead_v2` used for Q1/Q2 modelling is built **only from characters with a parseable status** — rows where the infobox status couldn't be parsed are dropped upstream (`build_pipeline.py`) before `is_dead_v2` is ever computed, so none of the 15,686-character modelling population carries a defaulted label. The "unparseable → alive" convention is a separate, Q3-only choice (see Unresolved rows, below and Q3 → Setup): the franchise-mortality denominator there intentionally keeps these characters and counts them as alive unless resolved by a registry or Haiku.
- 1,858 of those dropped rows (spread across ~20 franchises, concentrated in 4: The Wire, Prison Break, Boardwalk Empire, Lost) were labelled by **Claude Haiku 4.5**, "can't tell" allowed explicitly:
  - 123 dead / 1,223 alive / 512 undecided (27.6% undecidable)
  - stored in their own file (`data/gold/fill_unlabelled/haiku_fill.csv`) as a separate column, never merged into `is_dead`/`is_dead_v2`
- **What they are used for, and why:**
  - **Not used at all** in Q1/Q2: `is_dead`/`is_dead_v2` (the columns those models train and score against) are built directly from `characters_raw`/`onscreen_flags`, which never reference the Haiku file — no model is ever trained or evaluated on an LLM-produced label.
  - **Used only in Q3**, to turn a wide bounds range into a single point estimate for the 4 franchises where the infobox status mostly failed to parse (The Wire, Prison Break, Boardwalk Empire, Lost — see below). Without them, those 4 franchises would carry only a lower bound (unparsed = alive) and upper bound (unparsed = dead) spanning up to 85 points, which is not informative enough to compare against other franchises' mortality. The Haiku-resolved rows (`mortality_haiku_filled` in `src/q3_franchise/mortality.py`) narrow that range to one number per franchise, reported alongside the bounds rather than replacing them.
- **[Course: label quality / weak supervision]** — using an LLM only to fill labels a human/registry source cannot reach, and keeping those labels out of any trained/scored model, follows the principle that a noisier label source should never contaminate the evaluation of a cleaner one; it is used only to narrow a descriptive statistic, never as ground truth for a classifier.

### Data-quality issues and fixes

- **Issue: not every page is a character.**
  - Fix: 600-page stratified sample, labelled with Claude Haiku 4.5 against a written rubric (individual character? on-screen? dies?)
  - Reweighted to full corpus: 41% on-screen characters / 46% novel-or-reference-only / 12% not characters at all (creatures, vehicles, real people from the Young Indiana Jones wiki)
  - Inclusion rule: `is_onscreen = has_actor OR tmdb_match`
  - Scored against the annotated sample: **precision 0.79, recall 0.77** (actor field alone: recall 0.61; TMDB alone: recall 0.58)
  - **[Course: classifier evaluation / precision-recall trade-off]** — the OR of two weak signals is scored like any other classifier against a gold sample; it trades precision for recall deliberately (we'd rather admit a few non-characters than lose a fifth of the real ones)
  - Note: this is a rule-based filter scored against an LLM-annotated gold sample — not inter-annotator agreement between two human coders

- **Issue: 168 object pages** (portraits/paintings, Harry Potter + Star Wars wikis).
  - Fix: removed — dead by definition, a death classifier would learn nothing from them
  - Impact: Harry Potter mortality 23.0% → 16.7% once removed

- **Issue: 1,122 unnamed background roles** (e.g. "Unidentified Ugnaught worker").
  - Fix: removed — on-screen per TMDB, but not "characters" per our own annotation rubric
  - Impact: 6.7% of on-screen set, 22.7% mortality vs. 35.6% for named characters — left in, they drag the headline rate down

- **Final modelling population: 15,686 named, on-screen characters, 35.6% die.**
  - Both filters use only a row's own fields (an infobox tag, a name pattern) — never the label, never a cross-row statistic
  - **[Course: data leakage]** — because the filters never look at the label or at other rows, they are safe to apply before any train/test split exists; a filter built from label statistics would leak
  - Sensitivity check: `INCLUDE_UNNAMED_ROLES=1` reproduces the pipeline with unnamed roles kept in — numbers move slightly, no conclusion changes

- **Issue: coverage imbalance across franchises.**
  - Star Wars = 56% of raw scrape, still 12% after filtering → **[Course: grouped cross-validation]** used throughout Q1/Q2 specifically to prevent this one franchise from dominating both train and test
  - 5 franchises (Walking Dead, Stranger Things, Vikings, Spartacus, Westworld) undercounted (infobox template mismatch) → excluded from franchise-level analysis below a 30-character threshold

### Unresolved rows (10.8% of the wider Q3 population)

- 1,898 on-screen, named characters have unparseable status across the corpus — excluded entirely from the 15,686-character Q1/Q2 modelling population (see Labels, above), but kept in Q3's wider 17,584-character population, where they count toward the mortality denominator
- Unevenly distributed: 4 franchises (The Wire, Prison Break, Boardwalk Empire, Lost — 46%–85% unlabelled each) account for 1,139 of the 1,898; the remaining ~760 are small counts spread across roughly 20 other franchises (e.g. 158 in MCU, 132 in James Bond 007) with mostly-parseable status
- Label source priority: human-curated death registry → wiki infobox → Claude Haiku 4.5 (only for what neither settles)
- No registry covers the 4 concentrated franchises → their unresolved rows (1,139) are among the 1,858 sent to Haiku in batches of 100 (outcome under Labels, above)
- These 4 franchises are reported as a **range**, not a point estimate, in Q3

### External validation of the label

- `listofdeaths.fandom.com` — different wiki, different editors, organized by episode not by character → independent of the infobox fields the label reads
- Matched by name, GoT + The 100:

| franchise | matched 1:1 | registry says died, we say dead |
|---|---|---|
| Game of Thrones | 134 | 132 (98.5%) |
| The 100 | 68 | 65 (95.6%) |

- **197/202 agree (97.5%)**
- **[Course: agreement statistics]** — read as **recall** against the registry, not chance-corrected agreement: the registry lists only deaths, so the matched sample contains no negatives, and the lectures' warning that percent agreement overstates reliability under uneven category assignment applies directly here
- 2 of 5 disagreements look like registry errors (Jaqen H'ghar changes face, doesn't die; Ellaria Sand is imprisoned, not killed)
- Fandom's `Deceased` category is **not** an independent check: of 7,128 pages with parseable status, 96.6% already say dead via infobox too — same editors, same fact, two mechanisms. Shows internal consistency, not correctness.

---

## Question 1: Is it who you are, or what show you're in?

### Problem description

- Can a character's own attributes predict whether they die?
- Does franchise identity add predictive power beyond the character, or substitute for it?

### Solution

- **Models:** logistic regression, decision tree, random forest — 3 models × 9 feature sets.
  - **[Course: supervised learning]** — linear model as the floor, single tree for interpretable splits, forest against the lecture's warning that a single deep tree overfits.
- **Feature sets:** franchise metadata only; character attributes only; character without wiki-metadata proxies; character + franchise; franchise + network centrality; character + network; TF-IDF text only; character + text; everything (+369 one-hot Fandom category/infobox columns).
- **Character features:** gender, archetype (keyword rule over infobox title/occupation), alignment, prominence tier, billing order, episode count, page length, de-leaked infobox field count, has-this-field flags (`FEATURES.md` has exact definitions).
- **Network features:** per-franchise co-mention graph from page text — PageRank, HITS hub/authority, in/out degree, component size.
  - **[Course: graph centrality measures]** — PageRank/HITS chosen because they describe a character's structural position, independent of how much text was written about them.
- **Text leakage control:**
  - Problem: a dead character's page says "was killed by" — raw TF-IDF rediscovers the label.
  - Fix: strip every sentence mentioning death *or* survival ("still alive" is the label with the sign flipped), then remove any surviving death token.
  - Verified: resulting 300-term vocabulary contains no death/survival token.
  - Vectoriser fit **inside** the CV pipeline — vocabulary/IDF never see a test document.
  - **[Course: data leakage]** — same leakage-prevention principle as any feature pipeline, harder to spot here because it hides in near-synonyms of the label ("was killed", "still alive") rather than the label column itself.

### Evaluation

#### Evaluation Criteria

- **Metric:** PR-AUC (primary), ROC-AUC (secondary).
  - **[Course: evaluation under class imbalance]** — chosen before modelling: at 35.6% positive rate, a majority-class classifier is 64.4% "accurate" while useless, so accuracy is rejected up front; PR-AUC's chance level is the base rate rather than 0.5. Reporting two justified metrics (not one cherry-picked) follows the lecture's advice to justify a small number of measures.
- **Baselines:** majority-class classifier, prior-probability classifier.
- **Ground truth:** `is_dead_v2` from the wiki infobox / death registries (see Data → Labels) — never the Haiku-filled rows.
- **Success criterion:** feature set beats both baselines under both grouping rules in Experiment A (the CV comparison); the grouped-CV score specifically is what's used to judge "character vs. show," since it's the one that can't be won by memorizing franchise identity.
- **Guarding against spurious findings:**
  - All preprocessing (imputation, scaling, one-hot encoding) fit inside the CV pipeline, never across a fold boundary.
  - Grouped CV isolates franchise-memorization from real signal — collapse under grouped CV flags a feature as franchise-vocabulary rather than character signal.
  - Held-out test set (never touched by selection) checks whether the CV ranking itself is overfit.

#### Setup

There are **two separate experiments** below, both run under the same two grouping rules (random / grouped), but used for different purposes:

- **Experiment A — feature-set comparison (produces the 9-row ranking table in Results).** All 15,686 characters are used, split into 5 cross-validation folds, scored, then put back and re-split for the next feature set. Every feature set is scored this way, twice — once under each grouping rule:
  - *Random CV:* the 5 folds are stratified only on the label. A given franchise's characters can land in both the training folds and the validation fold in the same run → the model is allowed to learn that franchise's base rate, so this measures "how well can attributes + franchise memorization predict death."
  - *Grouped CV* (`GroupKFold` on franchise): the 5 folds are built so that **all of a franchise's characters go into exactly one fold** — a franchise is either entirely in training or entirely in validation, never split across the two. This measures "how well do character attributes predict death in a franchise the model has never seen a single row of."
  - **[Course: cross-validation design]** — same CV mechanism, two different grouping choices, because the lectures treat "what you group by" as the substantive modelling decision, not a technical detail.
- **Experiment B — held-out test estimate (produces the separate 2-row table further down in Results).** This does not reuse Experiment A's folds at all. It is one single, one-time train/test split, done independently for each grouping rule:
  - *Random rule:* one 80/20 split of the 15,686 characters, stratified on the label.
  - *Grouped rule:* one 80/20 split by whole franchise (see the box below for how franchises are assigned to each side).
  - In both cases: the winning feature-set + model from Experiment A ("everything" / random forest) is re-tuned by 5-fold CV **only inside the 80% trainval portion**, then scored exactly once on the untouched 20% test portion.
  - **[Course: overfitting / train-test discipline]** — Experiment A reports the *best* configuration per row by construction, which is an optimistic number; Experiment B's untouched 20% is the only way to check whether that optimism actually generalizes, per the lecture definition of overfitting as loss much worse off training data than on it.

> **Why franchises aren't just randomly assigned to the grouped 80/20 split:** franchise sizes range from 11 characters (Peaky Blinders) to 3,489 (MCU) — three orders of magnitude apart. If franchises were assigned to train/test by a coin flip, a couple of unlucky flips on the largest franchises could push the actual split to something like 95/5 instead of 80/20, since a few giant franchises dominate the character count. The fix: sort franchises largest-to-smallest, and assign each one, in turn, to whichever side (train or test) is currently furthest below its 80% or 20% target share of characters. This keeps the realized split close to 80/20 despite the extreme size differences, without ever splitting one franchise's characters across both sides.

- **No human subjects** were recruited for this project (no surveys, no user studies) — the spec's question about recruiting/bias-avoidance for human subjects doesn't apply here. The closest analogue, since there's no human sample to bias, is the known imbalance in *which franchises* are represented in the data (Star Wars is 56% of the raw scrape; five wikis are undercounted due to template mismatches) — that's a data bias, addressed by using grouped CV/grouped test splits (this section) and covered in full under Data and under Impediments below.

#### Results

**Experiment A — feature-set comparison (CV, all 15,686 characters, no data held out):**

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

- Character attributes beat franchise identity (0.633 vs. 0.556); combined beats either alone.
- Full model loses 0.198 PR-AUC moving from random → grouped CV (0.743 → 0.545) — much of the random-CV signal is per-show base-rate memorization, not character-level pattern.
- Network centrality transfers well: +0.094 PR-AUC (random CV) / +0.088 (grouped CV) added to franchise metadata.
- Fandom category features do the opposite: large random-CV gain, weak grouped-CV gain — franchise-specific vocabulary ("Jedi", "Death Eater") that means nothing elsewhere.
- Text transfers best of all (full treatment under Q2).

**Held-out test estimate:**

| split rule | chosen config | trainval CV PR-AUC | test PR-AUC | test ROC-AUC |
|---|---|---|---|---|
| random | everything / random forest | 0.732 | **0.752** | 0.850 |
| grouped | everything / random forest | 0.562 | **0.664** | 0.776 |

- Test exceeded CV under both rules — reassuring direction (overfitting the selection procedure would show test falling short of CV instead).
- Grouped-rule gap (0.562 → 0.664) shouldn't be over-read: only 38 franchises, result depends heavily on the test-set draw.

**Feature importance** (permutation, character-only forest): page length 0.167, archetype 0.111, gender 0.107, de-leaked field count 0.101, prominence tier 0.101, billing order 0.092.

- **[Course: permutation importance]** — used instead of raw tree-split importance because it's model-agnostic and measures actual predictive contribution, not just how often a feature is split on.
- The two strongest features (page length, field count) are wiki-page measurements, not character attributes — main caveat carried into the Conclusion.

**Gender, specifically:**

- Female → roughly half the odds of dying: **OR 0.564, 95% CI 0.516–0.616** (≈11.1pp lower death probability), adjusted for prominence, role, alignment, species, franchise fixed effects (logistic regression, n=14,106/15,686).
  - **[Course: confound adjustment]** — franchise fixed effects included specifically so the gender effect isn't just "franchises with more female leads happen to be less deadly."
- Unadjusted gap: 14.5pp — under a quarter of the raw difference is explained by men holding more disposable roles.
- Per-franchise: 22 franchises estimable separately, **11 reach p<0.05, none reverse direction** (uncorrected for multiple comparisons — claim is direction-consistency, not any one franchise's significance). Weak cases (Grey's Anatomy, Avatar, The Sopranos) are null, not reversed.
- Robustness: estimate moves by thousandths of an odds ratio across every inclusion-rule variant from the Data section.

#### Visualization

- **Figure 2** (`q1_model_comparison.png`) — PR-AUC by feature set, one panel per split rule (random left, grouped right). Read it by comparing bar *height loss* left-to-right per feature set: bars that barely shrink (network, text) carry transferable signal; bars that collapse (franchise, Fandom categories) were mostly memorizing franchise identity.
- **Figure 3** (`q1_feature_importance.png`) — permutation importance, character-only forest, colour-coded by whether a feature is a character attribute or a wiki-page measurement. The takeaway is in the colour split, not the ranking: the two tallest bars are page measurements, which is the project's central caveat.
- **Figure 4** (`q1_gender_forest.png`) — per-franchise odds ratios with 95% CIs for the gender effect. Every marker sits left of 1 (protective effect for women); read the width of each interval as a stand-in for that franchise's sample size — the pooled estimate, not any single wide-interval franchise, is the one to trust.

#### Impediments

- Checked one hypothesis that did **not** hold: pages named by relation ("Alastor Moody's mother") are not a distinct failure mode (29.8% mortality vs. 35.6% average; not overrepresented in errors).
- Ranked out-of-fold trainval errors (test set untouched) — 2 franchises overrepresented in the worst 100, failing in opposite directions:
  - **Spartacus**: 1.4% of trainval rows, 14% of worst-100 errors (10x). 47 rows |error| > 0.5, 40 are false positives (predicted death, survived). Base mortality 71%; centrality doesn't help — Agron and Nasir are both highly central, one dies, one doesn't.
  - **Grey's Anatomy**: 2.5x overrepresented; all 42 |error| > 0.5 rows are false negatives (predicted survival, died). Base mortality 9.3%.
- **[Course: error analysis / confusion matrix]** — the two franchises need the decision threshold moved in opposite directions (one is a false-positive problem, one false-negative), so no single global threshold fixes both — a direct application of confusion-matrix-based diagnosis rather than a single aggregate error rate.
- Left uncorrected deliberately — franchise-specific patches would hide a real limit of what wiki metadata supports.
- Folds are very unequal (no minimum franchise size imposed): Prison Break contributes 28 characters vs. MCU 3,489 — each counts as one fold.
- Missingness is franchise-level, not random: `n_relatives` = 0 across 6 franchises (33.7% of characters, incl. MCU) — those wikis don't use the family infobox keys. A held-out franchise can therefore lack a feature the model relied on, so grouped CV is a pessimistic, not clean, transfer estimate. Reported as-is because the conclusion rests on the *direction* of the CV drop, not its exact size.

---

## Question 2: Does the text about a character predict their death?

### Problem description

- Does unstructured wiki text carry death-prediction signal beyond structured attributes?

### Solution

- TF-IDF over each character's wiki description, same death/survival-stripping as Q1.
- Evaluated: text alone, and character + text — under both split rules, against the same baselines as Q1.
- **[Course: text feature leakage]** — same vectorizer-inside-pipeline discipline as Q1, since text features are more leakage-prone than structured ones.

### Evaluation

#### Evaluation Criteria

- Same metric/baselines/ground truth as Q1 (PR-AUC primary, ROC-AUC secondary, majority-class/prior baselines, `is_dead_v2`).
- Success criterion: text improves PR-AUC over character-only under **both** split rules — grouped-CV improvement specifically tests transfer to an unseen franchise, not just per-show vocabulary fit.
- Guard against spurious signal: `src/q2_text/coefficients.py` explicitly asserts no death/survival token survived stripping, so a text-based win can't trace back to label leakage.
  - **[Course: verifying leakage removal, not just assuming it]** — the check is code, not a claim, so the vocabulary can be audited directly rather than trusted.

#### Setup

- Reuses the random/grouped CV split protocol from Q1 (Setup) — same folds, same franchise grouping, so Q1 and Q2 results are directly comparable row-for-row.
- Vectoriser (TF-IDF, 300-term vocabulary cap) is fit **inside** each CV fold, never on the full corpus first, so no test document's vocabulary leaks into training.
- No human subjects.

#### Results

- `character + text` (0.545 grouped CV) ties the everything-model (369 more columns) as best grouped-CV configuration.
- Text alone (0.650) beats character attributes alone (0.633).
- Adding text to character attributes: +0.066 PR-AUC (random CV), +0.058 (grouped CV).
- Text is the only feature family that transfers to an unseen franchise as well as to a familiar one.
- Strongest coefficient: `antagonist` (+3.81). Other portable terms describe narrative position, not identity: `command`, `background`, `recurring`, `military`, `criminal` — same meaning in any franchise, hence transfer.
- Less portable terms: franchise fingerprints (`hydra`, `island`, `games`) or wiki structure (`relationships`, `category`, `characters`, `history` are section headings, not traits) mixed with franchise vocabulary (`hospital`, `debra`, `awakening`).

#### Visualization

- **Figure 5** (`q2_text_terms.png`) — strongest TF-IDF coefficients in each direction, split into two groups by design: portable narrative-role terms vs. franchise fingerprints. The takeaway is the grouping itself — it's what explains why text transfers (Results, above) while Fandom category features (Q1) don't.

#### Impediments

- Answers the question for descriptive wiki text, not dialogue.
- 1,136 episode transcripts across 11 shows exist, but episode-level death registry covers only 2 — largest transcript set (Grey's Anatomy, 464 episodes) has no outcome to align against.
- Most transcripts lack speaker labels — attribution is a prerequisite problem in itself.
- A model on 2 aligned shows wouldn't support a claim about the other 9.

---

## Question 3: Are some franchises just deadlier?

### Problem description

- Do mortality rates differ meaningfully by franchise?
- Do franchise properties (genre, medium, age, run length, rating) explain the difference?

### Solution

- **Regression:** one row per franchise (mortality + TMDB properties); weighted least squares on `logit(mortality)` vs. first release year, run span, number of titles, medium, mean audience rating.
  - Weighted by cast size, so 40-character and 4,000-character franchises don't count equally.
  - **[Course: pseudo-replication / non-independence]** — first attempt was a binomial GLM treating every one of the 17,584 characters as one independent yes/no trial, with each character carrying their franchise's properties (release year, medium, etc.) as covariates. That model returned p<1e-10 on every term — but a franchise's 3,647 MCU characters are not 3,647 independent tests of "does the MCU have high mortality," they're one show's rate copy-pasted 3,647 times. The model's own fit statistic flags this directly: Pearson χ²/df = 36.6 (it should be ≈1 if characters really were independent draws), meaning the data is far more clustered than the model assumes, so its p-values are far too small to trust.
    - Fix: collapse to one row per franchise (38 rows, not 17,584), regress `logit(mortality)` on the franchise properties directly, and weight each franchise's row by its cast size (so the MCU still counts for more than an 11-character franchise like Peaky Blinders) — without pretending there are 17,584 independent observations.
- **Clustering:** k-means over standardized franchise property vectors.
  - Standardized because raw Euclidean distance is meaningless when cast size runs to thousands and rating sits between 6–9.
  - k chosen by silhouette score, checked for stability across 10 seeds.
  - **[Course: clustering as exploratory, not confirmatory]** — treated as exploratory per the lecture's framing of clustering as subjective: 38 points in 8 dimensions admit several defensible groupings, so results are interpreted, not hypothesis-tested.

### Evaluation

#### Evaluation Criteria

- **Regression success criterion:** term-level significance (p<0.05) and overall R² — explicitly allows a null result, since the hypothesis under test is "franchise properties explain mortality," not "franchises differ."
- **Clustering success criterion:** silhouette score + stability across 10 seeds; a split is only informative if it also separates mortality rates.
- **Ground truth:** `is_dead_v2` where parseable, widened with Haiku-resolved rows for the 4 unresolved franchises (see Data → Labels) — reported as bounds + point estimate, never as a bare number, for those 4.
- **Guarding against spurious findings:**
  - Rejected the naive per-character GLM because pseudo-replication manufactured false significance (see Solution).
  - Ran a measurement-definition check on run span (see Impediments) that reversed an apparently-significant result — used as the project's own worked example of a variable that looked real but measured something else.
  - Every mortality estimate carries an uncertainty band (Wilson interval + label bounds) rather than a bare point estimate.

#### Setup

- **Population:** mortality computed over 17,584 on-screen named characters — wider than Q1/Q2's 15,686, because Q1/Q2 exclude characters with no usable label to *train* against, but excluding them from a *rate* would bias it.
- **Uncertainty handling:** for the 4 mostly-unlabelled franchises, report bounds (lower = unparsed→alive, upper = unparsed→dead) plus a point estimate using Haiku-resolved labels where available.
- No human subjects.

#### Results

| franchise | unlabelled | resolved | bounds | point estimate |
|---|---|---|---|---|
| The Wire | 214 | 200 (93.5%) | 14.3–99.2% | 16.3% |
| Prison Break | 155 | 143 (92.3%) | 6.6–91.3% | 21.3% |
| Boardwalk Empire | 440 | 329 (74.8%) | 19.7–99.8% | 23.2% |
| Lost | 330 | 203 (61.5%) | 23.2–69.0% | 26.6% |

- These bounds are wide enough that the 4 franchises carry little weight below.
- A flat "apply corpus death rate" shortcut isn't safe: matches Lost's resolved rate (26.6% vs. 27.1%) but not Prison Break's (21.3% vs. 13.8%).
- Mortality varies widely: Grey's Anatomy 9.3% to The 100 72.7% (franchises with ≥100 characters).
- **No franchise property is significant.** Run span closest at p=0.185; medium, title count, release year, rating further out. **R² = 0.079, adjusted R² = −0.065** across 38 franchises — worse than predicting the overall mean for every franchise.
- k-means: best silhouette k=2 (0.43, stable across 10 seeds) — splits 6 large multi-title film franchises from the other 32, at 35.4% vs. 37.1% mortality — not a meaningful gap given the label uncertainty above.
- **Conclusion for Q3:** franchises differ enormously in mortality; none of the available franchise metadata (TMDB fields) predicts which ones.

#### Visualization

- **Figure 6** (`q3_franchise_mortality.png`) — mortality per franchise with both sources of uncertainty layered on the same bar: a grey 95% Wilson interval (sampling uncertainty) and a blue band (how far the estimate could move if every unlabelled character turned out dead). The four wide-banded franchises are the ones the Results table above already flags as low-confidence.
- **Figure 7** (`q3_run_span.png`) — mortality against franchise run span, marker size proportional to cast size. The fit line is flat and the point cloud is diffuse — visually corroborating the p=0.185 / R²≈0 result rather than just asserting it.

#### Impediments

- **Measurement check that changed the answer:**
  - Run span must be first-to-last air date for TV (each TV show is one TMDB title, so release-year spread alone records 0 years regardless of actual run length).
  - Defined on release years alone: run span reaches p=0.022, adjusted R²=0.047.
  - Defined correctly: both vanish.
  - **[Course: construct validity]** — the apparent effect was separating multi-film franchises from television, not measuring duration; a direct illustration of the lecture point that a statistically significant term can still be measuring the wrong construct.
- The 4 mostly-unlabelled franchises (The Wire, Prison Break, Boardwalk Empire, Lost) can only be reported as bounds + a point estimate, never a clean number — addressed in Setup/Results above, not solved outright.
- Only 38 franchises total: both the regression (R², term p-values) and the clustering (which franchises land together) are sensitive to which few franchises are unusual, which is why results are reported with intervals/stability checks rather than as single confident numbers.

---

## Interactive Demo

**"Plot Armor Index"** — https://shaharspencer.github.io/needle-group-project/

- **What it is:** a small, dependent, in-browser logistic regression (fewer features than the project's real Q1 model) that scores individual characters, built so a reader can inspect specific predictions rather than only reading aggregate PR-AUC numbers.
- **Header stats bar:** number of held-out characters scored, number of franchises (38), overall base death rate, and test-set ROC-AUC, plus a plain-language line ("given one character who died and one who didn't, this model ranks the one who died as riskier about X% of the time").
- **"What drives this model" panel:** a horizontal bar chart of feature importance, computed as the spread (standard deviation) of each feature's log-odds contribution across the real characters shown — not the raw min-max coefficient range, because that would mechanically favor high-cardinality features like franchise (38 levels) over low-cardinality ones regardless of what the data says.
- **Tab 1 — "Look up a character":**
  - Left panel: a searchable, filterable list of real (held-out) characters — search by name or franchise; filter to "All" / "Got it right" / "Got it wrong" (correctness = does a 50%-threshold prediction match the actual outcome). A running count shows "N of M called correctly (X%) at a 50% threshold."
  - Right panel, once a character is selected:
    - A large percentage (predicted death risk), colour-coded red (likely dies) or blue (likely survives), with the character's name, franchise, role, and gender.
    - A horizontal risk meter with a labelled scale (certain survival — risk band — certain death).
    - A verdict line stating the character's actual outcome and whether the model's 50%-threshold call matched it.
    - A "Why" table: the 8 largest attributes by log-odds contribution, shown as red/blue bars (red = pushes toward dying, blue = toward surviving), each with a hover tooltip defining exactly what that attribute means and how it was derived.
    - An "Others the model scored the same" table: the 5 other characters closest to this one by predicted risk (not by similarity of traits), with their franchise, risk %, and actual outcome — explicitly framed as "the group the model can't tell apart," and flagging when their real outcomes diverge as a visible illustration of the model's limit.
    - A link to the character's source wiki page.
- **Tab 2 — "Build a character":** lets the reader set every model feature by hand (gender, role, alignment, franchise as dropdowns; billing order, story centrality, times mentioned, named relatives as sliders; species as a toggle), each with the same hover definitions, and see the predicted death risk and "Why" contribution table update live as the sliders move — a what-if tool rather than a lookup of a real character.
- **Integrity note built into the tool itself:** every character in the lookup tab is from the held-out test set, never a row the demo's model trained on, so "got it right / wrong" reflects real generalization rather than the model being asked about data it memorized.
- **Footer disclaimer, stated explicitly on the page:** the demo's model is a smaller logistic regression restricted to user-selectable attributes only; the project's actual Q1 model uses many more features and scores higher; predictions describe patterns in how these stories were written, not what any character "deserves."
- **[Course: interpretability / model transparency]** — the per-character contribution breakdown and the "others scored the same" comparison exist specifically to let a reader audit individual predictions rather than trust an aggregate metric alone, addressing the same "don't just report accuracy, look at the mistakes" principle applied throughout Q1's Impediments section.

---

## Future Work

Ordered by how much each would change the answers:

- **Community detection on the co-mention graph** — only centrality is extracted currently; community detection would test whether characters in the same narrative cluster share fates, which centrality alone can't (exactly where the Spartacus errors break down). **[Course: community detection]**.
- **Significance tests on the Q1 model comparison** — feature-set PR-AUCs currently ranked by point estimate only; a cross-fold significance test would settle whether e.g. 0.699 vs. 0.692 is a real difference. **[Course: significance testing on model comparisons]**.
- **Cost-sensitive loss** — Spartacus/Grey's Anatomy fail in opposite directions (false positive vs. false negative); a weighted loss would make the relative cost explicit. **[Course: cost-sensitive learning]**.
- **Inter-rater reliability figure** — Claude Haiku 4.5 labelled the 600-page sample and unresolved rows as a single coder; `src/labels/eval_labels.py` already computes Cohen's kappa if a second pass exists. **[Course: inter-annotator reliability]** — kappa, not raw agreement, given the uneven category split (41% on-screen / 12% non-character).
- **Dialogue** — blocked on the transcript speaker-attribution problem (Q2 Impediments).
- **Thin franchises and feature gaps** — fuller per-show scrapes for Walking Dead and Prison Break exist in `datasets/` but carry no page text/categories, unused; grouped CV should either handle per-franchise missingness explicitly or restrict to franchises with full feature coverage plus a minimum size threshold.
- **Prospective (not retrospective) features** — hardest extension: rebuild features from what was knowable partway through a series and predict forward, rather than from post-hoc wiki text.

---

## Conclusion

- Character attributes carry more signal than franchise identity: 0.633 PR-AUC (character alone) vs. 0.556 (franchise alone) vs. 0.743 (everything), on a 0.356 base rate.
- Held-out test: **0.752 PR-AUC / 0.850 ROC-AUC**, falling to **0.664 / 0.776** with whole franchises held out.
- Text is the single most useful addition — beats structured attributes alone, only family that transfers as well to an unseen franchise.
- Gender effect is real and stable: women ~half as likely to die, adjusted, across 11/22 franchises, no reversals.
- Franchise properties explain none of the variation in franchise-level mortality.
- **Main limitation:** the data is retrospective — wikis are written after the story ends, by fans who write more about characters they cared about, and our two strongest structured features (page length, infobox field count) measure fan attention, which death itself attracts. This shows what separates characters who died from those who didn't *in retrospective accounts*, not that a death was foreseeable from what a viewer knew at the time. The two families built from the story's own structure rather than how much was written about it — network centrality, narrative-role text terms — are also the two that survive the move to an unseen franchise.
