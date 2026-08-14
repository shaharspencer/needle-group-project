# Spoiler Alert: Who Makes It to the Finale?

**A Needle in a Data Haystack (67978) — Final Project, Group #56**

| Name | ID | CSE username | Email |
|---|---|---|---|
| Stav Abukarat | 322897687 | stava | stav.abukarat@mail.huji.ac.il |
| Shahar Spencer | 208067017 | shahar.spencer | shahar.spencer@mail.huji.ac.il |
| Bar Dalal | 212622260 | bar.dalal | bar.dalal@mail.huji.ac.il |
| Ido Oron | 208733634 | ido_oron | ido.oron@mail.huji.ac.il |

Interactive demo: **https://shaharspencer.github.io/needle-group-project/**

Code: see `README.md` for the full run order.

---

## Domain

Fictional characters die on purpose. Someone decides it, and that decision is
shaped by who the character is (how prominent, what role, what gender) and by
the show they're in — a Grey's Anatomy doctor and a Spartacus gladiator are not
facing the same odds. We wanted to know how much of that is visible in data, and
whether a character's own attributes or their franchise tells you more.

The unit of analysis is one character in one franchise; the outcome is binary,
does this character die during the story as aired. Everything comes from what
fans wrote on wikis plus cast data from TMDB.

---

## Data

**Where it came from.** We scraped 38 Fandom wikis through the MediaWiki API
(Game of Thrones, MCU, Star Wars, Harry Potter, Breaking Bad, and 33 more),
pulling each character page's infobox, full page text, and category tags.
That gave 68,463 pages. We enriched it with TMDB (280 titles, 28,151 cast
credits) for billing order, episode counts, genre and ratings, and pulled in
episode-level death registries and episode transcripts from Kaggle and
listofdeaths.fandom.com. The raw scrape is about 262 MB; the built modelling
table is 68,463 × 129.

**The label.** Death comes from the wikis' own infoboxes, not from us and not
from a model. Thirty wikis record a `status` string ("Deceased", "Alive");
eight instead record a date of death, where a filled field means dead. Both
conventions collapse into one binary `is_dead_v2`.

**Not every page is a character.** This is the largest data-quality problem we
faced. To size it we drew a 600-page stratified sample and labelled every page
with Claude Haiku 4.5 against a written rubric: is this an individual character,
does it appear on screen, does it die.

Reweighted to the full corpus, 41% of pages are on-screen characters, 46% are
characters who exist only in novels or reference books (Star Wars Legends, Harry
Potter companion books), and 12% are not characters at all — creature types,
vehicles, and real historical figures picked up from the Young Indiana Jones
wiki. We restrict to on-screen characters using
`is_onscreen = has_actor OR tmdb_match`, which scores precision 0.79 / recall
0.77 against that sample. Neither signal works alone (0.61 and 0.58 recall),
which is why the rule combines them.

![Composition of the scraped corpus](data/clean/figures/corpus_composition.png)

**Figure 1.** Three in five scraped pages are not an on-screen character. The
analysis population is the leftmost segment only. Plotting this was what made
the scale of the problem legible: the corpus is not a character list, it is a
wiki, and most of a wiki is not the thing you came for.

Two further groups pass that filter and still have to be excluded:

- **168 pages that are objects, not people.** Talking portraits on the Harry
  Potter wiki ("Witch in portrait") and background paintings on the Star Wars
  wiki. A portrait is dead by definition — whoever it depicts died before the
  painting existed — so it is trivially separable by a death classifier without
  that telling us anything about characters. These pages alone inflated Harry
  Potter's mortality by six points (23.0% → 16.7% once removed).
- **1,122 unnamed background roles**, titled by job rather than name
  ("Unidentified Ugnaught worker", "Unnamed pilot"). These do appear on screen —
  TMDB credits background extras, so the names match real credits — but they
  aren't characters in the sense we're measuring, and our own annotation rubric
  already counted unnamed roles as non-characters. They're 6.7% of the on-screen
  set, over a third of Star Wars, and die at 22.7% against 35.6% for named
  characters, so leaving them in dragged the headline mortality down.

**Final analysis population: 15,686 on-screen, named characters, 35.6% of whom
die.** Both filters are pure functions of a page's own fields (an infobox tag, a
name pattern) — never the label, never a statistic over other rows — so they're
safe to apply before any train/test split exists. Setting
`INCLUDE_UNNAMED_ROLES=1` reproduces the numbers with the unnamed roles kept in;
it moves everything slightly and changes no conclusion.

**Known biases.** Star Wars is 56% of the raw scrape and still 12% after
filtering, so grouped cross-validation and per-franchise reporting do real work
throughout. Coverage is also uneven across wikis: The Walking Dead contributes
49 rows against a wiki with hundreds, and Stranger Things, Vikings, Spartacus
and Westworld are all thinner than the shows warrant, because those wikis use
infobox templates our page discovery does not match. Franchise-level results for
those five describe an arbitrary sample of the wiki rather than the show, and we
exclude franchises under 30 characters from the franchise-level analysis for
this reason.

### Rows the infobox does not resolve

1,862 rows (11.9% of the population) had a status field that didn't parse into
anything usable, concentrated in four franchises: The Wire, Prison Break,
Boardwalk Empire and Lost, all between 46% and 85% unlabelled. Reporting those
franchises' mortality as-is would have been meaningless — Boardwalk Empire's
true rate was somewhere between 17% and 100% depending on how those rows
resolved.

Label sources are preferred in a fixed order: a human-curated death registry
where one exists, then the wiki infobox, then a language model (Claude Haiku
4.5) for whatever neither can settle. The registries do not reach these
franchises, so all 1,862 rows went to the model, batched 100 at a time, each
batch asked to read the wiki text and decide whether the character dies on
screen, with "can't tell" available as an explicit answer.

It returned a verdict on 1,858 of them: **123 dead, 1,223 alive, and 512 left
undecided — 27.6%.** Leaving "can't tell" available matters: a classifier that
resolves every case it is handed gives no signal about the cases it should not
resolve, and a quarter of these were genuinely undecidable from the page. The
death rate among the rows it did settle was 9.1%, close to the 8.5% the 600-page
sample implies for unlabelled rows.

These labels live in their own files (`data/gold/fill_unlabelled/`) and are
never merged into `is_dead`. They feed exactly one number: the Q3 mortality
estimates below.

### Checking the label against a separate source

The label and the features come from the same character pages, so we also check
it against a source that does not use those pages at all.
`listofdeaths.fandom.com` catalogues deaths episode by episode; we scraped its
Game of Thrones and The 100 entries (`legacy/scrapers/`) and matched them to our
characters by name. It is a different wiki, written by different editors and
organised around episodes rather than characters, so it is not derived from the
infobox fields the label reads.

| franchise | matched 1:1 | registry says died, we say dead |
|---|---|---|
| Game of Thrones | 134 | 132 (98.5%) |
| The 100 | 68 | 65 (95.6%) |

**197 of 202 agree, 97.5%.** Two of the five disagreements look more like the
registry's problem than ours (Jaqen H'ghar doesn't die, he changes face; Ellaria
Sand is imprisoned, not killed). This measures recall only — a name missing from
a registry might have died off screen or in the books — so it says nothing about
precision.

The Fandom `Deceased` category is not a second such source, despite looking like
one. 7,388 pages carry it, and of the 7,128 with a parseable status, 96.6%
already say dead — the same editors recording the same fact through two wiki
mechanisms. It establishes that the wikis are internally consistent, not that
the label is correct.

---

## Question 1: Is it who you are, or what show you're in?

**Problem.** Can we predict whether a character dies from their attributes, and
which matters more — the character or the franchise?

**Solution.** Nine feature sets crossed with three models (logistic
regression, decision tree, random forest), so the comparison isolates the
feature families rather than the estimator. The sets: franchise metadata only;
character attributes only; character without wiki-metadata proxies; character +
franchise; franchise + network centrality; character + network; TF-IDF over
page text alone; character + text; and everything, which adds 369 one-hot
Fandom category and infobox-field features.

Character features are gender, archetype (a keyword rule over infobox
title/occupation, since there's no clean occupation column), alignment,
prominence tier, billing order, episode count, page length, a de-leaked infobox
field count, and several has-this-field flags. Network features come from a
per-franchise co-mention graph built from page text: PageRank, HITS hub and
authority, in/out degree, component size. Every feature's exact definition and
computation is in `FEATURES.md`.

The text features require care about leakage. A dead character's opening
paragraph says "was killed by", so TF-IDF over raw text largely rediscovers the
label rather than predicting it. The text is therefore stripped twice before
vectorising: every sentence mentioning death *or survival* is dropped, then any
surviving death token is removed. Survival words matter as much as death words —
"still alive" is the label with the sign flipped. We verified that the resulting
300-term vocabulary contains no death or survival token, and that tense markers
are not carrying the label either — "is" and "was" are English stopwords and
already excluded. The vectoriser is fit inside the cross-validation pipeline, so
the vocabulary and IDF weights never see a test document.

Two cross-validation regimes, because they answer different questions. A
**random 5-fold split** lets the model learn each franchise's base rate, which
is the only setting where "a GoT character vs a Harry Potter character" is a
fair comparison. A **grouped split** (`GroupKFold` on franchise) holds out whole
franchises, so franchise identity is useless by construction and what's measured
is whether character attributes transfer to a show the model has never seen.

**Evaluation criteria.** PR-AUC as the headline, since the classes are
imbalanced (35.6% positive) and PR-AUC's chance level is the base rate rather
than 0.5. Two baselines: majority class and prior. Preprocessing (imputation,
scaling, one-hot encoding) all sits inside the CV pipeline so nothing is fit
across a fold boundary.

**Results.** PR-AUC, best model per feature set:

| features | random split | held-out franchise |
|---|---|---|
| everything | **0.743** | 0.545 |
| character + text | 0.699 | **0.545** |
| character + franchise | 0.692 | 0.485 |
| character + network | 0.680 | 0.519 |
| franchise + network | 0.650 | 0.486 |
| text only | 0.650 | 0.509 |
| character only | 0.633 | 0.487 |
| character, no wiki metadata | 0.590 | 0.504 |
| franchise only | 0.556 | 0.398 |
| baseline | 0.356 | 0.356 |

![PR-AUC by feature set under both cross-validation regimes](data/clean/figures/q1_model_comparison.png)

**Figure 2.** The same comparison as the table, drawn as two panels so the gap
between them is the point. Left: franchise identity is available to the model.
Right: whole franchises are held out. Every bar shortens on the right, and the
ones that shorten least are the families that describe the story rather than the
wiki.

Character attributes beat franchise identity (0.633 vs 0.556), and the two are
complementary — combining them beats either. So the answer is "both, and mostly
who you are," with a large caveat: the full model loses 0.198 PR-AUC when whole
franchises are held out (0.743 → 0.545). Much of what looks like character-level
signal on a random split is the model learning per-show base rates.

The feature families divide sharply on whether they transfer. Text is the
strongest addition in the table and is treated separately under Q2. Network
centrality also travels: adding it to franchise metadata is worth +0.094 on a
random split and +0.088 held-out, since PageRank over a co-mention graph
describes a character's position in their own story rather than the show's
vocabulary. Fandom category features show the opposite profile — a large mover
on a random split, much weaker held-out — because they encode franchise-specific
role vocabulary ("Jedi", "Death Eater") that means nothing in another franchise.

**A held-out estimate.** The comparison above is designed to rank feature
families, so it reports the best configuration in each row. A separate, stricter
protocol gives the performance estimate: an 80/20 `trainval`/`test` split —
stratified for the random regime, packed largest-deficit-first by franchise for
the grouped one, since franchise sizes span three orders of magnitude — with
model selection by k-fold *inside* trainval and the test set scored exactly once.

| regime | chosen | trainval CV PR-AUC | test PR-AUC | test ROC-AUC |
|---|---|---|---|---|
| random | everything / random forest | 0.732 | **0.752** | 0.850 |
| held-out franchise | everything / random forest | 0.562 | **0.664** | 0.776 |

Test scores came in above the cross-validated estimates that selected them in
both regimes, which is the direction that does not indicate overfitting to the
selection procedure.

**Feature importance** (permutation, character-only forest): page length 0.167,
archetype 0.111, gender 0.107, de-leaked field count 0.101, prominence tier
0.101, billing order 0.092.

![Permutation importance, character-only forest](data/clean/figures/q1_feature_importance.png)

**Figure 3.** Colour separates attributes of the character from measurements of
the wiki page. The two strongest features are the wiki ones, which is the single
most important caveat on this project and the reason the Conclusion treats these
results as descriptive rather than predictive.

### Gender, specifically

Being female roughly halves the odds of dying: **OR 0.564, 95% CI 0.516–0.616**,
about 11.1 percentage points lower death probability after adjusting for
prominence, role, alignment, species and franchise (logistic regression with
franchise fixed effects, n=14,106). The unadjusted gap is 14.5 points, so a bit
under a quarter of the raw difference is explained by men holding more disposable
roles.

Of 22 franchises where the effect could be estimated separately, **11 reach
p<0.05 and none point the other way.** The weak cases (Grey's Anatomy, Avatar,
The Sopranos) are null effects rather than reversals. The estimate is also
insensitive to the population definition: it moves by a few thousandths of an
odds ratio under each of the inclusion rules described in the Data section.

![Odds ratio for dying, women versus men, by franchise](data/clean/figures/q1_gender_forest.png)

**Figure 4.** A forest plot is the right form here because the claim is about
consistency, not size: what matters is that every point sits left of 1 and no
confidence interval crosses far to the right. Filled markers reach p<0.05, hollow
ones do not. The wide intervals belong to the small franchises, which is why the
pooled estimate carries the argument and the per-franchise panel supports it.

### What the model gets wrong

We ranked the out-of-fold trainval predictions by error (never test) and looked
for patterns in the worst 100. One candidate explanation did not hold: pages
named by relation to someone else ("Alastor Moody's mother") are not a distinct
failure mode. They are *less* likely to die than average (29.8% vs 35.6%) and
are not overrepresented among the largest errors.

Two franchises are, and they fail in opposite directions:

- **Spartacus** is 1.4% of trainval rows and 14% of the worst 100 errors, a 10x
  overrepresentation. Of its 47 errors above 0.5, 40 run one way: the model
  confidently predicts death for characters who survive. Base mortality is 71%,
  so "prominent + military + this franchise → dies" is right most of the time and
  wrong precisely for the show's leads. Centrality does not rescue it — Agron and
  Nasir are both highly central, and PageRank cannot distinguish "central and
  dies" from "central and survives".
- **Grey's Anatomy** is the mirror image: 2.5x overrepresented, and all 42 of its
  errors above 0.5 run the other way, with characters who die predicted as very
  likely to live. Base mortality is 9.3%, so the model has learned that almost
  nobody dies on this show — right nine times in ten, and wrong for every
  one-off death.

Both are base-rate anchoring in opposite directions, and no feature available to
us separates a central character who dies from a central character who survives.
We report it as a limit on what wiki metadata can support rather than patching
around two specific franchises.

---

## Question 2: Does the text about a character predict their death?

**Problem.** Does unstructured text carry signal about a character's fate beyond
what their structured attributes already say?

**Solution.** TF-IDF over each character's wiki description, death- and
survival-stripped by the two-pass procedure described in Q1, vectorised inside
the cross-validation pipeline so no test document informs the vocabulary or the
IDF weights. We evaluate it two ways: alone (`text`), and added to the character
attributes (`character + text`), under both CV regimes, against the same
baselines as Q1.

**Results.** Text is the most valuable feature family in the project. Adding it
to character attributes is worth +0.066 PR-AUC on a random split and +0.058 on a
held-out franchise, and `character + text` is the best held-out configuration in
the entire comparison at 0.545 — level with the everything-model, which carries
369 more columns and reaches 0.545 as well. Text alone (0.650) beats character
attributes alone (0.633) outright.

It is also the only family that transfers as well to an unseen franchise as to a
familiar one, and the fitted coefficients show partly why. The strongest term by
a wide margin is `antagonist` (+3.81), and several others describe narrative
position rather than identity — `command`, `background`, `recurring`, and lower
down `military` and `criminal`. Those mean the same thing in any franchise, which
is the property that survives being moved to a show the model has not seen.

The rest is less portable, and the figure below makes that plain rather than
hiding it. `hydra`, `island` and `games` sit high on the death side and are
franchise fingerprints, not roles. The survival side is mostly wiki structure —
`relationships`, `category`, `characters`, `history` are section headings that
describe the page rather than the character — mixed with more franchise
vocabulary (`hospital`, `debra`, `awakening`). Only part of this family is
carrying transferable signal, which is the honest reading of why text helps as
much as it does and no more.

![Strongest TF-IDF terms in each direction](data/clean/figures/q2_text_terms.png)

**Figure 5.** Worth reading term by term rather than as a block. The transferable
signal is real but partial: `antagonist`, `command`, `background` and `recurring`
would mean the same thing in a franchise the model has never seen, while `hydra`,
`island` and `games` on the same side would not.
`src/q2_text/coefficients.py` writes these weights and asserts that no death or
survival token survived the stripping, so the vocabulary can be audited rather
than taken on trust.

**Scope.** This answers the question for descriptive text about a character. It
does not answer it for dialogue. We hold transcripts for 1,136 episodes across 11
shows, but an episode-level death registry for only two of them, so the largest
transcript set (Grey's Anatomy, 464 episodes) has no outcome to align against;
most transcripts also carry no speaker labels, making attribution of a line to a
character a prerequisite problem in its own right. A dialogue model fit on the
two aligned shows would not support a claim about the other nine.

---

## Question 3: Are some franchises just deadlier?

**Problem.** Do mortality rates differ by franchise, and do franchise
properties (genre, medium, age, length, rating) explain the difference?

**Solution.** One row per franchise, with mortality plus TMDB properties. A
binomial GLM (weighted, so a franchise with 40 characters doesn't count the same
as one with 4,000) of deaths-out-of-characters on first release year, run span,
number of titles, medium, and mean audience rating. Then k-means over the
property vectors, with k chosen by silhouette and checked for stability across
10 seeds.

**Handling label uncertainty.** For the four franchises that are mostly
unlabelled, a single mortality figure would be indefensible, so we report the
bounds alongside the estimate: a lower bound with unparsed rows counted alive, an
upper bound with them counted dead, and a point estimate that uses the
model-supplied labels described above wherever they resolved a row.

| franchise | unlabelled | resolved | bounds | point estimate |
|---|---|---|---|---|
| The Wire | 214 | 200 (93.5%) | 14.3–99.2% | 16.3% |
| Prison Break | 155 | 143 (92.3%) | 6.6–91.3% | 21.3% |
| Boardwalk Empire | 440 | 329 (74.8%) | 19.7–99.8% | 23.2% |
| Lost | 330 | 203 (61.5%) | 23.2–69.0% | 26.6% |

The bounds are wide enough that these four franchises carry little weight in what
follows. Where the resolved labels can be compared against assuming the corpus
death rate applies uniformly to unlabelled rows, they agree for Lost (26.6% vs
27.1%) and diverge for Prison Break (21.3% vs 13.8%), which is the expected
pattern: a uniform assumption is least reliable for a single show with an unusual
mortality profile.

**Results.** Mortality does vary a lot by franchise — from Grey's Anatomy at
9.3% to The 100 at 72.7% among franchises with at least 100 characters. But
franchise *properties* explain none of it. No term in the model reaches
significance: run span is the closest at p=0.185, and medium, number of titles,
first release year and audience rating are all further away. **R² = 0.079, and
adjusted R² = −0.065** on n=38 — worse than predicting every franchise at the
overall mean.

![Franchise mortality with sampling and labelling uncertainty](data/clean/figures/q3_franchise_mortality.png)

**Figure 6.** Mortality per franchise, drawn with both kinds of uncertainty
rather than as a bare ranking. The grey bar is the 95% Wilson interval on the
estimate; the blue band is how far the figure could move if every character with
no recorded status turned out to have died. The four franchises with a long band
are the ones discussed above, and the band is the honest width of what we know
about them.

![Mortality against franchise run span](data/clean/figures/q3_run_span.png)

**Figure 7.** The null result, plotted so it can be checked rather than asserted.
Marker size is cast size. A flat fitted line through a diffuse cloud is what a
non-result looks like, and showing it is more informative than reporting p=0.185
alone.

One measurement choice matters enough to state, because it determines whether
this result is a null at all. Run span has to be the interval a franchise was in
production, which for a film series is the spread of its release years but for a
television series is first air date to last air date — a series is a single TMDB
title, so taking the spread of release years records every show as spanning zero
years regardless of how long it actually ran. Defined on release years alone,
run span reaches p=0.022 and the model an adjusted R² of 0.047; defined
correctly, both vanish. The apparent effect was the variable separating
multi-film franchises from television, not measuring duration.

k-means doesn't recover mortality groupings either. Best silhouette is k=2
(0.43, stable across 10 seeds), and that split mostly separates six large
multi-title film franchises from the other 32, landing at 35.4% and 37.1%
mortality — not a meaningful gap given the label uncertainty above.

The answer to Q3 is therefore negative, and cleanly so: franchises differ
enormously in how deadly they are, and none of the franchise metadata available
to us predicts which ones. Whatever drives the difference — tone, genre
convention, a showrunner's willingness to kill a lead — is not represented in
TMDB's fields.

---

## Conclusion

Character attributes carry more signal about death than franchise identity does,
and the two combine: 0.633 PR-AUC from character features alone against 0.556
from the franchise, and 0.743 from everything together, against a 0.356 base
rate. On the held-out test set the full model reaches **0.752 PR-AUC / 0.850
ROC-AUC** on a random split and **0.664 / 0.776** when whole franchises are held
out. The most useful single addition was text: TF-IDF over each
character's death-stripped wiki description, which on its own beats their
structured attributes and is the only feature family that helps as much on an
unseen franchise as on a familiar one. The gender effect is real and stable —
women are about half as likely to die, adjusted, consistent across 11 of 22
franchises with no reversals. Franchise-level properties, by contrast, explain
none of why some shows kill more than others: no term in the model reaches
significance and adjusted R² is negative.

The strongest constraint on all of this is what the data is. A wiki is written
after the story is over, and by people who write more about characters they cared
about. Our two most important structured features — page length and infobox
field count — therefore measure fan attention, and death attracts fan attention.
These results describe what separates characters who died from those who did not
in retrospective accounts; they do not establish that a death was foreseeable
from what a viewer knew at the time. The families that escape this are the ones
built from the story's own structure rather than from how much was written about
it — network centrality, and the narrative-role vocabulary the text features
pick up — and those are also the two that survive being moved to a franchise the
model has never seen.
