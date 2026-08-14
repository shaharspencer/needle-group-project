# Spoiler Alert: Who Makes It to the Finale?

**A Needle in a Data Haystack (67978) — Final Project, Group #56**

| Name | ID | CSE username | Email |
|---|---|---|---|
| Stav Abukarat | 322897687 | stava | stav.abukarat@mail.huji.ac.il |
| Shahar Spencer | 208067017 | shahar.spencer | shahar.spencer@mail.huji.ac.il |
| Bar Dalal | 212622260 | bar.dalal | bar.dalal@mail.huji.ac.il |
| Ido Oron | 208733634 | ido_oron | ido.oron@mail.huji.ac.il |

Demo: `docs/index.html` (also deployable on GitHub Pages).
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
fans wrote on wikis plus cast data from TMDB. We set out to answer three
questions and answered two.

---

## Data

**Where it came from.** We scraped 38 Fandom wikis through the MediaWiki API
(Game of Thrones, MCU, Star Wars, Harry Potter, Breaking Bad, and 33 more),
pulling each character page's infobox, full page text, and category tags.
That gave 68,463 pages. We enriched it with TMDB (280 titles, 28,151 cast
credits) for billing order, episode counts, genre and ratings, and pulled in
episode-level death registries and episode transcripts from Kaggle and
listofdeaths.fandom.com. The raw scrape is about 262 MB; the built modelling
table is 68,463 × 119.

**The label.** Death comes from the wikis' own infoboxes, not from us and not
from a model. Thirty wikis record a `status` string ("Deceased", "Alive");
eight instead record a date of death, where a filled field means dead. Both
conventions collapse into one binary `is_dead_v2`.

**Not every page is a character.** This turned out to be the single biggest
data problem in the project. From 600 annotated pages reweighted to the corpus,
roughly 40% are on-screen characters, 46% are characters who exist only in
novels or reference books (Star Wars Legends, Harry Potter companion books),
and 12% aren't characters at all — creature types, vehicles, real historical
figures picked up from the Young Indiana Jones wiki. We restrict to on-screen
characters using `is_onscreen = has_actor OR tmdb_match`, which scores
precision 0.79 / recall 0.77 against the annotated sample. Neither signal
works alone (0.61 and 0.58 recall).

Two further groups we found late, and only by clicking through our own demo and
noticing predictions that made no sense:

- **168 pages that are objects, not people.** Talking portraits on the Harry
  Potter wiki ("Witch in portrait"), background paintings on the Star Wars
  wiki. A portrait is definitionally dead — whoever it depicts died before the
  painting existed — so these were a free win for a death classifier with
  nothing to do with any character. They were inflating Harry Potter's
  mortality by six points on their own (23.0% → 16.7% once removed).
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

**Known biases we could not remove.** Star Wars is 56% of the raw scrape and
still 12% after filtering, so grouped cross-validation and per-franchise
reporting do real work throughout. Several wikis are badly under-collected
because our scraper looked for the wrong infobox template — we fixed four (see
Impediments), but The Walking Dead is still 49 rows against a wiki with
hundreds, and Stranger Things, Vikings, Spartacus and Westworld are all thinner
than they should be. Franchise-level results for those describe an arbitrary
sample of the wiki, not the show.

### Filling the gaps we couldn't parse

1,862 rows (11.9% of the population) had a status field that didn't parse into
anything usable, concentrated in four franchises: The Wire, Prison Break,
Boardwalk Empire and Lost, all between 46% and 85% unlabelled. Reporting those
franchises' mortality as-is would have been meaningless — Boardwalk Empire's
true rate was somewhere between 17% and 100% depending on how those rows
resolved.

We prefer label sources in a fixed order: a human-curated death registry where
one exists, then the wiki infobox, then Claude Haiku 4.5 for whatever neither
can settle. In practice the registries barely fire here (Game of Thrones has
zero unlabelled rows), so all 1,862 went to Haiku, batched 100 at a time across
19 subagents, each asked to read the wiki text and decide whether the character
dies on screen — and explicitly to answer "can't tell" instead of guessing.

It labelled 1,858 of them: **123 dead, 1,223 alive, and 512 left undecided —
27.5% of the batch.** We think the refusals are the most reassuring part of that
result; a model that resolves everything you hand it is a model that's guessing.
The death rate among what it did resolve was 9.1%, close to the 8.5% our
original 600-page annotation sample had independently estimated for unlabelled
rows in general.

These labels live in their own files (`data/gold/fill_unlabelled/`) and are
never merged into `is_dead`. They feed exactly one number: the Q3 mortality
estimates below.

### Checking the label against something we didn't write

The obvious objection is that we're grading our own homework. Two death
registries — Game of Thrones and The 100 — were compiled by other people, for
other reasons, before we touched anything, which makes them a genuine external
check.

| franchise | matched 1:1 | registry says died, we say dead |
|---|---|---|
| Game of Thrones | 134 | 132 (98.5%) |
| The 100 | 68 | 65 (95.6%) |

**197 of 202 agree, 97.5%.** Two of the five disagreements look more like the
registry's problem than ours (Jaqen H'ghar doesn't die, he changes face; Ellaria
Sand is imprisoned, not killed). This measures recall only — a name missing from
a registry might have died off screen or in the books — so it says nothing about
precision.

One thing this ruled out: the Fandom `Deceased` category looks like a second
independent signal, and we assumed it was for a while. It isn't. 7,388 pages
carry it, and of the 7,128 with a parseable status, 96.6% already say dead.
That's the same editors recording the same fact twice through two different
wiki mechanisms, not independent agreement. It confirms the wikis are
self-consistent and nothing else.

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

The text features need a word on leakage, because the obvious version of them
doesn't work. A dead character's opening paragraph says "was killed by", so
TF-IDF over the raw text mostly rediscovers the label — the same trap a violence
lexicon fell into earlier in this project. So the text is stripped twice before
vectorising: every sentence mentioning death *or survival* is dropped, then any
surviving death token is removed (survival words matter as much — "still alive"
is the label with the sign flipped). We checked the resulting 300-term
vocabulary contains no death or survival token, and that tense markers aren't
sneaking the label in either — "is" and "was" are English stopwords and already
excluded. The vectoriser is fit inside the cross-validation pipeline, so the
vocabulary and IDF weights never see a test document.

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
| everything | **0.744** | 0.545 |
| character + text | 0.704 | **0.547** |
| character + franchise | 0.693 | 0.483 |
| character + network | 0.687 | 0.518 |
| text only | 0.650 | 0.509 |
| franchise + network | 0.650 | 0.486 |
| character only | 0.643 | 0.487 |
| character, no wiki metadata | 0.595 | 0.501 |
| franchise only | 0.556 | 0.398 |
| baseline | 0.356 | 0.356 |

Character attributes beat franchise identity (0.643 vs 0.556), and the two are
complementary — combining them beats either. So the answer is "both, and mostly
who you are," with a large caveat: the full model loses 0.199 PR-AUC when whole
franchises are held out (0.744 → 0.545). Much of what looks like character-level
signal on a random split is the model learning per-show base rates.

**Text is the most valuable feature family we added, and the only one that
helps as much on unseen franchises as on seen ones.** Adding TF-IDF to character
attributes is worth +0.061 on a random split and +0.060 held-out, and
`character + text` is the best held-out set in the whole table (0.547) — beating
even the everything-model, which has 369 more columns. Text alone (0.650) also
beats character attributes alone (0.643). Reading the fitted coefficients shows
why it isn't just franchise vocabulary in disguise: the strongest term pushing
toward death is `antagonist` (+3.80), followed by `military`, `criminal`,
`command`, `background` and `recurring` — narrative-role words that mean the
same thing in any franchise. Being described as an antagonist is one of the
clearest death signals in the project.

Some of it *is* franchise vocabulary, to be fair. Terms pulling toward survival
include `hospital` (Grey's Anatomy), `thrones`, `dune` and `tony`, which are
franchise fingerprints, plus `category`, `relationships` and `personality`,
which are wiki section headings rather than anything about the character. That
mix is why text helps less than it might.

Fandom category features show the opposite profile: a big mover on a random
split, much weaker held-out, because they encode franchise-specific role
vocabulary ("Jedi", "Death Eater") that doesn't transfer. Network centrality is
the other family that travels: adding it to franchise metadata is worth +0.094
random and +0.088 held-out, since PageRank over a co-mention graph describes a
character's position in their own story rather than the show's vocabulary.

**An honest single number.** The table above picks the best of 27 configurations
and reports it, which is optimistic — you can't select on the same numbers you
report. For most of the project we didn't have an honest number, because we had
no held-out test set at all. We added one: an 80/20 `trainval`/`test` split
(stratified for the random regime, largest-deficit-first bin packing by
franchise for the grouped one, since franchise sizes span three orders of
magnitude), model selection by k-fold *inside* trainval only, then test scored
exactly once.

| regime | chosen | trainval CV PR-AUC | test PR-AUC | test ROC-AUC |
|---|---|---|---|---|
| random | everything / random forest | 0.734 | **0.755** | 0.851 |
| held-out franchise | everything / random forest | 0.562 | **0.663** | 0.774 |

Test came in above the CV estimate that selected it, in both regimes. We built
this expecting the opposite and would have reported it either way.

**Feature importance** (permutation, character-only forest): page length
0.155, gender 0.130, archetype 0.107, prominence tier 0.103, de-leaked field
count 0.099, billing order 0.088.

### Gender, specifically

Being female roughly halves the odds of dying: **OR 0.543, 95% CI 0.495–0.595**,
about 11.7 percentage points lower death probability after adjusting for
prominence, role, alignment, species and franchise (logistic regression with
franchise fixed effects). The unadjusted gap is 15.4 points, so a bit under a
quarter of the raw difference is explained by men holding more disposable roles.

Of 22 franchises where the effect could be estimated separately, **11 reach
p<0.05 and none point the other way.** The weak cases (Grey's Anatomy, Avatar,
The Sopranos) look like null effects rather than reversals. This was the most
stable result in the project — it moved by a few thousandths of an odds ratio
across every data-cleaning fix we made.

### What the model gets wrong

We pulled the worst 40 out-of-fold errors on trainval (never test) and looked
for patterns. One hypothesis didn't survive contact with the data: we thought
pages named by relation to someone else ("Alastor Moody's mother") would be a
distinct failure mode, but they're actually *less* likely to die than average
(28.9% vs 35.6%) and aren't overrepresented among the worst errors.

Two patterns did hold:

- **Spartacus** is 1.5% of trainval rows and 17% of the worst 100 errors, an 11x
  overrepresentation, and all 28 large errors run one direction: the model
  confidently predicts death for characters who survive. Base mortality is 71%,
  so "prominent + military + this franchise → dies" is right most of the time
  and wrong exactly for the show's leads. PageRank doesn't rescue it — Agron and
  Nasir are both highly central, and centrality can't distinguish "central and
  dies" from "central and survives."
- **Grey's Anatomy** is the mirror image: 2.5x overrepresented, and all 33
  errors run the other way, with characters who die predicted as very likely
  alive. Base mortality is 9.3%, so the model has learned "this show ≈ everyone
  lives," which is right nine times in ten and misses every one-off death.

Both are base-rate anchoring, and we don't have a feature that fixes it. We're
reporting it as a real limit on what wiki metadata can do rather than forcing a
patch that wouldn't generalise.

---

## Question 2: Does the text tell you who dies? (partly)

**Problem.** We proposed analysing episode transcripts — tf-idf and word2vec
over dialogue, plus a look at characters' last lines — to see whether language
predicts a character's fate.

**What we built instead.** We did not do the transcript version, and the reason
is alignment rather than text: we have transcripts for 1,136 episodes across 11
shows but a usable episode-level death registry for only two, and our largest
transcript set (Grey's Anatomy, 464 episodes) has nothing to align against. Most
transcripts also lack speaker labels, so attributing a line to a character is its
own unsolved problem before any modelling starts. We'd rather say that than ship
a dialogue model trained on two shows and call it an answer.

We did answer the underlying question with the text we do have per character:
TF-IDF over each character's wiki description, death-stripped as described in Q1.
That turns out to be the single most useful feature family in the project
(+0.061 PR-AUC on a random split, +0.060 held-out, and the best held-out set
overall at 0.547), and the fitted terms are interpretable — `antagonist`,
`military`, `criminal` push toward death; franchise words and wiki section
headings pull the other way.

So the honest status is: text about a character predicts their death better than
the character's structured attributes do. Whether *dialogue specifically* does is
still open, and it's the first thing we'd pick up next. `src/q2_text/` remains an
empty stub — the text features live in the Q1 pipeline, since that's where they're
evaluated.

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

**Handling label uncertainty.** For the four franchises that were mostly
unlabelled, we report three numbers: a lower bound (unparsed rows counted
alive), an upper bound (counted dead), and a point estimate. The point estimate
now uses the Haiku labels described earlier where they exist:

| franchise | unlabelled | resolved | old range | point estimate |
|---|---|---|---|---|
| The Wire | 214 | 200 (93.5%) | 10.5–99.2% | 16.3% |
| Prison Break | 155 | 143 (92.3%) | 3.8–91.3% | 21.3% |
| Boardwalk Empire | 440 | 329 (74.8%) | 16.6–99.8% | 23.2% |
| Lost | 330 | 203 (61.5%) | 20.3–69.0% | 26.6% |

Lost's filled estimate (26.6%) landing near the old flat-rate guess (27.1%) is a
reasonable check; Prison Break moving from 13.8% to 21.3% is where the flat
assumption had least reason to hold for one specific show.

**Results.** Mortality does vary a lot by franchise — from Grey's Anatomy at
9.3% to Spartacus at 73.3% among franchises with at least 100 characters. But
franchise *properties* explain none of it. No term in the model reaches
significance: run span is the closest at p=0.185, and medium, number of titles,
first release year and audience rating are all further away. **R² = 0.079, and
adjusted R² = −0.065** on n=38 — worse than predicting every franchise at the
overall mean.

The correction that produced this null came from the measurement rather than
from the model. `run_span` was the
spread of TMDB *release* years across a franchise's titles, which is fine for a
film series and meaningless for a television one: a show is a single title, so
Grey's Anatomy came out as spanning zero years despite running from 2005 to
2026, as did Lost, The Wire and thirteen others. Every TV franchise sat at zero
and the variable was effectively measuring "is this a multi-film franchise".
Reading `last_air_date` from TMDB instead moved run span from p=0.022 to
p=0.185 and took adjusted R² from 0.047 to below zero. The finding it removed
was the only significant one we had.

k-means doesn't recover mortality groupings either. Best silhouette is k=2, and
that split mostly separates large multi-title film franchises from everything
else, landing at 33.3% and 37.7% mortality — not a meaningful gap given the
label uncertainty above.

So the honest answer to Q3 is negative: franchises differ enormously in how
deadly they are, and the metadata we can get about a franchise doesn't tell you
which ones. Whatever drives it (tone, genre convention, showrunner choice) isn't
in TMDB's fields.

---

## Impediments

**A leak hiding inside a feature.** Infobox field count was one of our two
strongest features, and part of why was circular: a character who dies acquires
"Cause of death" and "Date of death" fields they otherwise wouldn't. Measured
over 63,975 labelled pages, the dead-minus-alive gap in field count was +1.78
raw; dropping death-named fields cuts it to +0.81. **55% of that feature's
separating power was the outcome written back into the input.** Removing it cost
us accuracy across the board, which is what a correction to a leak looks like.

This shapes what we can claim. Our two strongest features (page length, field
count) measure how much fans wrote about a character, and death causes fan
attention. So these results are **descriptive, not predictive**: they describe
what separates characters who died from those who didn't in wikis written after
the fact, not whether a death could have been called in advance. Network
centrality is the one family that mostly escapes this, and it's also the one
that transfers best across franchises.

**A scraper bug that quietly halved four wikis.** Page discovery asks each wiki
which pages transclude a configured infobox template. Transclusion follows
template redirects, but our parser matched the template name literally in the
wikitext, so any page using a redirect was found and then silently thrown away.
On the Matrix wiki, where pages write `{{Resistance character infobox}}` for
what redirects to `Infobox character`, we kept 18 of 500+ pages. Three other
wikis had the wrong template configured outright. After fixing: Matrix 18 → 495,
Dexter 262 → 1,125, The Boys 70 → 427, Peaky Blinders 14 → 61.

**A variable that was wrong for half the dataset.** Two of our fields were
computed in a way that happened to be correct for one kind of row and silently
wrong for another, and both survived every aggregate check we ran. `run_span`
took the spread of TMDB release years across a franchise's titles, which is
right for a film series and meaningless for a television one, since a show is a
single title: fifteen TV franchises were recorded as spanning zero years,
Grey's Anatomy among them. Gender was inferred from pronouns only when the
infobox field was *missing*, never when it was present but unparseable, so the
Dexter wiki — which writes an unparseable value on 1,058 of its 1,077 pages —
had no usable gender at all while its pages carried plenty of pronouns.

The general shape is worth stating because it is the one we would warn a reader
about: a field that is broken for an entire subgroup does not look like missing
data, it looks like a real value. Neither of these raised a null-rate flag.
`affiliation_alignment` and `n_relatives` have the same problem and we have not
fixed them — they are absent for 12 and 7 franchises respectively, covering 20%
and 37% of characters, because those wikis simply do not use those infobox
fields. Under grouped cross-validation this matters directly, since a held-out
franchise can be missing a feature the model learned to lean on.

---

## Conclusion

Character attributes carry more signal about death than franchise identity does,
and the two combine: 0.643 PR-AUC from character features alone against 0.556
from the franchise, and 0.744 from everything together, against a 0.356 base
rate. On a proper held-out test set the full model reaches **0.755 PR-AUC /
0.851 ROC-AUC** on a random split and **0.663 / 0.774** when whole franchises
are held out. The most useful single addition was text: TF-IDF over each
character's death-stripped wiki description, which on its own beats their
structured attributes and is the only feature family that helps as much on an
unseen franchise as on a familiar one. The gender effect is real and stable —
women are about half as likely to die, adjusted, consistent across 11 of 22
franchises with no reversals. Franchise-level properties, by contrast, explain
none of why some shows kill more than others: no term in the model reaches
significance and adjusted R² is negative.

Most of the work here went into finding out that our own numbers were wrong: a
feature that was partly restating the label, a population that included talking
portraits and background extras, four franchises whose mortality was
unmeasurable, a scraper that silently discarded most of four wikis, and a
franchise-level variable that recorded every television series as lasting no
time at all. None of these was visible in an aggregate metric — each one was
found by looking at individual rows, and each one had been quietly improving a
result. Every correction moved a number down, and one franchise-level result
disappeared entirely. We are more confident in what is left.
