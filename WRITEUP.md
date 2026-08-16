# Spoiler Alert: Who Makes It to the Finale?

**A Needle in a Data Haystack (67978) — Final Project, Group #56**

| Name | Email | CS id |
|---|---|---|
| Stav Abukarat | stav.abukarat@mail.huji.ac.il | stava |
| Shahar Spencer | shahar.spencer@mail.huji.ac.il | shahar.spencer |
| Bar Dalal | bar.dalal@mail.huji.ac.il | bar.dalal |
| Ido Oron | ido.oron@mail.huji.ac.il | ido_oron |

Demo: **https://shaharspencer.github.io/needle-group-project/**

## Domain

We study whether a fictional character dies on screen. The unit of observation
is one character in one franchise, and the label is binary: does the character
die at some point in the story as aired. We predict it from the character's own
attributes, the franchise they belong to, where they sit in their story's cast,
and how fan wikis describe them.

## Data

We scraped 38 Fandom wikis (Game of Thrones, the MCU, Star Wars, Harry Potter,
Breaking Bad and 33 more), giving 68,463 pages and about 262 MB of raw JSONL.
From the TMDB API we pulled 280 titles and 28,151 cast credits for billing
order, episode counts, genre and rating. Two Kaggle death registries and
`listofdeaths.fandom.com` are used only for validation.

The label `is_dead_v2` comes from infobox fields
(`src/labels/rebuild_label.py`): 30 wikis give a status string, eight a date
of death instead. Characters whose status will not parse are dropped from
Q1. For Q3 only, 1,858 such characters, concentrated in four franchises, were
labelled by Claude Haiku 4.5 with "can't tell" allowed (123 dead, 1,223
alive, 512 undecided).

### How good are the labels?

Every annotation pass was run by Haiku 4.5 against a written rubric
(`src/labels/ANNOTATION_RUBRIC.md`), coded twice independently over a
stratified sample of 600 pages. With no human coders, these figures measure
how repeatable the coding scheme is. We report Cohen's kappa rather than raw
agreement because when one label dominates, two coders can agree most of the
time just by both defaulting to it; kappa subtracts out that chance agreement
and raw agreement does not.

Our first rubric left four cases undecided, the largest being historical
figures played by actors in the story. Writing those rules down and
re-coding moved kappa on "does it die" from 0.38 to 0.67, and on "is it a
character" from 0.26 to 0.53 (Figure 2); raw agreement moved much less, 0.70
to 0.84.

Kappa between `is_dead_v2` and an annotation pass is 0.86 (n = 296, first
pass) and 0.65 (n = 405, second); we report both since the larger sample
gives the lower number, but either way this checks a model against the same
infoboxes, so the external check matters more: against
`listofdeaths.fandom.com`, a different wiki with different editors, 197 of
202 matched characters agree for Game of Thrones and The 100
(`src/labels/validate_registry.py`). The registry only lists characters who
died, so this measures recall, not our false-positive rate.

### Data quality

Most scraped pages are not on-screen characters: reweighted by stratum, the
corpus is 41% on-screen characters, 46% novel or reference-only, and 12% not
characters at all (Figure 1). We keep a page only if its infobox names an
actor or we matched it to a TMDB cast credit; against the annotated sample
that rule scores precision 0.79, recall 0.77. We also removed 168 object
pages (portraits, dead by definition) and 1,122 unnamed background roles.

Our final, clean analysis population is 15,686 named on-screen characters, of
whom 35.6% die. We watched out for one franchise dominating that population:
Star Wars is 56% of the raw scrape and still 12% after filtering, so we report
every Q1 result under grouped cross-validation as well as random.

## Q1: Do character attributes predict death better than franchise identity?

### Problem description

Does franchise identity add anything beyond the character, or does it just
substitute for it? And does a character's position in the story's social
structure matter beyond their own attributes?

### Our solution

We trained logistic regression, a decision tree and a random forest on 12
feature sets (`src/q1_character/models.py`), built from franchise identity,
character attributes, network centrality, community position and TF-IDF text,
plus combinations of those (`FEATURES.md` lists every feature). The largest
adds 369 one-hot columns from Fandom categories and infobox fields combined.

All three models use `class_weight="balanced"`, since only 35.6% of
characters die and without it a model can score well by getting survivors
right while ignoring deaths. We left logistic regression at C = 1.0, since
tuning it per feature set would confound the model with the features we are
comparing. The forest uses 400 trees, where the out-of-fold score stopped
moving.

The tree settings follow the supervised learning lecture's two overfitting
remedies, pre- and post-pruning, tried against an unpruned tree under grouped
CV. Unpruned, the tree needs 4,315 leaves at depth 36 for 0.375 PR-AUC,
barely above the 0.356 base rate. Pre-pruning (`max_depth=6`,
`min_samples_leaf=50`) reaches 0.462 with 43 leaves; post-pruning
(`ccp_alpha=0.001`) reaches 0.475 with 17. The two remedies are within fold
noise of each other on score, so we choose on size: post-pruning wins.

To capture where a character sits in the cast, we built a co-mention graph
per franchise from page text: an edge runs from A to B when every name token
of B appears on A's page, weighted by whether the mention goes both ways.
From it we took PageRank, HITS hub and authority scores, and in- and
out-degree. We partitioned each graph into communities using Louvain, which greedily
maximises modularity in O(n log n); we ruled out Girvan-Newman on cost, since
it recomputes edge betweenness after every removal and our Star Wars graph
has 42,268 nodes. Because Louvain's result depends on node order, we reran
every franchise at five seeds: membership agreement was 92% and modularity
varied by at most 0.008.

It finds 1,885 communities across the 38 franchises, with median modularity Q
of 0.426 and 36 of 38 clearing the Q > 0.3 threshold the lecture treats as
real structure; Peaky Blinders and Supernatural fall below it, so we treat
their community columns as noise. From each partition we take community size,
its share of the cast, embeddedness (how much of a character's co-mention
weight stays inside their own community) and degree rank, but not the
community id itself, since one-hot encoding an arbitrary integer would let the
model memorise per-franchise identity.

To prevent leakage, we drop every sentence that mentions death or survival, fit
the TF-IDF vectoriser inside each CV fold so no test vocabulary reaches
training, and then check the resulting 300-term vocabulary against the same
pattern that did the stripping (`src/q2_text/coefficients.py`).

### Evaluation criteria

We compare every model against two dummy classifiers that do no learning: one
always predicts "survives", the majority outcome, and the other guesses at
random in proportion to the base rate. Both score 0.356 PR-AUC, so any model
that cannot beat 0.356 has learned nothing. We report PR-AUC first and
ROC-AUC second, since accuracy would reward the majority dummy with 64.4% at
this base rate, while PR-AUC's chance level is the base rate itself and reads
directly against these baselines.

### Setup

We ran two experiments. The first is 5-fold CV over all 15,686 characters
under random CV (characters from one franchise fall in both train and
validation) and grouped CV (`GroupKFold` on franchise, holding out whole
franchises so the model is scored on one it has never seen). The second
takes the best CV configuration (all features, random forest), refits it with
the same hyperparameters on an 80% trainval split, and scores it once on the
untouched 20% test split (`src/q1_character/evaluate.py`).

### Results

In Figure 3 you can see PR-AUC for all 12 feature sets under both split rules.
Character attributes outscore franchise identity, 0.633 against 0.556, and the
two together beat either alone. The full model loses 0.189 moving from random
to grouped CV, so much of its random-CV score was memorised per-show base
rates. Franchise identity collapses to 0.398 on an unseen franchise, barely
above the 0.356 baseline: its one-hot column is all zeros there, leaving only
genre, medium, rating and year. Community position adds 0.024 over character
attributes alone, but only 0.004 once centrality is already in the model.

Ranking feature sets by a single number does not tell us whether the ordering
would survive a different fold assignment, so we rescored every set per fold
and compared each against character-only with a paired Wilcoxon signed-rank
test (`src/q1_character/significance.py`). Five paired folds cannot produce a
p-value below 0.0625, so we read the fold win count and the spread instead.
That spread is 0.009 to 0.013 across random folds and 0.034 to 0.082 across
grouped ones, wider than most of the gaps in the figure.

The full ten-set breakdown is in `q1_significance.csv`; the extremes are
everything (+0.077 grouped, 5/5 folds; +0.111 random, 5/5) and franchise only
(-0.104 grouped, 1/5; -0.085 random, 0/5). Six sets beat character-only on
all five random folds; only two manage it under the grouped rule, and every
network or community variant falls to three folds out of five. Several sets,
including community-only (-0.044 grouped) and franchise-only, come out
negative. We treat the community contribution to prediction as suggestive,
not established.

Louvain builds each partition from co-mentions alone, with no access to who
lives or dies, so we can check afterward whether membership lines up with
fate anyway (`src/q1_character/community.py`). Per franchise we compute

  D = P(same fate | same community) − P(same fate | different community)

against 2,000 within-franchise label shuffles, which hold the graph and the
death count fixed and only move who dies. The statistic is the pairwise
community-membership measure from class, applied to fate. We need the shuffle
because a positive D on its own could just be the base rate.

Thirteen of 33 franchises exceed the null at p < 0.05, largest in serialised
drama, led by Lost (D = 0.109, z = 15.1) and Breaking Bad (0.115, z = 9.3),
ahead of Dune, Game of Thrones and The 100. The cast-weighted mean across all
franchises is +0.012, well below any of those per-franchise highs.

The permutation p-value is one-sided and only ever flags positive clustering;
four franchises (the MCU, Grey's Anatomy, Ozark and Spartacus) have z-scores
below -2.3, which we read as significant in the opposite direction even
though this test does not flag them. We apply no correction for testing 33
franchises at once, and one of the thirteen, Supernatural, has a partition we
already called noise (Q = 0.234).

Neither modularity (Figure 5, right panel) nor fragmentation explains which
franchises land negative: fragmentation fits Ozark (81 communities for 146
characters) and Grey's Anatomy, but not the MCU, our least fragmented
franchise (21 communities for 3,489 characters). The four negative franchises
share no obvious genre, medium or mortality rate, from Spartacus at 72.5% to
Grey's Anatomy at 9.3%; some of this is likely just the multiple-comparisons
noise noted above.

On the held-out test, random split gives 0.756 PR-AUC / 0.849 ROC-AUC and
grouped gives 0.667 / 0.772, both above CV, so selection is unlikely to have
overfit, though with only 38 franchises the grouped figure depends heavily on
which ones landed in the test half.

To see which features carry the signal we shuffled each one in turn and measured
how far PR-AUC dropped. On the character-only forest that ranks page length
0.167, archetype 0.111, gender 0.107, de-leaked field count 0.101, prominence
tier 0.101 and billing order 0.092; the top two are both measurements of the
wiki page, not of the character. We measured on the fitted model instead of
out of fold, which is known to flatter continuous features like page length,
so the ordering is safer to trust than the values. `prominence_tier` is
itself derived from page length, so the shuffle splits credit between the two.

On gender (`src/q1_character/effects.py`), women have roughly half the odds of
dying (OR 0.564, 95% CI 0.516 to 0.616, about 11.1 percentage points), adjusted
for billing order, archetype, alignment, whether the character is human, and
franchise. The unadjusted gap is 14.5 points, so the covariates we adjust for
account for under a quarter of it. Archetype plausibly sits downstream of
gender rather than confounding it, since roles are not assigned evenly by
gender; if so, the adjusted number understates gender's total effect rather
than cleaning it. Of 22 franchises where the effect is estimable, none
reverse direction (Figure 7).

### Visualization

Figure 3 (`q1_model_comparison.png`) is the feature-set comparison, one panel
per split rule. Figure 4 (`q1_fold_spread.png`) shows the five folds behind
each bar, since means alone hide how much the grouped folds disagree. Figure
5 (`q1_community_fate.png`) gives fate concentration D per franchise against
its null, and against modularity to show the two are unrelated. Figure 6
(`q1_feature_importance.png`) is permutation importance, coloured by whether
the feature measures the character or the wiki page. Figure 7
(`q1_gender_forest.png`) is a forest plot of per-franchise gender odds ratios.
Figure 8 (`q1_cost_curves.png`) plots weighted loss against threshold at six
cost ratios, with the threshold each franchise would pick for itself beside
it.

### Impediments

Ranking out-of-fold errors turned up two franchises failing oppositely:
Spartacus gives 14% of the worst 100 errors from 1.4% of the data, almost all
false positives at 71% base mortality, where centrality cannot separate
anyone; Grey's Anatomy is overrepresented 2.5 times, almost all false
negatives against its 9.3% base rate. Scoring a threshold by `w1 x FN + FP`,
where w1 is how many false alarms one missed death is worth
(`src/q1_character/costs.py`), the best threshold moves from 0.80 at
w1 = 0.5 down to 0.13 at w1 = 10. At w1 = 3, Spartacus's own threshold (0.05)
and Grey's Anatomy's (0.64) diverge sharply from the default 0.50. We left
this uncorrected, since per-franchise thresholds need a labelled sample from
each new franchise, which grouped CV assumes we have none of.

## Q3: Are some franchises deadlier?

### Problem description

Do mortality rates differ by franchise, and do franchise properties (genre,
medium, rating, run length) explain the difference?

### Our solution

A binomial GLM that treats every character as an independent trial returned
p < 1e-10 on everything, which is too good to be true: a franchise's 3,647
MCU characters are not independent draws, so the model is wildly overconfident
(Pearson χ²/df = 36.6, versus an expected value near 1). We instead collapsed
to one row per franchise and regressed `logit(mortality)` on release year,
run span, title count, medium and rating, weighted by cast size
(`src/q3_franchise/mortality.py`). That drops us to 38 observations and costs
most of our power. A mixed-effects or quasibinomial model would have kept the
character-level rows while still correcting for the clustering, but we did
not try one. Cast size is also a rougher precision weight than the correct
n·p̂·(1−p̂) for a logit-transformed proportion.

Run span is measured as first to last air date. A release-year-based version
conflates multi-film franchises with long-running TV shows and gives a
spurious p = 0.022; testing both definitions and keeping the one that changes
significance is itself a caveat on the p = 0.185 we report from just 38
observations.

The eight properties we cluster on exclude mortality itself, since clustering
on the outcome would give clusters that differ in mortality by construction.
We standardised them first: `first_year` runs in the thousands and
`female_ratio` in [0,1], so unstandardised, Euclidean distance would mostly be
measuring release year. K came from ten k-means++ seeds, chosen by the
silhouette score; we also tracked average distance to centroid as a second
diagnostic but did not use it to pick k.

### Evaluation criteria

We count a franchise property as explaining mortality if its coefficient
reaches p < 0.05 with the franchise as the unit of analysis, and if the model
beats predicting the mean on adjusted R². The comparison for the clustering is
whether a partition built without mortality separates franchises that differ in
it.

### Setup

The Q3 population is 17,584 on-screen named characters, wider than Q1's
15,686, because dropping unlabelled characters from a *rate* would bias it. For
the four mostly-unlabelled franchises we report a range, treating unparsed
characters as alive for a lower bound and as dead for an upper bound, plus a
point estimate from the Haiku-resolved labels. Every estimate carries a Wilson
interval, which behaves well at small counts or extreme rates unlike a normal
approximation.

### Results

Among the 27 franchises with reliable status parsing, mortality runs from 9.3%
(Grey's Anatomy) to 72.6% (The 100), with The Walking Dead at 79.5% on 44
characters. No franchise property is significant. Run span comes closest at p =
0.185. R² is 0.079 and adjusted R² is −0.065, which means the model does worse
than just predicting the average mortality for every franchise.

The four mostly-unlabelled franchises carry the widest bounds and weakest
point estimates (Figure 11), from The Wire (14.3-99.2%, point estimate
16.3%) to Lost (23.2-69.0%, 26.6%).

The silhouette peaks at k = 2 (0.428, with 92.5% of franchise pairs assigned
consistently across ten seeds). The elbow curve supports no k at all, falling
smoothly from 2.41 at k = 2 to 1.21 at k = 8 with no bend, so the two selection
rules disagree about whether there is any structure to find.

For internal validation we used the three measures from the clustering
lecture; the k = 2 partition holds up on all of them: cohesion 2.82 against
separation 5.60, a correlation of −0.65 between the distance matrix and the
same-cluster adjacency matrix (negative is what we want, since same-cluster
pairs should be the close ones), and 94.7% pair agreement with Ward
agglomerative clustering cut at the same k.

The external check is weaker than it appears. Scored against `medium`, which
was not a clustered column, the partition gives weighted purity 0.684 and
entropy 0.804; the six-franchise cluster comes out pure film. But
`n_titles` is a clustered column, and film franchises here average 14.1
titles against 1.3 for TV, so `medium` is largely recoverable from what we
clustered on; that agreement is close to guaranteed and should not count as
external evidence.

The split separates six large multi-title film franchises from the other 32,
at 35.4% and 37.1% mortality, a difference within noise: the grouping the
data supports reflects franchise size and medium, independent of mortality.

### Visualization

Figure 11 (`q3_franchise_mortality.png`) gives mortality per franchise, each
bar carrying a Wilson interval and a band for the label bounds; the four very
wide bands are the franchises we could not label. Figure 12 (`q3_run_span.png`)
plots mortality against run span with marker size proportional to cast size,
and its flat fit line is the null result. Figure 13
(`q3_cluster_diagnostics.png`) is the elbow and silhouette curves with the Ward
dendrogram beside them.

### Impediments

The pseudo-replication and run-span-definition problems are described above.
The remaining one is coverage: four franchises have almost no parseable
status, so we can only give a range for them (The Wire: 14.3-99.2% on
infobox evidence alone). The Haiku labels narrow that, but those four stay
the weakest rows; re-scraping would not help, since those wikis do not carry
the fields.

## Interactive demo

**"Plot Armor Index"** — https://shaharspencer.github.io/needle-group-project/

The demo is a browser-only logistic regression over user-selectable
attributes, exported by `src/app/export_demo.py`, using fewer features than
the full Q1 model. One tab looks up a held-out test character and breaks its
score down per feature; the other lets you build a character by hand and
watch the risk update.

## Future work

- Rebuild the feature matrix from what a viewer could have known partway
  through a series, since everything we use now comes from wiki text written
  after the story ended.
- Speaker attribution across the 1,136 transcripts we hold would let the text
  feature draw on what characters say, not just what is written about them.
- Passing the cost matrix into the loss would let the model learn a different
  boundary, instead of moving the threshold after fitting.
- Two people coding the same 600 pages would separate rubric ambiguity from
  the model being inconsistent with itself.
- Clique percolation would let bridge characters belong to several
  communities, instead of the one Louvain assigns them.

## Conclusion

Character attributes carry more signal than franchise identity (0.633 PR-AUC
against 0.556, 0.744 combined, on a 0.356 base rate). The best model holds up
on a held-out test: 0.756 PR-AUC and 0.849 ROC-AUC, falling to 0.667
and 0.772 when whole franchises are held out instead. Text is the most useful
single addition and the one that transfers best. None of the five franchise
properties we tested came out significant, and the adjusted R² is negative,
so that model does worse than just predicting the average.

Wikis get written after the story ends, by fans who write more about the
characters they liked, and that limits our two strongest features: page
length and infobox field count both partly measure attention rather than the
character itself, and dead characters draw more of it. Features built from
the story's own structure are less exposed to this bias and hold up better on
a franchise the model has not seen, which is why we report grouped
cross-validation everywhere alongside random.