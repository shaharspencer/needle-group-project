# Annotation rubric (v2)

The coding scheme used for every annotated sample in this project: the 600-page
corpus-composition sample (`data/gold/`) and the unlabelled-status fill
(`data/gold/fill_unlabelled/`). Both passes of both samples code against this
file and nothing else.

Both runs need the same written scheme before their answers can be compared.
Otherwise, disagreement may come from different instructions rather than from
the pages themselves.

## Version history

**v1** was reconstructed after the fact from pass 1, which had been coded
against an undocumented scheme. Two Haiku runs using v1 reached kappa 0.26
on `is_character` and 0.51 on `appears_onscreen`, both below the 0.6 the
evaluation lecture treats as a usable floor. Reading the 338 disagreeing rows
showed the disagreement was concentrated in four cases v1 left open rather than
spread over hard rows.

**v2** resolves those four cases in the sections marked *Adjudication* below.
Its repeatability is measured using two fresh, separate Haiku runs (passes 3
and 4), both using this file.

## What the coder sees

Six columns, from `data/gold/chunks/chunk_NN.csv`:

| column | meaning |
|---|---|
| `char_id` | opaque id; echo it back unchanged |
| `name` | wiki page title |
| `franchise` | which franchise the wiki belongs to |
| `infobox_fields` | selected infobox keys and values, as scraped |
| `death_sentences` | sentences from the page that mention death |
| `text_excerpt` | first 1000 characters of the page |

The existing automatic label (`auto_is_dead`) and the parsed status field are
**withheld**. The coder judges the evidence, not the label under test.
Otherwise the evaluation measures how persuasive the label is rather than
whether it is right.

## Labels

Three labels per row. Code each one independently; do not let an answer to one
imply an answer to another beyond what the rules below state.

### `is_character`: 0 or 1

`1` when the page is about **one individual being** with an identity in the
story: a person, a named animal, a droid, an alien, a sentient object.

`0` when the page is about anything else:

- a species, breed, or creature *type* ("Dunkleosaurus", "Wookiee", "Dalek")
- a vehicle, weapon, location, organisation, faction, or event
- a member of the production: an actor, writer, director, or composer, credited
  as themselves
- a disambiguation, list, category, or other meta page

A page about a group of characters is `0`. A page about one member of that
group is `1`.

**Adjudication 1: real people depicted inside the fiction.** A historical or
otherwise real person who is *portrayed within the story* is `1`. Aethelflaed
in *Vikings* and Brilliant Chang in *Peaky Blinders* were real people, and both
are played by an actor and take part in the plot, so both are characters in the
sense this project measures. The distinction is participation in the narrative,
not whether the person also existed. A real person named only in passing, or
credited as themselves, stays `0`.

**Adjudication 2: unnamed individuals.** A page about one unnamed individual
("Sarah's lover", "Unidentified Ugnaught worker") is `1`. The page describes a
single being, which is what this label asks about. Whether unnamed roles belong
in the analysis is decided later and separately, by `is_unnamed_role` in
`src/constants.py`; the coder should not anticipate that filter.

### `appears_onscreen`: 0 or 1

`1` when the character appears in **filmed screen media** of the franchise:
live action or animation, film or television. Evidence includes a cast credit,
a "portrayed by" / "played by" infobox field, or text describing appearances by
episode or film.

`0` when the character exists only in print or interactive media (novels,
comics, short stories, reference guides, video games), or is mentioned in
dialogue but never shown.

Code `0` when `is_character` is `0`; a species does not appear on screen in the
sense this label measures.

Being on screen briefly still counts. The question is whether the character was
filmed, not how long for.

**Adjudication 3: what to do when the evidence is silent.** Most wiki pages
carry no cast credit, because most infoboxes are incomplete rather than because
the character was never filmed. Absence of a "portrayed by" field is therefore
not evidence for `0`.

Code `0` only on *positive* evidence that the character is confined to print or
interactive media: the text names a novel, comic, game, or reference guide as
the character's only appearance, or the franchise itself has no filmed
component. When the franchise is a filmed one and the text describes the
character taking part in episodes, films, or scenes, code `1` even with no cast
credit shown. When neither holds, fall back on the franchise: code `1` for a
franchise whose material is predominantly filmed.

### `dies_in_narrative`: 0, 1, or -1

`1` when the character **dies within the story as aired**, whether or not the
death is shown on camera.

`0` when the character is alive at the end of the material covered, or the
material gives no indication of death.

`-1` when the evidence genuinely cannot settle it. Use this rather than
guessing. Legitimate cases:

- the page describes a death but the character is later resurrected, restored,
  or revealed to have survived, and the page does not say which state is final
- the character is presumed dead, missing, or of unknown fate
- the excerpt is too short or too garbled to support a judgement

A character who dies and stays dead in an afterlife, vision, or flashback is
`1`. A character who is "killed" in a dream sequence, simulation, or alternate
timeline that the story does not treat as real is `0`.

**Adjudication 4: whose death the evidence describes.** The `death_sentences`
column collects every sentence on the page that mentions death, including
deaths of other people. A sentence saying the character killed someone, avenged
someone, or mourned someone is not evidence that the character died.

Code `1` only when the text states or clearly implies that *this* character
died. An empty or absent `death_sentences` column is evidence for `0`, not for
`-1`: the page had nothing to say about this character dying. Reserve `-1` for
pages that raise the question and leave it open, as listed above.

## Confidence

One of `high`, `medium`, `low`, describing how well the shown evidence supports
the three labels. `low` confidence is not the same as `-1`: `-1` means the
evidence is genuinely ambiguous, `low` means it is thin.

## Output

One JSON object per input row, one row per line, in `char_id` order:

```json
{"char_id": "e6288a6c8d09", "is_character": 0, "appears_onscreen": 0,
 "dies_in_narrative": 0, "confidence": "high",
 "rationale": "Creature type in video game, not individual character",
 "model": "claude-haiku-4-5-20251001", "prompt_version": "p1"}
```

`rationale` is one short clause. It is not scored, but it makes disagreements
between passes readable, which is the only practical way to tell a coding-scheme
problem from a hard row.

Set `prompt_version` to the rubric version the pass was coded under, so a later
reader can tell which passes are comparable.

## Passes

| pass | rubric | purpose |
|---|---|---|
| 1 | undocumented | the original labels, behind the corpus-composition figures |
| 2 | v1 | first reliability attempt; exposed the four open cases |
| 3 | v2 | repeatability of the documented scheme, Haiku run A |
| 4 | v2 | repeatability of the documented scheme, Haiku run B |

Haiku runs are separate and never see another pass's output. Agreement between
two passes measures model repeatability, not human inter-rater reliability or
correctness. `src/labels/reliability.py` reports Cohen's kappa per label;
comparison with a human-edited external source is handled separately by
`src/labels/validate_registry.py`.
