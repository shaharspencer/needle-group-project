#!/usr/bin/env node
/*
 * Scan a markdown file for the writing patterns that mark machine-drafted prose.
 *
 *   node tools/check_writing.js WRITEUP.md
 *
 * Wraps the detector from the avoid-ai-writing skill (MIT, see
 * tools/LICENSE-avoid-ai-writing). The skill itself lives in
 * .claude/skills/avoid-ai-writing/ and is what an agent loads when asked to
 * rewrite; this script is the deterministic check, so a claim that the prose
 * was cleaned up can be verified instead of taken on trust.
 *
 * Exit code 1 when the score exceeds THRESHOLD, so it can gate a commit.
 */
'use strict';

const fs = require('fs');
const path = require('path');
const detector = require('./ai_patterns.js');

const THRESHOLD = 20;

// The assignment allows six pages, figures excluded. Tables and headings count.
const PAGE_LIMIT = 6;

// Rough render model for 11pt single-spaced with 1in margins: a wrapped prose
// line holds about 95 characters and a page holds about 46 lines. A table row
// and a heading each occupy their own line, and headings carry a blank line.
// Crude, so treat it as a warning to check the exported PDF, not as truth.
const CHARS_PER_LINE = 95;
const LINES_PER_PAGE = 46;

function estimatePages(text) {
  const lines = text.split('\n');
  const tableRows = lines.filter((l) => l.startsWith('|')).length;
  const headings = lines.filter((l) => l.startsWith('#')).length;
  const bullets = lines.filter((l) => l.trim().startsWith('- ')).length;
  // The assignment excludes figures from the page limit, so embedded images
  // and their captions do not count. Captions run until the next blank line.
  const isImage = (l) => l.trim().startsWith('![');
  const kept = [];
  let inCaption = false;
  for (const l of lines) {
    if (isImage(l)) { inCaption = false; continue; }
    if (/^\*Figure \d/.test(l.trim())) { inCaption = !l.trim().endsWith('*'); continue; }
    if (inCaption) { inCaption = !l.trim().endsWith('*'); continue; }
    kept.push(l);
  }
  const prose = kept.filter((l) => !l.startsWith('|')).join('\n');

  const rendered =
    prose.length / CHARS_PER_LINE + tableRows + headings * 2 + bullets * 0.3;
  return { pages: rendered / LINES_PER_PAGE, tableRows, headings };
}

// FEATURES.md is a filename in this repo. The detector reads bare "features" as
// the inflated verb ("this release features...") and flags it, which is right
// in general and wrong here.
const FALSE_POSITIVES = [
  (issue) => issue.type === 'tier1-clarity' && /^FEATURES$/i.test(issue.text),
];

function main() {
  const target = process.argv[2];
  if (!target) {
    console.error('usage: node tools/check_writing.js <file.md>');
    process.exit(2);
  }

  const text = fs.readFileSync(target, 'utf8');
  const result = detector.analyzeText(text);

  const issues = result.issues.filter(
    (issue) => !FALSE_POSITIVES.some((rule) => rule(issue))
  );
  const suppressed = result.issues.length - issues.length;

  const size = estimatePages(text);
  const over = size.pages > PAGE_LIMIT;

  console.log(`${path.basename(target)}  ${result.stats.wordCount} words`);
  console.log(`  ~${size.pages.toFixed(1)} pages `
    + `(limit ${PAGE_LIMIT}, figures excluded)${over ? '  OVER' : ''}`);
  console.log(`  ${size.tableRows} table rows, ${size.headings} headings, `
    + 'both of which count toward the limit');
  console.log(`  score ${result.score}/100 (${result.label})`);
  console.log(`  classified ${result.document_classification}, `
    + `${result.confidence_category} confidence`);
  console.log(`  tier1 ${result.stats.tier1Count}  `
    + `tier2 ${result.stats.tier2Count}  tier3 ${result.stats.tier3Count}`);

  if (!issues.length) {
    console.log('\n  no issues');
  } else {
    console.log(`\n  ${issues.length} issue(s):`);
    for (const issue of issues) {
      console.log(`    [${issue.severity}] ${issue.type}: ${issue.text}`);
      if (issue.suggestion) console.log(`        -> ${issue.suggestion}`);
    }
  }
  if (suppressed) {
    console.log(`\n  ${suppressed} known false positive(s) suppressed`);
  }

  process.exit(result.score > THRESHOLD ? 1 : 0);
}

main();
