# directories-to-data

A pipeline that converts scanned historical city directories into structured, queryable
data using Gemini. Built for the Fondren Fellows "Directories to Data" project; source PDFs
for the two editions currently processed were provided by project mentors Sean Smith and
Norie Guthrie.

This document is aimed at a researcher who wants to run the pipeline on a **new** city
directory edition — a different year, a different city, or a different publisher.

## What This Is

Traditional approaches to digitizing historical directories — see Albers & Kappner (2023),
whose validation methodology this project's own accuracy measurement mirrors — pair a
trained OCR/handwriting-recognition model with hand-coded parsing rules (usually regex) that
turn a recognized line of text into structured fields. Both halves are typically
edition-specific: a new typeface, print quality, or entry format can mean retraining the
recognition model and rewriting the parsing rules.

This pipeline uses a vision-language model (Gemini) for both halves at once. A single
model call reads the page image and returns either verbatim text (stage 1) or structured
fields (stage 3), and adapting to a new edition's quirks is a matter of editing plain-language
prompt instructions, not retraining anything or writing new regex. In this repo, that's
exactly how the two editions processed so far were reconciled — see
[What Varied Across Editions](#what-varied-across-editions) below.

The honest tradeoff:

- **Cost model.** A trained OCR/regex pipeline is a one-time engineering cost with near-zero
  marginal cost per page afterward. This pipeline pays a per-page (stage 1) and per-batch-of-
  lines (stage 3) API cost on every run, for every edition, indefinitely.
- **Determinism.** A regex-based parser is fully deterministic — the same input always
  produces the same output. This pipeline is not: even at `temperature=0`, the model shows
  run-to-run variation on ambiguous characters (see
  [Known Limitations](#known-limitations)).
- **Adaptability.** In exchange, a new edition's format quirks are fixed by editing a prompt
  string and re-running, not by retraining a model or rewriting a parser.

## Pipeline Stages

Five stages, numbered 0-4. Each stage's output is the next stage's input. **Every stage is
resumable** — re-running skips work that's already done rather than redoing it (stage 1 skips
any page that already has an output file; stage 3 checkpoints by batch index and resumes
unless `--restart` is passed).

| Stage | Script | Reads | Writes |
|---|---|---|---|
| 0 | `scripts/00_extract_abbreviations.py` | One or more pages of the source PDF | `config/<year_vol>/abbreviations.json` |
| 1 | `scripts/01_ocr_entries.py` | Source PDF pages, `config/<year_vol>/abbreviations.json` (required) | `data/<year_vol>/pages/page_XXXX.json` (one file per page) |
| 2 | `scripts/02_merge_pages.py` | `data/<year_vol>/pages/*.json` | `data/<year_vol>/<year_vol>_entries.json` |
| 3 | `scripts/03_parse_entries.py` | `data/<year_vol>/<year_vol>_entries.json` | `data/<year_vol>/<year_vol>_parsed.json`, `data/<year_vol>/<year_vol>_parse_failures.json` (only if something failed to parse) |
| 4 | `scripts/04_validate_sample.py` | `data/<year_vol>/pages/*.json`, then hand-typed worksheets | `validation/<year_vol>/manual/*.txt`, `validation/<year_vol>/sample_pages.json`, `validation/<year_vol>/validation_report.json` |

**Stage 0 — bootstrap the abbreviations key.** A new edition needs
`config/<year_vol>/abbreviations.json` before stage 1 can run at all (stage 1 hard-fails
without it — the abbreviation list is injected into its prompt so the model can disambiguate
characters). Run once per edition:

```bash
python scripts/00_extract_abbreviations.py 1892_93 data/1892_93/1892-1893.pdf --page 100
```

**Stage 1 — verbatim transcription.** Slices requested pages out of the PDF and sends them to
Gemini as a standalone in-memory PDF (not rasterized images). Batches several pages per API
call but writes one output file per page, so a partial run resumes cleanly:

```bash
python scripts/01_ocr_entries.py 1892_93 data/1892_93/1892-1893.pdf --pages 100-103 --batch-size 4
```

**Stage 2 — merge.** Flattens every page file into one ordered list, tagging each line with
its source page:

```bash
python scripts/02_merge_pages.py 1892_93
```

**Stage 3 — field extraction.** Parses each line into structured fields in batches of 25
across a thread pool:

```bash
python scripts/03_parse_entries.py 1892_93 --restart --workers 8
```

**Stage 4 — accuracy measurement.** Not part of `run_all.sh`; run manually whenever you want
a citable accuracy number. See [Validation](#validation).

```bash
python scripts/04_validate_sample.py 1892_93 --sample 5
# ... hand-transcribe the worksheets from the PDF ...
python scripts/04_validate_sample.py 1892_93 --compare
```

Stages 1-3 are also wired together in `run_all.sh` for an unattended full run (see
[Running a New Edition](#running-a-new-edition)).

## Running a New Edition

This is the concrete, in-order path for a brand-new city directory PDF. Steps marked
**(by eye)** require a human looking at the actual PDF; everything else is a script.

1. **Place the PDF.** Source PDFs are gitignored — put the file at
   `data/<year_vol>/<name>.pdf` (e.g. `data/1892_93/1892-1893.pdf`). Nothing in the pipeline
   enforces this exact path, but every existing edition follows it.

2. **Scaffold `config/<year_vol>/page_layout.json`.** There is no script for this step today
   — write the file by hand. Minimum shape:

   ```json
   {
     "edition": {
       "city": "Houston",
       "year_label": "1892-93",
       "publisher": "Morrison & Fourmy"
     },
     "abbreviations": {
       "file": "1892-1893.pdf",
       "page": 100
     },
     "entries_source": {
       "file": "1892-1893.pdf",
       "page_start": 100,
       "page_end": 547
     }
   }
   ```

   The `edition` block is optional but recommended — stages 1 and 3 read it (via
   `edition_context()` in each script) to describe the edition in their prompts (e.g. "the
   1892-93 Houston city directory, published by Morrison & Fourmy"). If a field is missing,
   the prompt falls back to generic wording; if the whole block is absent, it falls back to
   "a historical city directory." Fill it from the PDF's own text — check whether it has an
   embedded text layer first (`fitz.open(pdf).load_page(0).get_text()` in a throwaway
   script); both 1900-01's and 1892-93's PDFs did, and the exact publisher name and year
   label were confirmed that way rather than guessed.

3. **Find the abbreviations page and the entries page range (by eye).** Open the PDF and
   locate the "Abbreviations Used in the Directory" table in the front matter — note its page
   number for `abbreviations.page`. Then find where the alphabetical resident/business
   listing actually starts and ends — note those for `entries_source.page_start`/`page_end`.
   There's no script for this; it's a skim of the PDF, or of its extracted text layer if it
   has one. In both editions processed so far, the abbreviations table and the first entries
   page turned out to be the same physical page.

4. **Run stage 0** to generate `abbreviations.json` from the page you just found:

   ```bash
   python scripts/00_extract_abbreviations.py 1892_93 data/1892_93/1892-1893.pdf --page 100
   ```

   It refuses to overwrite an existing `abbreviations.json` unless you pass `--force`.

5. **Test on a handful of pages before the full run.** Pick a few pages from the middle of
   `entries_source`'s range and inspect the output by hand:

   ```bash
   python scripts/01_ocr_entries.py 1892_93 data/1892_93/1892-1893.pdf --pages 323-326 --batch-size 4
   ```

   Then open `data/1892_93/pages/page_0323.json` etc. and actually read the lines. This step
   matters: it's exactly how the 1892-93 edition's running-page-header leakage was caught
   (every page's first "entry" was its own running header, e.g.
   `"288 [KEL] MORRISON & FOURMY'S [KEL]"`) before it polluted all 448 pages. If you find a
   systematic problem like that, it's a prompt fix in `build_prompt()` in
   `01_ocr_entries.py`, then re-test the same pages to confirm before scaling up.

6. **Run the full pipeline.**

   ```bash
   export GEMINI_API_KEY="..."   # or put it in .env, see Setup
   ./run_all.sh 1892_93 data/1892_93/1892-1893.pdf
   ```

   This runs stages 1-3 unattended and tees all output to
   `logs/run_<year_vol>_<timestamp>.log`.

7. **Validate.** Draw a random sample, hand-transcribe it from the PDF independent of the
   pipeline output, and score it — see [Validation](#validation).

## Output Schema

**`data/<year_vol>/<year_vol>_entries.json`** — flat list of every transcribed line, in
reading order, tagged with its source page:

```json
{
  "page": 56,
  "line": "ABRAHAMS ARMISTEAD L., attorney, notary, 9-11 Fox bldg, 317½ Main, phones 556, h. 1704 Dallas ave."
}
```

**`data/<year_vol>/<year_vol>_parsed.json`** — the same lines parsed into structured fields.
The line above becomes:

```json
{
  "page": 56,
  "raw": "ABRAHAMS ARMISTEAD L., attorney, notary, 9-11 Fox bldg, 317½ Main, phones 556, h. 1704 Dallas ave.",
  "surname": "ABRAHAMS",
  "given_name": "ARMISTEAD L.",
  "qualifier": null,
  "occupation": "attorney, notary, 9-11 Fox bldg, 317½ Main",
  "residence_type": "h",
  "address": "1704 Dallas ave.",
  "phone": "556",
  "race_marker": false,
  "entry_type": "resident",
  "notes": null
}
```

| Field | Description |
|---|---|
| `page` | Source PDF page number. Attached locally by the pipeline, not produced by the model — it's the provenance link back to the scan. |
| `raw` | The original verbatim line, exactly as it appears in `<year_vol>_entries.json`. Re-attached locally so every parsed record traces back to its source text. |
| `surname` | Family name as printed. For businesses/institutions, the first word or company name. |
| `given_name` | First name and any middle initials/suffixes as printed. `null` for businesses, institutions, and cross-references. |
| `qualifier` | Anything in parentheses directly after the name — e.g. `(wid John)` for widow, `(Mrs. F. B.)` for married name, or `(c)` for this era's racial designation marker. `null` if absent. |
| `occupation` | Occupation or business description as printed, abbreviations not expanded. `null` if absent. |
| `residence_type` | The abbreviation immediately before the address: `h` (head of household), `r` (resides), `bds` (boards), `rms` (rooms), or `null`. |
| `address` | Street address as printed. `null` if absent. |
| `phone` | Phone number as printed. `null` if absent. |
| `race_marker` | `true` if `(c)` appears in the entry, `false` otherwise. Preserved as printed — historical source data for research, not a modern editorial addition. |
| `entry_type` | One of `resident`, `business`, `cross_reference`, `partnership`, `other`. |
| `notes` | Anything that didn't fit the fields above (e.g. a death notice, an unusual format). `null` if nothing to note. |

`data/<year_vol>/<year_vol>_parse_failures.json` holds any line that couldn't be parsed even
after stage 3's split-and-retry gave up; it's absent entirely if nothing failed (both
editions currently in this repo have 0 unparseable lines and no such file).

## Validation

Methodology (see `scripts/04_validate_sample.py`, mirroring Albers & Kappner 2023): draw a
random sample of pages with a recorded, reproducible seed; a human hand-transcribes those
pages directly from the source PDF, without looking at the pipeline output; then diff the
two line-by-line for exact matches, near-matches (character similarity ≥ 0.90), and
missed/spurious lines, at both the line level and the character level.

| Edition | Sample | Lines | Exact | Near | Missed | Spurious | Line accuracy | Char accuracy |
|---|---|---|---|---|---|---|---|---|
| 1900_01 | 5 pages (seed 251711) | 365 | 335 | 26 | 4 | 4 | 91.78% | 99.76% |
| 1892_93 | 3 pages (seed 450146) | 129 | 122 | 7 | 0 | 0 | 94.57% | 99.91% |

(Numbers from `validation/1900_01/validation_report.json` and
`validation/1892_93/validation_report.json`.)

Most of the line-accuracy gap in both editions is **transcription convention differences**,
not pipeline errors: printed fraction glyphs (`½`) transcribed as `1/2`, ALL-CAPS business
name typesetting the human transcriber didn't consistently reproduce, curly-vs-straight
apostrophe autocorrect on the manual side, and inconsistent trailing periods. Character-level
accuracy — which is less sensitive to those formatting conventions — sits at 99.76% and
99.91% for the two editions respectively. Genuine single-character OCR-type errors do exist
in both samples (e.g. a misread street name, a transposed digit) but are a small minority of
the discrepancies.

## Editions Processed

| year_vol | City | Directory year | Pages | Entries | Line accuracy | Char accuracy |
|---|---|---|---|---|---|---|
| `1900_01` | Houston | 1900-1901 | 391 (pp. 55-445) | 26,556 | 91.78% | 99.76% |
| `1892_93` | Houston | 1892-93 | 448 (pp. 100-547) | 16,671 | 94.57% | 99.91% |

Both editions were published by Morrison & Fourmy, confirmed from each PDF's own front
matter, not assumed.

## What Varied Across Editions

Only two editions have been run, both Houston, both the same publisher — so this is a narrow
sample, but every difference below is a real, observed one, not a hypothetical:

- **Running page headers.** 1892-93's running header (page number + bracketed guide letters,
  e.g. `"288 [KEL] MORRISON & FOURMY'S [KEL]"` or `"[KEL] HOUSTON CITY DIRECTORY. [KEN] 289"`)
  was captured by the model as a spurious entry line on every single page tested. 1900-01
  never exhibited this. The fix was one new rule added to the shared prompt in
  `build_prompt()` (`01_ocr_entries.py`) — it now benefits both editions and any future one,
  not a per-edition branch.
- **Abbreviation set size and content.** 1900-01 has 104 entries in its abbreviations table
  (`config/1900_01/abbreviations.json`); 1892-93 has 86
  (`config/1892_93/abbreviations.json`). Some of the difference is genuine historical drift,
  not extraction inconsistency: 1892-93 defines `H. C. St. Ry.` as "Houston City Street Ry.",
  while 1900-01 instead defines `H. E. St Ry.` as "Houston Electric Street Railway" — Houston's
  streetcar system converted from horse-drawn to electric traction in the 1890s, so the two
  editions genuinely reference different named companies. Similarly, `Hts` (Heights) appears
  only in 1900-01's table, consistent with the Houston Heights suburb being developed later
  in that decade.
- **`blk` means something different per edition.** In 1892-93, `blk` is defined as "black"; in
  1900-01, it's defined as "block." Both were confirmed correct for their respective edition
  by direct inspection — this is not an extraction error in either file, it's the actual
  printed definition in that year's table.
- **The cross-reference convention.** 1900-01 uses a `☞ Name—See also X, Y, Z` convention for
  alphabetical cross-references (the hand-pointer symbol `☞` appears throughout its
  transcribed output). 1892-93's transcribed output contains **zero** occurrences of `☞` and
  zero instances of "See also" — its only "See ..." lines are advertisement pointers inside
  paid business listings (`"See advt."`, `"See back cover."`), a different convention
  entirely. A pipeline that assumed every edition has cross-reference lines to classify would
  be wrong for 1892-93.

**What transferred unchanged:** all five scripts (stages 0-4) required zero code changes
between editions. Every adjustment above was either a new `config/<year_vol>/` file (the
normal, expected way to onboard an edition) or one addition to a shared prompt string that
now benefits every edition, not a per-edition code branch.

## Known Limitations

- **Run-to-run variation.** Even at `temperature=0`, the model isn't fully deterministic on
  ambiguous input. Concretely: two lines in the 1900_01 run failed to parse (landed in
  `1900_01_parse_failures.json`) on one run, then parsed successfully with no changes to the
  input on a later full restart. Don't treat a single run's failure list as a permanent
  verdict on a line.
- **Single publisher, single city.** Every edition processed so far is a Houston directory
  from Morrison & Fourmy. Cross-publisher and cross-city generalization — a different
  typesetting house, a different city's naming/abbreviation conventions — is untested.
- **Cost profile.** Per-page (stage 1) and per-batch (stage 3) API calls mean this doesn't
  scale as cheaply as a trained local OCR model would for very large-scale digitization
  (thousands of pages across many volumes). It's well-suited to the scale processed so far
  (hundreds of pages per edition) and to adding new editions quickly, not necessarily to a
  large multi-city, multi-decade digitization project.
- **Front matter is out of scope beyond abbreviations.** `config/1900_01/page_layout.json`
  records where other front-matter sections live (street directory, ward boundaries, halls/
  parks/buildings, cemeteries, guide to streets), and `config/1900_01/toc_raw.json` /
  `toc_classified.json` map the volume's full table of contents — but no script consumes any
  of it yet. Only the General Directory of Names (the alphabetical resident/business listing)
  and the abbreviations key are actually processed.

## Setup

**Dependencies** (not pinned anywhere yet — install manually):

```bash
pip install pymupdf pydantic google-genai langsmith python-dotenv
```

**API key.** Create a repo-root `.env` (gitignored):

```
GEMINI_API_KEY=your-key-here
```

All three API-calling scripts (`00_extract_abbreviations.py`, `01_ocr_entries.py`,
`03_parse_entries.py`) load it automatically via `python-dotenv`. An exported shell variable
takes precedence over `.env`. `run_all.sh` sources `.env` under the same precedence before
its own fail-early check for the key.

**Optional LangSmith tracing.** All three API-calling scripts import `traceable` from
`langsmith` and decorate their model-calling functions with it, so `langsmith` must be
installed even if you don't want tracing (it's an import-time dependency). To actually see
traces, set LangSmith's own environment variables (its API key and tracing flag) per
[LangSmith's documentation](https://docs.smith.langchain.com/) — this repo doesn't configure
or require any LangSmith-specific setup beyond having the package importable.

**Source PDFs.** Gitignored (`*.pdf` in `.gitignore`) — never committed, and not distributed
with this repo. The two currently processed (`data/1900_01/1900-1901.pdf`,
`data/1892_93/1892-1893.pdf`) were provided by project mentors Sean Smith and Norie Guthrie
for the Fondren Fellows "Directories to Data" project. To process a new edition, supply its
PDF yourself at `data/<year_vol>/<name>.pdf`.
