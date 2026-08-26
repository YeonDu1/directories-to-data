# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Converts scanned historical city directories (Houston, TX — first edition processed is the
1900-1901 volume) into structured, research-usable data using Gemini. The one deliberately
narrow goal is getting the printed text right first, then parsing it — the pipeline does no
bounding boxes and no layout classification.

## Running the pipeline

```bash
export GEMINI_API_KEY="..."

# New edition only: bootstrap config/<year_vol>/abbreviations.json once, first
python scripts/00_extract_abbreviations.py 1900_01 data/1900_01/1900-1901.pdf --page 55

# Full unattended run (stages 1-3, tee'd to logs/run_<year_vol>_<stamp>.log)
./run_all.sh 1900_01 data/1900_01/1900-1901.pdf

# Or stage by stage
python scripts/01_ocr_entries.py 1900_01 data/1900_01/1900-1901.pdf --pages 50-51 --batch-size 2
python scripts/02_merge_pages.py 1900_01
python scripts/03_parse_entries.py 1900_01 --workers 8
```

Useful flags while iterating:
- `01`: `--pages "50-80"` / `"50,52,55"` (1-indexed PDF pages, overrides `page_layout.json`),
  `--batch-size N` (pages per API call), `--dry-run` (call the API, write nothing).
- `03`: `--limit 200` (spot-check the first N lines), `--restart` (discard checkpoint),
  `--workers N` (lower it on rate limits), `--batch-size N` (lines per API call).

Dependencies are not pinned anywhere; install manually:
`pip install pymupdf pydantic google-genai langsmith python-dotenv`.

The API key is read from `GEMINI_API_KEY`. All three API-calling scripts (0, 1, 3) call
`load_dotenv()`, so a repo-root `.env` (gitignored) is enough; an exported shell variable
overrides it. `run_all.sh` sources `.env` under the same precedence before its own
fail-early guard.

## Pipeline architecture

Three sequential stages plus a one-time bootstrapping step and a manual validation step.
Each stage's output is the next stage's input, and every stage is resumable — never assume
a re-run redoes work.

0. **`00_extract_abbreviations.py` — bootstrap a new edition's abbreviations key.** Not
   part of `run_all.sh`; run once per new edition before stage 1, which hard-fails without
   `config/<year_vol>/abbreviations.json`. Takes `--page N` or `--pages "N-M"` for the front
   matter page(s) containing the "Abbreviations Used in the Directory" table (falls back to
   `page_layout.json`'s `abbreviations` block, either `{"page": N}` or
   `{"page_start", "page_end"}`, if neither flag is given). Refuses to overwrite an existing
   `abbreviations.json` unless `--force` is passed. Same split-and-retry, explicit
   `MAX_OUTPUT_TOKENS`, `@traceable`, and `atomic_write` conventions as stages 1 and 3.
1. **`01_ocr_entries.py` — verbatim transcription.** Slices the requested pages out of the
   source PDF with PyMuPDF into a new in-memory PDF and sends *that* to Gemini (not
   rasterized JPEGs — this matches the AI Studio workflow that was validated for this
   project). Batches several pages per call but writes one file per page to
   `data/<year_vol>/pages/page_XXXX.json`, so an existing page file is skipped on re-run.
   `response_schema` is enforced via the Pydantic models, but the model has still been
   observed to return bare strings instead of `{"line": ...}` objects — `normalize_lines()`
   exists for that and should not be removed.
2. **`02_merge_pages.py` — merge.** Flattens all page files into
   `data/<year_vol>/<year_vol>_entries.json` as `{"page": N, "line": "..."}` records. The
   page number is the provenance link back to the scan and is carried through every later
   stage.
3. **`03_parse_entries.py` — field extraction.** Parses lines into structured fields
   (`surname`, `given_name`, `qualifier`, `occupation`, `residence_type`, `address`,
   `phone`, `race_marker`, `entry_type`, `notes`) in batches of 25 across a thread pool.
   Writes `<year_vol>_parsed.json` and `<year_vol>_parse_failures.json`.
   - `MAX_OUTPUT_TOKENS` is set explicitly on purpose: the SDK default is low, thinking
     tokens draw from the same budget, and leaving it unset was the actual cause of
     mid-JSON truncation. Do not remove it.
   - The model is told to return only an `n` index, not the original text; `raw` and `page`
     are re-attached locally to keep responses compact.
   - A batch that fails or returns a count mismatch is recursively split in half and
     retried; a single line that still fails lands in the failures file.
   - Progress is checkpointed by batch index to `data/<year_vol>/_parse_checkpoint.json`
     (gitignored) every 10 batches and on `KeyboardInterrupt`; output is reassembled in
     original batch order regardless of completion order. Writes go through `atomic_write`
     (temp file + rename) because the data dir may be cloud-synced.
4. **`04_validate_sample.py` — accuracy measurement.** Not part of `run_all.sh`.
   `--sample N` draws a reproducible random page sample and writes blank worksheets to
   `validation/<year_vol>/manual/`; a human transcribes those pages from the PDF alone,
   then `--compare` diffs them against pipeline output for a citable accuracy number
   (methodology follows Albers & Kappner 2023).

## Per-edition layout

Everything is keyed by a `year_vol` string (e.g. `1900_01`) that names both a config and a
data directory. Adding an edition means creating `config/<year_vol>/` and passing the new
key on the command line — no code changes.

- `config/<year_vol>/abbreviations.json` — required by stage 1; injected into the prompt so
  the model can disambiguate characters. Stage 1 hard-fails if it is missing or empty.
  Bootstrap it for a new edition with stage 0 (`00_extract_abbreviations.py`).
- `config/<year_vol>/page_layout.json` — `entries_source.page_start`/`page_end` is the
  default page range for stage 1 when `--pages` is omitted. Other keys record where
  front-matter sections (street directory, ward boundaries, guide to streets, …) live and
  which standalone PDF they were split into. An optional `edition` block
  (`{"city", "year_label", "publisher"}`) is read by stages 1 and 3's `edition_context()`
  to describe the edition in the prompt (e.g. "the 1900-1901 Houston city directory,
  published by Morrison & Fourmy"); missing fields fall back to generic wording.
- `config/<year_vol>/toc_raw.json` → `toc_classified.json` — the volume's table of contents,
  raw then annotated with `canonical_type`/`tier`/`needs_subscan`. Not yet consumed by any
  script; it is the map for extending coverage beyond the General Directory of Names.

## Conventions that matter

- Source PDFs and `logs/` are gitignored; the JSON in `data/` and `config/` is committed.
  Scripts assume the PDF sits at `data/<year_vol>/<name>.pdf` but nothing enforces it.
- Prompts are the substance of this project — most behavior lives in `build_prompt()` in
  stages 0 and 1 and `build_parse_prompt()`/`PARSE_PROMPT_TEMPLATE` in stage 3, not in
  Python logic. Changing prompt wording changes output shape, so treat prompt edits as the
  significant change.
- Transcription is strictly verbatim: abbreviations, ditto marks, and `[?]` uncertainty
  markers are preserved as printed and never expanded. Advertisements are skipped.
- Stage 1's prompt also skips running page headers (page number + guide letters, e.g.
  `"288 [KEL] MORRISON & FOURMY'S [KEL]"`). Added for the 1892-93 edition, whose header
  placement got captured as spurious entry lines on every page; 1900-01 never needed this
  rule, so don't assume every edition will hit the same failure mode.
- The `(c)` race marker and the `race_marker` field are intentionally retained as printed —
  they are historical source data preserved for research.
- All three model calls run at `temperature=0`; stages 0 and 1 use `thinking_level="high"`,
  stage 3 `"medium"`. `MODEL = "gemini-3.5-flash"` is hardcoded near the top of each script.
- All three API-calling scripts are wrapped with `@traceable` (LangSmith); `langsmith` must
  be installed even if tracing is not configured. Each also sets `max_output_tokens`
  explicitly and applies the same split-and-retry pattern on a malformed response, rather
  than dropping the whole batch/range.
- All paths are resolved relative to the current working directory — run scripts from the
  repo root.
