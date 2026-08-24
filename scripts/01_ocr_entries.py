"""
01_ocr_entries.py

Pure verbatim transcription of city directory entry pages using Gemini 3.5
Flash. No bounding boxes, no entry-type classification, just accurate
line-by-line text in reading order. Narrowed scope per project guidance,
focus on the one element common to every city directory regardless of
edition or city: getting the printed text right.

Batches multiple pages per API call to reduce total calls. Each page in
the batch is saved as its own output file so already-completed pages are
skipped automatically on re-runs.

If a batch response comes back malformed (bad JSON, wrong object count, a
page missing from the response), the batch is bisected and each half
retried independently, down to individual pages if needed, rather than
dropping every page in that batch.

Usage:
    # First run, small batch to validate before scaling up
    python scripts/01_ocr_entries.py 1900_01 data/1900_01/1900-1901.pdf \
        --pages 50-51 --batch-size 2

    # Once validated, larger range and batch size
    python scripts/01_ocr_entries.py 1900_01 data/1900_01/1900-1901.pdf \
        --pages 50-80 --batch-size 4

    # Uses entries_source range from page_layout.json if --pages omitted
    python scripts/01_ocr_entries.py 1900_01 data/1900_01/1900-1901.pdf

Set your API key before running, either in the repo-root .env file:

    GEMINI_API_KEY=your-key-here

or as a shell variable, which takes precedence over .env:

    export GEMINI_API_KEY="your-key-here"
"""

import os
import sys
import json
import argparse
from pathlib import Path

import fitz                          # pip install pymupdf
from pydantic import BaseModel
from google import genai
from google.genai import types
from langsmith import traceable      # pip install langsmith
from dotenv import load_dotenv    # pip install python-dotenv

# ---------------------------------------------------------------------------
# Output schema
# ---------------------------------------------------------------------------
#
# response_mime_type="application/json" alone only guarantees valid JSON
# syntax, it does not force the model to match the shape shown in the
# prompt text. Passing this schema as response_schema makes the API
# enforce the exact structure, so the shape can no longer drift between
# batches the way it did when this was left to an example in the prompt.

class LineEntry(BaseModel):
    line: str


class PageTranscription(BaseModel):
    page_position: int
    lines: list[LineEntry]
    page_complete: bool

# ---------------------------------------------------------------------------
# Client setup
# ---------------------------------------------------------------------------

# Reads GEMINI_API_KEY from the repo-root .env if it is not already in the
# environment. An exported shell variable still wins over the file.
load_dotenv()

client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
MODEL = "gemini-3.5-flash"   # verify exact string in AI Studio's model dropdown

# Set explicitly, the same fix used in 03_parse_entries.py: left unset, the
# SDK's low default combined with thinking tokens drawing from the same
# budget truncates responses well before the model's real output ceiling.
# Sized generously here since a batch response holds full verbatim text for
# several pages, not just compact structured fields like stage 3's.
MAX_OUTPUT_TOKENS = 65536

# ---------------------------------------------------------------------------
# PDF page extraction
# ---------------------------------------------------------------------------
#
# Pages are passed to Gemini as a standalone PDF, not as pre-rendered JPEGs.
# This matches the AI Studio workflow validated throughout this project,
# where every successful session uploaded a PDF directly and let Gemini
# handle the internal rendering, rather than a manually rasterized image
# at a fixed DPI that has not actually been tested here.

def extract_pages_as_pdf_bytes(pdf_path: str, page_numbers: list[int]) -> bytes:
    """
    Pull the given 1-indexed pages out of the source PDF and return them
    as a new, standalone PDF document in memory, in the order given.
    """
    src = fitz.open(pdf_path)
    total = len(src)
    invalid = [p for p in page_numbers if p < 1 or p > total]
    if invalid:
        raise ValueError(f"Page(s) {invalid} out of range (PDF has {total} pages).")

    new_doc = fitz.open()
    for page_number in page_numbers:
        new_doc.insert_pdf(src, from_page=page_number - 1, to_page=page_number - 1)

    pdf_bytes = new_doc.tobytes()
    new_doc.close()
    src.close()
    return pdf_bytes


def parse_pages(pages_arg: str, pdf_path: str) -> list[int]:
    """Parse "42", "42-50", "42,43,45", or "42-45,48" into a sorted list."""
    doc = fitz.open(pdf_path)
    total = len(doc)
    page_set = set()
    for part in pages_arg.split(","):
        part = part.strip()
        if "-" in part:
            start_s, end_s = part.split("-", 1)
            page_set.update(range(int(start_s), int(end_s) + 1))
        else:
            page_set.add(int(part))
    invalid = [p for p in page_set if p < 1 or p > total]
    if invalid:
        raise ValueError(f"Page(s) {invalid} out of range (PDF has {total} pages).")
    return sorted(page_set)

# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

def build_prompt(abbreviations: list[dict], page_count: int) -> str:
    abbrev_json = json.dumps(abbreviations, indent=2)
    return f"""\
These are {page_count} consecutive pages from the General Directory of
Names section of a historical city directory, in page order.

Use the abbreviation meanings below only to resolve genuinely ambiguous
characters during transcription. Do not expand abbreviations in your
output, transcribe exactly what is printed:

{abbrev_json}

For each page, return a JSON object. Return the full response as a
JSON array of these page objects in the same order as the pages were
provided:

[
  {{
    "page_position": 1,
    "lines": [
      {{"line": "Aarons Gustave, grocer, 815 Congress, h same"}}
    ],
    "page_complete": true
  }}
]

Rules:
- page_position is the position of the page within this upload, starting
  at 1.
- lines should include every individual resident or business directory
  entry on that page in reading order, left column top to bottom, then
  right column top to bottom. Include section headers and
  cross-references, transcribe those exactly as they appear.
- Do not include advertisements. This applies to large banner ads that
  span the full width of the page, typically at the very top or bottom,
  and also to smaller advertisement blocks that interrupt a column
  partway down, such as a boxed real estate, business, or product ad
  sitting between two directory entries. Skip all of these entirely,
  do not transcribe their text and do not count them as a line.
- Transcribe exactly as printed, character for character. Do not expand
  abbreviations, including ditto marks, transcribe them exactly as they
  appear.
- If a line is indented relative to where entries normally begin in
  that column, it is a continuation of the entry directly above it, not
  a new entry. Merge it into that entry's line.
- Occasionally a short fragment of text appears at the right edge of
  the line above an entry rather than indented below it. If you see
  this, that fragment belongs to the entry it sits above, append it to
  that entry rather than treating it as its own line.
- Mark any uncertain character with [?].
- If you are at risk of running out of space before finishing a page,
  stop after fully completing as many entries as fit and set
  page_complete to false for that page rather than cutting an entry
  off mid line.
- Return only the JSON array. No explanation, no markdown fences.\
"""

# ---------------------------------------------------------------------------
# Core extraction
# ---------------------------------------------------------------------------

@traceable(name="transcribe_page_batch")
def extract_batch(pdf_bytes: bytes, abbreviations: list[dict], page_count: int) -> list[dict]:
    """Send a batch of pages, as one in-memory PDF, to Gemini. Returns the parsed page array."""
    parts = [
        types.Part.from_bytes(mime_type="application/pdf", data=pdf_bytes),
        types.Part.from_text(text=build_prompt(abbreviations, page_count)),
    ]

    response = client.models.generate_content(
        model=MODEL,
        contents=[types.Content(role="user", parts=parts)],
        config=types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(thinking_level="high"),
            temperature=0,   # Gemini 3 guidance recommends testing against the
                              # default (unset / 1.0) instead, see note in chat
            response_mime_type="application/json",
            response_schema=list[PageTranscription],
            media_resolution="media_resolution_high",  # verify this string
                              # against your installed SDK version, this is a
                              # newer parameter and naming may have shifted
            max_output_tokens=MAX_OUTPUT_TOKENS,
        ),
    )

    pages = json.loads(response.text)
    if not isinstance(pages, list):
        raise ValueError(
            f"Expected a JSON list from Gemini, got {type(pages).__name__}. "
            f"Raw: {response.text[:300]}"
        )
    return pages


def page_warnings(pages) -> list[str]:
    """Non-fatal issues on successfully transcribed pages (missing fields, page_complete=false)."""
    warnings = []
    for p in pages:
        missing = {"page_position", "lines", "page_complete"} - set(p.keys())
        if missing:
            warnings.append(f"  page_position {p.get('page_position', '?')}: missing fields {missing}")
        if p.get("page_complete") is False:
            warnings.append(
                f"  page_position {p.get('page_position', '?')}: marked page_complete=false, "
                f"this page needs a follow-up call on its own."
            )
    return warnings


def transcribe_pages(
    pdf_path: str, page_numbers: list[int], abbreviations: list[dict]
) -> tuple[dict[int, dict], list[dict]]:
    """
    Transcribe a set of pages, splitting and retrying on failure.

    Ported from parse_lines() in 03_parse_entries.py: a malformed response
    (bad JSON, wrong object count, or a page missing from the response)
    used to drop every page in the batch. Instead, bisect the page list and
    retry each half independently, narrowing down to the specific page(s)
    actually causing the problem rather than losing the whole batch.

    Returns (results, failed):
      results: {page_number: page_object} for every page transcribed OK.
      failed:  [{"page": N, "error": "..."}] for pages that still failed
               even as a batch of one.
    """
    pdf_bytes = extract_pages_as_pdf_bytes(pdf_path, page_numbers)

    try:
        result_pages = extract_batch(pdf_bytes, abbreviations, len(page_numbers))
        if len(result_pages) != len(page_numbers):
            raise ValueError(f"count mismatch: sent {len(page_numbers)}, got {len(result_pages)}")

        results = {}
        for i, page_number in enumerate(page_numbers):
            match = next((p for p in result_pages if p.get("page_position") == i + 1), None)
            if match is None:
                raise ValueError(f"page_position {i + 1} missing from response for page {page_number}")
            results[page_number] = match
        return results, []

    except KeyboardInterrupt:
        raise
    except Exception as e:
        if len(page_numbers) == 1:
            return {}, [{"page": page_numbers[0], "error": str(e)[:200]}]
        mid = len(page_numbers) // 2
        r1, f1 = transcribe_pages(pdf_path, page_numbers[:mid], abbreviations)
        r2, f2 = transcribe_pages(pdf_path, page_numbers[mid:], abbreviations)
        return {**r1, **r2}, f1 + f2

# ---------------------------------------------------------------------------
# File I/O
# ---------------------------------------------------------------------------

def load_abbreviations(config_dir: Path) -> list[dict]:
    abbrev_path = config_dir / "abbreviations.json"
    if not abbrev_path.exists():
        raise FileNotFoundError(
            f"abbreviations.json not found at {abbrev_path}. "
            f"Restore it before running this script."
        )
    with open(abbrev_path, encoding="utf-8") as f:
        data = json.load(f)
    if not data:
        raise ValueError(
            f"{abbrev_path} is empty. Restore the abbreviations list before "
            f"running this script, the prompt depends on it."
        )
    return data


def load_page_layout(config_dir: Path) -> dict:
    layout_path = config_dir / "page_layout.json"
    if not layout_path.exists():
        return {}
    with open(layout_path) as f:
        return json.load(f)


def normalize_lines(raw_lines: list) -> list[dict]:
    """
    The model has been observed to return lines either as plain strings
    or as {"line": "..."} objects, regardless of the requested schema.
    Normalize either shape into a consistent object list before saving,
    so output files never end up mixed across batches or pages.
    """
    normalized = []
    for item in raw_lines:
        if isinstance(item, str):
            normalized.append({"line": item})
        elif isinstance(item, dict) and "line" in item:
            normalized.append({"line": item["line"]})
        else:
            normalized.append({"line": str(item)})
    return normalized


def save_page(lines: list, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    normalized = normalize_lines(lines)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(normalized, f, indent=2, ensure_ascii=False)

# ---------------------------------------------------------------------------
# Processing
# ---------------------------------------------------------------------------

def process_batch(
    pdf_path: str,
    page_numbers: list[int],
    abbreviations: list[dict],
    out_dir: Path,
    dry_run: bool = False,
) -> list[dict]:
    """
    Process one batch of pages. Skips pages that already have output files,
    only sending pages that still need work. Returns a list of summary
    dicts, one per page.
    """
    pages_to_fetch = []
    summaries = []

    for page_number in page_numbers:
        out_path = out_dir / f"page_{page_number:04d}.json"
        if out_path.exists() and not dry_run:
            with open(out_path) as f:
                existing = json.load(f)
            print(f"  Page {page_number}: already done ({len(existing)} lines) — skipping.")
            summaries.append({"page": page_number, "lines": len(existing), "skipped": True})
        else:
            pages_to_fetch.append(page_number)

    if not pages_to_fetch:
        return summaries

    print(f"  Fetching pages {pages_to_fetch} in one batch call...")
    results, failed = transcribe_pages(pdf_path, pages_to_fetch, abbreviations)

    warnings = page_warnings(results.values())
    if warnings:
        print(f"  Warnings:")
        for w in warnings:
            print(w)

    for page_number in pages_to_fetch:
        match = results.get(page_number)
        if match is None:
            continue  # reported below via `failed`

        lines = match.get("lines", [])
        print(f"  Page {page_number}: {len(lines)} lines, page_complete={match.get('page_complete')}")

        if not dry_run:
            out_path = out_dir / f"page_{page_number:04d}.json"
            save_page(lines, out_path)

        summaries.append({
            "page": page_number,
            "lines": len(lines),
            "page_complete": match.get("page_complete"),
            "skipped": False,
        })

    for f in failed:
        print(f"  Page {f['page']}: ERROR — {f['error']}")
        summaries.append({"page": f["page"], "error": f["error"]})

    return summaries

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def chunk_list(items: list, size: int):
    for i in range(0, len(items), size):
        yield items[i:i + size]


def main():
    parser = argparse.ArgumentParser(
        description="Verbatim transcription of city directory entry pages."
    )
    parser.add_argument("year_vol", help="Directory year-volume string, e.g. 1900_01")
    parser.add_argument("pdf", help="Path to the directory PDF file")
    parser.add_argument(
        "--pages",
        metavar="PAGES",
        help='Pages to process, e.g. "50-51", "50,52,55", "50-80". '
             'Overrides page_layout.json if both are present.',
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Pages per API call. Start small (2) to validate before scaling up. Default: 4",
    )
    parser.add_argument(
        "--out-dir",
        metavar="PATH",
        help="Output folder for JSON files (default: data/<year_vol>/pages/)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run extraction but do not write output files",
    )
    args = parser.parse_args()

    config_dir = Path("config") / args.year_vol
    out_dir = Path(args.out_dir) if args.out_dir else Path("data") / args.year_vol / "pages"

    print(f"Loading abbreviations from {config_dir} ...")
    abbreviations = load_abbreviations(config_dir)
    print(f"  {len(abbreviations)} abbreviations loaded.")

    if args.pages:
        pages = parse_pages(args.pages, args.pdf)
    else:
        layout = load_page_layout(config_dir)
        entries_source = layout.get("entries_source", {})
        start = entries_source.get("page_start")
        end = entries_source.get("page_end")
        if start is None or end is None:
            print(
                "No --pages flag given and page_layout.json does not define "
                "entries_source.page_start/page_end. Pass --pages explicitly."
            )
            sys.exit(1)
        pages = list(range(start, end + 1))
        print(f"Using page range from page_layout.json: {start}-{end} ({len(pages)} pages)")

    print(f"\nProcessing {len(pages)} page(s) from {args.pdf} in batches of {args.batch_size} -> {out_dir}")

    all_summaries = []
    for batch_num, batch in enumerate(chunk_list(pages, args.batch_size), start=1):
        print(f"\nBatch {batch_num}: pages {batch}")
        try:
            summaries = process_batch(args.pdf, batch, abbreviations, out_dir, dry_run=args.dry_run)
            all_summaries.extend(summaries)
        except Exception as e:
            print(f"  ERROR on batch {batch}: {e}")
            for p in batch:
                all_summaries.append({"page": p, "error": str(e)})

    print("\n=== Summary ===")
    print(f"  Pages processed : {sum(1 for s in all_summaries if not s.get('skipped') and 'error' not in s)}")
    print(f"  Pages skipped   : {sum(1 for s in all_summaries if s.get('skipped'))}")
    print(f"  Errors          : {sum(1 for s in all_summaries if 'error' in s)}")
    print(f"  Total lines     : {sum(s.get('lines', 0) for s in all_summaries)}")
    incomplete = [s for s in all_summaries if s.get("page_complete") is False]
    if incomplete:
        print(f"  Pages marked incomplete (need follow-up): {[s['page'] for s in incomplete]}")

    errors = [s for s in all_summaries if "error" in s]
    if errors:
        print("\nFailed pages:")
        for s in errors:
            print(f"  Page {s['page']}: {s['error']}")


if __name__ == "__main__":
    main()