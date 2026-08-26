"""
00_extract_abbreviations.py

Extracts the "Abbreviations Used in the Directory" key from a new
edition's front matter using Gemini, and writes it to
config/<year_vol>/abbreviations.json in the shape 01_ocr_entries.py
depends on: a JSON array of {"abbreviation": "...", "meaning": "..."}
objects.

This is a one-time bootstrapping step for a new edition, run before
01_ocr_entries.py, which hard-fails if abbreviations.json is missing or
empty. Front matter varies a lot page to page across editions/cities, so
this narrowly targets just the abbreviations table rather than trying to
handle every front-matter section.

Usage:
    # Extract from a single page of the edition's PDF
    python scripts/00_extract_abbreviations.py 1900_01 data/1900_01/1900-1901.pdf --page 31

    # Or a page range, if the abbreviations table spans multiple pages
    python scripts/00_extract_abbreviations.py 1900_01 data/1900_01/1900-1901.pdf --pages 31-32

    # Uses config/<year_vol>/page_layout.json's "abbreviations" block
    # ({"page": N} or {"page_start": N, "page_end": M}) if neither
    # --page nor --pages is given
    python scripts/00_extract_abbreviations.py 1900_01 data/1900_01/1900-1901.pdf

Refuses to overwrite an existing config/<year_vol>/abbreviations.json
unless --force is passed.

If a batch response comes back malformed (bad JSON, wrong shape), the
page range is bisected and each half retried independently, down to
individual pages if needed, the same split-and-retry pattern used in
01_ocr_entries.py and 03_parse_entries.py.

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
from dotenv import load_dotenv       # pip install python-dotenv

# ---------------------------------------------------------------------------
# Output schema
# ---------------------------------------------------------------------------

class AbbreviationEntry(BaseModel):
    abbreviation: str
    meaning: str

# ---------------------------------------------------------------------------
# Client setup
# ---------------------------------------------------------------------------

# Reads GEMINI_API_KEY from the repo-root .env if it is not already in the
# environment. An exported shell variable still wins over the file.
load_dotenv()

client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
MODEL = "gemini-3.5-flash"

# Set explicitly, the same fix used in 01_ocr_entries.py and
# 03_parse_entries.py: left unset, the SDK's low default combined with
# thinking tokens drawing from the same budget truncates responses well
# before the model's real output ceiling. An abbreviations table is short,
# but there's no reason to risk truncation over it.
MAX_OUTPUT_TOKENS = 8192

# ---------------------------------------------------------------------------
# PDF page extraction
# ---------------------------------------------------------------------------
#
# Same approach as 01_ocr_entries.py: pages are passed to Gemini as a
# standalone PDF, not pre-rendered JPEGs.

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
    """Parse "31", "31-32", or "31,33" into a sorted list of 1-indexed page numbers."""
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

def build_prompt(page_count: int) -> str:
    return f"""\
These are {page_count} consecutive pages, in page order, from the front
matter of a historical city directory, containing the "Abbreviations
Used in the Directory" key.

Extract every abbreviation/meaning pair in the table, in the order
printed, and return a JSON array of objects in this shape:

[
  {{"abbreviation": "acct.", "meaning": "accountant"}}
]

Rules:
- abbreviation: the abbreviation exactly as printed, including any
  trailing period.
- meaning: the expanded meaning exactly as printed, not paraphrased.
- If the table runs in two columns, read left column top to bottom,
  then right column top to bottom.
- Skip anything on the page that is not part of the abbreviations
  table itself (running headers, page numbers, unrelated front-matter
  text).
- Return only the JSON array. No explanation, no markdown fences.\
"""

# ---------------------------------------------------------------------------
# Core extraction
# ---------------------------------------------------------------------------

@traceable(name="extract_abbreviations_batch")
def extract_batch(pdf_bytes: bytes, page_count: int) -> list[dict]:
    """Send a page range, as one in-memory PDF, to Gemini. Returns the parsed abbreviation list."""
    parts = [
        types.Part.from_bytes(mime_type="application/pdf", data=pdf_bytes),
        types.Part.from_text(text=build_prompt(page_count)),
    ]

    response = client.models.generate_content(
        model=MODEL,
        contents=[types.Content(role="user", parts=parts)],
        config=types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(thinking_level="high"),
            temperature=0,
            response_mime_type="application/json",
            response_schema=list[AbbreviationEntry],
            media_resolution="media_resolution_high",  # verify this string
                              # against your installed SDK version, this is a
                              # newer parameter and naming may have shifted
            max_output_tokens=MAX_OUTPUT_TOKENS,
        ),
    )

    entries = json.loads(response.text)
    if not isinstance(entries, list):
        raise ValueError(
            f"Expected a JSON list from Gemini, got {type(entries).__name__}. "
            f"Raw: {response.text[:300]}"
        )
    return entries


def extract_abbreviations(pdf_path: str, page_numbers: list[int]) -> tuple[list[dict], list[dict]]:
    """
    Extract abbreviations from a page range, splitting and retrying on
    failure. Ported from transcribe_pages() in 01_ocr_entries.py /
    parse_lines() in 03_parse_entries.py: a malformed response no longer
    drops the whole page range, it bisects the range and retries each half
    independently, narrowing down to the specific page(s) actually causing
    the problem rather than losing everything.

    Returns (entries, failed):
      entries: [{"abbreviation": ..., "meaning": ...}, ...] collected from
               every page range that succeeded.
      failed:  [{"page": N, "error": "..."}] for pages that still failed
               even as a range of one.
    """
    pdf_bytes = extract_pages_as_pdf_bytes(pdf_path, page_numbers)

    try:
        entries = extract_batch(pdf_bytes, len(page_numbers))
        return entries, []

    except KeyboardInterrupt:
        raise
    except Exception as e:
        if len(page_numbers) == 1:
            return [], [{"page": page_numbers[0], "error": str(e)[:200]}]
        mid = len(page_numbers) // 2
        e1, f1 = extract_abbreviations(pdf_path, page_numbers[:mid])
        e2, f2 = extract_abbreviations(pdf_path, page_numbers[mid:])
        return e1 + e2, f1 + f2

# ---------------------------------------------------------------------------
# File I/O
# ---------------------------------------------------------------------------

def load_page_layout(config_dir: Path) -> dict:
    layout_path = config_dir / "page_layout.json"
    if not layout_path.exists():
        return {}
    with open(layout_path, encoding="utf-8") as f:
        return json.load(f)


def abbreviations_page_range(layout: dict) -> list[int] | None:
    """
    Read the "abbreviations" block from page_layout.json. Accepts either
    {"page": N} or {"page_start": N, "page_end": M}, matching the two
    conventions already used elsewhere in page_layout.json. Returns None
    if neither form is present.
    """
    block = layout.get("abbreviations", {})
    if "page" in block:
        return [block["page"]]
    if "page_start" in block and "page_end" in block:
        return list(range(block["page_start"], block["page_end"] + 1))
    return None


def atomic_write(path: Path, data):
    """
    Write via a temp file then rename. Protects against a half-written
    file if the run is interrupted, and avoids the write timeouts that
    can happen on cloud-synced folders.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    tmp.replace(path)

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Extract the abbreviations key from a directory's front matter."
    )
    parser.add_argument("year_vol", help="Directory year-volume string, e.g. 1900_01")
    parser.add_argument("pdf", help="Path to the directory PDF file")
    parser.add_argument(
        "--page",
        type=int,
        metavar="N",
        help="Single 1-indexed page number containing the abbreviations table.",
    )
    parser.add_argument(
        "--pages",
        metavar="PAGES",
        help='Page range, e.g. "31-32", if the table spans multiple pages. '
             "Overrides page_layout.json if given; --page overrides --pages.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite config/<year_vol>/abbreviations.json if it already exists.",
    )
    args = parser.parse_args()

    config_dir = Path("config") / args.year_vol
    out_path = config_dir / "abbreviations.json"

    if out_path.exists() and not args.force:
        print(f"ERROR: {out_path} already exists. Pass --force to overwrite it.")
        sys.exit(1)

    if args.page:
        pages = [args.page]
    elif args.pages:
        pages = parse_pages(args.pages, args.pdf)
    else:
        layout = load_page_layout(config_dir)
        pages = abbreviations_page_range(layout)
        if pages is None:
            print(
                "No --page/--pages flag given and page_layout.json does not "
                "define an abbreviations block with page or "
                "page_start/page_end. Pass --page explicitly."
            )
            sys.exit(1)
        print(f"Using page(s) from page_layout.json: {pages}")

    print(f"Extracting abbreviations from {args.pdf}, page(s) {pages} ...")
    entries, failed = extract_abbreviations(args.pdf, pages)

    if failed:
        print("\nFailed page(s):")
        for f in failed:
            print(f"  Page {f['page']}: {f['error']}")

    if not entries:
        print("\nNo abbreviations extracted. Not writing an empty file.")
        sys.exit(1)

    atomic_write(out_path, entries)
    print(f"\nWrote {len(entries)} abbreviations -> {out_path}")


if __name__ == "__main__":
    main()
