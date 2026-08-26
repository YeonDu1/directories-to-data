"""
03_parse_entries.py

Parses verbatim directory lines into structured fields using Gemini 3.5 Flash.

Two things make this much faster than the sequential version:

  1. max_output_tokens is set explicitly. Without it the SDK falls back to
     a low default, and since thinking tokens draw from the same budget,
     responses were being cut off mid-JSON well before the model's real
     65k output ceiling. That was the cause of the constant
     JSONDecodeErrors, not genuine overflow.

  2. Batches run concurrently in a thread pool. These calls are I/O bound
     (waiting on the network), so running several at once is close to a
     linear speedup.

Order is preserved regardless of which batch finishes first, and the
checkpoint tracks completed batches by index so an interrupted run
resumes without redoing work or corrupting ordering.

Output: data/<year_vol>/<year_vol>_parsed.json

Usage:
    python scripts/03_parse_entries.py 1900_01
    python scripts/03_parse_entries.py 1900_01 --workers 12
    python scripts/03_parse_entries.py 1900_01 --limit 200   # spot-check

Set your API key before running, either in the repo-root .env file:

    GEMINI_API_KEY=your-key-here

or as a shell variable, which takes precedence over .env:

    export GEMINI_API_KEY="your-key-here"
"""

import os
import sys
import json
import time
import argparse
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from google import genai
from google.genai import types
from langsmith import traceable      # pip install langsmith
from dotenv import load_dotenv    # pip install python-dotenv

# ---------------------------------------------------------------------------
# Client setup
# ---------------------------------------------------------------------------

# Reads GEMINI_API_KEY from the repo-root .env if it is not already in the
# environment. An exported shell variable still wins over the file.
load_dotenv()

client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
MODEL  = "gemini-3.5-flash"

BATCH_SIZE  = 25     # lines per API call
MAX_WORKERS = 8      # concurrent API calls

# Set explicitly. This is the fix for the truncation problem: without it
# the effective output budget was far below the model's real capacity.
MAX_OUTPUT_TOKENS = 32768

# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

def edition_context(layout: dict) -> str:
    """
    Describe this directory edition for the prompt, from page_layout.json's
    optional "edition" block ({"city", "year_label", "publisher"}). Falls
    back to generic wording for any field that's missing, so an edition
    with no block at all (or a partial one) still works.
    """
    edition = layout.get("edition", {})
    city = edition.get("city")
    year_label = edition.get("year_label")
    publisher = edition.get("publisher")

    if city and year_label:
        desc = f"the {year_label} {city} city directory"
    elif city:
        desc = f"a historical {city} city directory"
    elif year_label:
        desc = f"a historical city directory from {year_label}"
    else:
        desc = "a historical city directory"

    if publisher:
        desc += f", published by {publisher}"
    return desc


PARSE_PROMPT_TEMPLATE = """\
These are verbatim transcribed lines from __EDITION_DESC__. Each line is
one directory entry.

Parse each line into structured fields and return a JSON array with
one object per input line, in the same order:

[
  {
    "n": 1,
    "surname": "Bailey",
    "given_name": "Henry C. jr.",
    "qualifier": null,
    "occupation": "clk G. H. & S. A. Ry",
    "residence_type": "h",
    "address": "1116 Capitol ave.",
    "phone": null,
    "race_marker": false,
    "entry_type": "resident",
    "notes": null
  }
]

Field definitions:
- n: the number of the input line this object corresponds to, matching
  the numbering in the input list below. Always include this.
- surname: the family name as printed. For businesses or institutions,
  use the first word or company name.
- given_name: first name and any middle initials or suffixes as printed.
  Null for businesses, institutions, and cross-references.
- qualifier: anything in parentheses directly after the name, such as
  (wid John) for widow, (Mrs. F. B.) for married name, or (c) for the
  racial designation marker used in this directory. Null if absent.
- occupation: the occupation or business description as printed,
  abbreviations not expanded. Null if absent.
- residence_type: the abbreviation immediately before the address:
  "h" (head of household), "r" (resides), "bds" (boards),
  "rms" (rooms), or null if no residence type marker is present.
- address: the street address as printed. Null if absent.
- phone: any phone number mentioned, as printed. Null if absent.
- race_marker: true if (c) appears in this entry, false otherwise.
  This marker was used in this era to designate Black residents and is
  preserved here for historical research.
- entry_type: one of
    "resident"        individual person listing
    "business"        a business or institution without a named individual
    "cross_reference" a "See also..." or similar reference line
    "partnership"     a firm listing with named partners
    "other"           anything that does not fit the above
- notes: anything else in the entry that did not fit the fields above,
  such as a death notice or an unusual format. Null if nothing to note.

Rules:
- Return exactly one object per input line, in the same order, with n
  matching the input line number. Never merge, skip, or reorder lines.
- Do NOT echo the original line text back. The n field is enough to
  match it up. Keeping the output compact is important.
- Do not expand abbreviations. Field values exactly as printed.
- If a line cannot be parsed at all, set every field to null except n
  and entry_type, and set entry_type to "other".
- Return only the JSON array. No explanation, no markdown fences.
"""


def build_parse_prompt(edition_desc: str) -> str:
    return PARSE_PROMPT_TEMPLATE.replace("__EDITION_DESC__", edition_desc)

# ---------------------------------------------------------------------------
# API call
# ---------------------------------------------------------------------------

@traceable(name="parse_entry_batch")
def call_gemini(numbered_lines: str, prompt_text: str) -> list:
    """One API call. Returns the parsed JSON list."""
    response = client.models.generate_content(
        model=MODEL,
        contents=[
            types.Content(
                role="user",
                parts=[types.Part.from_text(
                    text=prompt_text + f"\n\nLines to parse:\n{numbered_lines}"
                )],
            )
        ],
        config=types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(thinking_level="medium"),
            temperature=0,
            response_mime_type="application/json",
            max_output_tokens=MAX_OUTPUT_TOKENS,
        ),
    )
    parsed = json.loads(response.text)
    if not isinstance(parsed, list):
        raise ValueError(f"Expected JSON list, got {type(parsed).__name__}")
    return parsed


def parse_lines(lines: list[dict], prompt_text: str, depth: int = 0) -> tuple[list[dict], list[dict]]:
    """
    Parse a list of entry lines. Returns (parsed, failed).

    With max_output_tokens set correctly this should rarely need to split,
    but the split-and-retry is kept as a safety net for genuinely
    oversized batches or transient API issues.
    """
    numbered = "\n".join(f"{i+1}. {item['line']}" for i, item in enumerate(lines))

    try:
        raw = call_gemini(numbered, prompt_text)
        if len(raw) != len(lines):
            raise ValueError(f"count mismatch: sent {len(lines)}, got {len(raw)}")

        results = []
        for i, obj in enumerate(raw):
            if not isinstance(obj, dict):
                obj = {}
            obj["raw"]  = lines[i]["line"]     # re-attach locally
            obj["page"] = lines[i]["page"]
            obj.pop("n", None)
            results.append(obj)
        return results, []

    except KeyboardInterrupt:
        raise
    except Exception as e:
        if len(lines) == 1:
            return [], [{**lines[0], "error": str(e)[:200]}]
        mid = len(lines) // 2
        l_ok, l_bad = parse_lines(lines[:mid], prompt_text, depth + 1)
        r_ok, r_bad = parse_lines(lines[mid:], prompt_text, depth + 1)
        return l_ok + r_ok, l_bad + r_bad


def normalize_parsed(obj: dict) -> dict:
    expected = [
        "page", "raw", "surname", "given_name", "qualifier",
        "occupation", "residence_type", "address", "phone",
        "race_marker", "entry_type", "notes",
    ]
    return {k: obj.get(k) for k in expected}

# ---------------------------------------------------------------------------
# File I/O
# ---------------------------------------------------------------------------

def load_entries(year_vol: str) -> list[dict]:
    p = Path("data") / year_vol / f"{year_vol}_entries.json"
    if not p.exists():
        raise FileNotFoundError(f"{p} not found. Run 02_merge_pages.py first.")
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def load_page_layout(config_dir: Path) -> dict:
    layout_path = config_dir / "page_layout.json"
    if not layout_path.exists():
        return {}
    with open(layout_path, encoding="utf-8") as f:
        return json.load(f)


def atomic_write(path: Path, data):
    """
    Write via a temp file then rename. Protects against a half-written
    file if the run is interrupted, and avoids the write timeouts that
    can happen on cloud-synced folders.
    """
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    tmp.replace(path)


def checkpoint_path(year_vol: str) -> Path:
    return Path("data") / year_vol / "_parse_checkpoint.json"


def load_checkpoint(path: Path) -> dict:
    """Returns {batch_index(str): {"parsed": [...], "failed": [...]}}"""
    if not path.exists():
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) and "parsed" not in data else {}
    except Exception:
        return {}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def chunk_list(items, size):
    return [items[i:i + size] for i in range(0, len(items), size)]


def main():
    ap = argparse.ArgumentParser(description="Parse directory lines into structured fields.")
    ap.add_argument("year_vol")
    ap.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    ap.add_argument("--workers", type=int, default=MAX_WORKERS,
                    help=f"Concurrent API calls (default {MAX_WORKERS}). Lower it if you hit rate limits.")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--restart", action="store_true")
    args = ap.parse_args()

    layout = load_page_layout(Path("config") / args.year_vol)
    edition_desc = edition_context(layout)
    prompt_text = build_parse_prompt(edition_desc)
    print(f"Edition: {edition_desc}")

    print(f"Loading entries from {args.year_vol}_entries.json ...")
    entries = load_entries(args.year_vol)
    if args.limit:
        entries = entries[: args.limit]
        print(f"  Limiting to first {args.limit} entries.")
    print(f"  {len(entries)} entries to parse.")

    ck_path = checkpoint_path(args.year_vol)
    if args.restart and ck_path.exists():
        ck_path.unlink()
        print("  Checkpoint discarded (--restart).")

    done = load_checkpoint(ck_path)
    batches = chunk_list(entries, args.batch_size)
    todo = [(i, b) for i, b in enumerate(batches) if str(i) not in done]

    if done:
        print(f"  Resuming: {len(done)} of {len(batches)} batches already done.")
    print(f"\nRunning {len(todo)} batches with {args.workers} concurrent workers ...\n")

    lock = threading.Lock()
    completed = 0
    start = time.time()

    def work(item):
        idx, batch = item
        ok, bad = parse_lines(batch, prompt_text)
        return idx, ok, bad

    try:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(work, item): item[0] for item in todo}
            for fut in as_completed(futures):
                idx, ok, bad = fut.result()
                with lock:
                    done[str(idx)] = {
                        "parsed": [normalize_parsed(o) for o in ok],
                        "failed": bad,
                    }
                    completed += 1
                    if completed % 10 == 0 or completed == len(todo):
                        elapsed = time.time() - start
                        rate = completed / elapsed if elapsed else 0
                        remaining = (len(todo) - completed) / rate if rate else 0
                        print(f"  {completed}/{len(todo)} batches  "
                              f"({rate*args.batch_size:.0f} entries/s, "
                              f"~{remaining/60:.1f} min left)")
                        atomic_write(ck_path, done)
    except KeyboardInterrupt:
        print("\n  Interrupted. Saving progress ...")
        atomic_write(ck_path, done)

    # Reassemble in original batch order
    all_parsed, all_failed = [], []
    for i in range(len(batches)):
        rec = done.get(str(i))
        if rec:
            all_parsed.extend(rec["parsed"])
            all_failed.extend(rec["failed"])

    out = Path("data") / args.year_vol / f"{args.year_vol}_parsed.json"
    atomic_write(out, all_parsed)
    print(f"\nSaved {len(all_parsed)} parsed entries -> {out}")

    fpath = Path("data") / args.year_vol / f"{args.year_vol}_parse_failures.json"
    if all_failed:
        atomic_write(fpath, all_failed)
        print(f"Recorded {len(all_failed)} unparseable lines -> {fpath}")
    elif fpath.exists():
        fpath.unlink()
        print(f"No unparseable lines this run — removed stale {fpath}")

    if len(done) == len(batches):
        atomic_write(ck_path, done)   # keep it; harmless and allows re-runs
        print("\n=== Parse complete ===")
    else:
        print(f"\n=== Partial: {len(done)}/{len(batches)} batches done, re-run to continue ===")

    print(f"  Parsed     : {len(all_parsed)}")
    print(f"  Unparseable: {len(all_failed)}")
    types_count = {}
    for o in all_parsed:
        t = o.get("entry_type") or "unknown"
        types_count[t] = types_count.get(t, 0) + 1
    for t, c in sorted(types_count.items(), key=lambda x: -x[1]):
        print(f"    {t:<18} {c}")


if __name__ == "__main__":
    main()