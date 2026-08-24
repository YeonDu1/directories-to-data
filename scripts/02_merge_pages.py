"""
02_merge_pages.py

Combines all per-page transcription files from data/<year_vol>/pages/
into one flat master JSON file: data/<year_vol>/<year_vol>_entries.json

Each entry in the master file retains its source page number so the
original location in the directory is always traceable.

Run this after 01_ocr_entries.py has finished processing all pages.

Usage:
    python scripts/02_merge_pages.py 1900_01
"""

import json
import sys
from pathlib import Path


def merge_pages(year_vol: str):
    pages_dir = Path("data") / year_vol / "pages"
    out_path  = Path("data") / year_vol / f"{year_vol}_entries.json"

    if not pages_dir.exists():
        print(f"ERROR: {pages_dir} does not exist. Run 01_ocr_entries.py first.")
        sys.exit(1)

    page_files = sorted(pages_dir.glob("page_*.json"))
    if not page_files:
        print(f"No page_XXXX.json files found in {pages_dir}.")
        sys.exit(1)

    print(f"Merging {len(page_files)} page files from {pages_dir} ...")

    all_entries = []
    empty_pages = []

    for page_file in page_files:
        page_num = int(page_file.stem.split("_")[1])
        with open(page_file, encoding="utf-8") as f:
            lines = json.load(f)

        if not lines:
            empty_pages.append(page_num)
            continue

        for item in lines:
            # Handle both {"line": "..."} shape and bare string shape defensively
            text = item["line"] if isinstance(item, dict) else str(item)
            all_entries.append({
                "page": page_num,
                "line": text,
            })

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_entries, f, indent=2, ensure_ascii=False)

    print(f"\n=== Merge complete ===")
    print(f"  Pages merged  : {len(page_files) - len(empty_pages)}")
    if empty_pages:
        print(f"  Empty pages   : {empty_pages}")
    print(f"  Total entries : {len(all_entries)}")
    print(f"  Output        : {out_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/02_merge_pages.py <year_vol>")
        print("  e.g. python scripts/02_merge_pages.py 1900_01")
        sys.exit(1)
    merge_pages(sys.argv[1])