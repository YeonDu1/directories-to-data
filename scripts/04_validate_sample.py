"""
04_validate_sample.py

Randomized human validation of transcription accuracy.

Two modes:

  --sample N   Randomly select N pages from the completed transcription
               output and create blank worksheets for you to transcribe
               by hand. Records which pages were chosen (and the random
               seed) so the sample is reproducible and auditable.

  --compare    Read your hand transcriptions back in, diff them line by
               line against the pipeline output, and report accuracy.

The point is to produce a real accuracy number that can be cited, rather
than an impression from spot-checking. This mirrors the validation logic
in Albers & Kappner (2023), comparing automated output against an
independent reference, scaled down to a hand-transcribed sample.

Usage:
    # Step 1 - draw the sample
    python scripts/04_validate_sample.py 1900_01 --sample 5

    # Step 2 - open each file in validation/1900_01/manual/ and type the
    #          page's entries by hand, one entry per line, looking ONLY at
    #          the scanned PDF page. Do not look at the pipeline output.

    # Step 3 - score it
    python scripts/04_validate_sample.py 1900_01 --compare
"""

import json
import random
import argparse
import sys
from difflib import SequenceMatcher
from pathlib import Path


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

def validation_dir(year_vol: str) -> Path:
    return Path("validation") / year_vol


def manual_dir(year_vol: str) -> Path:
    return validation_dir(year_vol) / "manual"


def sample_record_path(year_vol: str) -> Path:
    return validation_dir(year_vol) / "sample_pages.json"


def pages_dir(year_vol: str) -> Path:
    return Path("data") / year_vol / "pages"


# ---------------------------------------------------------------------------
# Loading pipeline output
# ---------------------------------------------------------------------------

def load_pipeline_page(year_vol: str, page_num: int) -> list[str]:
    """Return the pipeline's transcribed lines for one page, as plain strings."""
    path = pages_dir(year_vol) / f"page_{page_num:04d}.json"
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return [item["line"] if isinstance(item, dict) else str(item) for item in data]


def available_pages(year_vol: str) -> list[int]:
    """All page numbers that have non-empty pipeline output."""
    d = pages_dir(year_vol)
    if not d.exists():
        print(f"ERROR: {d} does not exist. Run 01_ocr_entries.py first.")
        sys.exit(1)

    pages = []
    for f in sorted(d.glob("page_*.json")):
        with open(f, encoding="utf-8") as fh:
            data = json.load(fh)
        # Skip pages that are empty or nearly empty (full-page ads), since
        # they are not a meaningful test of transcription accuracy.
        if len(data) >= 10:
            pages.append(int(f.stem.split("_")[1]))
    return pages


# ---------------------------------------------------------------------------
# Mode 1: draw the sample
# ---------------------------------------------------------------------------

def draw_sample(year_vol: str, n: int, seed: int | None):
    pages = available_pages(year_vol)
    if len(pages) < n:
        print(f"ERROR: only {len(pages)} eligible pages available, cannot sample {n}.")
        sys.exit(1)

    if seed is None:
        seed = random.randrange(1_000_000)
    rng = random.Random(seed)
    chosen = sorted(rng.sample(pages, n))

    mdir = manual_dir(year_vol)
    mdir.mkdir(parents=True, exist_ok=True)

    for page_num in chosen:
        worksheet = mdir / f"page_{page_num:04d}_manual.txt"
        if worksheet.exists():
            print(f"  Worksheet already exists, leaving alone: {worksheet.name}")
            continue
        worksheet.write_text(
            f"# Manual transcription worksheet - PDF page {page_num}\n"
            f"#\n"
            f"# Open page {page_num} of the source PDF and type every directory\n"
            f"# entry below, one entry per line, in reading order (left column\n"
            f"# top to bottom, then right column). Merge wrapped lines into one\n"
            f"# entry. Skip advertisements. Do NOT look at the pipeline output\n"
            f"# while doing this - that would defeat the purpose.\n"
            f"#\n"
            f"# Lines starting with # are ignored by the scorer.\n"
            f"\n",
            encoding="utf-8",
        )
        print(f"  Created {worksheet}")

    record = {
        "year_vol": year_vol,
        "seed": seed,
        "sample_size": n,
        "eligible_page_count": len(pages),
        "sampled_pages": chosen,
    }
    sample_record_path(year_vol).parent.mkdir(parents=True, exist_ok=True)
    with open(sample_record_path(year_vol), "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2)

    print(f"\nSampled {n} of {len(pages)} eligible pages (seed {seed}): {chosen}")
    print(f"Worksheets in: {mdir}")
    print(f"\nNext: transcribe each worksheet by hand, then run --compare")


# ---------------------------------------------------------------------------
# Mode 2: score it
# ---------------------------------------------------------------------------

def read_manual(path: Path) -> list[str]:
    """
    Read a hand-transcribed worksheet, dropping comments and blank lines.

    Accepts either plain text (one entry per line) or a JSON array of
    {"line": "..."} objects matching the pipeline's own page_XXXX.json
    shape, since that format is also natural to hand-type entry by entry.
    """
    body = "\n".join(
        raw for raw in path.read_text(encoding="utf-8").splitlines()
        if raw.strip() and not raw.strip().startswith("#")
    )

    try:
        data = json.loads(body)
    except json.JSONDecodeError:
        data = None

    if isinstance(data, list):
        return [
            item["line"] if isinstance(item, dict) and "line" in item else str(item)
            for item in data
        ]

    return [raw.strip() for raw in body.splitlines() if raw.strip()]


def similarity(a: str, b: str) -> float:
    """Character-level similarity ratio between two strings, 0.0 to 1.0."""
    return SequenceMatcher(None, a, b).ratio()


def align_and_score(manual: list[str], pipeline: list[str], threshold: float):
    """
    Align the two line lists using difflib, then classify each line as
    an exact match, a near match (same entry, minor character differences),
    a pipeline-only line (spurious), or a manual-only line (missed).
    """
    matcher = SequenceMatcher(None, manual, pipeline)

    exact = 0
    near = []          # (manual_line, pipeline_line, ratio)
    missed = []        # in manual, not in pipeline
    spurious = []      # in pipeline, not in manual

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            exact += (i2 - i1)
        elif tag == "replace":
            m_block = manual[i1:i2]
            p_block = pipeline[j1:j2]
            # Pair them off positionally within the replaced block
            for k in range(max(len(m_block), len(p_block))):
                m = m_block[k] if k < len(m_block) else None
                p = p_block[k] if k < len(p_block) else None
                if m is not None and p is not None:
                    ratio = similarity(m, p)
                    if ratio >= threshold:
                        near.append((m, p, ratio))
                    else:
                        missed.append(m)
                        spurious.append(p)
                elif m is not None:
                    missed.append(m)
                else:
                    spurious.append(p)
        elif tag == "delete":
            missed.extend(manual[i1:i2])
        elif tag == "insert":
            spurious.extend(pipeline[j1:j2])

    return exact, near, missed, spurious


def compare(year_vol: str, threshold: float):
    record_path = sample_record_path(year_vol)
    if not record_path.exists():
        print(f"ERROR: {record_path} not found. Run --sample first.")
        sys.exit(1)

    with open(record_path, encoding="utf-8") as f:
        record = json.load(f)

    mdir = manual_dir(year_vol)
    results = []

    print(f"Scoring {year_vol} against hand transcriptions in {mdir}\n")

    for page_num in record["sampled_pages"]:
        worksheet = mdir / f"page_{page_num:04d}_manual.txt"
        if not worksheet.exists():
            print(f"  Page {page_num}: worksheet missing, skipping.")
            continue

        manual = read_manual(worksheet)
        if not manual:
            print(f"  Page {page_num}: worksheet is empty, skipping.")
            continue

        pipeline = load_pipeline_page(year_vol, page_num)
        exact, near, missed, spurious = align_and_score(manual, pipeline, threshold)

        # Character-level accuracy across matched pairs
        char_scores = [1.0] * exact + [r for _, _, r in near]
        char_acc = sum(char_scores) / len(char_scores) if char_scores else 0.0

        line_acc = exact / len(manual) if manual else 0.0

        results.append({
            "page": page_num,
            "manual_lines": len(manual),
            "pipeline_lines": len(pipeline),
            "exact": exact,
            "near": len(near),
            "missed": len(missed),
            "spurious": len(spurious),
            "line_accuracy": line_acc,
            "char_accuracy": char_acc,
            "near_detail": near,
            "missed_detail": missed,
            "spurious_detail": spurious,
        })

        print(f"  Page {page_num}: {len(manual)} manual / {len(pipeline)} pipeline lines")
        print(f"    exact {exact}  near {len(near)}  missed {len(missed)}  spurious {len(spurious)}")
        print(f"    line accuracy {line_acc:.1%}   char accuracy {char_acc:.1%}")

    if not results:
        print("\nNo worksheets scored. Fill in at least one before comparing.")
        sys.exit(1)

    # Aggregate
    tot_manual   = sum(r["manual_lines"] for r in results)
    tot_exact    = sum(r["exact"] for r in results)
    tot_near     = sum(r["near"] for r in results)
    tot_missed   = sum(r["missed"] for r in results)
    tot_spurious = sum(r["spurious"] for r in results)

    overall_line = tot_exact / tot_manual if tot_manual else 0.0
    all_char = []
    for r in results:
        all_char.extend([1.0] * r["exact"] + [x[2] for x in r["near_detail"]])
    overall_char = sum(all_char) / len(all_char) if all_char else 0.0

    print("\n=== Overall ===")
    print(f"  Pages scored          : {len(results)}")
    print(f"  Manual lines total    : {tot_manual}")
    print(f"  Exact line matches    : {tot_exact}")
    print(f"  Near matches          : {tot_near}")
    print(f"  Missed by pipeline    : {tot_missed}")
    print(f"  Spurious in pipeline  : {tot_spurious}")
    print(f"  Line-level accuracy   : {overall_line:.2%}")
    print(f"  Character-level accuracy: {overall_char:.2%}")

    # Write full report
    out_path = validation_dir(year_vol) / "validation_report.json"
    report = {
        "year_vol": year_vol,
        "seed": record["seed"],
        "sampled_pages": record["sampled_pages"],
        "near_match_threshold": threshold,
        "totals": {
            "pages_scored": len(results),
            "manual_lines": tot_manual,
            "exact_matches": tot_exact,
            "near_matches": tot_near,
            "missed": tot_missed,
            "spurious": tot_spurious,
            "line_accuracy": overall_line,
            "char_accuracy": overall_char,
        },
        "per_page": [
            {k: v for k, v in r.items() if not k.endswith("_detail")}
            for r in results
        ],
        "discrepancies": [
            {
                "page": r["page"],
                "near_matches": [
                    {"manual": m, "pipeline": p, "similarity": round(ratio, 4)}
                    for m, p, ratio in r["near_detail"]
                ],
                "missed": r["missed_detail"],
                "spurious": r["spurious_detail"],
            }
            for r in results
            if r["near_detail"] or r["missed_detail"] or r["spurious_detail"]
        ],
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\nFull report with every discrepancy -> {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Randomized human validation of transcription accuracy."
    )
    parser.add_argument("year_vol", help="Directory year-volume string, e.g. 1900_01")
    parser.add_argument(
        "--sample",
        type=int,
        metavar="N",
        help="Draw a random sample of N pages and create blank worksheets",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducible sampling (recorded automatically if omitted)",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Score filled-in worksheets against pipeline output",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.90,
        help="Similarity ratio above which a non-exact line counts as a near match (default 0.90)",
    )
    args = parser.parse_args()

    if args.sample and args.compare:
        print("Pick one: --sample or --compare, not both.")
        sys.exit(1)

    if args.sample:
        draw_sample(args.year_vol, args.sample, args.seed)
    elif args.compare:
        compare(args.year_vol, args.threshold)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()