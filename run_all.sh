#!/bin/bash
#
# run_all.sh — full pipeline for one directory edition, unattended.
#
# Usage:
#   ./run_all.sh 1900_01 data/1900_01/1900-1901.pdf
#
# Each stage only runs if the previous one succeeded. All output goes to
# a timestamped log file in logs/ so you can read what happened after.

set -euo pipefail

YEAR_VOL="${1:?Usage: ./run_all.sh <year_vol> <pdf_path>}"
PDF_PATH="${2:?Usage: ./run_all.sh <year_vol> <pdf_path>}"

mkdir -p logs
STAMP=$(date +%Y%m%d_%H%M%S)
LOG="logs/run_${YEAR_VOL}_${STAMP}.log"

# The Python stages load .env themselves via python-dotenv, but this guard runs
# first, so load it here too. python-dotenv does not override variables already
# in the environment, so preserve an exported key across the source to match.
_exported_key="${GEMINI_API_KEY:-}"
if [ -f .env ]; then
    set -a
    # shellcheck disable=SC1091
    . ./.env
    set +a
fi
if [ -n "$_exported_key" ]; then
    export GEMINI_API_KEY="$_exported_key"
fi

# Fail early rather than 30 seconds into an overnight run
if [ -z "${GEMINI_API_KEY:-}" ]; then
    echo "ERROR: GEMINI_API_KEY is not set. Put it in .env or export it before running."
    exit 1
fi

if [ ! -f "$PDF_PATH" ]; then
    echo "ERROR: PDF not found at $PDF_PATH"
    exit 1
fi

{
    echo "=================================================="
    echo "Pipeline run for $YEAR_VOL"
    echo "Started: $(date)"
    echo "PDF: $PDF_PATH"
    echo "=================================================="

    echo ""
    echo ">>> STAGE 1/3: Transcription (01_ocr_entries.py)"
    echo ">>> $(date)"
    python -u scripts/01_ocr_entries.py "$YEAR_VOL" "$PDF_PATH" --batch-size 4

    echo ""
    echo ">>> STAGE 2/3: Merge (02_merge_pages.py)"
    echo ">>> $(date)"
    python -u scripts/02_merge_pages.py "$YEAR_VOL"

    echo ""
    echo ">>> STAGE 3/3: Parse (03_parse_entries.py)"
    echo ">>> $(date)"
    python -u scripts/03_parse_entries.py "$YEAR_VOL" --restart --workers 8

    echo ""
    echo "=================================================="
    echo "ALL STAGES COMPLETE"
    echo "Finished: $(date)"
    echo "=================================================="

    echo ""
    echo "Output files:"
    ls -lh "data/${YEAR_VOL}/${YEAR_VOL}_entries.json" 2>/dev/null || true
    ls -lh "data/${YEAR_VOL}/${YEAR_VOL}_parsed.json" 2>/dev/null || true
    ls -lh "data/${YEAR_VOL}/${YEAR_VOL}_parse_failures.json" 2>/dev/null || true
    echo "Transcribed pages: $(ls data/${YEAR_VOL}/pages/*.json 2>/dev/null | wc -l)"

} 2>&1 | tee "$LOG"

echo ""
echo "Log saved to: $LOG"