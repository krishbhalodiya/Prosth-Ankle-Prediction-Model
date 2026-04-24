#!/usr/bin/env bash
# Export a notebook to PDF without needing a full LaTeX install.
#
# Usage:
#   scripts/export_pdf.sh [notebook.ipynb] [out.pdf]
#
# Defaults:
#   notebook = Project_Documentation_Spring_2026.ipynb
#   out      = <notebook basename>.pdf
#
# This uses nbconvert's "webpdf" exporter (headless Chromium via Playwright),
# so xelatex / MacTeX / tcolorbox are NOT required. The Cursor/VS Code
# "Export to PDF" button uses the LaTeX path instead and will fail on a
# BasicTeX install unless you `sudo tlmgr install tcolorbox adjustbox
# caption enumitem eurosym jknapltx parskip pgf rsfs soul titling
# trimspaces ucs ulem upquote needspace environ cancel etoolbox`.

set -euo pipefail

cd "$(dirname "$0")/.."

NOTEBOOK="${1:-Project_Documentation_Spring_2026.ipynb}"
DEFAULT_OUT="${NOTEBOOK%.ipynb}.pdf"
OUT="${2:-$DEFAULT_OUT}"

PYTHON="venv/bin/python"
if [ ! -x "$PYTHON" ]; then
    echo "Error: $PYTHON not found. Activate the project venv first." >&2
    exit 1
fi

if ! "$PYTHON" -c "import playwright" >/dev/null 2>&1; then
    echo ">> Installing playwright ..."
    "$PYTHON" -m pip install --quiet "playwright>=1.40"
fi

if [ ! -d "$HOME/Library/Caches/ms-playwright" ] || \
   ! ls "$HOME/Library/Caches/ms-playwright"/chromium* >/dev/null 2>&1; then
    echo ">> Installing chromium for playwright ..."
    "$PYTHON" -m playwright install chromium
fi

OUT_DIR="$(dirname "$OUT")"
OUT_NAME="$(basename "${OUT%.pdf}")"

echo ">> Converting $NOTEBOOK -> $OUT"
"$PYTHON" -m nbconvert "$NOTEBOOK" \
    --to webpdf \
    --output-dir "${OUT_DIR:-.}" \
    --output "$OUT_NAME"

FINAL="${OUT_DIR:-.}/${OUT_NAME}.pdf"
if [ -f "$FINAL" ]; then
    echo ">> Done: $FINAL ($(du -h "$FINAL" | cut -f1))"
else
    echo "!! nbconvert reported success but $FINAL is missing." >&2
    exit 1
fi
