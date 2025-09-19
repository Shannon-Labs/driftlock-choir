#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
OUT_DIR="${ROOT_DIR}/dist/pct_packet"
FIG_SRC="${ROOT_DIR}/patent/figures"
FIG_OUT_PNG="${OUT_DIR}/figures/png"
FIG_OUT_SVG="${OUT_DIR}/figures/svg"

mkdir -p "$OUT_DIR" "$FIG_OUT_PNG" "$FIG_OUT_SVG"

echo "⟹ Preparing PCT spec"
cp -f "${ROOT_DIR}/patent/PROVISIONAL_PATENT_APPLICATION.html" "$OUT_DIR/spec.html"
cp -f "${ROOT_DIR}/patent/ABSTRACT.txt" "$OUT_DIR/abstract.txt" 2>/dev/null || true
cp -f "${ROOT_DIR}/patent/Assignment_and_Entity.md" "$OUT_DIR/assignment.md" 2>/dev/null || true

echo "⟹ Exporting SVG figures to PNG (600 DPI, B/W)"
STRICT_BW=1 "${ROOT_DIR}/scripts/export_patent_assets.sh" >/dev/null 2>&1 || true

echo "⟹ Copying figure outputs"
cp -rf "${ROOT_DIR}/dist/patent_packet/figures/svg" "$FIG_OUT_SVG" || true
cp -rf "${ROOT_DIR}/dist/patent_packet/figures/png" "$FIG_OUT_PNG" || true

echo "⟹ Building drawings.pdf"
if command -v convert >/dev/null 2>&1; then
  convert "$FIG_OUT_PNG"/*.png "$OUT_DIR/drawings.pdf" 2>/dev/null || echo "⚠︎ convert failed to build drawings.pdf"
fi

echo "⟹ Generating spec.pdf"
if command -v chromium >/dev/null 2>&1; then
  chromium --headless --disable-gpu --print-to-pdf="$OUT_DIR/spec.pdf" "file://$OUT_DIR/spec.html" || true
elif command -v google-chrome >/dev/null 2>&1; then
  google-chrome --headless --disable-gpu --print-to-pdf="$OUT_DIR/spec.pdf" "file://$OUT_DIR/spec.html" || true
elif command -v wkhtmltopdf >/dev/null 2>&1; then
  wkhtmltopdf "$OUT_DIR/spec.html" "$OUT_DIR/spec.pdf" || true
else
  echo "⚠︎ No HTML→PDF tool found (chromium/google-chrome/wkhtmltopdf). spec.pdf not generated."
fi

echo "⟹ Wrote $OUT_DIR"

