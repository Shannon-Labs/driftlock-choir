#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
OUT_DIR="${ROOT_DIR}/dist/patent_packet"
FIG_SRC="${ROOT_DIR}/patent/figures"
FIG_OUT_PNG="${OUT_DIR}/figures/png"
FIG_OUT_SVG="${OUT_DIR}/figures/svg"

mkdir -p "$OUT_DIR" "$FIG_OUT_PNG" "$FIG_OUT_SVG"

echo "⟹ Copying source materials"
cp -f "${ROOT_DIR}/patent/PROVISIONAL_PATENT_APPLICATION.html" "$OUT_DIR/spec.html"
cp -f "${ROOT_DIR}/patent/PROVISIONAL_PATENT_APPLICATION.md" "$OUT_DIR/spec.md"
cp -f "${ROOT_DIR}/patent/ABSTRACT.txt" "$OUT_DIR/abstract.txt" 2>/dev/null || true
cp -f "${ROOT_DIR}/patent/PATENT_SUMMARY.md" "$OUT_DIR/summary.md" 2>/dev/null || true
cp -f "${ROOT_DIR}/patent/Assignment_and_Entity.md" "$OUT_DIR/assignment.md" 2>/dev/null || true

echo "⟹ Exporting SVG figures to PNG (600 DPI, B/W style)"
convert_one() {
  local in_svg="$1"; local out_png="$2";
  if command -v inkscape >/dev/null 2>&1; then
    inkscape "$in_svg" --export-type=png --export-filename="$out_png" --export-dpi=600 >/dev/null 2>&1 && return 0
  fi
  if command -v rsvg-convert >/dev/null 2>&1; then
    rsvg-convert -d 600 -p 600 -o "$out_png" "$in_svg" && return 0
  fi
  if command -v convert >/dev/null 2>&1; then
    if [ "${STRICT_BW:-0}" = "1" ]; then
      convert -density 600 "$in_svg" -alpha off -colors 2 -monochrome "$out_png" && return 0
    else
      convert -density 600 "$in_svg" -colorspace Gray -type Grayscale -alpha off "$out_png" && return 0
    fi
  fi
  # Try Python + cairosvg if available
  python3 - <<'PY' "$in_svg" "$out_png" || true
import sys
try:
    import cairosvg
except Exception:
    sys.exit(1)
in_svg, out_png = sys.argv[1], sys.argv[2]
cairosvg.svg2png(url=in_svg, write_to=out_png, dpi=600)
PY
  if [ -f "$out_png" ]; then return 0; fi
  return 1
}

shopt -s nullglob
for svg in "$FIG_SRC"/*.svg; do
  base="$(basename "$svg")"
  cp -f "$svg" "$FIG_OUT_SVG/$base"
  png="$FIG_OUT_PNG/${base%.svg}.png"
  if convert_one "$svg" "$png"; then
    echo "✔ ${base} → $(basename "$png")"
  else
    echo "⚠︎ Could not convert $base to PNG (no converter found)."
  fi
done

echo "⟹ Assembling figures PDF (if ImageMagick available)"
if command -v convert >/dev/null 2>&1; then
  convert "$FIG_OUT_PNG"/*.png "$OUT_DIR/drawings.pdf" 2>/dev/null || echo "⚠︎ convert failed to build drawings.pdf"
fi

echo "⟹ Generating spec PDF from HTML (Chrome/Chromium/wkhtmltopdf if available)"
if command -v chromium >/dev/null 2>&1; then
  chromium --headless --disable-gpu --print-to-pdf="$OUT_DIR/spec.pdf" "file://$OUT_DIR/spec.html" || true
elif command -v google-chrome >/dev/null 2>&1; then
  google-chrome --headless --disable-gpu --print-to-pdf="$OUT_DIR/spec.pdf" "file://$OUT_DIR/spec.html" || true
elif command -v wkhtmltopdf >/dev/null 2>&1; then
  wkhtmltopdf "$OUT_DIR/spec.html" "$OUT_DIR/spec.pdf" || true
else
  echo "⚠︎ No HTML→PDF tool found (chromium/google-chrome/wkhtmltopdf). spec.pdf not generated."
fi

echo "⟹ Creating packet checklist"
cat > "$OUT_DIR/FILING_CHECKLIST.txt" <<'TXT'
Patent Center (EFS-Web successor) filing checklist — Provisional

[ ] Application data sheet (ADS) — complete in Patent Center UI
[ ] Specification PDF — spec.pdf (or upload spec.html and convert client-side)
[ ] Drawings PDF — drawings.pdf (ensure black/white line art)
[ ] Abstract — abstract.txt (≤150 words)
[ ] Entity/assignment details — assignment.md
[ ] Figures — individual PNG/SVG backups (in figures/)

Notes:
- Ensure the specification and drawings reference numerals match (see spec and figure SVGs).
- If PDFs not generated, use Chrome: File → Print → Save as PDF on spec.html, ensure margins adequate.
- Keep performance language illustrative (“example”, “non‑limiting”).
TXT

echo "⟹ Done: $OUT_DIR"
