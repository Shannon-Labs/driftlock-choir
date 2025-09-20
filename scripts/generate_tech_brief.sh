#!/bin/bash

# Generate Technical Brief PDF from Markdown
# For a16z Speedrun reviewers who want offline docs

echo "📄 Generating Driftlock Technical Brief..."

# Check if pandoc is installed
if ! command -v pandoc &> /dev/null; then
    echo "⚠️  Pandoc not found. Installing..."
    if [[ "$OSTYPE" == "darwin"* ]]; then
        brew install pandoc
    else
        echo "Please install pandoc: https://pandoc.org/installing.html"
        exit 1
    fi
fi

# Create output directory
mkdir -p dist/briefs

# Generate main technical brief
echo "Converting docs/results_extended_010.md to PDF..."
pandoc docs/results_extended_010.md \
    -o dist/briefs/driftlock_tech_brief_2025.pdf \
    --pdf-engine=pdflatex \
    --template=default \
    -V geometry:margin=1in \
    -V colorlinks=true \
    -V linkcolor=blue \
    -V urlcolor=blue \
    --toc \
    --metadata title="Driftlock: 22 Picosecond Wireless Synchronization" \
    --metadata author="Hunter Bown, Shannon Labs" \
    --metadata date="September 2025" \
    2>/dev/null || {
    # Fallback to HTML if LaTeX not available
    echo "LaTeX not available, generating HTML instead..."
    pandoc docs/results_extended_010.md \
        -o dist/briefs/driftlock_tech_brief_2025.html \
        --standalone \
        --toc \
        --metadata title="Driftlock: 22 Picosecond Wireless Synchronization" \
        --metadata author="Hunter Bown, Shannon Labs" \
        --metadata date="September 2025"
}

# Combine key docs into comprehensive brief
echo "Creating comprehensive technical package..."
cat > dist/briefs/full_technical_package.md << EOF
# Driftlock Technical Package
*a16z Speedrun Submission - September 2025*

---

EOF

# Append results
cat docs/results_extended_010.md >> dist/briefs/full_technical_package.md

echo -e "\n\n---\n\n" >> dist/briefs/full_technical_package.md

# Append quickstart
echo "## Quickstart Guide" >> dist/briefs/full_technical_package.md
cat docs/quickstart.md >> dist/briefs/full_technical_package.md

echo -e "\n\n---\n\n" >> dist/briefs/full_technical_package.md

# Append technical summary from README
echo "## Technical Overview" >> dist/briefs/full_technical_package.md
sed -n '/## Technical Innovation/,/## Quick Start/p' README.md >> dist/briefs/full_technical_package.md

# Convert comprehensive package
pandoc dist/briefs/full_technical_package.md \
    -o dist/briefs/driftlock_full_package_2025.pdf \
    --pdf-engine=xelatex \
    -V geometry:margin=1in \
    -V colorlinks=true \
    2>/dev/null || {
    pandoc dist/briefs/full_technical_package.md \
        -o dist/briefs/driftlock_full_package_2025.html \
        --standalone \
        --toc
}

echo "✅ Technical briefs generated in dist/briefs/"
echo ""
echo "Available documents:"
ls -la dist/briefs/
echo ""
echo "📧 Ready to attach to investor emails!"