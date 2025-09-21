#!/bin/bash
# Nightly Pulse Script - Run acceptance pipeline and archive results.

set -e

# Capture timestamp and git hash
TIMESTAMP=$(date -u +"%Y%m%d_%H%M%S")
GIT_HASH=$(git rev-parse --short HEAD)
OUTPUT_DIR="driftlock choir_choir_sim/outputs/history/${TIMESTAMP}_${GIT_HASH}"

mkdir -p "$OUTPUT_DIR"

# Run pulse acceptance
echo "Running nightly pulse at $TIMESTAMP (commit $GIT_HASH)"
python scripts/pulse_acceptance.py

# Copy outputs to history
cp -r driftlock choir_choir_sim/outputs/pulse/* "$OUTPUT_DIR/"

# Generate archive zip
zip -r "${OUTPUT_DIR}.zip" "$OUTPUT_DIR"

# Cleanup old runs (keep 30 days)
find driftlock choir_choir_sim/outputs/history -type d -mtime +30 -exec rm -rf {} + 2>/dev/null || true

# Upload to S3 (assume AWS CLI configured)
aws s3 cp "${OUTPUT_DIR}.zip" s3://driftlock choir-artifacts/history/ --acl public-read

echo "Nightly pulse complete. Archive: ${OUTPUT_DIR}.zip uploaded to S3."