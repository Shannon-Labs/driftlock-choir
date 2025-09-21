#!/usr/bin/env python3
"""
Package Artifacts for Investor Handoffs or S3 Sync.

Zips the latest acceptance + baseline MP4s, telemetry JSONL, composite spritesheet, digest, and site_data JSON.
"""

import argparse
from pathlib import Path
import shutil
import zipfile

def main():
    parser = argparse.ArgumentParser(description="Package Driftlock artifacts.")
    parser.add_argument("--output-dir", default="deploy/artifacts", help="Output directory for zip")
    parser.add_argument("--zip-name", default="driftlock_artifacts.zip", help="Zip filename")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Source paths
    sources = [
        "driftlock_choir_sim/outputs/movies/demo_choir_sim.mp4",
        "driftlock_choir_sim/outputs/movies/baseline_choir_sim.mp4",
        "driftlock_choir_sim/outputs/movies/demo_teaser_choir_sim.mp4",
        "driftlock_choir_sim/outputs/movies/baseline_teaser_choir_sim.mp4",
        "driftlock_choir_sim/outputs/comparisons/side_by_side/comparison.mp4",
        "driftlock_choir_sim/outputs/comparisons/side_by_side/spritesheet.png",
        "driftlock_choir_sim/outputs/pulse/digest.md",
        "driftlock_choir_sim/outputs/pulse/acceptance_summary.json",
        "docs/site_data/index.json"
    ]

    zip_path = output_dir / args.zip_name
    with zipfile.ZipFile(zip_path, 'w') as zf:
        for source in sources:
            if Path(source).exists():
                zf.write(source, Path(source).name)
            else:
                print(f"Warning: {source} not found, skipping.")

    print(f"Artifacts packaged to {zip_path}")

if __name__ == "__main__":
    main()