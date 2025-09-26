#!/usr/bin/env python3
"""Run a sweep over the pathfinder relative threshold."""

import subprocess
import sys

THRESHOLDS = [-18.0, -15.0, -12.0, -9.0]
PROFILES = ["INDOOR_OFFICE", "URBAN_CANYON"]

def main():
    """Run the sweep."""
    for threshold in THRESHOLDS:
        for profile in PROFILES:
            tag = f"thresh_db_{str(threshold).replace('-', 'm')}"
            cmd = [
                "./scripts/run_profile_sweep.py",
                "--profiles",
                profile,
                "--pathfinder-relative-threshold-db",
                str(threshold),
                "--tag",
                tag,
            ]
            print(f"Running sweep for {profile} with threshold {threshold} dB")
            subprocess.run(cmd, check=True)

if __name__ == "__main__":
    main()
