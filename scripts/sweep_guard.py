#!/usr/bin/env python3
"""Run a sweep over the pathfinder guard interval."""

import subprocess
import sys

GUARD_INTERVALS = [30.0, 60.0, 90.0, 120.0]
PROFILES = ["INDOOR_OFFICE", "URBAN_CANYON"]

def main():
    """Run the sweep."""
    for guard_interval in GUARD_INTERVALS:
        for profile in PROFILES:
            tag = f"guard_ns_{guard_interval}"
            cmd = [
                "./scripts/run_profile_sweep.py",
                "--profiles",
                profile,
                "--pathfinder-guard-interval-ns",
                str(guard_interval),
                "--tag",
                tag,
            ]
            print(f"Running sweep for {profile} with guard interval {guard_interval} ns")
            subprocess.run(cmd, check=True)

if __name__ == "__main__":
    main()
