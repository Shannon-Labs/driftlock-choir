#!/usr/bin/env python3
"""Run a sweep over the pathfinder pre-guard interval."""

import subprocess
import sys

PRE_GUARD_INTERVALS = [0.0, 20.0, 40.0, 60.0, 80.0, 100.0]
PROFILES = ["INDOOR_OFFICE", "URBAN_CANYON"]

def main():
    """Run the sweep."""
    for pre_guard_interval in PRE_GUARD_INTERVALS:
        for profile in PROFILES:
            tag = f"pre_guard_ns_{pre_guard_interval}"
            cmd = [
                "./scripts/run_profile_sweep.py",
                "--profiles",
                profile,
                "--pathfinder-pre-guard-ns",
                str(pre_guard_interval),
                "--pathfinder-guard-interval-ns",
                "120.0",
                "--tag",
                tag,
            ]
            print(f"Running sweep for {profile} with pre-guard interval {pre_guard_interval} ns")
            subprocess.run(cmd, check=True)

if __name__ == "__main__":
    main()
