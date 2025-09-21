#!/usr/bin/env python3
"""
Smoke test runner for DriftLock simulations.
Executes quick phase1 atomic handshake and small MC run using smoke.yaml.
Fails on non-zero exit codes; outputs telemetry paths for CI.
"""

import os
import subprocess
import sys
from pathlib import Path


def run_command(cmd: list[str], env: dict[str, str]) -> subprocess.CompletedProcess:
    """Execute a subprocess command, raising with detailed logging on failure."""
    print(f"Running: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True, env=env)
    except subprocess.CalledProcessError as exc:
        print(exc.stdout)
        print(exc.stderr, file=sys.stderr)
        raise
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)
    return result


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    smoke_root = repo_root / "results" / "smoke"
    phase1_dir = smoke_root / "phase1"
    mc_root = smoke_root / "mc"
    phase1_dir.mkdir(parents=True, exist_ok=True)
    mc_root.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env.setdefault("PYTHONPATH", str(repo_root))

    phase1_cmd = [
        sys.executable,
        "sim/phase1.py",
        "--atomic-mode",
        "--num-trials",
        "5",
        "--results-dir",
        str(phase1_dir),
    ]
    run_command(phase1_cmd, env)

    mc_cmd = [
        sys.executable,
        "scripts/run_mc.py",
        "all",
        "-c",
        "sim/configs/smoke.yaml",
        "-o",
        str(mc_root),
        "-r",
        "smoke",
    ]
    run_command(mc_cmd, env)

    telemetry_files = sorted(mc_root.rglob("*.jsonl"))
    if telemetry_files:
        print("Collected telemetry files:")
        for path in telemetry_files:
            print(f"  - {path.relative_to(repo_root)}")
    else:
        print("WARNING: no telemetry JSONL files found under results/smoke/mc")

    print("Smoke tests completed successfully.")


if __name__ == "__main__":
    main()
