"""Pytest configuration ensuring local packages are importable.

This repository hosts multiple python packages that rely on being
importable without installation.  The tests expect the `src/`
namespace as well as the simulation utilities under
`driftlock_choir_sim/` to be directly importable.  When pytest is
invoked from the repository root the interpreter only sees the current
working directory, so we extend ``sys.path`` to include those module
roots."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
SRC_ROOT = REPO_ROOT / "src"
SIM_ROOT = REPO_ROOT / "driftlock_choir_sim"

for path in (SIM_ROOT, SRC_ROOT):
    if path.is_dir():
        str_path = str(path)
        if str_path not in sys.path:
            sys.path.insert(0, str_path)
