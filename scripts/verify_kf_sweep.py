#!/usr/bin/env python3
"""Validate Phase 2 KF sweep summaries against expected best combos."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Sequence


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('summary', type=Path, help='Path to kf_sweep_summary.json')
    parser.add_argument('--expected-min', type=float, help='Expected global minimum RMSE (ps)')
    parser.add_argument('--expected-best-mean', type=float, help='Expected best mean RMSE (ps)')
    parser.add_argument('--expected-clock', type=float, help='Expected clock gain for best mean combo')
    parser.add_argument('--expected-freq', type=float, help='Expected frequency gain for best mean combo')
    parser.add_argument('--expected-iterations', type=int, help='Expected iteration count for best mean combo')
    parser.add_argument('--tolerance', type=float, default=0.05, help='Tolerance for RMSE checks (ps)')
    return parser.parse_args()


def _load_summary(path: Path) -> Dict[str, Any]:
    try:
        with path.open('r', encoding='utf-8') as handle:
            return json.load(handle)
    except FileNotFoundError as exc:
        raise SystemExit(f"Summary file not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Failed to parse JSON from {path}: {exc}") from exc


def _best_mean_combo(stats: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    if not stats:
        raise SystemExit('No combo statistics available in summary file')
    return min(stats, key=lambda item: float(item['mean_rmse_ps']))


def _check_close(label: str, value: float, expected: float, tolerance: float) -> None:
    delta = value - expected
    print(f"{label}: {value:.4f} ps (expected {expected:.4f} ps, delta {delta:+.4f} ps)")
    if abs(delta) > tolerance:
        raise SystemExit(f"{label} deviates from expectation by more than {tolerance} ps")


def _check_equal(label: str, value: float, expected: float, rel_tol: float = 1e-6) -> None:
    if math.isclose(value, expected, rel_tol=rel_tol, abs_tol=1e-9):
        print(f"{label}: {value} (matches expected {expected})")
    else:
        raise SystemExit(f"{label} mismatch: got {value}, expected {expected}")


def main() -> None:
    args = _parse_args()
    data = _load_summary(args.summary)
    summary = data.get('summary')
    if summary is None:
        raise SystemExit('Summary payload missing "summary" key')

    combo_stats = summary.get('combo_stats')
    best_mean = _best_mean_combo(combo_stats)

    min_rmse = float(summary['rmse_ps']['min'])
    best_mean_rmse = float(best_mean['mean_rmse_ps'])
    baseline_improvement = best_mean.get('mean_improvement_ps')

    print(f"Loaded summary: {args.summary}")
    print(f"Global min RMSE: {min_rmse:.4f} ps")
    print(
        "Best mean combo: clock_gain={gain_clock} freq_gain={gain_freq} iterations={iterations} mean_rmse={mean:.4f} ps".format(
            gain_clock=best_mean['gain_clock'],
            gain_freq=best_mean['gain_freq'],
            iterations=best_mean['iterations'],
            mean=best_mean_rmse,
        )
    )
    if baseline_improvement is not None:
        print(f"Mean improvement vs baseline: {baseline_improvement:+.4f} ps")

    if args.expected_min is not None:
        _check_close('Global min RMSE', min_rmse, args.expected_min, args.tolerance)
    if args.expected_best_mean is not None:
        _check_close('Best mean RMSE', best_mean_rmse, args.expected_best_mean, args.tolerance)
    if args.expected_clock is not None:
        _check_equal('Clock gain', float(best_mean['gain_clock']), args.expected_clock)
    if args.expected_freq is not None:
        _check_equal('Frequency gain', float(best_mean['gain_freq']), args.expected_freq)
    if args.expected_iterations is not None:
        _check_equal('Iteration count', int(best_mean['iterations']), args.expected_iterations, rel_tol=0.0)


if __name__ == '__main__':
    main()
