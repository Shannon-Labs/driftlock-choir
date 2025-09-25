#!/usr/bin/env python3
"""Guard/aperture sweep harness for Pathfinder tuning.

This driver scans Pathfinder guard intervals, aperture durations, and detector
thresholds for challenging multipath profiles. Results are captured as JSON
artifacts under ``results/tuning_temp`` so downstream analysis can pick up the
latest runs without clobbering previous diagnostics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from itertools import product
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = REPO_ROOT / "results" / "tuning_temp"
DEFAULT_INDOOR_GUARDS_NS = (40.0, 60.0, 80.0, 100.0)
DEFAULT_APERTURES_NS = (80.0, 120.0, 160.0, 200.0)
DEFAULT_RELATIVE_THRESH_DB = (-20.0, -18.0, -16.0)
DEFAULT_NOISE_GUARDS = (3.0, 4.0, 5.0)
DEFAULT_PROFILE = "INDOOR_OFFICE"
URBAN_PROFILE = "URBAN_CANYON"
DEFAULT_NUM_TRIALS = 64
DEFAULT_TOP_K = 3


@dataclass(frozen=True)
class PathfinderParams:
    guard_interval_ns: float
    aperture_duration_ns: float
    relative_threshold_db: float
    noise_guard_multiplier: float

    def slug(self) -> str:
        """Create a filesystem-friendly slug describing the parameter combo."""
        parts = [
            f"g{int(round(self.guard_interval_ns))}",
            f"a{int(round(self.aperture_duration_ns))}",
            f"thr{self._encode_float(self.relative_threshold_db)}",
            f"ng{self._encode_float(self.noise_guard_multiplier)}",
        ]
        return "_".join(parts)

    @staticmethod
    def _encode_float(value: float) -> str:
        if value == 0.0:
            return "0"
        scaled = f"{value:.1f}" if value != math.trunc(value) else f"{int(value)}"
        scaled = scaled.replace("-", "m").replace(".", "p")
        return scaled


@dataclass
class SweepMetrics:
    profile: str
    params: PathfinderParams
    tau_bias_ns: Optional[float]
    tau_bias_abs_ns: Optional[float]
    tau_rmse_ns: Optional[float]
    rmse_over_crlb: Optional[float]
    deltaf_bias_hz: Optional[float]
    deltaf_rmse_hz: Optional[float]
    coarse_bias_forward_ns: Optional[float]
    coarse_bias_reverse_ns: Optional[float]
    aperture_rate_forward: Optional[float]
    aperture_rate_reverse: Optional[float]
    first_to_peak_forward_ns: Optional[float]
    first_to_peak_reverse_ns: Optional[float]
    json_path: Path
    runtime_s: float
    first_path_blend: float

    def as_dict(self) -> Dict[str, object]:
        payload = asdict(self)
        payload["json_path"] = str(self.json_path)
        return payload


def _stat_mean(stats: Optional[Dict[str, float]]) -> Optional[float]:
    if not isinstance(stats, dict):
        return None
    return stats.get("mean")


def _extract_metrics(
    profile: str,
    params: PathfinderParams,
    summary: Dict[str, object],
    runtime_s: float,
    output_path: Path,
    first_path_blend: float,
) -> SweepMetrics:
    two_way = summary.get("two_way_metrics", {}) if isinstance(summary, dict) else {}
    directional = summary.get("directional_metrics", {}) if isinstance(summary, dict) else {}

    forward = directional.get("forward", {}) if isinstance(directional, dict) else {}
    reverse = directional.get("reverse", {}) if isinstance(directional, dict) else {}

    tau_bias = _stat_mean(two_way.get("tau_bias_ns"))
    tau_bias_abs = abs(tau_bias) if tau_bias is not None else None

    metrics = SweepMetrics(
        profile=profile,
        params=params,
        tau_bias_ns=tau_bias,
        tau_bias_abs_ns=tau_bias_abs,
        tau_rmse_ns=two_way.get("tau_rmse_ns"),
        rmse_over_crlb=two_way.get("rmse_over_crlb"),
        deltaf_bias_hz=_stat_mean(two_way.get("deltaf_bias_hz")),
        deltaf_rmse_hz=two_way.get("deltaf_rmse_hz"),
        coarse_bias_forward_ns=_stat_mean(forward.get("coarse_bias_ns")),
        coarse_bias_reverse_ns=_stat_mean(reverse.get("coarse_bias_ns")),
        aperture_rate_forward=forward.get("pathfinder_aperture_rate"),
        aperture_rate_reverse=reverse.get("pathfinder_aperture_rate"),
        first_to_peak_forward_ns=_stat_mean(forward.get("pathfinder_first_to_peak_ns")),
        first_to_peak_reverse_ns=_stat_mean(reverse.get("pathfinder_first_to_peak_ns")),
        json_path=output_path,
        runtime_s=runtime_s,
        first_path_blend=first_path_blend,
    )
    return metrics


def _handshake_command(
    profile: str,
    params: PathfinderParams,
    num_trials: int,
    rng_seed: int,
    disable_simple_search: bool,
    first_path_blend: float,
    output_path: Path,
) -> List[str]:
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "run_handshake_diag.py"),
        "--channel-profile",
        profile,
        "--num-trials",
        str(num_trials),
        "--rng-seed",
        str(rng_seed),
        "--pathfinder-relative-threshold-db",
        str(params.relative_threshold_db),
        "--pathfinder-noise-guard-multiplier",
        str(params.noise_guard_multiplier),
        "--pathfinder-guard-interval-ns",
        str(params.guard_interval_ns),
        "--pathfinder-aperture-duration-ns",
        str(params.aperture_duration_ns),
        "--pathfinder-first-path-blend",
        str(first_path_blend),
        "--output-json",
        str(output_path),
        "--output-dir",
        str(RESULT_ROOT),
    ]
    if disable_simple_search:
        cmd.append("--pathfinder-disable-simple-search")
    return cmd


def _timestamp_slug() -> str:
    now = datetime.now(timezone.utc)
    return now.strftime("%Y%m%d_%H%M%S_%f")


def _output_path(profile: str, params: PathfinderParams, first_path_blend: float) -> Path:
    blend_tag = PathfinderParams._encode_float(first_path_blend)
    digest = hashlib.sha1(f"{profile}:{params.slug()}:{first_path_blend}".encode("utf-8")).hexdigest()[:10]
    filename = f"tdl_diag_{profile.lower()}_{params.slug()}_b{blend_tag}_{_timestamp_slug()}_{digest}.json"
    return RESULT_ROOT / filename


def run_diagnostic(
    profile: str,
    params: PathfinderParams,
    num_trials: int,
    rng_seed: int,
    disable_simple_search: bool,
    first_path_blend: float,
) -> SweepMetrics:
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    output_path = _output_path(profile, params, first_path_blend)
    cmd = _handshake_command(profile, params, num_trials, rng_seed, disable_simple_search, first_path_blend, output_path)

    start = time.time()
    result = subprocess.run(cmd, cwd=str(REPO_ROOT), capture_output=True, text=True)
    runtime = time.time() - start

    if result.returncode != 0:
        raise RuntimeError(
            f"run_handshake_diag failed for {profile} with params {params}:\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )

    with output_path.open("r", encoding="utf-8") as handle:
        summary = json.load(handle)
    return _extract_metrics(profile, params, summary, runtime, output_path, first_path_blend)


def build_param_grid(guards: Sequence[float], apertures: Sequence[float], thresholds: Sequence[float], noise_guards: Sequence[float]) -> List[PathfinderParams]:
    grid = []
    for guard_ns, aperture_ns, threshold_db, noise_mult in product(guards, apertures, thresholds, noise_guards):
        grid.append(
            PathfinderParams(
                guard_interval_ns=float(guard_ns),
                aperture_duration_ns=float(aperture_ns),
                relative_threshold_db=float(threshold_db),
                noise_guard_multiplier=float(noise_mult),
            )
        )
    return grid


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep Pathfinder guard/aperture parameters")
    parser.add_argument("--profile", default=DEFAULT_PROFILE, help="Primary profile to sweep (default INDOOR_OFFICE)")
    parser.add_argument("--num-trials", type=int, default=DEFAULT_NUM_TRIALS, help="Monte Carlo trials per configuration")
    parser.add_argument("--rng-seed", type=int, default=2025, help="Deterministic seed for reproducibility")
    parser.add_argument("--guards", type=float, nargs="+", default=DEFAULT_INDOOR_GUARDS_NS, help="Guard intervals (ns)")
    parser.add_argument("--apertures", type=float, nargs="+", default=DEFAULT_APERTURES_NS, help="Aperture durations (ns)")
    parser.add_argument("--thresholds", type=float, nargs="+", default=DEFAULT_RELATIVE_THRESH_DB, help="Relative thresholds (dB)")
    parser.add_argument("--noise-guards", type=float, nargs="+", default=DEFAULT_NOISE_GUARDS, help="Noise guard multipliers")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K, help="Number of top configs to evaluate on URBAN_CANYON")
    parser.add_argument("--urban-trials", type=int, default=DEFAULT_NUM_TRIALS, help="Trials for URBAN_CANYON verification runs")
    parser.add_argument("--disable-simple-search", action="store_true", help="Force aperture-only behaviour")
    parser.add_argument("--first-path-blend", type=float, default=0.05, help="Blend factor between peak and first-path when seeding the coarse estimator (scaled heuristically per profile)")
    return parser.parse_args()


def summarize_results(indoor_results: List[SweepMetrics], urban_results: List[SweepMetrics], args: argparse.Namespace) -> Path:
    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "primary_profile": args.profile,
        "num_trials_primary": args.num_trials,
        "num_trials_urban": args.urban_trials,
        "rng_seed": args.rng_seed,
        "disable_simple_search": args.disable_simple_search,
        "first_path_blend": args.first_path_blend,
        "guards_ns": list(map(float, args.guards)),
        "apertures_ns": list(map(float, args.apertures)),
        "thresholds_db": list(map(float, args.thresholds)),
        "noise_guards": list(map(float, args.noise_guards)),
        "indoor_results": [metrics.as_dict() for metrics in indoor_results],
        "urban_results": [metrics.as_dict() for metrics in urban_results],
    }
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    manifest = RESULT_ROOT / f"pathfinder_sweep_summary_{_timestamp_slug()}.json"
    with manifest.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    return manifest


def print_ranked_results(metrics: Iterable[SweepMetrics]) -> None:
    print("\n=== Ranked Results ===")
    for idx, metric in enumerate(metrics, start=1):
        tau_bias = "N/A" if metric.tau_bias_ns is None else f"{metric.tau_bias_ns:+.3f} ns"
        tau_rmse = "N/A" if metric.tau_rmse_ns is None else f"{metric.tau_rmse_ns:.3f} ns"
        deltaf_bias = "N/A" if metric.deltaf_bias_hz is None else f"{metric.deltaf_bias_hz:+.1f} Hz"
        print(
            f"{idx:2d}. {metric.profile:>13} | {metric.params.slug():25s} | "
            f"bias {tau_bias:>12s} | rmse {tau_rmse:>10s} | Δf {deltaf_bias:>12s} | file {metric.json_path.name}"
        )


def main() -> None:
    args = parse_arguments()
    param_grid = build_param_grid(args.guards, args.apertures, args.thresholds, args.noise_guards)
    print(f"Sweeping {len(param_grid)} Pathfinder configurations on {args.profile}")
    indoor_results: List[SweepMetrics] = []

    for idx, params in enumerate(param_grid, start=1):
        print(f"[{idx}/{len(param_grid)}] {params.slug()} ...", end=" ")
        try:
            metrics = run_diagnostic(args.profile, params, args.num_trials, args.rng_seed, args.disable_simple_search, args.first_path_blend)
        except Exception as exc:  # noqa: BLE001
            print("FAILED")
            print(f"    {exc}")
            continue
        indoor_results.append(metrics)
        bias = "N/A" if metrics.tau_bias_ns is None else f"{metrics.tau_bias_ns:+.3f} ns"
        rmse = "N/A" if metrics.tau_rmse_ns is None else f"{metrics.tau_rmse_ns:.3f} ns"
        print(f"bias {bias}, rmse {rmse}")

    indoor_results.sort(key=lambda m: m.tau_bias_abs_ns if m.tau_bias_abs_ns is not None else float("inf"))

    top_k = min(max(args.top_k, 0), len(indoor_results))
    print_ranked_results(indoor_results[:top_k])

    urban_results: List[SweepMetrics] = []
    if top_k:
        print(f"\nEvaluating top {top_k} configurations on {URBAN_PROFILE}")
    for idx, metrics in enumerate(indoor_results[:top_k], start=1):
        print(f"[{idx}/{top_k}] {metrics.params.slug()} ...", end=" ")
        try:
            urban_metric = run_diagnostic(URBAN_PROFILE, metrics.params, args.urban_trials, args.rng_seed, args.disable_simple_search, args.first_path_blend)
        except Exception as exc:  # noqa: BLE001
            print("FAILED")
            print(f"    {exc}")
            continue
        urban_results.append(urban_metric)
        bias = "N/A" if urban_metric.tau_bias_ns is None else f"{urban_metric.tau_bias_ns:+.3f} ns"
        rmse = "N/A" if urban_metric.tau_rmse_ns is None else f"{urban_metric.tau_rmse_ns:.3f} ns"
        print(f"bias {bias}, rmse {rmse}")

    manifest = summarize_results(indoor_results, urban_results, args)
    print(f"\nSweep summary written to {manifest}")
    print_ranked_results(urban_results)


if __name__ == "__main__":
    main()
