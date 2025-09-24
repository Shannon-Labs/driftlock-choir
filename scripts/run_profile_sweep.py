#!/usr/bin/env python3
"""Run a seeded TDL profile sweep and emit JSON manifests."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Iterable, List, Tuple

from run_handshake_diag import run_diagnostic


DEFAULT_PROFILES = ["IDEAL", "URBAN_CANYON", "INDOOR_OFFICE"]


def _build_args(namespace: argparse.Namespace, profile: str, output_json: Path) -> argparse.Namespace:
    args = argparse.Namespace(
        channel_profile=profile,
        num_trials=namespace.num_trials,
        rng_seed=namespace.rng_seed,
        distance_m=namespace.distance_m,
        snr_db=namespace.snr_db,
        retune_offsets_hz=tuple(namespace.retune_offsets_hz),
        delta_t_us=tuple(namespace.delta_t_us),
        disable_coarse=namespace.disable_coarse,
        coarse_bw_hz=namespace.coarse_bw_hz,
        coarse_duration_us=namespace.coarse_duration_us,
        coarse_variance_floor_ps=namespace.coarse_variance_floor_ps,
        beat_duration_us=namespace.beat_duration_us,
        baseband_rate_factor=namespace.baseband_rate_factor,
        min_baseband_rate_hz=namespace.min_baseband_rate_hz,
        min_adc_rate_hz=namespace.min_adc_rate_hz,
        filter_relative_bw=namespace.filter_relative_bw,
        phase_noise_psd=namespace.phase_noise_psd,
        jitter_rms_fs=namespace.jitter_rms_fs,
        mac_preamble_len=namespace.mac_preamble_len,
        mac_narrowband_len=namespace.mac_narrowband_len,
        mac_guard_us=namespace.mac_guard_us,
        pathfinder_relative_threshold_db=namespace.pathfinder_relative_threshold_db,
        pathfinder_noise_guard_multiplier=namespace.pathfinder_noise_guard_multiplier,
        pathfinder_guard_interval_ns=namespace.pathfinder_guard_interval_ns,
        pathfinder_alpha=namespace.pathfinder_alpha,
        pathfinder_beta=namespace.pathfinder_beta,
        use_phase_slope_fit=namespace.use_phase_slope_fit,
        output_dir=namespace.output_dir,
        output_json=output_json,
    )
    return args


def _ensure_output_dir(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    return root


def run_sweep(namespace: argparse.Namespace) -> List[Path]:
    manifests: List[Path] = []
    profiles: Iterable[str] = namespace.profiles or DEFAULT_PROFILES
    output_root = _ensure_output_dir(namespace.output_dir)

    for profile in profiles:
        manifest_name = f"tdl_diag_{profile.lower()}_{namespace.tag}.json" if namespace.tag else f"tdl_diag_{profile.lower()}.json"
        manifest_path = output_root / manifest_name
        args = _build_args(namespace, profile, manifest_path)
        summary = run_diagnostic(args)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with manifest_path.open('w', encoding='utf-8') as handle:
            json.dump(summary, handle, indent=2)
        print(f"[tdl-sweep] wrote {manifest_path}")
        manifests.append(manifest_path)
    return manifests


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Run seeded TDL profile sweep (handshake diagnostics).')
    parser.add_argument('--profiles', nargs='+', default=DEFAULT_PROFILES, help='Profiles to sweep (default: IDEAL URBAN_CANYON INDOOR_OFFICE).')
    parser.add_argument('--snr-db', type=float, default=40.0)
    parser.add_argument('--rng-seed', type=int, default=2025)
    parser.add_argument('--num-trials', type=int, default=64)
    parser.add_argument('--distance-m', type=float, default=120.0)
    parser.add_argument('--retune-offsets-hz', type=float, nargs='+', default=[1e6, 5e6])
    parser.add_argument('--delta-t-us', type=float, nargs='+', default=[0.0, 1.5])
    parser.add_argument('--disable-coarse', action='store_true')
    parser.add_argument('--coarse-bw-hz', type=float, default=40e6)
    parser.add_argument('--coarse-duration-us', type=float, default=5.0)
    parser.add_argument('--coarse-variance-floor-ps', type=float, default=50.0)
    parser.add_argument('--beat-duration-us', type=float, default=20.0)
    parser.add_argument('--baseband-rate-factor', type=float, default=20.0)
    parser.add_argument('--min-baseband-rate-hz', type=float, default=200_000.0)
    parser.add_argument('--min-adc-rate-hz', type=float, default=20_000.0)
    parser.add_argument('--filter-relative-bw', type=float, default=1.4)
    parser.add_argument('--phase-noise-psd', type=float, default=-80.0)
    parser.add_argument('--jitter-rms-fs', type=float, default=1000.0)
    parser.add_argument('--mac-preamble-len', type=int, default=1024)
    parser.add_argument('--mac-narrowband-len', type=int, default=512)
    parser.add_argument('--mac-guard-us', type=float, default=10.0)
    parser.add_argument('--pathfinder-relative-threshold-db', type=float, default=-12.0)
    parser.add_argument('--pathfinder-noise-guard-multiplier', type=float, default=6.0)
    parser.add_argument('--pathfinder-guard-interval-ns', type=float, default=30.0)
    parser.add_argument('--pathfinder-alpha', type=float, default=0.3)
    parser.add_argument('--pathfinder-beta', type=float, default=0.5)
    parser.add_argument('--use-phase-slope-fit', action='store_true')
    parser.add_argument('--output-dir', type=Path, default=Path('results/profile_sweep'))
    parser.add_argument('--tag', type=str, default=None, help='Optional tag appended to manifest filenames.')
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    run_sweep(args)


if __name__ == '__main__':
    main()
