#!/usr/bin/env python3
"""TDL profile handshake diagnostics.

This utility runs repeated Chronometric handshake simulations against a
specified tapped-delay-line (TDL) profile and emits aggregate bias metrics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

# Ensure src/ is importable when script executed directly
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / 'src') not in os.sys.path:
    os.sys.path.insert(0, str(ROOT / 'src'))

from alg.chronometric_handshake import (
    ChronometricHandshakeConfig,
    ChronometricHandshakeSimulator,
    ChronometricNode,
    ChronometricNodeConfig,
)
from mac.scheduler import MacSlots


def _build_nodes() -> Tuple[ChronometricNode, ChronometricNode]:
    """Mirror the canonical two-node configuration used in tests."""
    node_a = ChronometricNode(
        ChronometricNodeConfig(
            node_id=0,
            carrier_freq_hz=2.4e9,
            phase_offset_rad=0.15,
            clock_bias_s=8e-12,
            freq_error_ppm=0.4,
        )
    )
    node_b = ChronometricNode(
        ChronometricNodeConfig(
            node_id=1,
            carrier_freq_hz=2.4e9 + 100e3,
            phase_offset_rad=1.1,
            clock_bias_s=-6e-12,
            freq_error_ppm=-0.25,
        )
    )
    return node_a, node_b


def _ensure_float(value: Optional[float]) -> Optional[float]:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return None
    return float(value)


def _summary_stats(values: Iterable[Optional[float]]) -> Dict[str, float]:
    data = [float(v) for v in values if v is not None and np.isfinite(v)]
    if not data:
        return {}
    arr = np.asarray(data, dtype=float)
    return {
        'mean': float(arr.mean()),
        'median': float(np.median(arr)),
        'std': float(arr.std(ddof=0)),
        'p95': float(np.percentile(arr, 95)),
        'min': float(arr.min()),
        'max': float(arr.max()),
    }


def _fraction(values: Iterable[bool]) -> float:
    seq = list(values)
    if not seq:
        return float('nan')
    return float(np.mean(seq))


def _categorical_distribution(values: Iterable[Optional[str]]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for val in values:
        if val is None:
            continue
        key = str(val)
        counts[key] = counts.get(key, 0) + 1
    return counts


def _stats_to_ns(stats: Dict[str, float]) -> Dict[str, float]:
    return {key: val / 1000.0 for key, val in stats.items()}


def _rmse(values: Iterable[Optional[float]]) -> Optional[float]:
    data = [float(v) for v in values if v is not None and np.isfinite(v)]
    if not data:
        return None
    return float(np.sqrt(np.mean(np.square(data))))


def _git_sha() -> Optional[str]:
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    except (subprocess.SubprocessError, FileNotFoundError):
        return None


def run_diagnostic(args: argparse.Namespace) -> Dict[str, Any]:
    if args.debug:
        logging.basicConfig(level=logging.DEBUG, format='[%(levelname)s] %(message)s')
    rng = np.random.default_rng(args.rng_seed)

    mac = MacSlots(
        preamble_len=args.mac_preamble_len,
        narrowband_len=args.mac_narrowband_len,
        guard_us=args.mac_guard_us,
    )

    cfg = ChronometricHandshakeConfig(
        beat_duration_s=args.beat_duration_us * 1e-6,
        baseband_rate_factor=args.baseband_rate_factor,
        min_baseband_rate_hz=args.min_baseband_rate_hz,
        min_adc_rate_hz=args.min_adc_rate_hz,
        filter_relative_bw=args.filter_relative_bw,
        phase_noise_psd=args.phase_noise_psd,
        jitter_rms_s=args.jitter_rms_fs * 1e-15,
        retune_offsets_hz=tuple(args.retune_offsets_hz),
        coarse_enabled=not args.disable_coarse,
        coarse_bandwidth_hz=args.coarse_bw_hz,
        coarse_duration_s=args.coarse_duration_us * 1e-6,
        coarse_variance_floor_ps=args.coarse_variance_floor_ps,
        coarse_preamble_mode=args.coarse_preamble_mode,
        coarse_formant_profile=args.coarse_formant_profile,
        coarse_formant_fundamental_hz=args.coarse_formant_fundamental_hz,
        coarse_formant_harmonic_count=args.coarse_formant_harmonic_count,
        coarse_formant_include_fundamental=args.coarse_formant_include_fundamental,
        coarse_formant_scale=args.coarse_formant_scale,
        coarse_formant_phase_jitter=args.coarse_formant_phase_jitter,
        coarse_formant_missing_fundamental=not args.disable_formant_missing_fundamental,
        channel_profile=args.channel_profile,
        delta_t_schedule_us=tuple(args.delta_t_us),
        mac=mac,
        pathfinder_relative_threshold_db=args.pathfinder_relative_threshold_db,
        pathfinder_noise_guard_multiplier=args.pathfinder_noise_guard_multiplier,
        pathfinder_guard_interval_s=args.pathfinder_guard_interval_ns * 1e-9,
        pathfinder_pre_guard_ns=args.pathfinder_pre_guard_ns,
        pathfinder_aperture_duration_ns=args.pathfinder_aperture_duration_ns,
        pathfinder_first_path_blend=args.pathfinder_first_path_blend,
        pathfinder_use_simple_search=not args.pathfinder_disable_simple_search,
        use_phase_slope_fit=args.use_phase_slope_fit,
    )

    simulator = ChronometricHandshakeSimulator(cfg)
    node_a, node_b = _build_nodes()

    directional_records: Dict[str, Dict[str, List[Any]]] = {
        'forward': {
            'tau_bias_ps': [],
            'deltaf_bias_hz': [],
            'coarse_bias_ps': [],
            'first_path_error_ps': [],
            'peak_path_error_ps': [],
            'alias_resolved': [],
            'coarse_locked': [],
            'guard_hit': [],
            'tau_var_s2': [],
            'pathfinder_used_aperture': [],
            'pathfinder_first_to_peak_ns': [],
            'pathfinder_missing_fundamental_hz': [],
            'pathfinder_dominant_hz': [],
            'pathfinder_formant_label': [],
            'pathfinder_formant_score': [],
        },
        'reverse': {
            'tau_bias_ps': [],
            'deltaf_bias_hz': [],
            'coarse_bias_ps': [],
            'first_path_error_ps': [],
            'peak_path_error_ps': [],
            'alias_resolved': [],
            'coarse_locked': [],
            'guard_hit': [],
            'tau_var_s2': [],
            'pathfinder_used_aperture': [],
            'pathfinder_first_to_peak_ns': [],
            'pathfinder_missing_fundamental_hz': [],
            'pathfinder_dominant_hz': [],
            'pathfinder_formant_label': [],
            'pathfinder_formant_score': [],
        },
    }

    two_way_bias_ps: List[float] = []
    two_way_deltaf_bias_hz: List[float] = []
    two_way_tau_var_s2: List[float] = []

    for _ in range(args.num_trials):
        result, _ = simulator.run_two_way(
            node_a=node_a,
            node_b=node_b,
            distance_m=args.distance_m,
            snr_db=args.snr_db,
            rng=rng,
        )
        two_way_bias_ps.append((result.tof_est_s - result.tof_true_s) * 1e12)
        two_way_deltaf_bias_hz.append(result.delta_f_est_hz - result.delta_f_true_hz)
        if result.forward.effective_tau_variance_s2 is not None and result.reverse.effective_tau_variance_s2 is not None:
            combined_var = 0.25 * (result.forward.effective_tau_variance_s2 + result.reverse.effective_tau_variance_s2)
            two_way_tau_var_s2.append(combined_var)

        for name, measurement in (('forward', result.forward), ('reverse', result.reverse)):
            record = directional_records[name]
            record['tau_bias_ps'].append((measurement.tau_est_s - measurement.tau_true_s) * 1e12)
            record['deltaf_bias_hz'].append(measurement.delta_f_est_hz - measurement.delta_f_true_hz)
            if measurement.coarse_tau_est_s is not None:
                record['coarse_bias_ps'].append((measurement.coarse_tau_est_s - measurement.tau_true_s) * 1e12)
            else:
                record['coarse_bias_ps'].append(None)
            if measurement.pathfinder is not None:
                pathfinder = measurement.pathfinder
                record['first_path_error_ps'].append((pathfinder.first_path_s - measurement.tau_true_s) * 1e12)
                record['peak_path_error_ps'].append((pathfinder.peak_path_s - measurement.tau_true_s) * 1e12)
                record['pathfinder_used_aperture'].append(bool(pathfinder.used_aperture_fallback))
                record['pathfinder_first_to_peak_ns'].append((pathfinder.peak_path_s - pathfinder.first_path_s) * 1e9)
                record['pathfinder_missing_fundamental_hz'].append(pathfinder.missing_fundamental_hz)
                record['pathfinder_dominant_hz'].append(pathfinder.dominant_harmonic_hz)
                record['pathfinder_formant_label'].append(pathfinder.formant_label)
                record['pathfinder_formant_score'].append(pathfinder.formant_score)
            else:
                record['first_path_error_ps'].append(None)
                record['peak_path_error_ps'].append(None)
                record['pathfinder_used_aperture'].append(None)
                record['pathfinder_first_to_peak_ns'].append(None)
                record['pathfinder_missing_fundamental_hz'].append(None)
                record['pathfinder_dominant_hz'].append(None)
                record['pathfinder_formant_label'].append(None)
                record['pathfinder_formant_score'].append(None)
            record['alias_resolved'].append(bool(measurement.alias_resolved))
            record['coarse_locked'].append(
                None if measurement.coarse_locked is None else bool(measurement.coarse_locked)
            )
            record['guard_hit'].append(bool(measurement.coarse_guard_hit))
            record['tau_var_s2'].append(float(measurement.effective_tau_variance_s2))

    config_descriptor = {
        'channel_profile': args.channel_profile,
        'num_trials': args.num_trials,
        'rng_seed': args.rng_seed,
        'distance_m': args.distance_m,
        'snr_db': args.snr_db,
        'retune_offsets_hz': list(args.retune_offsets_hz),
        'delta_t_schedule_us': list(args.delta_t_us),
        'coarse_enabled': not args.disable_coarse,
        'coarse_bw_hz': args.coarse_bw_hz,
        'coarse_duration_us': args.coarse_duration_us,
        'coarse_preamble_mode': args.coarse_preamble_mode,
        'coarse_formant_profile': args.coarse_formant_profile,
        'coarse_formant_fundamental_hz': args.coarse_formant_fundamental_hz,
        'coarse_formant_harmonic_count': args.coarse_formant_harmonic_count,
        'coarse_formant_include_fundamental': args.coarse_formant_include_fundamental,
        'coarse_formant_scale': args.coarse_formant_scale,
        'coarse_formant_phase_jitter': args.coarse_formant_phase_jitter,
        'coarse_formant_missing_fundamental': not args.disable_formant_missing_fundamental,
        'pathfinder_alpha': args.pathfinder_alpha,
        'pathfinder_beta': args.pathfinder_beta,
        'pathfinder_relative_threshold_db': args.pathfinder_relative_threshold_db,
        'pathfinder_noise_guard_multiplier': args.pathfinder_noise_guard_multiplier,
        'pathfinder_guard_interval_ns': args.pathfinder_guard_interval_ns,
        'pathfinder_pre_guard_ns': args.pathfinder_pre_guard_ns,
        'pathfinder_aperture_duration_ns': args.pathfinder_aperture_duration_ns,
        'pathfinder_first_path_blend': args.pathfinder_first_path_blend,
        'pathfinder_use_simple_search': not args.pathfinder_disable_simple_search,
        'use_phase_slope_fit': args.use_phase_slope_fit,
        'debug_logging': args.debug,
    }
    config_hash = hashlib.sha1(json.dumps(config_descriptor, sort_keys=True).encode('utf-8')).hexdigest()
    tau_rmse_ps = _rmse(two_way_bias_ps)
    deltaf_rmse = _rmse(two_way_deltaf_bias_hz)
    tw_crlb_ns = None
    if two_way_tau_var_s2:
        tw_crlb_ns = float(np.sqrt(np.mean(two_way_tau_var_s2)) * 1e9)

    failure_reasons: List[str] = []
    failure_details: Dict[str, Any] = {}

    summary: Dict[str, Any] = {
        'profile': args.channel_profile,
        'num_trials': args.num_trials,
        'distance_m': args.distance_m,
        'snr_db': args.snr_db,
        'retune_offsets_hz': list(args.retune_offsets_hz),
        'delta_t_schedule_us': list(args.delta_t_us),
        'coarse_enabled': not args.disable_coarse,
        'coarse_bw_hz': args.coarse_bw_hz,
        'coarse_duration_us': args.coarse_duration_us,
        'coarse_preamble_mode': args.coarse_preamble_mode,
        'coarse_formant_profile': args.coarse_formant_profile,
        'coarse_formant_fundamental_hz': args.coarse_formant_fundamental_hz,
        'coarse_formant_harmonic_count': args.coarse_formant_harmonic_count,
        'coarse_formant_include_fundamental': args.coarse_formant_include_fundamental,
        'coarse_formant_scale': args.coarse_formant_scale,
        'coarse_formant_phase_jitter': args.coarse_formant_phase_jitter,
        'coarse_formant_missing_fundamental': not args.disable_formant_missing_fundamental,
        'pathfinder_relative_threshold_db': args.pathfinder_relative_threshold_db,
        'pathfinder_noise_guard_multiplier': args.pathfinder_noise_guard_multiplier,
        'pathfinder_guard_interval_ns': args.pathfinder_guard_interval_ns,
        'pathfinder_aperture_duration_ns': args.pathfinder_aperture_duration_ns,
        'pathfinder_first_path_blend': args.pathfinder_first_path_blend,
        'pathfinder_use_simple_search': not args.pathfinder_disable_simple_search,
        'pathfinder_alpha': args.pathfinder_alpha,
        'pathfinder_beta': args.pathfinder_beta,
        'use_phase_slope_fit': args.use_phase_slope_fit,
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'directional_metrics': {},
        'two_way_metrics': {
            'tau_bias_ns': _stats_to_ns(_summary_stats(two_way_bias_ps)),
            'deltaf_bias_hz': _summary_stats(two_way_deltaf_bias_hz),
            'tau_rmse_ns': tau_rmse_ps / 1000.0 if tau_rmse_ps is not None else None,
            'deltaf_rmse_hz': deltaf_rmse,
            'crlb_tau_ns': tw_crlb_ns,
            'rmse_over_crlb': (tau_rmse_ps / 1000.0) / tw_crlb_ns if (tau_rmse_ps is not None and tw_crlb_ns and tw_crlb_ns > 0) else None,
        },
        'git_sha': _git_sha(),
        'config_hash': config_hash,
        'rng_seed': args.rng_seed,
    }

    for name, record in directional_records.items():
        locked_values = [bool(val) for val in record['coarse_locked'] if val is not None]
        guard_values = [bool(val) for val in record['guard_hit']]
        tau_rmse = _rmse(record['tau_bias_ps'])
        deltaf_rmse = _rmse(record['deltaf_bias_hz'])
        locked_rate = _fraction(locked_values) if locked_values else None
        guard_rate = _fraction(guard_values) if guard_values else None
        first_path_values = [val for val in record['first_path_error_ps'] if val is not None]
        first_path_within_5 = [abs(val) < 5_000.0 for val in first_path_values]
        first_path_within_10 = [abs(val) < 10_000.0 for val in first_path_values]
        first_path_negative = [val < 0.0 for val in first_path_values]
        crlb_ns = None
        if record['tau_var_s2']:
            crlb_ns = float(np.sqrt(np.mean(record['tau_var_s2'])) * 1e9)
        fallback_values = [bool(val) for val in record['pathfinder_used_aperture'] if val is not None]
        summary['directional_metrics'][name] = {
            'tau_bias_ns': _stats_to_ns(_summary_stats(record['tau_bias_ps'])),
            'deltaf_bias_hz': _summary_stats(record['deltaf_bias_hz']),
            'coarse_bias_ns': _stats_to_ns(_summary_stats(record['coarse_bias_ps'])),
            'first_path_error_ns': _stats_to_ns(_summary_stats(record['first_path_error_ps'])),
            'peak_path_error_ns': _stats_to_ns(_summary_stats(record['peak_path_error_ps'])),
            'alias_success_rate': _fraction(record['alias_resolved']),
            'coarse_locked_rate': locked_rate,
            'guard_hit_rate': guard_rate,
            'tau_rmse_ns': tau_rmse / 1000.0 if tau_rmse is not None else None,
            'deltaf_rmse_hz': deltaf_rmse,
            'crlb_tau_ns': crlb_ns,
            'rmse_over_crlb': (tau_rmse / 1000.0) / crlb_ns if (tau_rmse is not None and crlb_ns and crlb_ns > 0) else None,
            'pathfinder_aperture_rate': _fraction(fallback_values) if fallback_values else None,
            'pathfinder_first_to_peak_ns': _summary_stats(record['pathfinder_first_to_peak_ns']),
            'pathfinder_missing_fundamental_hz': _summary_stats(record['pathfinder_missing_fundamental_hz']),
            'pathfinder_dominant_hz': _summary_stats(record['pathfinder_dominant_hz']),
            'pathfinder_formant_score': _summary_stats(record['pathfinder_formant_score']),
            'pathfinder_formant_labels': _categorical_distribution(record['pathfinder_formant_label']) or None,
            'first_path_within_5ns_rate': _fraction(first_path_within_5) if first_path_within_5 else None,
            'first_path_within_10ns_rate': _fraction(first_path_within_10) if first_path_within_10 else None,
            'first_path_negative_rate': _fraction(first_path_negative) if first_path_negative else None,
        }

    all_locked = [bool(val) for values in directional_records.values() for val in values['coarse_locked'] if val is not None]
    all_guard = [bool(val) for values in directional_records.values() for val in values['guard_hit']]
    summary['coarse_locked'] = all(all_locked) if all_locked else None
    summary['guard_hit'] = any(all_guard)

    ratio = summary['two_way_metrics']['rmse_over_crlb']
    if ratio is not None:
        if ratio < 1.0:
            failure_reasons.append('WARN_CRLB_LOW')
        elif ratio > 1.5:
            failure_reasons.append('FAIL_CRLB_HIGH')
            failure_details['crlb_ratio_tau'] = ratio
    else:
        failure_reasons.append('WARN_NO_CRLB')

    bias_caps_ns = {
        'IDEAL': 0.2,
        'INDOOR': 1.2,
        'INDOOR_OFFICE': 1.2,
        'URBAN': 0.8,
        'URBAN_CANYON': 0.8,
    }
    profile_key = args.channel_profile.upper()
    mean_bias = summary['two_way_metrics']['tau_bias_ns'].get('mean')
    if mean_bias is not None and profile_key in bias_caps_ns:
        if abs(mean_bias) > bias_caps_ns[profile_key]:
            failure_reasons.append('FAIL_BIAS_CAP')
            failure_details['tau_bias_mean_ns'] = mean_bias
            failure_details['bias_cap_ns'] = bias_caps_ns[profile_key]

    over_cap_trials: List[int] = []
    if profile_key in bias_caps_ns:
        cap_ps = bias_caps_ns[profile_key] * 1000.0
        for idx, value in enumerate(two_way_bias_ps):
            if value is None:
                continue
            if abs(value) > cap_ps:
                over_cap_trials.append(idx)
        if over_cap_trials:
            failure_details['bias_cap_trials'] = over_cap_trials

    if summary['coarse_locked'] is False:
        failure_reasons.append('WARN_COARSE_UNLOCKED')
    if summary['guard_hit']:
        failure_reasons.append('WARN_GUARD_HIT')

    status = 'OK' if not failure_reasons else ';'.join(failure_reasons)
    if failure_details:
        summary['failure_details'] = failure_details

    summary['status'] = status

    if 'rmse_over_crlb' in summary['two_way_metrics'] and summary['two_way_metrics']['rmse_over_crlb'] > 1.5:
        print("WARNING: Performance guardrails FAILED - RMSE/CRLB > 1.5")

    mean_bias_ns = summary['two_way_metrics']['tau_bias_ns'].get('mean', 0)
    if abs(mean_bias_ns) > 0.2:
        print("WARNING: Performance guardrails FAILED - Bias > 0.2 ns")

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description='Run Chronometric handshake diagnostics for a TDL profile.')
    parser.add_argument('--channel-profile', required=True, help='Profile name defined in src/chan/tdl.py (e.g. IDEAL).')
    parser.add_argument('--num-trials', type=int, default=64, help='Number of Monte Carlo trials to run.')
    parser.add_argument('--rng-seed', type=int, default=2025, help='Base RNG seed.')
    parser.add_argument('--distance-m', type=float, default=120.0, help='Link distance in meters.')
    parser.add_argument('--snr-db', type=float, default=30.0, help='SNR in dB for the handshake tone.')
    parser.add_argument('--retune-offsets-hz', type=float, nargs='+', default=(1e6, 5e6), help='Retune offsets in Hz.')
    parser.add_argument('--delta-t-us', dest='delta_t_us', type=float, nargs='+', default=(0.0, 1.5), help='Delta-t schedule in microseconds.')
    parser.add_argument('--disable-coarse', action='store_true', help='Disable coarse delay finder and Pathfinder conditioning.')
    parser.add_argument('--coarse-bw-hz', type=float, default=40e6)
    parser.add_argument('--coarse-duration-us', type=float, default=5.0)
    parser.add_argument('--coarse-variance-floor-ps', type=float, default=50.0)
    parser.add_argument('--coarse-preamble-mode', choices=['zadoff', 'formant'], default='zadoff')
    parser.add_argument('--coarse-formant-profile', type=str, default='A', help='Vowel profile (A/E/I/O/U) used when --coarse-preamble-mode=formant.')
    parser.add_argument('--coarse-formant-fundamental-hz', type=float, default=25_000.0)
    parser.add_argument('--coarse-formant-harmonic-count', type=int, default=12)
    parser.add_argument('--coarse-formant-include-fundamental', action='store_true')
    parser.add_argument('--coarse-formant-scale', type=float, default=1_000.0, help='Scaling applied to canonical vowel formants (Hz->Hz).')
    parser.add_argument('--coarse-formant-phase-jitter', type=float, default=0.0, help='Uniform phase jitter (radians) applied per harmonic.')
    parser.add_argument('--disable-formant-missing-fundamental', action='store_true', help='Disable missing-fundamental decoding even when using formant preambles.')
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
    parser.add_argument('--pathfinder-pre-guard-ns', type=float, default=0.0)
    parser.add_argument('--pathfinder-aperture-duration-ns', type=float, default=100.0)
    parser.add_argument('--pathfinder-first-path-blend', type=float, default=0.05, help='Blend factor between the pathfinder peak (0.0) and first-path (1.0) timestamps when seeding the coarse hint. Actual blend is scaled by profile heuristics.')
    parser.add_argument('--pathfinder-disable-simple-search', action='store_true', help='Skip the forward threshold scan so the aperture window is always used.')
    parser.add_argument('--pathfinder-alpha', type=float, default=0.3)
    parser.add_argument('--pathfinder-beta', type=float, default=0.5)
    parser.add_argument('--use-phase-slope-fit', action='store_true', help='Enable multi-carrier phase-slope fusion.')
    parser.add_argument('--debug', action='store_true', help='Enable verbose estimator diagnostics.')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'results' / 'phase1' / 'tdl_profiles')
    parser.add_argument('--output-json', type=Path, default=None, help='Explicit path for JSON summary (overrides --output-dir).')

    args = parser.parse_args()
    summary = run_diagnostic(args)

    output_path: Optional[Path]
    if args.output_json is not None:
        output_path = args.output_json
    else:
        timestamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
        output_dir = args.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        filename = f"tdl_diag_{args.channel_profile.lower()}_{timestamp}.json"
        output_path = output_dir / filename

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open('w', encoding='utf-8') as handle:
            json.dump(summary, handle, indent=2)
        print(f"Saved diagnostic summary to {output_path}")

    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
