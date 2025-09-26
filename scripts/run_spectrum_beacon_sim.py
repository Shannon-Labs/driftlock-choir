#!/usr/bin/env python3
"""Simulate vowel-coded spectrum beacons using the missing-fundamental decoder.

This script explores a potential secondary application of Project Aperture × Formant:
encoding spectrum-occupancy metadata in vowel-shaped preambles and decoding it with
Pathfinder's missing-fundamental analysis. The simulator emits synthetic channels
(with configurable multipath delay spreads, SNRs, and optional beacon absences)
then measures label recovery, missing-fundamental accuracy, and false-positive
rates when the beacon is silent.

Usage example::

    python scripts/run_spectrum_beacon_sim.py \
        --profiles A E I O U \
        --num-trials 1000 \
        --snr-db 25 \
        --max-extra-paths 3 \
        --max-delay-ns 80 \
        --empty-prob 0.2 \
        --output results/project_aperture_formant/URBAN_CANYON/beacon_study.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from phy.formants import (
    FormantAnalysisResult,
    FormantSynthesisConfig,
    analyze_missing_fundamental,
    build_formant_library,
    synthesize_formant_preamble,
)


@dataclass
class TrialResult:
    """Outcome of a single beacon transmission (or absence)."""

    has_beacon: bool
    true_label: Optional[str]
    detected: bool
    predicted_label: Optional[str]
    score: Optional[float]
    missing_f0_hz: Optional[float]
    dominant_hz: Optional[float]
    delay_ns: float
    snr_db: float


_DEF_PROFILES = ("A", "E", "I", "O", "U")


def _summary_stats(values: Iterable[float]) -> Dict[str, float]:
    data = [float(v) for v in values if np.isfinite(v)]
    if not data:
        return {}
    arr = np.asarray(data, dtype=float)
    return {
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "std": float(arr.std(ddof=0)),
        "p95": float(np.percentile(arr, 95)),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def _fraction(events: Iterable[bool]) -> Optional[float]:
    seq = list(events)
    if not seq:
        return None
    return float(np.mean(seq))


def _simulate_channel(
    waveform: np.ndarray,
    sample_rate: float,
    max_extra_paths: int,
    max_delay_ns: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Apply a random multipath channel to the baseband waveform."""

    base = waveform
    if max_extra_paths <= 0 or max_delay_ns <= 0.0:
        return base

    max_delay_s = max_delay_ns * 1e-9
    max_offset_samples = int(np.ceil(max_delay_s * sample_rate))
    if max_offset_samples <= 0:
        return base

    taps = [np.array([1.0 + 0.0j])]
    delays = [0]

    num_paths = rng.integers(low=0, high=max_extra_paths + 1)
    for _ in range(int(num_paths)):
        delay = rng.integers(low=1, high=max_offset_samples + 1)
        amp = (rng.normal(scale=0.6) + 1j * rng.normal(scale=0.6)) / np.sqrt(2.0)
        taps.append(np.pad(np.array([amp], dtype=np.complex128), (delay, 0)))
        delays.append(delay)

    channel = np.zeros(max(delays) + 1, dtype=np.complex128)
    for tap in taps:
        channel[: tap.size] += tap

    return np.convolve(base, channel, mode="full")


def _inject_awgn(waveform: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    if snr_db is None:
        return waveform
    power = np.mean(np.abs(waveform) ** 2)
    if power <= 0.0:
        return waveform
    snr_linear = 10.0 ** (snr_db / 10.0)
    noise_power = power / snr_linear
    noise = (
        rng.normal(scale=np.sqrt(noise_power / 2.0), size=waveform.shape)
        + 1j * rng.normal(scale=np.sqrt(noise_power / 2.0), size=waveform.shape)
    )
    return waveform + noise


def _run_trial(
    args: argparse.Namespace,
    profiles: List[str],
    library: Dict[str, object],
    rng: np.random.Generator,
) -> TrialResult:
    has_beacon = rng.random() > args.empty_prob
    snr_db = float(rng.uniform(args.snr_db_min, args.snr_db_max))

    if has_beacon:
        label = str(rng.choice(profiles))
        synth_config = FormantSynthesisConfig(
            profile=label,
            fundamental_hz=args.fundamental_hz,
            harmonic_count=args.harmonics,
            include_fundamental=args.include_fundamental,
            formant_scale=args.formant_scale,
            phase_jitter=args.phase_jitter,
        )
        waveform, _ = synthesize_formant_preamble(
            length=args.symbol_length,
            sample_rate=args.sample_rate,
            config=synth_config,
        )
        received = _simulate_channel(
            waveform,
            sample_rate=args.sample_rate,
            max_extra_paths=args.max_extra_paths,
            max_delay_ns=args.max_delay_ns,
            rng=rng,
        )
        delay_ns = (len(received) - len(waveform)) / args.sample_rate * 1e9
    else:
        label = None
        received = np.zeros(args.symbol_length, dtype=np.complex128)
        delay_ns = 0.0

    received = _inject_awgn(received, snr_db=snr_db, rng=rng)
    segment = received[: args.analysis_length]

    analysis: Optional[FormantAnalysisResult]
    descriptors = list(library.values())
    analysis = analyze_missing_fundamental(segment, args.sample_rate, descriptors, top_peaks=args.top_peaks)

    detected = False
    predicted_label = analysis.label if analysis else None
    score = analysis.score if analysis else None
    missing_f0 = analysis.missing_fundamental_hz if analysis else None
    dominant_hz = analysis.dominant_hz if analysis else None

    if analysis is not None:
        passes_checks = True
        label_key = analysis.label.upper()
        descriptor = library.get(label_key)

        if args.score_threshold is not None:
            if analysis.score is None or analysis.score > args.score_threshold:
                passes_checks = False

        if passes_checks and args.missing_f0_tolerance_hz is not None:
            if descriptor is None or missing_f0 is None or abs(missing_f0 - descriptor.fundamental_hz) > args.missing_f0_tolerance_hz:
                passes_checks = False

        if passes_checks and args.dominant_tolerance_hz is not None:
            if descriptor is None or dominant_hz is None or abs(dominant_hz - descriptor.dominant_hz) > args.dominant_tolerance_hz:
                passes_checks = False

        if passes_checks:
            detected = True
        else:
            # downrank detection if validation failed
            predicted_label = None


    return TrialResult(
        has_beacon=has_beacon,
        true_label=label,
        detected=detected,
        predicted_label=predicted_label,
        score=score,
        missing_f0_hz=missing_f0,
        dominant_hz=dominant_hz,
        delay_ns=delay_ns,
        snr_db=snr_db,
    )


def _aggregate(results: List[TrialResult], args: argparse.Namespace) -> Dict[str, object]:
    success_scores: List[float] = []
    success_missing: List[float] = []
    success_dominant: List[float] = []
    success_snr: List[float] = []

    false_scores: List[float] = []
    false_missing: List[float] = []

    true_positive = 0
    detected_trials = 0
    false_positive = 0

    per_label: Dict[str, Dict[str, List[float]]] = {}

    for trial in results:
        if trial.has_beacon:
            if trial.detected:
                detected_trials += 1
                if trial.predicted_label == trial.true_label:
                    true_positive += 1
                    if trial.score is not None:
                        success_scores.append(trial.score)
                    if trial.missing_f0_hz is not None:
                        success_missing.append(trial.missing_f0_hz)
                    if trial.dominant_hz is not None:
                        success_dominant.append(trial.dominant_hz)
                    success_snr.append(trial.snr_db)
                    label = trial.true_label or "-"
                    bucket = per_label.setdefault(label, {
                        "count": 0,
                        "scores": [],
                        "missing": [],
                        "dominant": [],
                        "snr": [],
                    })
                    bucket["count"] += 1
                    if trial.score is not None:
                        bucket["scores"].append(trial.score)
                    if trial.missing_f0_hz is not None:
                        bucket["missing"].append(trial.missing_f0_hz)
                    if trial.dominant_hz is not None:
                        bucket["dominant"].append(trial.dominant_hz)
                    bucket["snr"].append(trial.snr_db)
            else:
                # undetected beacon
                pass
        else:
            if trial.detected:
                false_positive += 1
                if trial.score is not None:
                    false_scores.append(trial.score)
                if trial.missing_f0_hz is not None:
                    false_missing.append(trial.missing_f0_hz)

    total_beacons = sum(1 for r in results if r.has_beacon)
    total_empty = len(results) - total_beacons

    aggregate: Dict[str, object] = {
        "num_trials": len(results),
        "num_beacons": total_beacons,
        "num_empty": total_empty,
        "detected_rate": _fraction(r.detected for r in results if r.has_beacon),
        "label_accuracy": (true_positive / total_beacons) if total_beacons else None,
        "false_positive_rate": (false_positive / total_empty) if total_empty else None,
        "detected_trials": detected_trials,
        "true_positive_trials": true_positive,
        "score_stats_success": _summary_stats(success_scores),
        "score_stats_false": _summary_stats(false_scores),
        "missing_f0_hz_success": _summary_stats(success_missing),
        "missing_f0_hz_false": _summary_stats(false_missing),
        "dominant_hz_success": _summary_stats(success_dominant),
        "snr_success": _summary_stats(success_snr),
        "config": {
            "profiles": args.profiles,
            "sample_rate": args.sample_rate,
            "symbol_length": args.symbol_length,
            "analysis_length": args.analysis_length,
            "fundamental_hz": args.fundamental_hz,
            "harmonics": args.harmonics,
            "include_fundamental": args.include_fundamental,
            "formant_scale": args.formant_scale,
            "phase_jitter": args.phase_jitter,
            "max_extra_paths": args.max_extra_paths,
            "max_delay_ns": args.max_delay_ns,
            "snr_db_min": args.snr_db_min,
            "snr_db_max": args.snr_db_max,
            "empty_prob": args.empty_prob,
            "top_peaks": args.top_peaks,
            "rng_seed": args.rng_seed,
        },
    }

    aggregate["per_label"] = {}
    for label, bucket in sorted(per_label.items()):
        aggregate["per_label"][label] = {
            "count": bucket["count"],
            "score": _summary_stats(bucket["scores"]),
            "missing_f0_hz": _summary_stats(bucket["missing"]),
            "dominant_hz": _summary_stats(bucket["dominant"]),
            "snr_db": _summary_stats(bucket["snr"]),
        }

    return aggregate


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profiles", nargs="+", default=_DEF_PROFILES, help="Vowel profiles to beacon (default: A E I O U)")
    parser.add_argument("--num-trials", type=int, default=512, help="Total number of Monte Carlo trials")
    parser.add_argument("--snr-db", dest="snr_db", type=float, nargs="*", default=[25.0], help="One or two values for SNR range (dB)")
    parser.add_argument("--sample-rate", type=float, default=2_000_000.0, help="Sample rate in Hz")
    parser.add_argument("--symbol-length", type=int, default=4096, help="Synthesized beacon length (samples)")
    parser.add_argument("--analysis-length", type=int, default=2048, help="Samples analyzed by the decoder")
    parser.add_argument("--fundamental-hz", type=float, default=25_000.0)
    parser.add_argument("--harmonics", type=int, default=12)
    parser.add_argument("--include-fundamental", action="store_true", help="Include the literal fundamental in synthesis")
    parser.add_argument("--formant-scale", type=float, default=1_000.0)
    parser.add_argument("--phase-jitter", type=float, default=0.0, help="Uniform phase jitter (rad) per harmonic")
    parser.add_argument("--max-extra-paths", type=int, default=3, help="Maximum number of non-line-of-sight taps")
    parser.add_argument("--max-delay-ns", type=float, default=80.0, help="Maximum multipath delay spread (ns)")
    parser.add_argument("--empty-prob", type=float, default=0.2, help="Probability that no beacon is present")
    parser.add_argument("--top-peaks", type=int, default=6, help="Number of FFT peaks fed to the analyzer")
    parser.add_argument("--score-threshold", type=float, default=None, help="Optional maximum score for declaring detection")
    parser.add_argument("--missing-f0-tolerance-hz", type=float, default=None, help="Reject detections when |missing_f0 - fundamental| exceeds this (Hz)")
    parser.add_argument("--dominant-tolerance-hz", type=float, default=None, help="Reject detections when |dominant - descriptor| exceeds this (Hz)")
    parser.add_argument("--rng-seed", type=int, default=2025)
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON output path")
    parser.add_argument("--dump-trials", type=Path, default=None, help="Optional path to dump per-trial records (JSON lines)")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.num_trials <= 0:
        raise SystemExit("--num-trials must be positive")
    if args.analysis_length <= 0:
        raise SystemExit("--analysis-length must be positive")
    if args.analysis_length > args.symbol_length:
        raise SystemExit("--analysis-length cannot exceed --symbol-length")

    if not args.profiles:
        raise SystemExit("At least one profile must be specified")
    profiles = [profile.upper() for profile in args.profiles]

    if len(args.snr_db) == 1:
        snr_min = snr_max = float(args.snr_db[0])
    elif len(args.snr_db) >= 2:
        snr_min, snr_max = sorted(float(v) for v in args.snr_db[:2])
    else:
        snr_min = snr_max = 25.0
    args.snr_db_min = snr_min
    args.snr_db_max = snr_max

    rng = np.random.default_rng(args.rng_seed)

    library = build_formant_library(
        fundamental_hz=args.fundamental_hz,
        harmonic_count=args.harmonics,
        include_fundamental=args.include_fundamental,
        formant_scale=args.formant_scale,
    )

    results: List[TrialResult] = []
    for _ in range(args.num_trials):
        results.append(_run_trial(args, profiles, library, rng))

    summary = _aggregate(results, args)
    summary["score_threshold"] = args.score_threshold

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"Saved spectrum beacon summary to {args.output}")

    if args.dump_trials:
        args.dump_trials.parent.mkdir(parents=True, exist_ok=True)
        with args.dump_trials.open("w", encoding="utf-8") as handle:
            for trial in results:
                record = {
                    "has_beacon": trial.has_beacon,
                    "true_label": trial.true_label,
                    "detected": trial.detected,
                    "predicted_label": trial.predicted_label,
                    "score": trial.score,
                    "missing_f0_hz": trial.missing_f0_hz,
                    "dominant_hz": trial.dominant_hz,
                    "delay_ns": trial.delay_ns,
                    "snr_db": trial.snr_db,
                }
                handle.write(json.dumps(record) + "\n")
        print(f"Dumped {len(results)} trials to {args.dump_trials}")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
