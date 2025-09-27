#!/usr/bin/env python3
"""Enhanced spectrum beacon simulation with improved formant robustness.

This script extends the basic beacon simulation with enhanced formant discrimination
features including adaptive bandwidth control, harmonic weighting, and multipath
awareness to address I/E vowel confusion challenges.

Usage example::

    python scripts/run_enhanced_spectrum_beacon_sim.py \
        --profiles A E I O U \
        --num-trials 1000 \
        --snr-db 20 35 \
        --channel-profile URBAN_CANYON \
        --enhanced-features all \
        --output results/enhanced_beacon_validation.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.phy.formants_enhanced import (
    EnhancedFormantAnalysisResult,
    EnhancedFormantSynthesisConfig,
    analyze_enhanced_missing_fundamental,
    build_enhanced_formant_library,
    calculate_formant_bandwidth,
    synthesize_enhanced_formant_preamble,
)
from src.chan.tdl import TDL_PROFILES, tdl_from_profile


@dataclass
class EnhancedTrialResult:
    """Enhanced outcome of a single beacon transmission with multipath awareness."""
    
    has_beacon: bool
    true_label: Optional[str]
    detected: bool
    predicted_label: Optional[str]
    score: Optional[float]
    confidence: Optional[float]
    missing_f0_hz: Optional[float]
    dominant_hz: Optional[float]
    delay_ns: float
    snr_db: float
    multipath_delay_ns: float
    formant_coherence: Optional[float]
    harmonic_agreement: Optional[float]
    # Enhanced metrics
    multipath_discrimination_score: Optional[float]
    adaptive_bandwidth_applied: bool


_DEF_PROFILES = ("A", "E", "I", "O", "U")
_ENHANCED_FEATURES = {"adaptive_bandwidth", "harmonic_weighting", "multipath_discrimination", "all"}


def _summary_stats(values: Iterable[float]) -> Dict[str, float]:
    """Calculate summary statistics for a set of values."""
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
    """Calculate fraction of True values in a sequence."""
    seq = list(events)
    if not seq:
        return None
    return float(np.mean(seq))


def _simulate_enhanced_channel(
    waveform: np.ndarray,
    sample_rate: float,
    channel_profile: str,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, float]:
    """Apply realistic multipath channel using TDL profiles."""
    
    base = waveform
    if channel_profile.upper() == "IDEAL":
        return base, 0.0
    
    # Create TDL channel from profile
    tdl = tdl_from_profile(channel_profile, rng)
    
    # Apply channel to waveform
    fs = sample_rate
    channel_output = tdl.apply_to_waveform(base, fs)
    
    # Calculate maximum multipath delay
    max_delay_ns = float(np.max(tdl.delays_s) * 1e9) if len(tdl.delays_s) > 0 else 0.0
    
    return channel_output, max_delay_ns


def _inject_awgn(waveform: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    """Inject additive white Gaussian noise."""
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


def _calculate_multipath_discrimination(
    formant_coherence: float,
    harmonic_agreement: float,
    multipath_delay_ns: float,
    snr_db: float,
) -> float:
    """Calculate multipath discrimination score.
    
    Higher scores indicate better discrimination against multipath-induced confusion.
    """
    if formant_coherence is None or harmonic_agreement is None:
        return 0.0
    
    # Base discrimination from formant coherence
    discrimination = formant_coherence * 0.6 + harmonic_agreement * 0.4
    
    # Penalty for severe multipath
    if multipath_delay_ns > 100.0:  # Severe multipath
        multipath_penalty = min(1.0, multipath_delay_ns / 500.0)
        discrimination *= (1.0 - multipath_penalty * 0.3)
    
    # Boost for high SNR
    if snr_db > 30.0:
        discrimination *= 1.1
    
    return float(np.clip(discrimination, 0.0, 1.0))


def _run_enhanced_trial(
    args: argparse.Namespace,
    profiles: List[str],
    library: Dict[str, object],
    rng: np.random.Generator,
) -> EnhancedTrialResult:
    """Run a single enhanced beacon trial with multipath awareness."""
    
    has_beacon = rng.random() > args.empty_prob
    snr_db = float(rng.uniform(args.snr_db_min, args.snr_db_max))
    
    if has_beacon:
        label = str(rng.choice(profiles))
        synth_config = EnhancedFormantSynthesisConfig(
            profile=label,
            fundamental_hz=args.fundamental_hz,
            harmonic_count=args.harmonics,
            include_fundamental=args.include_fundamental,
            formant_scale=args.formant_scale,
            phase_jitter=args.phase_jitter,
            adaptive_bandwidth=args.adaptive_bandwidth,
            prosodic_variation=args.prosodic_variation,
        )
        waveform, _ = synthesize_enhanced_formant_preamble(
            length=args.symbol_length,
            sample_rate=args.sample_rate,
            config=synth_config,
        )
        
        # Apply realistic multipath channel
        received, multipath_delay_ns = _simulate_enhanced_channel(
            waveform,
            sample_rate=args.sample_rate,
            channel_profile=args.channel_profile,
            rng=rng,
        )
        delay_ns = (len(received) - len(waveform)) / args.sample_rate * 1e9
    else:
        label = None
        received = np.zeros(args.symbol_length, dtype=np.complex128)
        delay_ns = 0.0
        multipath_delay_ns = 0.0

    received = _inject_awgn(received, snr_db=snr_db, rng=rng)
    segment = received[: args.analysis_length]

    # Enhanced analysis with multipath awareness
    analysis: Optional[EnhancedFormantAnalysisResult]
    descriptors = list(library.values())
    analysis = analyze_enhanced_missing_fundamental(
        segment, args.sample_rate, descriptors, 
        top_peaks=args.top_peaks, snr_estimate=snr_db
    )

    detected = False
    predicted_label = analysis.label if analysis else None
    score = analysis.score if analysis else None
    confidence = analysis.confidence if analysis else None
    missing_f0 = analysis.missing_fundamental_hz if analysis else None
    dominant_hz = analysis.dominant_hz if analysis else None
    formant_coherence = analysis.formant_coherence if analysis else None
    harmonic_agreement = analysis.harmonic_agreement if analysis else None

    multipath_discrimination = None
    if analysis is not None:
        multipath_discrimination = _calculate_multipath_discrimination(
            formant_coherence, harmonic_agreement, multipath_delay_ns, snr_db
        )
        
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

        # Enhanced confidence threshold
        if passes_checks and args.confidence_threshold is not None:
            if confidence is None or confidence < args.confidence_threshold:
                passes_checks = False

        if passes_checks:
            detected = True
        else:
            # downrank detection if validation failed
            predicted_label = None

    return EnhancedTrialResult(
        has_beacon=has_beacon,
        true_label=label,
        detected=detected,
        predicted_label=predicted_label,
        score=score,
        confidence=confidence,
        missing_f0_hz=missing_f0,
        dominant_hz=dominant_hz,
        delay_ns=delay_ns,
        snr_db=snr_db,
        multipath_delay_ns=multipath_delay_ns,
        formant_coherence=formant_coherence,
        harmonic_agreement=harmonic_agreement,
        multipath_discrimination_score=multipath_discrimination,
        adaptive_bandwidth_applied=args.adaptive_bandwidth,
    )


def _aggregate_enhanced_results(results: List[EnhancedTrialResult], args: argparse.Namespace) -> Dict[str, object]:
    """Aggregate enhanced results with detailed performance metrics."""
    
    success_scores: List[float] = []
    success_confidences: List[float] = []
    success_multipath_scores: List[float] = []
    success_formant_coherence: List[float] = []
    success_harmonic_agreement: List[float] = []

    false_scores: List[float] = []
    false_confidences: List[float] = []

    true_positive = 0
    detected_trials = 0
    false_positive = 0

    per_label: Dict[str, Dict[str, List[float]]] = {}
    confusion_matrix: Dict[str, Dict[str, int]] = {}

    for trial in results:
        if trial.has_beacon:
            if trial.detected:
                detected_trials += 1
                if trial.predicted_label == trial.true_label:
                    true_positive += 1
                    if trial.score is not None:
                        success_scores.append(trial.score)
                    if trial.confidence is not None:
                        success_confidences.append(trial.confidence)
                    if trial.multipath_discrimination_score is not None:
                        success_multipath_scores.append(trial.multipath_discrimination_score)
                    if trial.formant_coherence is not None:
                        success_formant_coherence.append(trial.formant_coherence)
                    if trial.harmonic_agreement is not None:
                        success_harmonic_agreement.append(trial.harmonic_agreement)
                    
                    # Update confusion matrix
                    true_label = trial.true_label or "-"
                    pred_label = trial.predicted_label or "-"
                    if true_label not in confusion_matrix:
                        confusion_matrix[true_label] = {}
                    confusion_matrix[true_label][pred_label] = confusion_matrix[true_label].get(pred_label, 0) + 1
                    
                    label = trial.true_label or "-"
                    bucket = per_label.setdefault(label, {
                        "count": 0,
                        "scores": [],
                        "confidences": [],
                        "multipath_scores": [],
                        "formant_coherence": [],
                        "harmonic_agreement": [],
                        "snr": [],
                    })
                    bucket["count"] += 1
                    if trial.score is not None:
                        bucket["scores"].append(trial.score)
                    if trial.confidence is not None:
                        bucket["confidences"].append(trial.confidence)
                    if trial.multipath_discrimination_score is not None:
                        bucket["multipath_scores"].append(trial.multipath_discrimination_score)
                    if trial.formant_coherence is not None:
                        bucket["formant_coherence"].append(trial.formant_coherence)
                    if trial.harmonic_agreement is not None:
                        bucket["harmonic_agreement"].append(trial.harmonic_agreement)
                    bucket["snr"].append(trial.snr_db)
            else:
                # undetected beacon
                pass
        else:
            if trial.detected:
                false_positive += 1
                if trial.score is not None:
                    false_scores.append(trial.score)
                if trial.confidence is not None:
                    false_confidences.append(trial.confidence)

    total_beacons = sum(1 for r in results if r.has_beacon)
    total_empty = len(results) - total_beacons

    # Calculate confusion rates
    confusion_rates = {}
    for true_label, pred_counts in confusion_matrix.items():
        total = sum(pred_counts.values())
        confusion_rates[true_label] = {
            pred_label: count / total for pred_label, count in pred_counts.items()
        }

    aggregate: Dict[str, object] = {
        "num_trials": len(results),
        "num_beacons": total_beacons,
        "num_empty": total_empty,
        "detected_rate": _fraction(r.detected for r in results if r.has_beacon),
        "label_accuracy": (true_positive / total_beacons) if total_beacons else None,
        "false_positive_rate": (false_positive / total_empty) if total_empty else None,
        "detected_trials": detected_trials,
        "true_positive_trials": true_positive,
        # Enhanced metrics
        "avg_confidence_success": _summary_stats(success_confidences),
        "avg_confidence_false": _summary_stats(false_confidences),
        "multipath_discrimination_score": _summary_stats(success_multipath_scores),
        "formant_coherence": _summary_stats(success_formant_coherence),
        "harmonic_agreement": _summary_stats(success_harmonic_agreement),
        "confusion_matrix": confusion_rates,
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
            "channel_profile": args.channel_profile,
            "snr_db_min": args.snr_db_min,
            "snr_db_max": args.snr_db_max,
            "empty_prob": args.empty_prob,
            "top_peaks": args.top_peaks,
            "adaptive_bandwidth": args.adaptive_bandwidth,
            "prosodic_variation": args.prosodic_variation,
            "rng_seed": args.rng_seed,
        },
    }

    aggregate["per_label"] = {}
    for label, bucket in sorted(per_label.items()):
        aggregate["per_label"][label] = {
            "count": bucket["count"],
            "score": _summary_stats(bucket["scores"]),
            "confidence": _summary_stats(bucket["confidences"]),
            "multipath_discrimination": _summary_stats(bucket["multipath_scores"]),
            "formant_coherence": _summary_stats(bucket["formant_coherence"]),
            "harmonic_agreement": _summary_stats(bucket["harmonic_agreement"]),
            "snr_db": _summary_stats(bucket["snr"]),
        }

    return aggregate


def _parse_args() -> argparse.Namespace:
    """Parse command line arguments for enhanced beacon simulation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profiles", nargs="+", default=_DEF_PROFILES, 
                       help="Vowel profiles to beacon (default: A E I O U)")
    parser.add_argument("--num-trials", type=int, default=512, 
                       help="Total number of Monte Carlo trials")
    parser.add_argument("--snr-db", dest="snr_db", type=float, nargs="*", default=[25.0], 
                       help="One or two values for SNR range (dB)")
    parser.add_argument("--channel-profile", type=str, default="URBAN_CANYON",
                       choices=list(TDL_PROFILES.keys()),
                       help="TDL channel profile for multipath simulation")
    parser.add_argument("--sample-rate", type=float, default=2_000_000.0, 
                       help="Sample rate in Hz")
    parser.add_argument("--symbol-length", type=int, default=4096, 
                       help="Synthesized beacon length (samples)")
    parser.add_argument("--analysis-length", type=int, default=2048, 
                       help="Samples analyzed by the decoder")
    parser.add_argument("--fundamental-hz", type=float, default=25_000.0)
    parser.add_argument("--harmonics", type=int, default=12)
    parser.add_argument("--include-fundamental", action="store_true", 
                       help="Include the literal fundamental in synthesis")
    parser.add_argument("--formant-scale", type=float, default=1_000.0)
    parser.add_argument("--phase-jitter", type=float, default=0.0, 
                       help="Uniform phase jitter (rad) per harmonic")
    parser.add_argument("--empty-prob", type=float, default=0.2, 
                       help="Probability that no beacon is present")
    parser.add_argument("--top-peaks", type=int, default=8, 
                       help="Number of FFT peaks fed to the analyzer")
    parser.add_argument("--score-threshold", type=float, default=None, 
                       help="Optional maximum score for declaring detection")
    parser.add_argument("--confidence-threshold", type=float, default=0.5,
                       help="Minimum confidence threshold for detection")
    parser.add_argument("--missing-f0-tolerance-hz", type=float, default=None, 
                       help="Reject detections when |missing_f0 - fundamental| exceeds this (Hz)")
    parser.add_argument("--dominant-tolerance-hz", type=float, default=None, 
                       help="Reject detections when |dominant - descriptor| exceeds this (Hz)")
    # Enhanced features
    parser.add_argument("--adaptive-bandwidth", action="store_true", default=True,
                       help="Enable adaptive bandwidth control based on channel conditions")
    parser.add_argument("--prosodic-variation", action="store_true", default=True,
                       help="Enable temporal prosodic variation for disambiguation")
    parser.add_argument("--rng-seed", type=int, default=2025)
    parser.add_argument("--output", type=Path, default=None, 
                       help="Optional JSON output path")
    parser.add_argument("--dump-trials", type=Path, default=None, 
                       help="Optional path to dump per-trial records (JSON lines)")
    return parser.parse_args()


def main() -> None:
    """Main enhanced beacon simulation function."""
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

    # Build enhanced formant library with channel awareness
    library = build_enhanced_formant_library(
        fundamental_hz=args.fundamental_hz,
        harmonic_count=args.harmonics,
        include_fundamental=args.include_fundamental,
        formant_scale=args.formant_scale,
        snr_db=(snr_min + snr_max) / 2.0,  # Use average SNR for bandwidth calculation
        multipath_delay_ns=100.0 if args.channel_profile != "IDEAL" else 0.0,
    )

    results: List[EnhancedTrialResult] = []
    for _ in range(args.num_trials):
        results.append(_run_enhanced_trial(args, profiles, library, rng))

    summary = _aggregate_enhanced_results(results, args)
    summary["score_threshold"] = args.score_threshold
    summary["confidence_threshold"] = args.confidence_threshold

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"Saved enhanced spectrum beacon summary to {args.output}")

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
                    "confidence": trial.confidence,
                    "missing_f0_hz": trial.missing_f0_hz,
                    "dominant_hz": trial.dominant_hz,
                    "delay_ns": trial.delay_ns,
                    "snr_db": trial.snr_db,
                    "multipath_delay_ns": trial.multipath_delay_ns,
                    "formant_coherence": trial.formant_coherence,
                    "harmonic_agreement": trial.harmonic_agreement,
                    "multipath_discrimination_score": trial.multipath_discrimination_score,
                    "adaptive_bandwidth_applied": trial.adaptive_bandwidth_applied,
                }
                handle.write(json.dumps(record) + "\n")
        print(f"Dumped {len(results)} enhanced trials to {args.dump_trials}")

    # Print enhanced summary
    print("ENHANCED SPECTRUM BEACON SIMULATION RESULTS")
    print("=" * 50)
    print(f"Channel Profile: {args.channel_profile}")
    print(f"SNR Range: {args.snr_db_min}-{args.snr_db_max} dB")
    print(f"Detection Rate: {summary['detected_rate']:.1%}")
    print(f"Label Accuracy: {summary['label_accuracy']:.1%}")
    print(f"False Positive Rate: {summary['false_positive_rate']:.1%}")
    
    if 'avg_confidence_success' in summary and summary['avg_confidence_success']:
        print(f"Average Confidence: {summary['avg_confidence_success']['mean']:.3f}")
    
    if 'multipath_discrimination_score' in summary and summary['multipath_discrimination_score']:
        print(f"Multipath Discrimination: {summary['multipath_discrimination_score']['mean']:.3f}")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()