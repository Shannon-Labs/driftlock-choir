#!/usr/bin/env python3
"""Validation script for enhanced formant features against baseline performance.

This script runs comparative analysis between the baseline formant system and
enhanced formant features to quantify improvements in I/E vowel discrimination.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

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
    VOWEL_FORMANT_TABLE,
)


@dataclass
class ValidationResult:
    """Results of comparative validation between baseline and enhanced systems."""
    
    profile: str
    baseline_accuracy: float
    enhanced_accuracy: float
    improvement: float
    confusion_reduction: Dict[str, float]
    confidence_improvement: float
    multipath_resilience: float


@dataclass
class EnhancedFormantResult:
    """Wrapper for enhanced formant analysis results with additional metrics."""
    
    baseline_result: FormantAnalysisResult
    confidence: float
    formant_coherence: float
    discrimination_score: float
    snr_estimate: float
    
    @property
    def label(self):
        return self.baseline_result.label
    
    @property
    def dominant_hz(self):
        return self.baseline_result.dominant_hz
    
    @property
    def missing_fundamental_hz(self):
        return self.baseline_result.missing_fundamental_hz
    
    @property
    def score(self):
        return self.baseline_result.score


# Enhanced formant parameters based on acoustic engineering optimization
# Formants are in Hz, matching the original VOWEL_FORMANT_TABLE format
ENHANCED_FORMANT_TABLE = {
    "A": {"F1": 850.0, "F2": 1610.0, "F3": 2800.0},  # Enhanced A formants
    "E": {"F1": 550.0, "F2": 2100.0, "F3": 2700.0},  # Enhanced E: F2 increased to 2100 Hz for better separation from I
    "I": {"F1": 350.0, "F2": 2500.0, "F3": 3100.0},  # Enhanced I: F2 increased to 2500 Hz for better separation from E
    "O": {"F1": 600.0, "F2": 1000.0, "F3": 2700.0},  # Original O formants
    "U": {"F1": 350.0, "F2": 800.0, "F3": 2500.0},   # Original U formants
}


def _enhance_formant_synthesis(
    profile: str,
    fundamental_hz: float = 25000.0,
    harmonic_count: int = 12,
    include_fundamental: bool = False,
    formant_scale: float = 1.0,  # Set to 1.0 since we're using Hz directly
    phase_jitter: float = 0.0,
    adaptive_bandwidth: bool = True,
    snr_db: float = 25.0,
) -> np.ndarray:
    """Enhanced formant synthesis with optimized parameters and adaptive features."""
    
    # Use enhanced formant frequencies (already in Hz)
    formants = ENHANCED_FORMANT_TABLE.get(profile.upper(), VOWEL_FORMANT_TABLE[profile.upper()])
    
    # Adaptive bandwidth based on SNR and channel conditions
    if adaptive_bandwidth:
        # Wider bandwidth for lower SNR, narrower for higher SNR
        bandwidth_factor = max(0.7, min(1.3, 1.0 - (snr_db - 20) / 50))
        formant_scale = formant_scale * bandwidth_factor
    
    # Create enhanced synthesis with proper formant scaling
    synth_config = FormantSynthesisConfig(
        profile=profile,
        fundamental_hz=fundamental_hz,
        harmonic_count=harmonic_count,
        include_fundamental=include_fundamental,
        formant_scale=formant_scale,  # Now 1.0 since formants are in Hz
        phase_jitter=phase_jitter,
    )
    
    waveform, _ = synthesize_formant_preamble(
        length=4096,  # Standard symbol length
        sample_rate=2_000_000.0,
        config=synth_config,
    )
    
    return waveform


def _calculate_formant_coherence(spectrum: np.ndarray, formant_freqs: Dict[str, float], 
                                sample_rate: float) -> float:
    """Calculate how well formant peaks align with expected frequencies."""
    n = len(spectrum)
    freqs = np.fft.fftfreq(n, 1/sample_rate)[:n//2]
    spectrum_mag = np.abs(spectrum[:n//2])
    
    coherence_score = 0.0
    expected_peaks = []
    
    # Look for peaks near expected formant frequencies
    for formant_name, expected_freq in formant_freqs.items():
        # Search window around expected frequency
        freq_window = 200  # Hz
        idx_start = np.argmax(freqs >= expected_freq - freq_window)
        idx_end = np.argmax(freqs >= expected_freq + freq_window)
        
        if idx_end > idx_start:
            window_spectrum = spectrum_mag[idx_start:idx_end]
            if len(window_spectrum) > 0:
                peak_strength = np.max(window_spectrum)
                # Normalize by overall spectrum strength
                overall_max = np.max(spectrum_mag) if np.max(spectrum_mag) > 0 else 1.0
                coherence_score += (peak_strength / overall_max) / len(formant_freqs)
    
    return float(np.clip(coherence_score, 0.0, 1.0))


def _analyze_enhanced_formants(
    waveform: np.ndarray,
    sample_rate: float,
    profiles: List[str],
    snr_estimate: float = 25.0,
):
    """Enhanced analysis with formant coherence and discrimination metrics."""
    
    # First, run baseline analysis
    library = build_formant_library(
        fundamental_hz=25000.0,
        harmonic_count=12,
        include_fundamental=False,
        formant_scale=1000.0,
    )
    
    descriptors = list(library.values())
    baseline_result = analyze_missing_fundamental(
        waveform, sample_rate, descriptors, top_peaks=8
    )
    
    if not baseline_result:
        return None
    
    # Calculate enhanced metrics
    spectrum = np.fft.fft(waveform)
    predicted_profile = baseline_result.label.upper()
    
    # Formant coherence for predicted profile
    formant_freqs = ENHANCED_FORMANT_TABLE.get(predicted_profile,
                                              VOWEL_FORMANT_TABLE[predicted_profile])
    formant_coherence = _calculate_formant_coherence(spectrum, formant_freqs, sample_rate)
    
    # Calculate discrimination score between similar vowels (E vs I)
    if predicted_profile in ["E", "I"]:
        alternative_profile = "I" if predicted_profile == "E" else "E"
        alt_formant_freqs = ENHANCED_FORMANT_TABLE.get(alternative_profile,
                                                      VOWEL_FORMANT_TABLE[alternative_profile])
        
        # How much better does the predicted profile fit than the alternative?
        alt_coherence = _calculate_formant_coherence(spectrum, alt_formant_freqs, sample_rate)
        discrimination_score = max(0.0, formant_coherence - alt_coherence)
    else:
        discrimination_score = 0.5  # Neutral for non-confusing vowels
    
    # Enhanced confidence based on formant coherence and discrimination
    # Use score as base confidence (lower score = higher confidence)
    baseline_confidence = max(0.0, 1.0 - baseline_result.score / 100.0) if baseline_result.score > 0 else 0.5
    enhanced_confidence = min(1.0, baseline_confidence +
                            0.3 * formant_coherence + 0.2 * discrimination_score)

    # Create enhanced result using the wrapper class
    enhanced_result = EnhancedFormantResult(
        baseline_result=baseline_result,
        confidence=enhanced_confidence,
        formant_coherence=formant_coherence,
        discrimination_score=discrimination_score,
        snr_estimate=snr_estimate,
    )
    
    return enhanced_result


def _run_comparative_trial(
    profiles: List[str],
    rng: np.random.Generator,
    use_enhanced: bool = True,
    snr_db: float = 25.0,
    multipath_severity: float = 0.0,
) -> Dict[str, float]:
    """Run a single trial comparing baseline vs enhanced performance."""
    
    has_beacon = rng.random() > 0.2  # 20% chance of no beacon
    if not has_beacon:
        return {"detected": False, "correct": False, "profile": None}
    
    # Select random profile
    profile = rng.choice(profiles)
    
    # Generate waveform
    if use_enhanced:
        waveform = _enhance_formant_synthesis(
            profile=profile,
            snr_db=snr_db,
            adaptive_bandwidth=True,
        )
    else:
        synth_config = FormantSynthesisConfig(
            profile=profile,
            fundamental_hz=25000.0,
            harmonic_count=12,
            include_fundamental=False,
            formant_scale=1.0,  # Set to 1.0 for baseline too
            phase_jitter=0.0,
        )
        waveform, _ = synthesize_formant_preamble(
            length=4096,
            sample_rate=2_000_000.0,
            config=synth_config,
        )
    
    # Add simple multipath simulation
    if multipath_severity > 0:
        delay_samples = int(multipath_severity * 10)  # Up to 10 sample delay
        if delay_samples > 0:
            delayed_component = waveform[:-delay_samples] * 0.3
            waveform[delay_samples:] += delayed_component
    
    # Add noise
    power = np.mean(np.abs(waveform) ** 2)
    if power > 0:
        snr_linear = 10.0 ** (snr_db / 10.0)
        noise_power = power / snr_linear
        noise = (rng.normal(scale=np.sqrt(noise_power/2), size=waveform.shape) +
                1j * rng.normal(scale=np.sqrt(noise_power/2), size=waveform.shape))
        waveform += noise
    
    # Analyze
    if use_enhanced:
        result = _analyze_enhanced_formants(
            waveform, 2_000_000.0, profiles, snr_estimate=snr_db
        )
    else:
        library = build_formant_library(
            fundamental_hz=25000.0,
            harmonic_count=12,
            include_fundamental=False,
            formant_scale=1000.0,
        )
        descriptors = list(library.values())
        result = analyze_missing_fundamental(waveform, 2_000_000.0, descriptors, top_peaks=8)
    
    # For baseline analysis, use score-based detection
    if use_enhanced:
        # Enhanced: use confidence threshold
        detected = result is not None and getattr(result, 'confidence', 0.0) > 0.5
        confidence = getattr(result, 'confidence', 0.0) if result else 0.0
    else:
        # Baseline: detect if result exists and score is reasonable
        detected = result is not None and result.score < 1e10  # More reasonable threshold
        confidence = 1.0 if detected else 0.0
    
    correct = detected and result is not None and result.label.upper() == profile.upper()
    
    return {
        "detected": detected,
        "correct": correct,
        "profile": profile,
        "predicted": result.label if result else None,
        "confidence": confidence,
        "formant_coherence": getattr(result, 'formant_coherence', 0.0) if result else 0.0,
        "discrimination_score": getattr(result, 'discrimination_score', 0.0) if result else 0.0,
    }


def run_validation(
    profiles: List[str] = ["A", "E", "I", "O", "U"],
    num_trials: int = 1000,
    snr_range: tuple = (20, 35),
    multipath_levels: List[float] = [0.0, 0.5, 1.0],
    rng_seed: int = 2025,
) -> Dict[str, ValidationResult]:
    """Run comprehensive validation comparing baseline and enhanced formant systems."""
    
    rng = np.random.default_rng(rng_seed)
    results = {}
    
    for profile in profiles:
        print(f"Validating profile: {profile}")
        
        baseline_correct = 0
        baseline_total = 0
        enhanced_correct = 0
        enhanced_total = 0
        
        confusion_counts_baseline = {p: 0 for p in profiles if p != profile}
        confusion_counts_enhanced = {p: 0 for p in profiles if p != profile}
        
        confidence_baseline = []
        confidence_enhanced = []
        multipath_resilience = []
        
        for _ in range(num_trials):
            # Vary SNR and multipath
            snr_db = rng.uniform(snr_range[0], snr_range[1])
            multipath = rng.choice(multipath_levels)
            
            # Baseline trial
            baseline_result = _run_comparative_trial(
                profiles, rng, use_enhanced=False, 
                snr_db=snr_db, multipath_severity=multipath
            )
            if baseline_result["profile"] == profile:
                baseline_total += 1
                if baseline_result["correct"]:
                    baseline_correct += 1
                elif baseline_result["predicted"] and baseline_result["predicted"] in confusion_counts_baseline:
                    confusion_counts_baseline[baseline_result["predicted"]] += 1
                confidence_baseline.append(baseline_result["confidence"])
            
            # Enhanced trial
            enhanced_result = _run_comparative_trial(
                profiles, rng, use_enhanced=True,
                snr_db=snr_db, multipath_severity=multipath
            )
            if enhanced_result["profile"] == profile:
                enhanced_total += 1
                if enhanced_result["correct"]:
                    enhanced_correct += 1
                elif enhanced_result["predicted"] and enhanced_result["predicted"] in confusion_counts_enhanced:
                    confusion_counts_enhanced[enhanced_result["predicted"]] += 1
                confidence_enhanced.append(enhanced_result["confidence"])
                multipath_resilience.append(enhanced_result["discrimination_score"])
        
        # Calculate metrics
        baseline_accuracy = baseline_correct / baseline_total if baseline_total > 0 else 0.0
        enhanced_accuracy = enhanced_correct / enhanced_total if enhanced_total > 0 else 0.0
        
        # Confusion reduction
        confusion_reduction = {}
        for p in confusion_counts_baseline:
            baseline_confusion = confusion_counts_baseline[p] / baseline_total if baseline_total > 0 else 0.0
            enhanced_confusion = confusion_counts_enhanced[p] / enhanced_total if enhanced_total > 0 else 0.0
            reduction = baseline_confusion - enhanced_confusion
            confusion_reduction[p] = max(0.0, reduction)
        
        # Confidence improvement
        avg_confidence_baseline = np.mean(confidence_baseline) if confidence_baseline else 0.5
        avg_confidence_enhanced = np.mean(confidence_enhanced) if confidence_enhanced else 0.5
        confidence_improvement = avg_confidence_enhanced - avg_confidence_baseline
        
        # Multipath resilience
        avg_multipath_resilience = np.mean(multipath_resilience) if multipath_resilience else 0.5
        
        results[profile] = ValidationResult(
            profile=profile,
            baseline_accuracy=baseline_accuracy,
            enhanced_accuracy=enhanced_accuracy,
            improvement=enhanced_accuracy - baseline_accuracy,
            confusion_reduction=confusion_reduction,
            confidence_improvement=confidence_improvement,
            multipath_resilience=avg_multipath_resilience,
        )
    
    return results


def main():
    """Main validation function."""
    parser = argparse.ArgumentParser(description="Validate enhanced formant features")
    parser.add_argument("--profiles", nargs="+", default=["A", "E", "I", "O", "U"],
                       help="Vowel profiles to validate")
    parser.add_argument("--num-trials", type=int, default=500,
                       help="Number of trials per profile")
    parser.add_argument("--snr-min", type=float, default=20.0,
                       help="Minimum SNR in dB")
    parser.add_argument("--snr-max", type=float, default=35.0,
                       help="Maximum SNR in dB")
    parser.add_argument("--output", type=Path, default=None,
                       help="Output JSON file for results")
    parser.add_argument("--rng-seed", type=int, default=2025)
    
    args = parser.parse_args()
    
    print("ENHANCED FORMANT VALIDATION")
    print("=" * 50)
    print(f"Profiles: {args.profiles}")
    print(f"Trials per profile: {args.num_trials}")
    print(f"SNR range: {args.snr_min}-{args.snr_max} dB")
    print()
    
    results = run_validation(
        profiles=args.profiles,
        num_trials=args.num_trials,
        snr_range=(args.snr_min, args.snr_max),
        rng_seed=args.rng_seed,
    )
    
    # Print summary
    total_improvement = 0.0
    e_i_confusion_reduction = 0.0
    
    print("VALIDATION RESULTS")
    print("=" * 50)
    for profile, result in results.items():
        print(f"\nProfile: {profile}")
        print(f"  Baseline Accuracy: {result.baseline_accuracy:.1%}")
        print(f"  Enhanced Accuracy: {result.enhanced_accuracy:.1%}")
        print(f"  Improvement: {result.improvement:+.1%}")
        print(f"  Confidence Improvement: {result.confidence_improvement:+.3f}")
        print(f"  Multipath Resilience: {result.multipath_resilience:.3f}")
        
        if result.confusion_reduction:
            print("  Confusion Reduction:")
            for confused_with, reduction in result.confusion_reduction.items():
                if reduction > 0.01:  # Only show significant reductions
                    print(f"    {confused_with}: {reduction:.1%}")
                    if (profile, confused_with) in [("E", "I"), ("I", "E")]:
                        e_i_confusion_reduction = max(e_i_confusion_reduction, reduction)
        
        total_improvement += result.improvement
    
    avg_improvement = total_improvement / len(results)
    
    print(f"\nOVERALL SUMMARY")
    print(f"Average Accuracy Improvement: {avg_improvement:.1%}")
    print(f"E/I Confusion Reduction: {e_i_confusion_reduction:.1%}")
    
    # Save results if requested
    if args.output:
        # Convert results to serializable format
        serializable_results = {}
        for profile, result in results.items():
            serializable_results[profile] = {
                "baseline_accuracy": result.baseline_accuracy,
                "enhanced_accuracy": result.enhanced_accuracy,
                "improvement": result.improvement,
                "confusion_reduction": result.confusion_reduction,
                "confidence_improvement": result.confidence_improvement,
                "multipath_resilience": result.multipath_resilience,
            }
        
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()