#!/usr/bin/env python3
"""Extended vocal beacon simulation - Phase 1 of Musical-RF Architecture.

This script tests the extended vocal techniques (diphthongs, ornaments, 
consonant-vowel combinations) building on the proven Italian vowel foundation.

Based on the breakthrough that improved vowel "I" from 0% to 57.9% detection,
we now expand to 30+ additional beacon types with dynamic spectral signatures.

Usage:

    python scripts/run_extended_vocal_beacons.py \
        --vocal-types diphthongs ornaments consonant-vowels \
        --num-trials 200 \
        --snr-db 20 35 \
        --output results/project_aperture_formant/extended_vocals/test.json
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

# Ensure src/ is importable
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / 'src') not in sys.path:
    sys.path.insert(0, str(ROOT / 'src'))

from phy.vocal_extensions import (
    COMMON_EXTENDED_VOCALS,
    build_extended_vocal_library,
    DiphthongConfig,
    VocalOrnamentConfig, 
    ConsonantVowelConfig,
    create_diphthong_descriptor,
    create_vocal_ornament_descriptor,
    create_consonant_vowel_descriptor,
)
from phy.formants import (
    FormantDescriptor,
    FormantSynthesisConfig,
    build_formant_library,
    synthesize_formant_preamble,
    analyze_missing_fundamental,
)


def get_extended_vocal_profiles(vocal_types: List[str]) -> List[str]:
    """Get list of extended vocal profiles based on requested types."""
    
    profiles = []
    
    if "diphthongs" in vocal_types:
        profiles.extend(["AI", "AU", "EI", "OU"])
        
    if "ornaments" in vocal_types:
        profiles.extend(["TRILL_A", "TRILL_E", "TRILL_I", "VIBRATO_A", "VIBRATO_I"])
        
    if "consonant-vowels" in vocal_types:
        profiles.extend(["MA", "LA", "NA", "MI", "LI"])
        
    if "italian-vowels" in vocal_types:
        profiles.extend(["A", "E", "I", "O", "U"])  # Include proven foundation
    
    return profiles


def simulate_extended_vocal_beacon(
    profile: str,
    library: Dict[str, FormantDescriptor],
    sample_rate: float = 2e6,
    symbol_length: int = 4096,
    analysis_length: int = 2048,
    snr_db: float = 25.0,
    max_extra_paths: int = 4,
    max_delay_ns: float = 120.0,
    rng: Optional[np.random.Generator] = None,
) -> Dict[str, object]:
    """Simulate a single extended vocal beacon transmission."""
    
    if rng is None:
        rng = np.random.default_rng()
    
    if profile not in library:
        return {
            "profile": profile,
            "success": False,
            "error": f"Unknown profile: {profile}",
        }
    
    descriptor = library[profile]
    
    # Synthesize the extended vocal beacon
    config = FormantSynthesisConfig(
        profile=profile,
        fundamental_hz=descriptor.fundamental_hz,
        harmonic_count=len(descriptor.harmonics_hz),
        include_fundamental=descriptor.include_fundamental,
        formant_scale=1.0,  # Already scaled in descriptor
        phase_jitter=0.0,   # Start with clean signals
    )
    
    try:
        waveform, synth_library = synthesize_formant_preamble(
            symbol_length, sample_rate, config
        )
    except Exception as e:
        return {
            "profile": profile,
            "success": False, 
            "error": f"Synthesis failed: {e}",
        }
    
    # Add multipath channel simulation (simple version)
    if max_extra_paths > 0 and max_delay_ns > 0:
        max_delay_s = max_delay_ns * 1e-9
        max_offset_samples = int(np.ceil(max_delay_s * sample_rate))
        
        if max_offset_samples > 0:
            num_paths = rng.integers(0, max_extra_paths + 1)
            for _ in range(int(num_paths)):
                delay_samples = rng.integers(1, max_offset_samples + 1)
                amplitude = rng.normal(0, 0.3) + 1j * rng.normal(0, 0.3)
                
                # Add delayed copy
                if delay_samples < len(waveform):
                    delayed_waveform = np.zeros_like(waveform)
                    delayed_waveform[delay_samples:] = waveform[:-delay_samples] * amplitude
                    waveform = waveform + delayed_waveform
    
    # Add AWGN
    signal_power = np.mean(np.abs(waveform) ** 2)
    if signal_power > 0 and np.isfinite(snr_db):
        snr_linear = 10 ** (snr_db / 10)
        noise_power = signal_power / snr_linear
        noise = (
            rng.normal(0, np.sqrt(noise_power / 2), waveform.shape)
            + 1j * rng.normal(0, np.sqrt(noise_power / 2), waveform.shape)
        )
        waveform = waveform + noise
    
    # Analyze the received signal
    analysis_segment = waveform[:analysis_length]
    
    try:
        result = analyze_missing_fundamental(
            analysis_segment,
            sample_rate,
            list(library.values()),
            top_peaks=6,
        )
    except Exception as e:
        return {
            "profile": profile,
            "success": False,
            "error": f"Analysis failed: {e}",
        }
    
    if result is None:
        return {
            "profile": profile,
            "success": False,
            "detected_profile": None,
            "score": None,
            "missing_f0_hz": None,
            "dominant_hz": None,
        }
    
    # Determine success
    success = result.label == profile
    
    return {
        "profile": profile,
        "success": success,
        "detected_profile": result.label,
        "score": result.score,
        "missing_f0_hz": result.missing_fundamental_hz,
        "dominant_hz": result.dominant_hz,
        "snr_db": snr_db,
    }


def run_extended_vocal_experiment(
    vocal_types: List[str],
    num_trials: int,
    snr_db_range: tuple,
    max_extra_paths: int = 4,
    max_delay_ns: float = 120.0,
    rng_seed: int = 2025,
) -> Dict[str, object]:
    """Run comprehensive extended vocal beacon experiment."""
    
    rng = np.random.default_rng(rng_seed)
    
    # Get profiles to test
    profiles = get_extended_vocal_profiles(vocal_types)
    if not profiles:
        raise ValueError("No profiles selected for testing")
    
    print(f"Testing {len(profiles)} extended vocal profiles: {profiles}")
    
    # Build combined library (Italian vowels + extended vocals)
    fundamental_hz = 25000.0
    harmonic_count = 12
    formant_scale = 1000.0
    
    # Start with proven Italian vowels
    base_library = build_formant_library(
        fundamental_hz, harmonic_count, False, formant_scale
    )
    
    # Add extended vocal techniques
    extended_library = build_extended_vocal_library(
        fundamental_hz, harmonic_count, formant_scale
    )
    
    # Combine libraries
    full_library = {**base_library, **extended_library}
    
    print(f"Built library with {len(full_library)} total vocal profiles")
    
    # Run trials
    results = []
    snr_min, snr_max = snr_db_range
    
    for trial_idx in range(num_trials):
        # Random profile and SNR for this trial
        profile = str(rng.choice(profiles))
        snr_db = float(rng.uniform(snr_min, snr_max))
        
        result = simulate_extended_vocal_beacon(
            profile=profile,
            library=full_library,
            snr_db=snr_db,
            max_extra_paths=max_extra_paths,
            max_delay_ns=max_delay_ns,
            rng=rng,
        )
        
        result["trial"] = trial_idx
        results.append(result)
        
        if (trial_idx + 1) % 50 == 0:
            print(f"Completed {trial_idx + 1}/{num_trials} trials")
    
    # Analyze results
    successful_trials = [r for r in results if r.get("success", False)]
    total_attempts = len([r for r in results if r.get("success") is not None])
    
    accuracy = len(successful_trials) / total_attempts if total_attempts > 0 else 0
    
    # Per-profile analysis
    profile_stats = {}
    for profile in profiles:
        profile_trials = [r for r in results if r["profile"] == profile]
        profile_successes = [r for r in profile_trials if r.get("success", False)]
        
        if profile_trials:
            profile_stats[profile] = {
                "total_trials": len(profile_trials),
                "successful_trials": len(profile_successes),
                "accuracy": len(profile_successes) / len(profile_trials),
                "avg_score": np.mean([r.get("score", 0) for r in profile_successes]) if profile_successes else 0,
            }
    
    summary = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "vocal_types": vocal_types,
            "profiles_tested": profiles,
            "num_trials": num_trials,
            "snr_range_db": snr_db_range,
            "max_extra_paths": max_extra_paths,
            "max_delay_ns": max_delay_ns,
            "rng_seed": rng_seed,
        },
        "overall_results": {
            "total_trials": len(results),
            "successful_trials": len(successful_trials),
            "overall_accuracy": accuracy,
            "profiles_tested": len(profiles),
        },
        "profile_results": profile_stats,
        "detailed_results": results,
    }
    
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--vocal-types",
        nargs="+",
        choices=["diphthongs", "ornaments", "consonant-vowels", "italian-vowels"],
        default=["diphthongs", "ornaments", "consonant-vowels", "italian-vowels"],
        help="Types of extended vocal techniques to test",
    )
    parser.add_argument(
        "--num-trials", type=int, default=200, help="Number of trials to run"
    )
    parser.add_argument(
        "--snr-db",
        type=float,
        nargs=2,
        default=[20.0, 35.0],
        help="SNR range in dB (min max)",
    )
    parser.add_argument(
        "--max-extra-paths", type=int, default=4, help="Maximum multipath reflections"
    )
    parser.add_argument(
        "--max-delay-ns", type=float, default=120.0, help="Maximum multipath delay (ns)"
    )
    parser.add_argument(
        "--rng-seed", type=int, default=2025, help="Random number generator seed"
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="Output JSON file path"
    )
    
    args = parser.parse_args()
    
    print("🎵 Extended Vocal Beacon Simulation - Phase 1 Musical-RF Architecture")
    print("=" * 70)
    print(f"Building on Italian vowel success: I detection 0% → 57.9%")
    print(f"Testing vocal types: {args.vocal_types}")
    print(f"Trials: {args.num_trials}, SNR: {args.snr_db[0]}-{args.snr_db[1]} dB")
    print()
    
    try:
        results = run_extended_vocal_experiment(
            vocal_types=args.vocal_types,
            num_trials=args.num_trials,
            snr_db_range=tuple(args.snr_db),
            max_extra_paths=args.max_extra_paths,
            max_delay_ns=args.max_delay_ns,
            rng_seed=args.rng_seed,
        )
        
        # Save results
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w") as f:
            json.dump(results, f, indent=2)
        
        print(f"Results saved to: {args.output}")
        print()
        print("🎪 EXTENDED VOCAL BEACON RESULTS")
        print("=" * 40)
        
        overall = results["overall_results"]
        print(f"Total profiles tested: {overall['profiles_tested']}")
        print(f"Overall accuracy: {overall['overall_accuracy']:.1%}")
        print(f"Successful trials: {overall['successful_trials']}/{overall['total_trials']}")
        print()
        
        print("Per-Profile Performance:")
        for profile, stats in results["profile_results"].items():
            print(f"  {profile:12s}: {stats['accuracy']:.1%} accuracy ({stats['successful_trials']}/{stats['total_trials']} trials)")
        
        print()
        print("🚀 Next: Implement string section for Phase 2!")
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())