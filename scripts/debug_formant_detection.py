#!/usr/bin/env python3
"""Debug script to understand formant detection issues."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from phy.formants import (
    FormantSynthesisConfig,
    synthesize_formant_preamble,
    analyze_missing_fundamental,
    build_formant_library,
    VOWEL_FORMANT_TABLE,
)

def debug_formant_detection():
    """Debug the formant detection for each vowel profile."""
    
    profiles = ["A", "E", "I", "O", "U"]
    sample_rate = 2_000_000.0
    
    print("FORMANT DETECTION DEBUG")
    print("=" * 50)
    
    for profile in profiles:
        print(f"\nTesting profile: {profile}")
        
        # Generate waveform
        synth_config = FormantSynthesisConfig(
            profile=profile,
            fundamental_hz=25000.0,
            harmonic_count=12,
            include_fundamental=False,
            formant_scale=1000.0,
            phase_jitter=0.0,
        )
        
        waveform, library = synthesize_formant_preamble(
            length=4096,
            sample_rate=sample_rate,
            config=synth_config,
        )
        
        # Analyze the waveform
        descriptors = list(library.values())
        result = analyze_missing_fundamental(
            waveform, sample_rate, descriptors, top_peaks=8
        )
        
        if result:
            print(f"  Detected: {result.label}")
            print(f"  Dominant Hz: {result.dominant_hz:.1f}")
            print(f"  Missing Fundamental: {result.missing_fundamental_hz:.1f}")
            print(f"  Score: {result.score:.3f}")
            print(f"  Correct: {result.label == profile}")
        else:
            print("  No result detected")
        
        # Check formant frequencies
        formants = VOWEL_FORMANT_TABLE[profile]
        print(f"  Expected Formants: {formants} Hz")

if __name__ == "__main__":
    debug_formant_detection()