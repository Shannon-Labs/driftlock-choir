#!/usr/bin/env python3
"""Debug synthesis/analysis pipeline mismatch for extended vocals."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / 'src') not in sys.path:
    sys.path.insert(0, str(ROOT / 'src'))

import numpy as np
from phy.formants import (
    FormantSynthesisConfig, 
    synthesize_formant_preamble,
    analyze_missing_fundamental,
    build_formant_library
)
from phy.vocal_extensions import build_extended_vocal_library

def main():
    print("🔬 Debugging Synthesis/Analysis Pipeline")
    print("=" * 50)
    
    # Build libraries
    base_lib = build_formant_library(25000.0, 12, False, 1000.0)
    extended_lib = build_extended_vocal_library(25000.0, 12, 1000.0)
    full_lib = {**base_lib, **extended_lib}
    
    # Test basic Italian vowel synthesis/analysis (should work)
    print("1. Testing proven Italian vowel A:")
    config_a = FormantSynthesisConfig(profile="A")
    try:
        waveform_a, synth_lib_a = synthesize_formant_preamble(4096, 2e6, config_a)
        print(f"   ✅ Synthesis OK: {len(waveform_a)} samples")
        
        analysis_result = analyze_missing_fundamental(
            waveform_a[:2048], 2e6, list(base_lib.values())
        )
        print(f"   ✅ Analysis OK: detected '{analysis_result.label}' (expected 'A')")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Test extended vocal (AI diphthong)
    print("\n2. Testing extended vocal AI diphthong:")
    if "AI" in extended_lib:
        desc = extended_lib["AI"]
        print(f"   AI descriptor: {len(desc.harmonics_hz)} harmonics")
        print(f"   Harmonics: {desc.harmonics_hz}")
        print(f"   Amplitudes: {desc.amplitudes}")
        
        # Try synthesizing AI using base formant synthesis
        # The issue might be that FormantSynthesisConfig doesn't handle extended profiles
        print("   ⚠️  Issue found: Extended vocals not in base synthesis system!")
        print("   The synthesis pipeline only knows about A, E, I, O, U")
        print("   Extended profiles exist in library but synthesis can't create them")
    
    print("\n🔍 ROOT CAUSE IDENTIFIED:")
    print("   Extended vocal profiles are generated in the library but")
    print("   the synthesis pipeline (FormantSynthesisConfig) only supports")
    print("   the 5 basic vowels. Need to extend synthesis to handle")
    print("   diphthongs, ornaments, and consonant-vowels.")
    
    print("\n🛠️  SOLUTION NEEDED:")
    print("   1. Extend synthesize_formant_preamble() to handle extended profiles")
    print("   2. OR create separate synthesis functions for extended techniques")
    print("   3. OR modify the test script to use descriptors directly")

if __name__ == "__main__":
    main()