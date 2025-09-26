#!/usr/bin/env python3
"""Debug extended vocal synthesis and analysis pipeline."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / 'src') not in sys.path:
    sys.path.insert(0, str(ROOT / 'src'))

from phy.vocal_extensions import build_extended_vocal_library, COMMON_EXTENDED_VOCALS
from phy.formants import build_formant_library

def main():
    print("🔍 Debugging Extended Vocal Pipeline")
    print("=" * 40)
    
    # Test library building
    print("Building libraries...")
    base_lib = build_formant_library(25000.0, 12, False, 1000.0)
    extended_lib = build_extended_vocal_library(25000.0, 12, 1000.0)
    
    print(f"Base library: {len(base_lib)} vowels")
    print(f"Extended library: {len(extended_lib)} techniques")
    
    # Check a few extended entries
    print("\nExtended library entries:")
    for key, desc in list(extended_lib.items())[:5]:
        print(f"  {key}: {len(desc.harmonics_hz)} harmonics, dominant = {desc.dominant_hz:.0f} Hz")
    
    # The issue might be that extended vocals aren't being found in analysis
    # Let's check if the profiles match what we're testing
    print("\nCommon extended vocal configs:")
    for key in list(COMMON_EXTENDED_VOCALS.keys())[:5]:
        print(f"  {key}: {COMMON_EXTENDED_VOCALS[key]}")

if __name__ == "__main__":
    main()