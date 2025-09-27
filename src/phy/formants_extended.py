"""Extended Musical-RF formant library with diphthongs, tones, and vocal ornaments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence, Tuple, Union
import numpy as np
from numpy.typing import NDArray

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from formants import VOWEL_FORMANT_TABLE, FormantDescriptor, FormantAnalysisResult


# Chinese Mandarin vowel formants (Hz) - Research-based values
CHINESE_VOWEL_FORMANTS: Mapping[str, Tuple[float, float, float]] = {
    # Standard Mandarin vowels with acoustic optimization for RF
    "a_zh": (730.0, 1090.0, 2440.0),    # Chinese /a/ - central vowel
    "o_zh": (520.0, 920.0, 2560.0),     # Chinese /o/ - back rounded
    "e_zh": (460.0, 2280.0, 2850.0),    # Chinese /e/ - mid front
    "i_zh": (270.0, 2290.0, 3010.0),    # Chinese /i/ - close front
    "u_zh": (330.0, 820.0, 2200.0),     # Chinese /u/ - close back
    "ü_zh": (310.0, 1770.0, 2300.0),    # Chinese /ü/ - rounded front (unique!)
}

# Diphthong formant transitions - Start and end formants with transition parameters
DIPHTHONG_FORMANTS: Mapping[str, Dict[str, Union[Tuple[float, float, float], float]]] = {
    # Italian-style diphthongs
    "ai": {
        "start": (650.0, 1080.0, 2650.0),    # A formants
        "end": (340.0, 1870.0, 2800.0),      # I formants
        "transition_time": 0.3,              # 30% of signal duration
    },
    "au": {
        "start": (650.0, 1080.0, 2650.0),    # A formants
        "end": (350.0, 600.0, 2700.0),       # U formants
        "transition_time": 0.4,
    },
    "ei": {
        "start": (400.0, 1700.0, 2600.0),    # E formants
        "end": (340.0, 1870.0, 2800.0),      # I formants
        "transition_time": 0.35,
    },
    "ou": {
        "start": (400.0, 800.0, 2600.0),     # O formants
        "end": (350.0, 600.0, 2700.0),       # U formants
        "transition_time": 0.45,
    },
    # Chinese-style combinations
    "ao_zh": {
        "start": (730.0, 1090.0, 2440.0),    # Chinese a
        "end": (520.0, 920.0, 2560.0),       # Chinese o
        "transition_time": 0.35,
    },
    "ei_zh": {
        "start": (460.0, 2280.0, 2850.0),    # Chinese e
        "end": (270.0, 2290.0, 3010.0),      # Chinese i
        "transition_time": 0.3,
    },
}

# Chinese tone contours - Pitch modulation patterns for RF
MANDARIN_TONE_CONTOURS: Mapping[str, Dict[str, Union[str, List[float]]]] = {
    "tone1": {
        "name": "High Level (mā)",
        "pattern": "level",
        "pitch_contour": [1.0, 1.0, 1.0, 1.0, 1.0],  # Constant high pitch
        "f0_multiplier": 1.2,  # 20% higher fundamental
    },
    "tone2": {
        "name": "Rising (má)",
        "pattern": "rising",
        "pitch_contour": [0.7, 0.8, 0.9, 1.0, 1.1],  # Rising pitch
        "f0_multiplier": 1.0,
    },
    "tone3": {
        "name": "Falling-Rising (mǎ)",
        "pattern": "dipping",
        "pitch_contour": [0.9, 0.7, 0.6, 0.8, 1.0],  # V-shaped
        "f0_multiplier": 0.9,
    },
    "tone4": {
        "name": "Falling (mà)",
        "pattern": "falling",
        "pitch_contour": [1.1, 1.0, 0.9, 0.8, 0.7],  # Falling pitch
        "f0_multiplier": 1.0,
    },
}

# Vocal ornaments - Temporal modulations for RF signals
VOCAL_ORNAMENTS: Mapping[str, Dict[str, Union[str, float, List[float]]]] = {
    "trill": {
        "type": "frequency_modulation",
        "rate_hz": 5.0,          # 5 Hz trill rate
        "depth": 0.05,           # 5% frequency deviation
        "pattern": "sinusoidal",
    },
    "vibrato": {
        "type": "frequency_modulation", 
        "rate_hz": 6.5,          # 6.5 Hz vibrato rate
        "depth": 0.03,           # 3% frequency deviation
        "pattern": "sinusoidal",
    },
    "mordent": {
        "type": "frequency_jump",
        "duration": 0.1,         # 10% of signal duration
        "jump_ratio": 1.12,      # Minor second interval (12% up)
        "pattern": "main-upper-main",
    },
    "turn": {
        "type": "frequency_sequence",
        "duration": 0.15,        # 15% of signal duration
        "sequence": [1.0, 1.12, 1.0, 0.89, 1.0],  # Upper neighbor, main, lower neighbor, main
        "pattern": "melodic_ornament",
    },
}

# Consonant spectral signatures for timing markers
CONSONANT_SIGNATURES: Mapping[str, Dict[str, Union[str, float, Tuple[float, ...]]]] = {
    # Plosives - Transient burst signatures
    "p": {
        "type": "plosive",
        "burst_duration": 0.02,   # 20ms burst
        "burst_frequencies": (1500.0, 3000.0, 4500.0),  # Burst spectrum
        "silence_duration": 0.01, # 10ms silence before burst
    },
    "t": {
        "type": "plosive", 
        "burst_duration": 0.015,
        "burst_frequencies": (2000.0, 4000.0, 6000.0),  # Higher frequency burst
        "silence_duration": 0.01,
    },
    "k": {
        "type": "plosive",
        "burst_duration": 0.025,
        "burst_frequencies": (1000.0, 2500.0, 3500.0),  # Lower frequency burst
        "silence_duration": 0.015,
    },
    # Fricatives - Continuous noise signatures
    "f": {
        "type": "fricative",
        "noise_duration": 0.08,   # 80ms fricative
        "noise_frequencies": (1200.0, 2400.0, 4800.0),  # Fricative spectrum
        "amplitude_envelope": "exponential_decay",
    },
    "s": {
        "type": "fricative",
        "noise_duration": 0.1,
        "noise_frequencies": (4000.0, 6000.0, 8000.0),  # High frequency fricative
        "amplitude_envelope": "steady",
    },
    "sh": {
        "type": "fricative",
        "noise_duration": 0.09,
        "noise_frequencies": (2000.0, 4000.0, 6000.0),  # Mid-high fricative
        "amplitude_envelope": "steady",
    },
}


@dataclass(frozen=True)
class ExtendedFormantDescriptor:
    """Extended formant descriptor with support for temporal modulation."""
    
    label: str
    base_formants: Tuple[float, float, float]
    formant_scale: float = 90_000.0
    
    # Temporal modulation parameters
    is_diphthong: bool = False
    diphthong_params: Optional[Dict] = None
    
    # Mandarin tone parameters
    has_tone: bool = False
    tone_contour: Optional[List[float]] = None
    f0_multiplier: float = 1.0
    
    # Vocal ornament parameters
    has_ornament: bool = False
    ornament_type: Optional[str] = None
    ornament_params: Optional[Dict] = None
    
    # Consonant timing parameters
    has_consonant: bool = False
    consonant_signature: Optional[Dict] = None


def create_extended_formant_library() -> Dict[str, ExtendedFormantDescriptor]:
    """Create comprehensive Musical-RF formant library with all extensions."""
    
    library: Dict[str, ExtendedFormantDescriptor] = {}
    
    # Add base Italian vowels
    for vowel, formants in VOWEL_FORMANT_TABLE.items():
        library[vowel] = ExtendedFormantDescriptor(
            label=vowel,
            base_formants=formants,
        )
    
    # Add Chinese vowels
    for vowel, formants in CHINESE_VOWEL_FORMANTS.items():
        library[vowel] = ExtendedFormantDescriptor(
            label=vowel,
            base_formants=formants,
        )
    
    # Add diphthongs
    for diph, params in DIPHTHONG_FORMANTS.items():
        library[diph] = ExtendedFormantDescriptor(
            label=diph,
            base_formants=params["start"],
            is_diphthong=True,
            diphthong_params=params,
        )
    
    # Add toned vowels (Chinese vowels with Mandarin tones)
    for vowel, formants in CHINESE_VOWEL_FORMANTS.items():
        for tone, tone_params in MANDARIN_TONE_CONTOURS.items():
            toned_label = f"{vowel}_{tone}"
            library[toned_label] = ExtendedFormantDescriptor(
                label=toned_label,
                base_formants=formants,
                has_tone=True,
                tone_contour=tone_params["pitch_contour"],
                f0_multiplier=tone_params["f0_multiplier"],
            )
    
    # Add vowels with ornaments
    for vowel, formants in VOWEL_FORMANT_TABLE.items():
        for ornament, ornament_params in VOCAL_ORNAMENTS.items():
            ornamented_label = f"{vowel}_{ornament}"
            library[ornamented_label] = ExtendedFormantDescriptor(
                label=ornamented_label,
                base_formants=formants,
                has_ornament=True,
                ornament_type=ornament,
                ornament_params=ornament_params,
            )
    
    return library


def get_library_statistics() -> Dict[str, int]:
    """Get statistics about the extended formant library."""
    
    library = create_extended_formant_library()
    
    stats = {
        "total_signatures": len(library),
        "base_vowels": len(VOWEL_FORMANT_TABLE),
        "chinese_vowels": len(CHINESE_VOWEL_FORMANTS),
        "diphthongs": len(DIPHTHONG_FORMANTS),
        "toned_vowels": len(CHINESE_VOWEL_FORMANTS) * len(MANDARIN_TONE_CONTOURS),
        "ornamented_vowels": len(VOWEL_FORMANT_TABLE) * len(VOCAL_ORNAMENTS),
        "consonant_signatures": len(CONSONANT_SIGNATURES),
    }
    
    # Calculate potential interference patterns (combinatorial scaling)
    n_signals = stats["total_signatures"]
    stats["interference_patterns"] = n_signals * (n_signals - 1) // 2
    
    return stats


def synthesize_extended_formant_waveform(
    descriptor: ExtendedFormantDescriptor,
    length: int,
    sample_rate: float,
    fundamental_hz: float = 50_000_000.0,
) -> NDArray[np.complex128]:
    """Synthesize complex RF waveform with extended formant features."""
    
    time_axis = np.arange(length, dtype=float) / sample_rate
    waveform = np.zeros(length, dtype=np.complex128)
    
    # Scale formants to RF frequencies
    f1, f2, f3 = descriptor.base_formants
    f1_rf = f1 * descriptor.formant_scale
    f2_rf = f2 * descriptor.formant_scale  
    f3_rf = f3 * descriptor.formant_scale
    
    # Apply fundamental frequency with tone modulation if present
    current_f0 = fundamental_hz * descriptor.f0_multiplier
    
    if descriptor.has_tone and descriptor.tone_contour:
        # Apply Mandarin tone contour
        tone_modulation = np.interp(
            np.linspace(0, 1, length),
            np.linspace(0, 1, len(descriptor.tone_contour)),
            descriptor.tone_contour
        )
        f0_modulated = current_f0 * tone_modulation
    else:
        f0_modulated = np.full(length, current_f0)
    
    # Generate base formant structure
    if descriptor.is_diphthong and descriptor.diphthong_params:
        # Handle diphthong transition
        start_formants = descriptor.diphthong_params["start"]
        end_formants = descriptor.diphthong_params["end"]  
        transition_time = descriptor.diphthong_params["transition_time"]
        
        transition_point = int(length * transition_time)
        
        # Interpolate formant frequencies over time
        f1_dip = np.concatenate([
            np.linspace(start_formants[0] * descriptor.formant_scale, 
                       end_formants[0] * descriptor.formant_scale, transition_point),
            np.full(length - transition_point, end_formants[0] * descriptor.formant_scale)
        ])
        f2_dip = np.concatenate([
            np.linspace(start_formants[1] * descriptor.formant_scale,
                       end_formants[1] * descriptor.formant_scale, transition_point),
            np.full(length - transition_point, end_formants[1] * descriptor.formant_scale)
        ])
        
        formant_freqs = [f1_dip, f2_dip]
        amplitudes = [1.0, 0.8]
    else:
        # Static formants
        formant_freqs = [f1_rf, f2_rf, f3_rf]
        amplitudes = [1.0, 0.8, 0.6]
    
    # Synthesize formant components
    for i, (freq, amp) in enumerate(zip(formant_freqs, amplitudes)):
        if isinstance(freq, np.ndarray):
            # Time-varying frequency (diphthong)
            phase = np.cumsum(2 * np.pi * freq / sample_rate)
        else:
            # Static frequency
            phase = 2 * np.pi * freq * time_axis
        
        # Apply vocal ornaments
        if descriptor.has_ornament and descriptor.ornament_params:
            ornament = descriptor.ornament_params
            if ornament["type"] == "frequency_modulation":
                # Add vibrato/trill modulation
                mod_depth = ornament["depth"]
                mod_rate = ornament["rate_hz"]
                modulation = 1 + mod_depth * np.sin(2 * np.pi * mod_rate * time_axis)
                if isinstance(freq, np.ndarray):
                    phase = np.cumsum(2 * np.pi * freq * modulation / sample_rate)
                else:
                    phase = 2 * np.pi * freq * modulation * time_axis
        
        component = amp * np.exp(1j * phase)
        waveform += component
    
    # Normalize
    norm = np.linalg.norm(waveform)
    if norm > 0:
        waveform = waveform / norm
        
    return waveform.astype(np.complex128)


if __name__ == "__main__":
    # Demonstrate extended library capabilities
    library = create_extended_formant_library()
    stats = get_library_statistics()
    
    print("MUSICAL-RF EXTENDED FORMANT LIBRARY")
    print("=" * 50)
    print(f"Total RF Signatures: {stats['total_signatures']}")
    print(f"Base Italian Vowels: {stats['base_vowels']}")
    print(f"Chinese Vowels: {stats['chinese_vowels']}")
    print(f"Diphthongs: {stats['diphthongs']}")
    print(f"Toned Vowels (Mandarin): {stats['toned_vowels']}")
    print(f"Ornamented Vowels: {stats['ornamented_vowels']}")
    print(f"Consonant Signatures: {stats['consonant_signatures']}")
    print(f"Potential Interference Patterns: {stats['interference_patterns']:,}")
    print()
    print("Sample signatures:")
    for i, (label, descriptor) in enumerate(library.items()):
        if i < 10:  # Show first 10
            print(f"  {label}: {descriptor.base_formants}")
        elif i == 10:
            print(f"  ... and {len(library) - 10} more signatures")
            break