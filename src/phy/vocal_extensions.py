"""Extended vocal techniques for spectrum beacon diversity.

This module implements Phase 1 of the Grand Musical-RF Architecture:
extending the proven Italian vowel foundation with advanced vocal techniques
including diphthongs, ornaments, and consonant-vowel combinations.

Based on the breakthrough Italian vowel optimization that improved vowel "I" 
detection from 0% to 57.9%, we now expand to dynamic spectral signatures.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from .formants import VOWEL_FORMANT_TABLE, FormantDescriptor, FormantSynthesisConfig


@dataclass(frozen=True)
class DiphthongConfig:
    """Configuration for vowel-to-vowel gliding synthesis."""
    
    start_vowel: str
    end_vowel: str
    transition_duration_ms: float = 50.0
    glide_profile: str = "linear"  # "linear", "exponential", "s_curve"


@dataclass(frozen=True)
class VocalOrnamentConfig:
    """Configuration for classical vocal ornaments (trills, vibrato, etc.)."""
    
    base_vowel: str
    ornament_type: str  # "trill", "vibrato", "staccato", "legato"
    modulation_rate_hz: float = 8.0
    modulation_depth_cents: float = 25.0
    envelope_shape: str = "smooth"


@dataclass(frozen=True)
class ConsonantVowelConfig:
    """Configuration for consonant-vowel combinations."""
    
    consonant: str  # "M", "L", "N", "R", etc.
    vowel: str
    transition_duration_ms: float = 30.0
    consonant_emphasis: float = 0.3


# Consonant formant approximations for common sounds
CONSONANT_FORMANT_TABLE: Dict[str, Tuple[float, float, float]] = {
    # Nasal consonants - add nasal resonance around 250Hz and 2500Hz
    "M": (250.0, 1000.0, 2500.0),  # Bilabial nasal
    "N": (300.0, 1500.0, 2600.0),  # Alveolar nasal  
    "NG": (270.0, 2300.0, 2900.0), # Velar nasal
    
    # Liquid consonants - formant transitions
    "L": (400.0, 1200.0, 2400.0),  # Lateral liquid
    "R": (500.0, 1300.0, 1800.0),  # Rhotic liquid (American R)
    
    # Fricatives - noise-like but with formant coloring
    "S": (4000.0, 8000.0, 12000.0),  # High-frequency fricative
    "SH": (2500.0, 4000.0, 6000.0),  # Lower fricative
    "F": (1500.0, 4000.0, 8000.0),   # Labiodental fricative
}


def create_diphthong_descriptor(
    config: DiphthongConfig,
    fundamental_hz: float,
    harmonic_count: int,
    formant_scale: float,
) -> FormantDescriptor:
    """Create a FormantDescriptor for a diphthong (vowel-to-vowel glide)."""
    
    if config.start_vowel not in VOWEL_FORMANT_TABLE:
        raise ValueError(f"Unknown start vowel: {config.start_vowel}")
    if config.end_vowel not in VOWEL_FORMANT_TABLE:
        raise ValueError(f"Unknown end vowel: {config.end_vowel}")
    
    start_formants = VOWEL_FORMANT_TABLE[config.start_vowel]
    end_formants = VOWEL_FORMANT_TABLE[config.end_vowel]
    
    # For now, use the average of start and end formants
    # TODO: Implement time-varying formants for true gliding
    avg_formants = tuple(
        (start + end) / 2.0 * formant_scale 
        for start, end in zip(start_formants, end_formants)
    )
    
    # Generate harmonic series based on averaged formants - FIX: Add more harmonics
    partials: Dict[float, float] = {}
    
    for harmonic_idx in range(1, harmonic_count + 1):
        harmonic_freq = fundamental_hz * harmonic_idx
        
        # Calculate amplitude based on proximity to formant frequencies
        amplitude = 0.0
        for formant_freq in avg_formants:
            if formant_freq > 0:
                # Gaussian envelope around each formant - WIDER bandwidth for more harmonics
                freq_diff = abs(harmonic_freq - formant_freq)
                formant_bw = formant_freq * 0.2  # Increased from 0.1 to 0.2
                amplitude += np.exp(-(freq_diff / formant_bw) ** 2)
        
        # LOWERED threshold to include more harmonics
        if amplitude > 0.005:  # Was 0.01, now 0.005
            partials[harmonic_freq] = float(amplitude)
    
    # ENSURE we always have multiple harmonics
    if len(partials) < 3:
        # Add fundamental and a few harmonics if too sparse
        for i in range(1, 4):
            partials[fundamental_hz * i] = 1.0 / i
    
    # Sort by frequency and normalize
    harmonic_freqs, amplitudes = zip(*sorted(partials.items()))
    amplitudes_arr = np.asarray(amplitudes, dtype=float)
    if amplitudes_arr.max() > 0.0:
        amplitudes_arr = amplitudes_arr / amplitudes_arr.max()
    
    label = f"{config.start_vowel}{config.end_vowel}"
    
    return FormantDescriptor(
        label=label,
        fundamental_hz=float(fundamental_hz),
        harmonics_hz=tuple(float(f) for f in harmonic_freqs),
        amplitudes=tuple(float(a) for a in amplitudes_arr),
        include_fundamental=False,
    )


def create_vocal_ornament_descriptor(
    config: VocalOrnamentConfig,
    fundamental_hz: float,
    harmonic_count: int,
    formant_scale: float,
) -> FormantDescriptor:
    """Create a FormantDescriptor for vocal ornaments (trills, vibrato, etc.)."""
    
    if config.base_vowel not in VOWEL_FORMANT_TABLE:
        raise ValueError(f"Unknown base vowel: {config.base_vowel}")
    
    base_formants = VOWEL_FORMANT_TABLE[config.base_vowel]
    scaled_formants = tuple(f * formant_scale for f in base_formants)
    
    # Generate partials with ornament-specific modifications
    partials: Dict[float, float] = {}
    
    for harmonic_idx in range(1, harmonic_count + 1):
        harmonic_freq = fundamental_hz * harmonic_idx
        
        # Base amplitude from formant proximity
        amplitude = 0.0
        for formant_freq in scaled_formants:
            if formant_freq > 0:
                freq_diff = abs(harmonic_freq - formant_freq)
                formant_bw = formant_freq * 0.1
                amplitude += np.exp(-(freq_diff / formant_bw) ** 2)
        
        # Apply ornament-specific modifications
        if config.ornament_type == "trill":
            # Trills create amplitude modulation
            amplitude *= (1.0 + 0.3 * np.sin(2 * np.pi * config.modulation_rate_hz))
        elif config.ornament_type == "vibrato":
            # Vibrato creates frequency modulation (approximated as amplitude change)
            vibrato_cents = config.modulation_depth_cents
            vibrato_factor = 2 ** (vibrato_cents / 1200)  # Cents to frequency ratio
            amplitude *= (1.0 + 0.2 * (vibrato_factor - 1.0))
        elif config.ornament_type == "staccato":
            # Staccato creates sharp attack/decay
            amplitude *= 0.8  # Slightly reduced for percussive quality
        
        if amplitude > 0.01:
            partials[harmonic_freq] = float(amplitude)
    
    if not partials:
        partials[fundamental_hz] = 1.0
    
    # Sort and normalize
    harmonic_freqs, amplitudes = zip(*sorted(partials.items()))
    amplitudes_arr = np.asarray(amplitudes, dtype=float)
    if amplitudes_arr.max() > 0.0:
        amplitudes_arr = amplitudes_arr / amplitudes_arr.max()
    
    label = f"{config.ornament_type.upper()}_{config.base_vowel}"
    
    return FormantDescriptor(
        label=label,
        fundamental_hz=float(fundamental_hz),
        harmonics_hz=tuple(float(f) for f in harmonic_freqs),
        amplitudes=tuple(float(a) for a in amplitudes_arr),
        include_fundamental=False,
    )


def create_consonant_vowel_descriptor(
    config: ConsonantVowelConfig,
    fundamental_hz: float,
    harmonic_count: int,
    formant_scale: float,
) -> FormantDescriptor:
    """Create a FormantDescriptor for consonant-vowel combinations."""
    
    if config.consonant not in CONSONANT_FORMANT_TABLE:
        raise ValueError(f"Unknown consonant: {config.consonant}")
    if config.vowel not in VOWEL_FORMANT_TABLE:
        raise ValueError(f"Unknown vowel: {config.vowel}")
    
    consonant_formants = CONSONANT_FORMANT_TABLE[config.consonant]
    vowel_formants = VOWEL_FORMANT_TABLE[config.vowel]
    
    # Blend consonant and vowel formants based on emphasis
    c_weight = config.consonant_emphasis
    v_weight = 1.0 - c_weight
    
    blended_formants = tuple(
        (c_f * c_weight + v_f * v_weight) * formant_scale
        for c_f, v_f in zip(consonant_formants, vowel_formants)
    )
    
    # Generate harmonic series
    partials: Dict[float, float] = {}
    
    for harmonic_idx in range(1, harmonic_count + 1):
        harmonic_freq = fundamental_hz * harmonic_idx
        
        amplitude = 0.0
        for formant_freq in blended_formants:
            if formant_freq > 0:
                freq_diff = abs(harmonic_freq - formant_freq)
                formant_bw = formant_freq * 0.15  # Wider bandwidth for consonants
                amplitude += np.exp(-(freq_diff / formant_bw) ** 2)
        
        # Add some noise-like character for consonants
        if config.consonant in ["S", "SH", "F"] and harmonic_freq > 2000:
            amplitude *= 1.2  # Emphasize high frequencies for fricatives
        
        if amplitude > 0.01:
            partials[harmonic_freq] = float(amplitude)
    
    if not partials:
        partials[fundamental_hz] = 1.0
    
    # Sort and normalize
    harmonic_freqs, amplitudes = zip(*sorted(partials.items()))
    amplitudes_arr = np.asarray(amplitudes, dtype=float)
    if amplitudes_arr.max() > 0.0:
        amplitudes_arr = amplitudes_arr / amplitudes_arr.max()
    
    label = f"{config.consonant}{config.vowel}"
    
    return FormantDescriptor(
        label=label,
        fundamental_hz=float(fundamental_hz),
        harmonics_hz=tuple(float(f) for f in harmonic_freqs),
        amplitudes=tuple(float(a) for a in amplitudes_arr),
        include_fundamental=False,
    )


def build_extended_vocal_library(
    fundamental_hz: float,
    harmonic_count: int,
    formant_scale: float,
) -> Dict[str, FormantDescriptor]:
    """Build a library of extended vocal techniques beyond basic Italian vowels."""
    
    library: Dict[str, FormantDescriptor] = {}
    
    # Common diphthongs
    diphthongs = [
        DiphthongConfig("A", "I", 40.0),  # "AI" as in "eye"
        DiphthongConfig("A", "U", 50.0),  # "AU" as in "cow" 
        DiphthongConfig("E", "I", 30.0),  # "EI" rising glide
        DiphthongConfig("O", "U", 45.0),  # "OU" as in "go"
        DiphthongConfig("I", "A", 35.0),  # "IA" falling glide
    ]
    
    for config in diphthongs:
        descriptor = create_diphthong_descriptor(
            config, fundamental_hz, harmonic_count, formant_scale
        )
        library[descriptor.label] = descriptor
    
    # Vocal ornaments on each Italian vowel
    base_vowels = ["A", "E", "I", "O", "U"]
    ornament_types = ["trill", "vibrato", "staccato"]
    
    for vowel in base_vowels:
        for ornament in ornament_types:
            config = VocalOrnamentConfig(
                base_vowel=vowel,
                ornament_type=ornament,
                modulation_rate_hz=8.0 if ornament == "trill" else 6.0,
                modulation_depth_cents=25.0 if ornament == "vibrato" else 15.0,
            )
            descriptor = create_vocal_ornament_descriptor(
                config, fundamental_hz, harmonic_count, formant_scale
            )
            library[descriptor.label] = descriptor
    
    # Consonant-vowel combinations
    consonants = ["M", "N", "L", "R"]  # Start with common, well-defined consonants
    vowels = ["A", "E", "I", "O", "U"]
    
    for consonant in consonants:
        for vowel in vowels:
            config = ConsonantVowelConfig(
                consonant=consonant,
                vowel=vowel,
                transition_duration_ms=25.0,
                consonant_emphasis=0.25,  # Subtle consonant coloring
            )
            descriptor = create_consonant_vowel_descriptor(
                config, fundamental_hz, harmonic_count, formant_scale
            )
            library[descriptor.label] = descriptor
    
    return library


# Pre-defined common extended vocal profiles for easy access
COMMON_EXTENDED_VOCALS = {
    # Diphthongs - dynamic spectral sweeps
    "AI": DiphthongConfig("A", "I", 40.0),
    "AU": DiphthongConfig("A", "U", 50.0), 
    "EI": DiphthongConfig("E", "I", 30.0),
    "OU": DiphthongConfig("O", "U", 45.0),
    
    # Trills - rapid alternations for timing
    "TRILL_A": VocalOrnamentConfig("A", "trill", 8.0, 20.0),
    "TRILL_E": VocalOrnamentConfig("E", "trill", 8.0, 20.0),
    "TRILL_I": VocalOrnamentConfig("I", "trill", 8.0, 20.0),
    
    # Vibrato - frequency modulation for robustness
    "VIBRATO_A": VocalOrnamentConfig("A", "vibrato", 6.0, 25.0),
    "VIBRATO_I": VocalOrnamentConfig("I", "vibrato", 6.0, 25.0),
    
    # Consonant-vowel - speech-like transitions
    "MA": ConsonantVowelConfig("M", "A", 25.0, 0.3),
    "LA": ConsonantVowelConfig("L", "A", 20.0, 0.2),
    "NA": ConsonantVowelConfig("N", "A", 25.0, 0.3),
    "MI": ConsonantVowelConfig("M", "I", 25.0, 0.3),
    "LI": ConsonantVowelConfig("L", "I", 20.0, 0.2),
}