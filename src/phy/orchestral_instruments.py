"""Orchestral instrument models for Musical-RF Architecture.

This module implements Phase 2+ of the Grand Musical-RF Architecture:
string, woodwind, brass, and percussion sections with their characteristic
spectral signatures optimized for RF beacon applications.

Each instrument family provides unique RF benefits:
- Strings: Rich harmonic content for robust missing-fundamental detection
- Woodwinds: Spectral purity and orthogonal signal spaces  
- Brass: High-power transmission and multipath penetration
- Percussion: Timing synchronization and transient markers
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np

from .formants import FormantDescriptor


class InstrumentFamily(Enum):
    """Musical instrument families with distinct acoustic characteristics."""
    STRINGS = "strings"
    WOODWINDS = "woodwinds" 
    BRASS = "brass"
    PERCUSSION = "percussion"


@dataclass(frozen=True)
class MusicalNote:
    """Represents a musical note with standard frequency relationships."""
    
    name: str  # e.g., "C4", "A#5", "Bb3"
    frequency_hz: float
    octave: int
    
    @classmethod
    def from_midi(cls, midi_note: int) -> "MusicalNote":
        """Create MusicalNote from MIDI note number (A4 = 69, 440 Hz)."""
        # MIDI note 69 is A4 at 440 Hz
        frequency = 440.0 * (2 ** ((midi_note - 69) / 12))
        
        octave = (midi_note // 12) - 1
        note_names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
        note_name = note_names[midi_note % 12]
        
        name = f"{note_name}{octave}"
        
        return cls(name=name, frequency_hz=frequency, octave=octave)
    
    @classmethod 
    def from_name(cls, name: str) -> "MusicalNote":
        """Create MusicalNote from name like 'C4', 'F#5', 'Bb3'."""
        # Parse note name and octave
        name = name.strip().upper()
        
        # Handle flats (b) and sharps (#)
        if 'B' in name and len(name) > 2 and name[1] == 'B':
            # Bb, Db, etc. - flat notation
            note_base = name[0]
            octave = int(name[2])
            semitone_offset = -1  # Flat
        elif '#' in name:
            # C#, F#, etc. - sharp notation  
            note_base = name[0]
            octave = int(name[2])
            semitone_offset = 1  # Sharp
        else:
            # Natural note
            note_base = name[0]
            octave = int(name[1])
            semitone_offset = 0
        
        # Convert to MIDI note number
        note_values = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
        
        if note_base not in note_values:
            raise ValueError(f"Invalid note base: {note_base}")
        
        midi_note = (octave + 1) * 12 + note_values[note_base] + semitone_offset
        
        return cls.from_midi(midi_note)


@dataclass(frozen=True)
class InstrumentConfig:
    """Configuration for a specific orchestral instrument."""
    
    name: str
    family: InstrumentFamily
    fundamental_note: MusicalNote
    harmonic_series: List[float]  # Relative amplitudes of harmonics 1, 2, 3, ...
    spectral_envelope: str = "natural"  # "natural", "bright", "warm", "mellow"
    dynamic_range_db: float = 40.0  # pp to ff dynamic range


# Predefined instrument configurations based on acoustic research
ORCHESTRAL_INSTRUMENTS = {
    
    # === STRING SECTION ===
    # Rich harmonic content, excellent for missing-fundamental detection
    
    "violin_e5": InstrumentConfig(
        name="Violin E5",
        family=InstrumentFamily.STRINGS,
        fundamental_note=MusicalNote.from_name("E5"),  # 659 Hz
        harmonic_series=[1.0, 0.8, 0.6, 0.45, 0.3, 0.2, 0.15, 0.1],  # Rich harmonics
        spectral_envelope="bright",
        dynamic_range_db=45.0,
    ),
    
    "violin_a4": InstrumentConfig(
        name="Violin A4", 
        family=InstrumentFamily.STRINGS,
        fundamental_note=MusicalNote.from_name("A4"),  # 440 Hz
        harmonic_series=[1.0, 0.7, 0.5, 0.4, 0.25, 0.15, 0.1, 0.05],
        spectral_envelope="bright",
        dynamic_range_db=45.0,
    ),
    
    "viola_c4": InstrumentConfig(
        name="Viola C4",
        family=InstrumentFamily.STRINGS, 
        fundamental_note=MusicalNote.from_name("C4"),  # 261 Hz
        harmonic_series=[1.0, 0.6, 0.4, 0.3, 0.2, 0.1, 0.05, 0.03],  # Warmer profile
        spectral_envelope="warm",
        dynamic_range_db=42.0,
    ),
    
    "cello_c3": InstrumentConfig(
        name="Cello C3",
        family=InstrumentFamily.STRINGS,
        fundamental_note=MusicalNote.from_name("C3"),  # 131 Hz  
        harmonic_series=[1.0, 0.8, 0.7, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05],  # Rich bass
        spectral_envelope="warm",
        dynamic_range_db=50.0,
    ),
    
    "double_bass_e1": InstrumentConfig(
        name="Double Bass E1", 
        family=InstrumentFamily.STRINGS,
        fundamental_note=MusicalNote.from_name("E1"),  # 41 Hz
        harmonic_series=[1.0, 0.9, 0.8, 0.6, 0.5, 0.3, 0.2, 0.1],  # Strong fundamentals
        spectral_envelope="warm",
        dynamic_range_db=48.0,
    ),
    
    # === WOODWIND SECTION ===
    # Diverse spectral signatures, orthogonal signal spaces
    
    "flute_c5": InstrumentConfig(
        name="Flute C5",
        family=InstrumentFamily.WOODWINDS,
        fundamental_note=MusicalNote.from_name("C5"),  # 523 Hz
        harmonic_series=[1.0, 0.1, 0.05, 0.02],  # Nearly pure tone
        spectral_envelope="bright",
        dynamic_range_db=35.0,
    ),
    
    "clarinet_c4": InstrumentConfig( 
        name="Clarinet C4",
        family=InstrumentFamily.WOODWINDS,
        fundamental_note=MusicalNote.from_name("C4"),  # 261 Hz
        harmonic_series=[1.0, 0.0, 0.7, 0.0, 0.4, 0.0, 0.2, 0.0],  # Odd harmonics only!
        spectral_envelope="natural",
        dynamic_range_db=40.0,
    ),
    
    "oboe_a4": InstrumentConfig(
        name="Oboe A4",
        family=InstrumentFamily.WOODWINDS, 
        fundamental_note=MusicalNote.from_name("A4"),  # 440 Hz
        harmonic_series=[1.0, 0.9, 0.8, 0.6, 0.5, 0.3, 0.2, 0.1],  # Rich, nasal
        spectral_envelope="bright",
        dynamic_range_db=38.0,
    ),
    
    "bassoon_c3": InstrumentConfig(
        name="Bassoon C3",
        family=InstrumentFamily.WOODWINDS,
        fundamental_note=MusicalNote.from_name("C3"),  # 131 Hz
        harmonic_series=[1.0, 0.8, 0.7, 0.5, 0.4, 0.2, 0.1, 0.05],  # Rich low end
        spectral_envelope="warm", 
        dynamic_range_db=42.0,
    ),
    
    # === BRASS SECTION ===
    # High-power transmission, multipath penetration
    
    "trumpet_c5": InstrumentConfig(
        name="Trumpet C5",
        family=InstrumentFamily.BRASS,
        fundamental_note=MusicalNote.from_name("C5"),  # 523 Hz
        harmonic_series=[1.0, 0.8, 0.9, 0.7, 0.6, 0.5, 0.3, 0.2],  # Brilliant, cutting
        spectral_envelope="bright",
        dynamic_range_db=50.0,
    ),
    
    "french_horn_f3": InstrumentConfig( 
        name="French Horn F3",
        family=InstrumentFamily.BRASS,
        fundamental_note=MusicalNote.from_name("F3"),  # 175 Hz
        harmonic_series=[1.0, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05],  # Mellow, blended
        spectral_envelope="mellow",
        dynamic_range_db=45.0,
    ),
    
    "trombone_c3": InstrumentConfig(
        name="Trombone C3", 
        family=InstrumentFamily.BRASS,
        fundamental_note=MusicalNote.from_name("C3"),  # 131 Hz
        harmonic_series=[1.0, 0.8, 0.7, 0.6, 0.4, 0.3, 0.2, 0.1],  # Powerful, smooth
        spectral_envelope="natural",
        dynamic_range_db=48.0,
    ),
    
    "tuba_c2": InstrumentConfig(
        name="Tuba C2",
        family=InstrumentFamily.BRASS, 
        fundamental_note=MusicalNote.from_name("C2"),  # 65 Hz
        harmonic_series=[1.0, 0.9, 0.8, 0.7, 0.5, 0.3, 0.2, 0.1],  # Fundamental power
        spectral_envelope="warm",
        dynamic_range_db=52.0,
    ),
}


def create_instrument_descriptor(
    config: InstrumentConfig,
    rf_fundamental_hz: float = 25000.0,
    formant_scale: float = 1000.0,
) -> FormantDescriptor:
    """Create a FormantDescriptor for an orchestral instrument."""
    
    # Map musical frequency to RF carrier structure
    musical_freq = config.fundamental_note.frequency_hz
    
    # Generate harmonic frequencies and amplitudes
    harmonics_hz = []
    amplitudes = []
    
    for harmonic_idx, amplitude in enumerate(config.harmonic_series, 1):
        if amplitude > 0.01:  # Include harmonics above threshold
            # Map musical harmonics to RF spectral structure
            harmonic_freq = rf_fundamental_hz * harmonic_idx
            
            # Apply spectral envelope shaping
            if config.spectral_envelope == "bright":
                envelope_factor = 1.0 + 0.2 * (harmonic_idx - 1)  # Emphasize upper harmonics
            elif config.spectral_envelope == "warm":
                envelope_factor = 1.0 / (1.0 + 0.1 * (harmonic_idx - 1))  # De-emphasize upper
            elif config.spectral_envelope == "mellow":
                envelope_factor = 1.0 / (1.0 + 0.15 * (harmonic_idx - 1))  # Even warmer
            else:  # natural
                envelope_factor = 1.0
            
            final_amplitude = amplitude * envelope_factor
            
            harmonics_hz.append(harmonic_freq)
            amplitudes.append(final_amplitude)
    
    # Normalize amplitudes
    if amplitudes:
        max_amp = max(amplitudes)
        amplitudes = [a / max_amp for a in amplitudes]
    else:
        # Fallback: just fundamental
        harmonics_hz = [rf_fundamental_hz]
        amplitudes = [1.0]
    
    return FormantDescriptor(
        label=config.name.replace(" ", "_").upper(),
        fundamental_hz=float(rf_fundamental_hz),
        harmonics_hz=tuple(float(f) for f in harmonics_hz),
        amplitudes=tuple(float(a) for a in amplitudes),
        include_fundamental=False,
    )


def build_orchestral_library(
    rf_fundamental_hz: float = 25000.0,
    formant_scale: float = 1000.0,
    families: Optional[List[InstrumentFamily]] = None,
) -> Dict[str, FormantDescriptor]:
    """Build a library of orchestral instrument descriptors."""
    
    if families is None:
        families = list(InstrumentFamily)
    
    library = {}
    
    for instrument_key, config in ORCHESTRAL_INSTRUMENTS.items():
        if config.family in families:
            descriptor = create_instrument_descriptor(
                config, rf_fundamental_hz, formant_scale
            )
            library[descriptor.label] = descriptor
    
    return library


def get_instrument_families_info() -> Dict[str, Dict[str, object]]:
    """Get information about each instrument family's RF benefits."""
    
    return {
        "strings": {
            "rf_benefits": [
                "Rich harmonic content for robust missing-fundamental detection",
                "Wide frequency range for spectral diversity",
                "Excellent multipath resilience (concert hall optimized)", 
                "Dynamic range control through bowing techniques",
            ],
            "coordination_role": "Harmonic foundation and melodic agility",
            "instruments": [k for k, v in ORCHESTRAL_INSTRUMENTS.items() if v.family == InstrumentFamily.STRINGS],
        },
        "woodwinds": {
            "rf_benefits": [
                "Diverse spectral signatures (pure tones to rich harmonics)",
                "Orthogonal signal spaces (odd vs even harmonics)",
                "Agile frequency modulation capabilities", 
                "Clear fundamental frequencies for precise coordination",
            ],
            "coordination_role": "Spectral purity and melodic agility",
            "instruments": [k for k, v in ORCHESTRAL_INSTRUMENTS.items() if v.family == InstrumentFamily.WOODWINDS],
        },
        "brass": {
            "rf_benefits": [
                "High-power transmission capabilities",
                "Excellent multipath penetration",
                "Brilliant spectral signatures that cut through interference",
                "Wide dynamic range for adaptive power control",
            ],
            "coordination_role": "Power projection and signal penetration", 
            "instruments": [k for k, v in ORCHESTRAL_INSTRUMENTS.items() if v.family == InstrumentFamily.BRASS],
        },
    }


# Musical ensemble combinations for advanced RF coordination
ORCHESTRAL_ENSEMBLES = {
    "string_quartet": ["violin_a4", "violin_e5", "viola_c4", "cello_c3"],
    "woodwind_quintet": ["flute_c5", "oboe_a4", "clarinet_c4", "bassoon_c3"],
    "brass_quartet": ["trumpet_c5", "french_horn_f3", "trombone_c3", "tuba_c2"],
    "chamber_orchestra": [
        "violin_e5", "violin_a4", "viola_c4", "cello_c3",  # Strings
        "flute_c5", "oboe_a4", "clarinet_c4",             # Woodwinds
        "french_horn_f3",                                  # Brass
    ],
    "symphony_orchestra": list(ORCHESTRAL_INSTRUMENTS.keys()),  # Full orchestra!
}


def create_ensemble_coordination(
    ensemble_name: str,
    library: Dict[str, FormantDescriptor],
) -> Dict[str, object]:
    """Create coordination configuration for a musical ensemble."""
    
    if ensemble_name not in ORCHESTRAL_ENSEMBLES:
        raise ValueError(f"Unknown ensemble: {ensemble_name}")
    
    instruments = ORCHESTRAL_ENSEMBLES[ensemble_name]
    
    # Create ensemble descriptor
    ensemble_config = {
        "name": ensemble_name,
        "instruments": instruments,
        "coordination_type": "simultaneous" if len(instruments) <= 4 else "sectional",
        "rf_benefits": {
            "signal_diversity": len(instruments),
            "spectral_coverage": "wide" if len(instruments) > 6 else "focused",
            "redundancy": "high" if len(instruments) > 8 else "moderate",
        },
    }
    
    # Add family distribution
    families = {}
    for inst_key in instruments:
        if inst_key in ORCHESTRAL_INSTRUMENTS:
            family = ORCHESTRAL_INSTRUMENTS[inst_key].family.value
            families[family] = families.get(family, 0) + 1
    
    ensemble_config["family_distribution"] = families
    
    return ensemble_config