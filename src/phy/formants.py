"""Formant-inspired spectral synthesis and analysis helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray


VOWEL_FORMANT_TABLE: Mapping[str, Tuple[float, float, float]] = {
    # Pure Italian vowel formants for sustained vowels (Hz).
    # Based on classical vocal pedagogy and Ingo Titze's work at University of Iowa
    # Optimized for acoustic distinctiveness in RF beacon applications
    "A": (700.0, 1220.0, 2600.0),   # /a/ - open central, pure
    "E": (450.0, 2100.0, 2900.0),   # /e/ - mid-front, distinct from /i/
    "I": (300.0, 2700.0, 3400.0),   # /i/ - close front, maximum forward/shrill
    "O": (500.0, 900.0, 2400.0),    # /o/ - close-mid back, pure rounded
    "U": (350.0, 750.0, 2200.0),    # /u/ - close back, maximum dark/rounded
}

DEFAULT_FUNDAMENTAL_HZ = 25_000.0
DEFAULT_FORMANT_SCALE = 1_000.0
DEFAULT_HARMONIC_COUNT = 12


@dataclass(frozen=True)
class FormantDescriptor:
    """Metadata describing the harmonic scaffold for a vowel profile."""

    label: str
    fundamental_hz: float
    harmonics_hz: Tuple[float, ...]
    amplitudes: Tuple[float, ...]
    include_fundamental: bool = False

    @property
    def dominant_hz(self) -> float:
        if not self.harmonics_hz:
            return float('nan')
        idx = int(np.argmax(self.amplitudes))
        return float(self.harmonics_hz[idx])


@dataclass(frozen=True)
class FormantSynthesisConfig:
    """Configuration for generating vowel-coded coarse preambles."""

    profile: str
    fundamental_hz: float = DEFAULT_FUNDAMENTAL_HZ
    harmonic_count: int = DEFAULT_HARMONIC_COUNT
    include_fundamental: bool = False
    formant_scale: float = DEFAULT_FORMANT_SCALE
    phase_jitter: float = 0.0


@dataclass(frozen=True)
class FormantAnalysisResult:
    """Outcome of missing-fundamental analysis in the aperture window."""

    label: str
    dominant_hz: float
    missing_fundamental_hz: float
    score: float


def build_formant_library(
    fundamental_hz: float,
    harmonic_count: int,
    include_fundamental: bool,
    formant_scale: float,
) -> Dict[str, FormantDescriptor]:
    """Return descriptors for all supported vowel profiles."""

    if harmonic_count < 1:
        raise ValueError('harmonic_count must be positive')

    library: Dict[str, FormantDescriptor] = {}

    for label, raw_formants in VOWEL_FORMANT_TABLE.items():
        scaled_formants = tuple(float(f) * formant_scale for f in raw_formants)
        partials: Dict[float, float] = {}

        start_harmonic = 1 if include_fundamental else 2
        for n in range(start_harmonic, start_harmonic + harmonic_count):
            freq = float(fundamental_hz) * float(n)
            partials[freq] = max(partials.get(freq, 0.0), 0.2)

        emphasis = 1.0
        for formant in scaled_formants:
            partials[formant] = max(partials.get(formant, 0.0), emphasis)
            emphasis = max(0.3, emphasis - 0.2)

        harmonic_freqs, amplitudes = zip(*sorted(partials.items()))
        amplitudes_arr = np.asarray(amplitudes, dtype=float)
        if amplitudes_arr.max() > 0.0:
            amplitudes_arr = amplitudes_arr / amplitudes_arr.max()

        descriptor = FormantDescriptor(
            label=label,
            fundamental_hz=float(fundamental_hz),
            harmonics_hz=tuple(float(val) for val in harmonic_freqs),
            amplitudes=tuple(float(val) for val in amplitudes_arr),
            include_fundamental=include_fundamental,
        )
        library[label] = descriptor

    return library


def synthesize_formant_preamble(
    length: int,
    sample_rate: float,
    config: FormantSynthesisConfig,
) -> Tuple[NDArray[np.complex128], Dict[str, FormantDescriptor]]:
    """Generate a complex baseband waveform shaped by vowel formants."""

    if length <= 0:
        raise ValueError('length must be positive')
    if sample_rate <= 0.0:
        raise ValueError('sample_rate must be positive')

    profile = config.profile.upper()
    if profile not in VOWEL_FORMANT_TABLE:
        raise ValueError(f'Unsupported formant profile: {profile}')

    library = build_formant_library(
        fundamental_hz=config.fundamental_hz,
        harmonic_count=config.harmonic_count,
        include_fundamental=config.include_fundamental,
        formant_scale=config.formant_scale,
    )

    descriptor = library[profile]
    harmonics = descriptor.harmonics_hz
    amplitudes = descriptor.amplitudes

    time_axis = np.arange(length, dtype=float) / float(sample_rate)
    waveform = np.zeros(length, dtype=np.complex128)

    rng: Optional[np.random.Generator]
    rng = np.random.default_rng(0)
    jitter = float(np.clip(config.phase_jitter, 0.0, np.pi))

    for amp, freq in zip(amplitudes, harmonics):
        if amp <= 0.0:
            continue
        phase = 0.0
        if jitter > 0.0:
            phase = float(rng.uniform(-jitter, jitter))
        waveform += amp * np.exp(1j * (2.0 * np.pi * freq * time_axis + phase))

    norm = np.linalg.norm(waveform)
    if norm > 0.0:
        waveform = waveform / norm

    return waveform.astype(np.complex128), library


def analyze_missing_fundamental(
    segment: NDArray[np.complex128],
    sample_rate: float,
    descriptors: Sequence[FormantDescriptor],
    top_peaks: int = 6,
) -> Optional[FormantAnalysisResult]:
    """Infer the most likely vowel profile using harmonic cues."""

    if segment.size < 8 or sample_rate <= 0.0:
        return None

    window = np.hanning(segment.size)
    spectrum = np.fft.fft(segment * window)
    freqs = np.fft.fftfreq(segment.size, d=1.0 / float(sample_rate))
    pos_mask = freqs > 0.0
    spectrum = spectrum[pos_mask]
    freqs = freqs[pos_mask]
    magnitudes = np.abs(spectrum)
    if magnitudes.size == 0 or not np.any(np.isfinite(magnitudes)):
        return None
    if top_peaks <= 0:
        top_peaks = 1
    top_peaks = min(top_peaks, magnitudes.size)
    top_indices = np.argpartition(magnitudes, -top_peaks)[-top_peaks:]
    top_indices = top_indices[np.argsort(magnitudes[top_indices])[::-1]]
    top_freqs = freqs[top_indices]
    dominant_hz = float(top_freqs[0])

    best_descriptor: Optional[FormantDescriptor] = None
    best_score = float('inf')

    for descriptor in descriptors:
        expected = np.asarray(descriptor.harmonics_hz, dtype=float)
        amplitudes = np.asarray(descriptor.amplitudes, dtype=float)
        usable = min(expected.size, top_freqs.size)
        if usable == 0:
            continue
        expected = expected[:usable]
        amplitudes = amplitudes[:usable]
        observed = np.sort(top_freqs[:usable])
        weight = 1.0 / np.maximum(amplitudes, 1e-3)
        score = float(np.mean(((observed - expected) ** 2) * (weight ** 2)))
        score += 0.1 * abs(dominant_hz - descriptor.dominant_hz)
        if score < best_score:
            best_score = score
            best_descriptor = descriptor

    if best_descriptor is None:
        return None

    harmonic_number = max(int(round(dominant_hz / best_descriptor.fundamental_hz)), 1)
    missing_fundamental = dominant_hz / float(harmonic_number)

    return FormantAnalysisResult(
        label=best_descriptor.label,
        dominant_hz=dominant_hz,
        missing_fundamental_hz=missing_fundamental,
        score=float(best_score),
    )
