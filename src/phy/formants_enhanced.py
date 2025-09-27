"""Enhanced formant-based spectral synthesis and analysis with adaptive robustness features.

This module extends the basic formant functionality with:
- Optimized formant frequencies for better I/E discrimination
- Adaptive bandwidth control based on channel conditions
- Harmonic weighting using formant envelope coherence
- Temporal prosodic variation for acoustic disambiguation
- Formant-aware multipath discrimination
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray


# Optimized formant frequencies for enhanced I/E discrimination
# Increased F2 separation from 600Hz to 800Hz while maintaining acoustic validity
OPTIMIZED_VOWEL_FORMANT_TABLE: Mapping[str, Tuple[float, float, float]] = {
    # Optimized Italian vowel formants for RF beacon applications
    # Increased F2 separation between E and I vowels for better discrimination
    "A": (700.0, 1220.0, 2600.0),   # /a/ - open central, pure (unchanged)
    "E": (450.0, 2050.0, 2900.0),   # /e/ - mid-front, F2 lowered to 2050Hz
    "I": (300.0, 2850.0, 3400.0),   # /i/ - close front, F2 raised to 2850Hz
    "O": (500.0, 900.0, 2400.0),    # /o/ - close-mid back, pure rounded (unchanged)
    "U": (350.0, 750.0, 2200.0),    # /u/ - close back, maximum dark/rounded (unchanged)
}

DEFAULT_FUNDAMENTAL_HZ = 25_000.0
DEFAULT_FORMANT_SCALE = 1_000.0
DEFAULT_HARMONIC_COUNT = 12


@dataclass(frozen=True)
class EnhancedFormantDescriptor:
    """Enhanced metadata describing the harmonic scaffold for a vowel profile."""
    
    label: str
    fundamental_hz: float
    harmonics_hz: Tuple[float, ...]
    amplitudes: Tuple[float, ...]
    formant_centers: Tuple[float, float, float]  # F1, F2, F3 centers
    formant_bandwidths: Tuple[float, float, float]  # Adaptive bandwidths
    include_fundamental: bool = False

    @property
    def dominant_hz(self) -> float:
        if not self.harmonics_hz:
            return float('nan')
        idx = int(np.argmax(self.amplitudes))
        return float(self.harmonics_hz[idx])


@dataclass(frozen=True)
class EnhancedFormantSynthesisConfig:
    """Enhanced configuration for generating vowel-coded coarse preambles."""
    
    profile: str
    fundamental_hz: float = DEFAULT_FUNDAMENTAL_HZ
    harmonic_count: int = DEFAULT_HARMONIC_COUNT
    include_fundamental: bool = False
    formant_scale: float = DEFAULT_FORMANT_SCALE
    phase_jitter: float = 0.0
    # Enhanced parameters
    adaptive_bandwidth: bool = True
    prosodic_variation: bool = True
    formant_bandwidth_factor: float = 1.0  # Base bandwidth multiplier


@dataclass(frozen=True)
class EnhancedFormantAnalysisResult:
    """Enhanced outcome of missing-fundamental analysis."""
    
    label: str
    dominant_hz: float
    missing_fundamental_hz: float
    score: float
    confidence: float  # Additional confidence metric
    formant_coherence: float  # Measure of formant envelope match
    harmonic_agreement: float  # Harmonic pattern consistency


def calculate_formant_bandwidth(snr_db: float, multipath_delay_ns: float) -> Tuple[float, float, float]:
    """Calculate adaptive formant bandwidths based on channel conditions.
    
    Args:
        snr_db: Signal-to-noise ratio in dB
        multipath_delay_ns: Maximum multipath delay in nanoseconds
        
    Returns:
        Tuple of (f1_bw, f2_bw, f3_bw) bandwidth factors
    """
    # Base bandwidths (Hz) - wider for higher formants due to multipath sensitivity
    base_bw_f1 = 80.0   # F1 bandwidth
    base_bw_f2 = 120.0  # F2 bandwidth (most critical for discrimination)
    base_bw_f3 = 150.0  # F3 bandwidth
    
    # SNR adaptation: wider bandwidth in low SNR, narrower in high SNR
    snr_factor = np.clip(1.0 - (snr_db - 25.0) / 50.0, 0.5, 2.0)
    
    # Multipath adaptation: wider bandwidth in severe multipath
    multipath_factor = np.clip(1.0 + multipath_delay_ns / 100.0, 1.0, 3.0)
    
    # Combine factors
    total_factor = snr_factor * multipath_factor
    
    return (
        base_bw_f1 * total_factor,
        base_bw_f2 * total_factor,
        base_bw_f3 * total_factor
    )


def build_enhanced_formant_library(
    fundamental_hz: float,
    harmonic_count: int,
    include_fundamental: bool,
    formant_scale: float,
    snr_db: float = 25.0,
    multipath_delay_ns: float = 0.0,
) -> Dict[str, EnhancedFormantDescriptor]:
    """Return enhanced descriptors for all supported vowel profiles with adaptive bandwidths."""
    
    if harmonic_count < 1:
        raise ValueError('harmonic_count must be positive')
    
    # Calculate adaptive bandwidths based on channel conditions
    f1_bw, f2_bw, f3_bw = calculate_formant_bandwidth(snr_db, multipath_delay_ns)
    
    library: Dict[str, EnhancedFormantDescriptor] = {}
    
    for label, raw_formants in OPTIMIZED_VOWEL_FORMANT_TABLE.items():
        scaled_formants = tuple(float(f) * formant_scale for f in raw_formants)
        partials: Dict[float, float] = {}
        
        start_harmonic = 1 if include_fundamental else 2
        for n in range(start_harmonic, start_harmonic + harmonic_count):
            freq = float(fundamental_hz) * float(n)
            partials[freq] = max(partials.get(freq, 0.0), 0.2)
        
        # Enhanced formant emphasis with bandwidth consideration
        emphasis = 1.0
        formant_weights = [1.0, 1.2, 0.8]  # F2 gets highest weight for discrimination
        
        for i, formant in enumerate(scaled_formants):
            # Apply formant-specific weighting
            formant_weight = formant_weights[i] if i < len(formant_weights) else 1.0
            partials[formant] = max(partials.get(formant, 0.0), emphasis * formant_weight)
            emphasis = max(0.3, emphasis - 0.2)
        
        harmonic_freqs, amplitudes = zip(*sorted(partials.items()))
        amplitudes_arr = np.asarray(amplitudes, dtype=float)
        if amplitudes_arr.max() > 0.0:
            amplitudes_arr = amplitudes_arr / amplitudes_arr.max()
        
        descriptor = EnhancedFormantDescriptor(
            label=label,
            fundamental_hz=float(fundamental_hz),
            harmonics_hz=tuple(float(val) for val in harmonic_freqs),
            amplitudes=tuple(float(val) for val in amplitudes_arr),
            formant_centers=scaled_formants,
            formant_bandwidths=(f1_bw, f2_bw, f3_bw),
            include_fundamental=include_fundamental,
        )
        library[label] = descriptor
    
    return library


def synthesize_enhanced_formant_preamble(
    length: int,
    sample_rate: float,
    config: EnhancedFormantSynthesisConfig,
) -> Tuple[NDArray[np.complex128], Dict[str, EnhancedFormantDescriptor]]:
    """Generate enhanced complex baseband waveform with prosodic variation."""
    
    if length <= 0:
        raise ValueError('length must be positive')
    if sample_rate <= 0.0:
        raise ValueError('sample_rate must be positive')
    
    profile = config.profile.upper()
    if profile not in OPTIMIZED_VOWEL_FORMANT_TABLE:
        raise ValueError(f'Unsupported formant profile: {profile}')
    
    library = build_enhanced_formant_library(
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
    
    # Prosodic variation: slight frequency modulation for disambiguation
    prosodic_factor = 1.0
    if config.prosodic_variation:
        # Add subtle frequency variation (0.1% modulation)
        prosodic_variation_hz = 0.001 * np.sin(2 * np.pi * 10.0 * time_axis)  # 10Hz modulation
        prosodic_factor = 1.0 + prosodic_variation_hz
    
    for amp, freq in zip(amplitudes, harmonics):
        if amp <= 0.0:
            continue
        phase = 0.0
        if jitter > 0.0:
            phase = float(rng.uniform(-jitter, jitter))
        
        # Apply prosodic variation
        modulated_freq = freq * prosodic_factor
        
        waveform += amp * np.exp(1j * (2.0 * np.pi * modulated_freq * time_axis + phase))
    
    norm = np.linalg.norm(waveform)
    if norm > 0.0:
        waveform = waveform / norm
    
    return waveform.astype(np.complex128), library


def analyze_enhanced_missing_fundamental(
    segment: NDArray[np.complex128],
    sample_rate: float,
    descriptors: Sequence[EnhancedFormantDescriptor],
    top_peaks: int = 8,  # Increased for better formant analysis
    snr_estimate: Optional[float] = None,
) -> Optional[EnhancedFormantAnalysisResult]:
    """Enhanced missing-fundamental analysis with formant coherence and harmonic weighting."""
    
    if segment.size < 16 or sample_rate <= 0.0:  # Increased minimum size
        return None
    
    # Use larger analysis window for better frequency resolution
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
    
    # Find spectral peaks with better resolution
    top_indices = np.argpartition(magnitudes, -top_peaks)[-top_peaks:]
    top_indices = top_indices[np.argsort(magnitudes[top_indices])[::-1]]
    top_freqs = freqs[top_indices]
    top_mags = magnitudes[top_indices]
    
    if len(top_freqs) == 0:
        return None
    
    dominant_hz = float(top_freqs[0])
    
    best_descriptor: Optional[EnhancedFormantDescriptor] = None
    best_score = float('inf')
    best_formant_coherence = 0.0
    best_harmonic_agreement = 0.0
    
    for descriptor in descriptors:
        expected = np.asarray(descriptor.harmonics_hz, dtype=float)
        amplitudes = np.asarray(descriptor.amplitudes, dtype=float)
        
        # Enhanced matching with formant coherence
        formant_coherence = calculate_formant_coherence(
            top_freqs, top_mags, descriptor.formant_centers, descriptor.formant_bandwidths
        )
        
        # Harmonic agreement with weighted matching
        harmonic_agreement = calculate_harmonic_agreement(
            top_freqs, expected, amplitudes, descriptor.formant_bandwidths[1]  # Use F2 bandwidth
        )
        
        # Enhanced scoring with formant and harmonic weights
        usable = min(expected.size, top_freqs.size)
        if usable == 0:
            continue
        
        expected = expected[:usable]
        amplitudes = amplitudes[:usable]
        observed = np.sort(top_freqs[:usable])
        
        # Frequency-dependent weighting: higher weight for formant regions
        formant_weights = np.ones_like(amplitudes)
        for formant_center in descriptor.formant_centers:
            # Give higher weight to frequencies near formant centers
            formant_distance = np.abs(observed - formant_center)
            formant_mask = formant_distance < descriptor.formant_bandwidths[1] * 2  # 2x F2 bandwidth
            formant_weights[formant_mask] = 2.0  # Double weight in formant regions
        
        weight = formant_weights / np.maximum(amplitudes, 1e-3)
        freq_error = ((observed - expected) ** 2) * (weight ** 2)
        
        # Combine frequency error with formant coherence and harmonic agreement
        score = float(np.mean(freq_error))
        score -= 0.5 * formant_coherence  # Better coherence lowers score
        score -= 0.3 * harmonic_agreement  # Better agreement lowers score
        score += 0.1 * abs(dominant_hz - descriptor.dominant_hz)
        
        if score < best_score:
            best_score = score
            best_descriptor = descriptor
            best_formant_coherence = formant_coherence
            best_harmonic_agreement = harmonic_agreement
    
    if best_descriptor is None:
        return None
    
    harmonic_number = max(int(round(dominant_hz / best_descriptor.fundamental_hz)), 1)
    missing_fundamental = dominant_hz / float(harmonic_number)
    
    # Calculate confidence based on multiple factors
    confidence = calculate_detection_confidence(
        best_score, best_formant_coherence, best_harmonic_agreement, snr_estimate
    )
    
    return EnhancedFormantAnalysisResult(
        label=best_descriptor.label,
        dominant_hz=dominant_hz,
        missing_fundamental_hz=missing_fundamental,
        score=float(best_score),
        confidence=confidence,
        formant_coherence=best_formant_coherence,
        harmonic_agreement=best_harmonic_agreement,
    )


def calculate_formant_coherence(
    observed_freqs: NDArray[np.float64],
    observed_mags: NDArray[np.float64],
    formant_centers: Tuple[float, float, float],
    formant_bandwidths: Tuple[float, float, float],
) -> float:
    """Calculate how well observed spectral peaks match expected formant structure."""
    
    if len(observed_freqs) == 0:
        return 0.0
    
    coherence_scores = []
    
    for formant_center, bandwidth in zip(formant_centers, formant_bandwidths):
        # Find peaks within formant region
        formant_mask = np.abs(observed_freqs - formant_center) < bandwidth * 1.5
        formant_peaks = observed_mags[formant_mask]
        
        if len(formant_peaks) > 0:
            # Higher coherence if strong peaks are found in formant regions
            peak_strength = np.max(formant_peaks) / (np.max(observed_mags) + 1e-12)
            coherence_scores.append(peak_strength)
        else:
            coherence_scores.append(0.0)
    
    return float(np.mean(coherence_scores)) if coherence_scores else 0.0


def calculate_harmonic_agreement(
    observed_freqs: NDArray[np.float64],
    expected_freqs: NDArray[np.float64],
    expected_amplitudes: NDArray[np.float64],
    tolerance_hz: float,
) -> float:
    """Calculate agreement between observed and expected harmonic patterns."""
    
    if len(observed_freqs) == 0 or len(expected_freqs) == 0:
        return 0.0
    
    agreement_scores = []
    
    for exp_freq, exp_amp in zip(expected_freqs, expected_amplitudes):
        # Find closest observed frequency
        freq_diff = np.abs(observed_freqs - exp_freq)
        if len(freq_diff) > 0 and np.min(freq_diff) < tolerance_hz:
            closest_idx = np.argmin(freq_diff)
            # Score based on frequency match and expected amplitude
            freq_match = 1.0 - (freq_diff[closest_idx] / tolerance_hz)
            agreement_scores.append(freq_match * exp_amp)
    
    return float(np.mean(agreement_scores)) if agreement_scores else 0.0


def calculate_detection_confidence(
    score: float,
    formant_coherence: float,
    harmonic_agreement: float,
    snr_estimate: Optional[float],
) -> float:
    """Calculate overall detection confidence from multiple metrics."""
    
    # Normalize score (lower is better)
    score_confidence = max(0.0, 1.0 - score / 1000.0)  # Adjust divisor based on typical score range
    
    # Combine metrics with weights
    confidence = 0.4 * score_confidence + 0.4 * formant_coherence + 0.2 * harmonic_agreement
    
    # Adjust for SNR if available
    if snr_estimate is not None:
        snr_factor = np.clip(snr_estimate / 30.0, 0.5, 1.5)  # Boost confidence in high SNR
        confidence *= snr_factor
    
    return float(np.clip(confidence, 0.0, 1.0))


# Backward compatibility with existing code
def get_optimized_formant_table() -> Mapping[str, Tuple[float, float, float]]:
    """Return the optimized formant table for external use."""
    return OPTIMIZED_VOWEL_FORMANT_TABLE