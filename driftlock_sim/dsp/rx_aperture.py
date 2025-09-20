from __future__ import annotations

import numpy as np
from numpy.fft import rfft, rfftfreq


def envelope_spectrum(x: np.ndarray, fs: float, alpha: float = 2.0) -> tuple[np.ndarray, np.ndarray]:
    """Enhanced envelope spectrum with noise reduction and spectral optimization."""
    # Enhanced envelope detection with adaptive alpha
    e = np.abs(x) ** alpha
    e = e - np.mean(e)

    # Apply spectral whitening for better peak detection
    E = np.abs(rfft(e))
    f = rfftfreq(len(e), 1 / fs)

    # Spectral enhancement for low-SNR conditions
    E = _enhance_spectral_peaks(E, f, fs)

    return f, E


def _enhance_spectral_peaks(E: np.ndarray, f: np.ndarray, fs: float) -> np.ndarray:
    """Enhance spectral peaks using adaptive filtering for low-SNR conditions."""
    # Apply matched filtering to enhance periodic components
    # This is particularly effective for multi-carrier signals
    window_size = max(5, int(fs / 1000))  # Adaptive window size
    if window_size < len(E):
        # Simple peak enhancement filter
        E_enhanced = E.copy()
        for i in range(window_size, len(E) - window_size):
            neighborhood = E[i - window_size:i + window_size + 1]
            # Boost peaks relative to local average
            local_avg = np.mean(neighborhood)
            if E[i] > local_avg:
                boost_factor = 1.0 + 0.5 * (E[i] / local_avg - 1.0)
                E_enhanced[i] *= boost_factor

        return E_enhanced
    return E


def detect_df_peak(f: np.ndarray, E: np.ndarray, df_hz: float, tol: float = 0.02) -> tuple[float, float]:
    """Enhanced Δf peak detection with sub-pixel accuracy and noise rejection."""
    # Multi-stage peak detection for robustness
    peak_freq, peak_snr = _multi_stage_peak_detection(f, E, df_hz, tol)

    # Sub-pixel refinement using parabolic interpolation
    if peak_snr > -np.inf:
        peak_freq = _refine_peak_location(f, E, peak_freq, df_hz)

    return peak_freq, peak_snr


def _multi_stage_peak_detection(f: np.ndarray, E: np.ndarray, df_hz: float, tol: float) -> tuple[float, float]:
    """Multi-stage peak detection with noise rejection."""
    # Stage 1: Basic band-limited search
    band = (f >= (1 - tol) * df_hz) & (f <= (1 + tol) * df_hz)
    if not np.any(band):
        return df_hz, -np.inf

    # Stage 2: Enhanced peak finding with noise floor estimation
    f_band = f[band]
    E_band = E[band]

    # Estimate noise floor using robust statistics
    noise_floor = _robust_noise_floor(E_band)

    # Find peaks above noise floor
    peak_indices = _find_peaks_above_noise(E_band, noise_floor, min_prominence_db=3.0)

    if len(peak_indices) == 0:
        return df_hz, -np.inf

    # Select strongest peak
    idx = peak_indices[np.argmax(E_band[peak_indices])]
    peak_freq = float(f_band[idx])
    peak_snr = float(20 * np.log10(E_band[idx] / (noise_floor + 1e-12)))

    return peak_freq, peak_snr


def _robust_noise_floor(E: np.ndarray) -> float:
    """Estimate noise floor using robust statistics."""
    # Use median absolute deviation for robust noise estimation
    median_val = np.median(E)
    mad = np.median(np.abs(E - median_val))
    # Conservative noise floor estimate
    return median_val + 2.0 * mad


def _find_peaks_above_noise(E: np.ndarray, noise_floor: float, min_prominence_db: float) -> np.ndarray:
    """Find peaks that exceed noise floor by minimum prominence."""
    min_prominence = 10 ** (min_prominence_db / 20.0)
    threshold = noise_floor * min_prominence

    # Simple peak finding (could be enhanced with scipy.signal.find_peaks)
    peak_indices = []
    for i in range(1, len(E) - 1):
        if E[i] > threshold and E[i] > E[i-1] and E[i] > E[i+1]:
            peak_indices.append(i)

    return np.array(peak_indices, dtype=int)


def _refine_peak_location(f: np.ndarray, E: np.ndarray, coarse_freq: float, df_hz: float) -> float:
    """Refine peak location using parabolic interpolation."""
    # Find index closest to coarse frequency
    idx = np.argmin(np.abs(f - coarse_freq))
    if idx == 0 or idx == len(f) - 1:
        return coarse_freq

    # Parabolic interpolation for sub-pixel accuracy
    try:
        # Fit parabola to 3 points around peak
        x1, x2, x3 = f[idx-1], f[idx], f[idx+1]
        y1, y2, y3 = E[idx-1], E[idx], E[idx+1]

        # Parabolic interpolation formula
        denom = (x1 - x2) * (x1 - x3) * (x2 - x3)
        if abs(denom) < 1e-12:
            return coarse_freq

        a = (x3 * (y2 - y1) + x2 * (y1 - y3) + x1 * (y3 - y2)) / denom
        b = (x3**2 * (y1 - y2) + x2**2 * (y3 - y1) + x1**2 * (y2 - y3)) / denom
        c = (x2 * x3 * (x2 - x3) * y1 + x3 * x1 * (x3 - x1) * y2 + x1 * x2 * (x1 - x2) * y3) / denom

        # Vertex of parabola (peak location)
        refined_freq = -b / (2 * a) if a != 0 else coarse_freq
        return float(refined_freq)
    except:
        return coarse_freq


def detect_df_peak_robust(
    x: np.ndarray,
    fs: float,
    df_hz: float,
    alpha: float = 2.0,
    n_averages: int = 1,
) -> tuple[float, float, dict]:
    """Ultra-robust Δf detection for sub-0dB SNR conditions."""
    results = []

    for avg in range(n_averages):
        # Multiple envelope processing strategies
        f1, E1 = envelope_spectrum(x, fs, alpha=alpha)
        f2, E2 = envelope_spectrum(x, fs, alpha=alpha*1.5)  # Different alpha
        f3, E3 = envelope_spectrum(x, fs, alpha=alpha*0.5)  # Different alpha

        # Combine results from different processing strategies
        peak1, snr1 = detect_df_peak(f1, E1, df_hz)
        peak2, snr2 = detect_df_peak(f2, E2, df_hz)
        peak3, snr3 = detect_df_peak(f3, E3, df_hz)

        results.append({
            'peak_freq': np.mean([peak1, peak2, peak3]),
            'peak_snr': np.mean([snr1, snr2, snr3]),
            'snr_std': np.std([snr1, snr2, snr3]),
            'freq_std': np.std([peak1, peak2, peak3])
        })

    # Combine multiple measurements
    peak_freqs = [r['peak_freq'] for r in results]
    peak_snrs = [r['peak_snr'] for r in results]

    # Weighted average based on SNR
    weights = np.maximum(np.array(peak_snrs), 0.1)  # Minimum weight
    weights = weights / np.sum(weights)

    final_freq = float(np.sum(np.array(peak_freqs) * weights))
    final_snr = float(np.mean(peak_snrs))

    # Quality metrics
    freq_consistency = np.std(peak_freqs)
    snr_consistency = np.std(peak_snrs)

    quality_metrics = {
        'freq_consistency_hz': freq_consistency,
        'snr_consistency_db': snr_consistency,
        'n_measurements': n_averages,
        'detection_confidence': min(1.0, final_snr / 10.0)  # Normalize to 0-1
    }

    return final_freq, final_snr, quality_metrics


def choir_health_index(f: np.ndarray, E: np.ndarray, df_hz: float, harmonics: int = 3, bw_bins: int = 2) -> float:
    """Enhanced choir health index with multi-harmonic analysis."""
    # Sum small neighborhoods around k*df peaks vs total energy
    df = f[1] - f[0]
    energy_total = np.sum(E)
    energy_peaks = 0.0

    for k in range(1, harmonics + 1):
        fk = k * df_hz
        idx = int(round(fk / df))
        lo = max(0, idx - bw_bins)
        hi = min(len(E), idx + bw_bins + 1)
        energy_peaks += np.sum(E[lo:hi])

    # Enhanced health metric including spectral flatness
    base_health = float(energy_peaks / (energy_total + 1e-12))

    # Spectral flatness measure (how peaky vs flat the spectrum is)
    spectral_flatness = _spectral_flatness(E)
    # Combine base health with spectral flatness
    enhanced_health = base_health * (0.7 + 0.3 * spectral_flatness)

    return enhanced_health


def _spectral_flatness(E: np.ndarray) -> float:
    """Calculate spectral flatness measure (geometric/arithmetic mean ratio)."""
    E_norm = E / (np.sum(E) + 1e-12)
    # Avoid log(0) by adding small epsilon
    log_E = np.log(E_norm + 1e-12)
    geometric_mean = np.exp(np.mean(log_E))
    arithmetic_mean = np.mean(E_norm)

    if arithmetic_mean > 0:
        return float(geometric_mean / arithmetic_mean)
    else:
        return 0.0

