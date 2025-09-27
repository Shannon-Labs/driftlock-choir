"""Wideband pathfinder utilities for isolating first-arrival multipath components."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from numpy.typing import NDArray
from scipy import signal

from .formants import FormantAnalysisResult, analyze_missing_fundamental
from .preamble import Preamble

_EPS = np.finfo(float).eps


@dataclass(frozen=True)
class PathfinderConfig:
    """Configuration knobs for the wideband pathfinder."""

    relative_threshold_db: float = -12.0
    noise_guard_multiplier: float = 6.0
    smoothing_kernel: int = 5
    guard_interval_s: float = 30e-9
    pre_guard_interval_s: float = 0.0
    aperture_duration_ns: float = 100.0  # Duration of the search window in nanoseconds
    use_simple_search: bool = True  # Disable to force aperture-based fallback
    aperture_lead_ns: float = 60.0  # Extra lead when falling back to the aperture window
    aperture_trail_ns: float = 40.0  # Trailing padding behind peak when scanning backwards
    threshold_relax_db: float = 6.0  # Additional dB slack applied once noise guards are satisfied
    local_floor_percentile: float = 60.0  # Percentile used to estimate within-window noise floor
    fractional_oversample: int = 4  # Oversample factor for fractional lag refinement
    minimum_peak_ratio: float = 0.18  # Minimum |first|/|peak| ratio for accepting early candidates


@dataclass(frozen=True)
class PathfinderResult:
    """Summary of the detected first-arrival path."""

    first_path_s: float
    first_path_amplitude: float
    peak_path_s: float
    peak_path_amplitude: float
    detection_threshold: float
    lag_samples_first: int
    lag_samples_peak: int
    used_aperture_fallback: bool = False
    missing_fundamental_hz: Optional[float] = None
    dominant_harmonic_hz: Optional[float] = None
    formant_label: Optional[str] = None
    formant_score: Optional[float] = None

    @property
    def peak_to_first_ratio(self) -> float:
        denom = max(self.peak_path_amplitude, _EPS)
        return float(self.first_path_amplitude / denom)


def find_first_arrival(
    received: NDArray[np.complex128],
    preamble: Preamble,
    sample_rate: float,
    config: PathfinderConfig | None = None,
) -> PathfinderResult:
    """Finds the first-arriving path using a hybrid approach."""
    if config is None:
        config = PathfinderConfig()
    if received.ndim != 1:
        raise ValueError('received must be 1-D array')

    corr = signal.convolve(received, preamble.matched_filter, mode='full')
    magnitude = np.abs(corr)
    if config.smoothing_kernel > 1 and config.smoothing_kernel % 2 == 1:
        magnitude = signal.medfilt(magnitude, kernel_size=config.smoothing_kernel)

    lags = np.arange(-(preamble.length - 1), received.size)

    # --- Threshold Calculation ---
    peak_idx = int(np.argmax(magnitude))
    peak_amp = float(magnitude[peak_idx])
    if peak_amp <= _EPS:
        return PathfinderResult(
            first_path_s=0.0,
            first_path_amplitude=0.0,
            peak_path_s=0.0,
            peak_path_amplitude=0.0,
            detection_threshold=0.0,
            lag_samples_first=0,
            lag_samples_peak=0,
        )

    lead_samples = max(int(round(config.aperture_lead_ns * 1e-9 * sample_rate)), 0)
    trail_samples = max(int(round(config.aperture_trail_ns * 1e-9 * sample_rate)), 0)
    base_aperture_samples = max(int(round(config.aperture_duration_ns * 1e-9 * sample_rate)), 1)

    noise_slice = magnitude[:max(preamble.length, 1)]
    noise_floor = float(np.median(noise_slice)) if noise_slice.size else 0.0
    rel_threshold = peak_amp * (10.0 ** (config.relative_threshold_db / 20.0))

    lag_zero_idx = int(np.searchsorted(lags, 0))

    # Guard interval expressed in samples for quick comparisons.
    guard_samples: Optional[int]
    if config.guard_interval_s > 0.0:
        guard_samples = max(int(round(config.guard_interval_s * sample_rate)), 0)
    else:
        guard_samples = None
    guard_limit_samples: Optional[int]
    if guard_samples is not None and guard_samples > 3:
        guard_limit_samples = guard_samples
    else:
        guard_limit_samples = None

    pre_guard_samples: int = 0
    if config.pre_guard_interval_s > 0.0:
        pre_guard_samples = max(int(round(config.pre_guard_interval_s * sample_rate)), 0)

    local_floor = noise_floor
    if config.local_floor_percentile > 0.0:
        lookback = max(lead_samples, base_aperture_samples)
        local_start = max(lag_zero_idx + pre_guard_samples - lookback, 0)
        local_end = max(lag_zero_idx + pre_guard_samples, local_start + 1)
        local_slice = magnitude[local_start:local_end]
        if local_slice.size:
            percentile = float(np.clip(config.local_floor_percentile, 0.0, 100.0))
            local_floor = max(local_floor, float(np.percentile(local_slice, percentile)))

    guard_threshold = max(local_floor, noise_floor) * config.noise_guard_multiplier
    detection_threshold = max(rel_threshold, guard_threshold, _EPS)
    if config.threshold_relax_db > 0.0:
        relaxed_db = config.relative_threshold_db - float(config.threshold_relax_db)
        relaxed_threshold = peak_amp * (10.0 ** (relaxed_db / 20.0))
        detection_threshold = max(min(detection_threshold, relaxed_threshold), guard_threshold, _EPS)

    # --- Step 1: Try a simple forward search first ---
    start_search_idx = lag_zero_idx + pre_guard_samples
    first_idx = -1
    required_window_samples = 0
    if config.use_simple_search:
        for idx in range(start_search_idx, len(magnitude)):
            if magnitude[idx] > detection_threshold:
                # Check if it's a local peak
                prev_val = magnitude[idx - 1] if idx > 0 else 0
                next_val = magnitude[idx + 1] if idx < len(magnitude) - 1 else 0
                if magnitude[idx] >= prev_val and magnitude[idx] >= next_val:
                    first_idx = idx
                    break  # Found it!

        # If the candidate lies outside the configured guard window, check if it still
        # meets the relaxed gating criteria; otherwise punt to the aperture search.
        if first_idx != -1 and guard_samples is not None:
            if peak_idx - first_idx > guard_samples:
                candidate_val = magnitude[first_idx]
                ratio = candidate_val / peak_amp if peak_amp > _EPS else 0.0
                delta_fraction = 1.0
                required_ratio_guard = config.minimum_peak_ratio + 0.2 * min(delta_fraction, 1.5)
                required_ratio_guard = float(np.clip(required_ratio_guard, config.minimum_peak_ratio, 0.7))
                candidate_time = lags[first_idx] / sample_rate
                if ratio >= required_ratio_guard and candidate_time >= -5e-9:
                    pass  # Keep this early candidate despite exceeding guard
                else:
                    required_window_samples = max(required_window_samples, peak_idx - first_idx)
                    first_idx = -1

    # --- Step 2: If no simple path found, use the robust aperture search ---
    used_aperture = False
    min_aperture_idx = start_search_idx

    if first_idx == -1:
        used_aperture = True
        window_lead = max(required_window_samples, base_aperture_samples, lead_samples)
        if (
            guard_limit_samples is not None
            and guard_limit_samples >= required_window_samples
            and window_lead > guard_limit_samples
        ):
            window_lead = guard_limit_samples
        window_trail = max(trail_samples, base_aperture_samples // 4)
        start_window_idx = max(min_aperture_idx, peak_idx - window_lead)
        end_window_idx = min(len(magnitude), peak_idx + window_trail + 1)

        aperture_slice = magnitude[start_window_idx:end_window_idx]
        if aperture_slice.size:
            # Scan from the leading edge and take the first local peak that clears the
            # relaxed threshold so genuinely early arrivals win over later, stronger taps.
            for offset, value in enumerate(aperture_slice):
                if value < detection_threshold:
                    continue
                idx = start_window_idx + offset
                if guard_limit_samples is not None and (peak_idx - idx) > guard_limit_samples:
                    continue
                ratio = value / peak_amp if peak_amp > _EPS else 0.0
                delta_fraction = 0.0
                if window_lead > 0:
                    delta_fraction = max(peak_idx - idx, 0) / float(window_lead)
                required_ratio = config.minimum_peak_ratio + 0.1 * min(delta_fraction, 1.5)
                required_ratio = float(np.clip(required_ratio, config.minimum_peak_ratio, 0.6))
                if ratio < required_ratio:
                    continue
                candidate_time = lags[idx] / sample_rate
                if candidate_time < -5e-9:
                    continue
                prev_val = magnitude[idx - 1] if idx > 0 else value
                next_val = magnitude[idx + 1] if idx < len(magnitude) - 1 else value
                if value >= prev_val and value >= next_val:
                    first_idx = idx
                    break

        if first_idx == -1:
            # Search backwards from peak for local maxima above threshold.
            candidate_idx = -1
            for idx in range(peak_idx - 1, start_window_idx - 1, -1):
                if magnitude[idx] < detection_threshold:
                    continue
                if guard_limit_samples is not None and (peak_idx - idx) > guard_limit_samples:
                    continue
                prev_val = magnitude[idx - 1] if idx > 0 else magnitude[idx]
                next_val = magnitude[idx + 1] if idx < len(magnitude) - 1 else magnitude[idx]
                ratio = magnitude[idx] / peak_amp if peak_amp > _EPS else 0.0
                delta_fraction = 0.0
                if window_lead > 0:
                    delta_fraction = max(peak_idx - idx, 0) / float(window_lead)
                required_ratio = config.minimum_peak_ratio + 0.1 * min(delta_fraction, 1.5)
                required_ratio = float(np.clip(required_ratio, config.minimum_peak_ratio, 0.6))
                if ratio < required_ratio:
                    continue
                candidate_time = lags[idx] / sample_rate
                if candidate_time < -5e-9:
                    continue
                if magnitude[idx] >= prev_val and magnitude[idx] >= next_val:
                    candidate_idx = idx
                    break

            if candidate_idx != -1:
                first_idx = candidate_idx
            else:
                # Fall back to threshold crossing with progressive expansion.
                for idx in range(peak_idx, start_window_idx - 1, -1):
                    if magnitude[idx] < detection_threshold:
                        candidate = min(idx + 1, len(magnitude) - 1)
                        if guard_limit_samples is not None and (peak_idx - candidate) > guard_limit_samples:
                            continue
                        value = magnitude[candidate]
                        if value < detection_threshold:
                            continue
                        ratio = value / peak_amp if peak_amp > _EPS else 0.0
                        delta_fraction = 0.0
                        if window_lead > 0:
                            delta_fraction = max(peak_idx - candidate, 0) / float(window_lead)
                        required_ratio = config.minimum_peak_ratio + 0.1 * min(delta_fraction, 1.5)
                        required_ratio = float(np.clip(required_ratio, config.minimum_peak_ratio, 0.6))
                        if ratio < required_ratio:
                            continue
                        candidate_time = lags[candidate] / sample_rate
                        if candidate_time < -5e-9:
                            continue
                        first_idx = candidate
                        break
                else:
                    for idx in range(start_window_idx - 1, min_aperture_idx - 1, -1):
                        if magnitude[idx] < detection_threshold:
                            candidate = min(idx + 1, len(magnitude) - 1)
                            if guard_limit_samples is not None and (peak_idx - candidate) > guard_limit_samples:
                                continue
                            value = magnitude[candidate]
                            if value < detection_threshold:
                                continue
                            ratio = value / peak_amp if peak_amp > _EPS else 0.0
                            delta_fraction = 0.0
                            if window_lead > 0:
                                delta_fraction = max(peak_idx - candidate, 0) / float(window_lead)
                            required_ratio = config.minimum_peak_ratio + 0.1 * min(delta_fraction, 1.5)
                            required_ratio = float(np.clip(required_ratio, config.minimum_peak_ratio, 0.6))
                            if ratio < required_ratio:
                                continue
                            candidate_time = lags[candidate] / sample_rate
                            if candidate_time < -5e-9:
                                continue
                            first_idx = candidate
                            break
                    else:
                        first_idx = min_aperture_idx
                        if guard_limit_samples is not None:
                            first_idx = max(peak_idx - guard_limit_samples, first_idx)

    # --- Refine and Return ---
    refined_lag, refined_amp = _refine_peak_location(
        lags,
        magnitude,
        corr,
        first_idx,
        max(int(config.fractional_oversample), 1),
    )
    first_time = refined_lag / sample_rate
    first_amp = refined_amp
    
    peak_lag = lags[peak_idx]
    peak_time = peak_lag / sample_rate

    formant_analysis: Optional[FormantAnalysisResult] = None
    analyze_segment = bool(preamble.metadata and preamble.metadata.get('formant_analyze'))
    if analyze_segment:
        library = preamble.metadata.get('formant_library') if preamble.metadata else None
        if isinstance(library, dict) and library:
            window_samples = int(config.aperture_duration_ns * 1e-9 * sample_rate)
            window_radius = max(window_samples // 2, 16)
            start_idx = max(first_idx - window_radius, 0)
            end_idx = min(first_idx + window_radius, received.size)
            segment = received[start_idx:end_idx]
            if segment.size >= 8:
                formant_analysis = analyze_missing_fundamental(
                    segment,
                    sample_rate,
                    list(library.values()),
                )

        if formant_analysis is None and preamble.metadata:
            descriptor = preamble.metadata.get('formant_descriptor')
            if descriptor is not None:
                dominant_idx = int(np.argmax(descriptor.amplitudes))
                dominant_hz = float(descriptor.harmonics_hz[dominant_idx])
                harmonic_number = max(int(round(dominant_hz / descriptor.fundamental_hz)), 1)
                missing_hz = dominant_hz / float(harmonic_number)
                formant_analysis = FormantAnalysisResult(
                    label=descriptor.label,
                    dominant_hz=dominant_hz,
                    missing_fundamental_hz=missing_hz,
                    score=float('inf'),
                )

    return PathfinderResult(
        first_path_s=first_time,
        first_path_amplitude=first_amp,
        peak_path_s=peak_time,
        peak_path_amplitude=peak_amp,
        detection_threshold=float(detection_threshold),
        lag_samples_first=int(lags[first_idx]),
        lag_samples_peak=int(peak_lag),
        used_aperture_fallback=used_aperture,
        missing_fundamental_hz=None if formant_analysis is None else formant_analysis.missing_fundamental_hz,
        dominant_harmonic_hz=None if formant_analysis is None else formant_analysis.dominant_hz,
        formant_label=None if formant_analysis is None else formant_analysis.label,
        formant_score=None if formant_analysis is None else formant_analysis.score,
    )

def _parabolic_refine(lags: NDArray, magnitude: NDArray, peak_idx: int) -> tuple[float, float]:
    if peak_idx <= 0 or peak_idx >= len(magnitude) - 1:
        lag = float(lags[peak_idx])
        amp = float(magnitude[peak_idx])
        return lag, amp

    x1, x2, x3 = float(lags[peak_idx - 1]), float(lags[peak_idx]), float(lags[peak_idx + 1])
    y1, y2, y3 = float(magnitude[peak_idx - 1]), float(magnitude[peak_idx]), float(magnitude[peak_idx + 1])

    denom = (x1 - x2) * (x1 - x3) * (x2 - x3)
    if abs(denom) < 1e-12:
        lag = float(lags[peak_idx])
        amp = float(magnitude[peak_idx])
        return lag, amp

    a = (x3 * (y2 - y1) + x2 * (y1 - y3) + x1 * (y3 - y2)) / denom
    b = (x3**2 * (y1 - y2) + x2**2 * (y3 - y1) + x1**2 * (y2 - y3)) / denom
    vertex = -b / (2 * a) if abs(a) > 1e-12 else float(lags[peak_idx])
    refined_index = float(vertex - lags[0])
    refined_index = float(np.clip(refined_index, 0.0, len(magnitude) - 1.0))
    refined_lag = float(lags[0]) + refined_index
    refined_amp = float(np.interp(refined_index, np.arange(len(magnitude)), magnitude))
    return refined_lag, refined_amp


def _refine_peak_location(
    lags: NDArray,
    magnitude: NDArray,
    corr: NDArray,
    peak_idx: int,
    oversample: int,
) -> tuple[float, float]:
    oversample = int(max(oversample, 1))
    parabolic_lag, parabolic_amp = _parabolic_refine(lags, magnitude, peak_idx)
    if oversample <= 1:
        return parabolic_lag, parabolic_amp

    start = max(peak_idx - 2, 0)
    end = min(peak_idx + 3, corr.size)
    segment = corr[start:end]
    if segment.size < 3:
        return parabolic_lag, parabolic_amp

    try:
        upsampled = signal.resample_poly(segment, oversample, 1)
    except Exception:
        return parabolic_lag, parabolic_amp

    if upsampled.size == 0:
        return parabolic_lag, parabolic_amp

    up_mag = np.abs(upsampled)
    local_idx = int(np.argmax(up_mag))
    refined_index = start + local_idx / oversample
    refined_lag = float(lags[0]) + refined_index
    refined_amp = float(up_mag[local_idx])
    return refined_lag, refined_amp
