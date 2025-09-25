"""Wideband pathfinder utilities for isolating first-arrival multipath components."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from numpy.typing import NDArray
from scipy import signal

from .preamble import Preamble

_EPS = np.finfo(float).eps


@dataclass(frozen=True)
class PathfinderConfig:
    """Configuration knobs for the wideband pathfinder."""

    relative_threshold_db: float = -12.0
    noise_guard_multiplier: float = 6.0
    smoothing_kernel: int = 5
    guard_interval_s: float = 30e-9
    aperture_duration_ns: float = 100.0  # Duration of the search window in nanoseconds
    use_simple_search: bool = True  # Disable to force aperture-based fallback


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

    noise_slice = magnitude[:max(preamble.length, 1)]
    noise_floor = float(np.median(noise_slice))
    rel_threshold = peak_amp * (10.0 ** (config.relative_threshold_db / 20.0))
    detection_threshold = max(rel_threshold, noise_floor * config.noise_guard_multiplier, _EPS)

    # Guard interval expressed in samples for quick comparisons.
    guard_samples: Optional[int]
    if config.guard_interval_s > 0.0:
        guard_samples = max(int(round(config.guard_interval_s * sample_rate)), 0)
    else:
        guard_samples = None

    # --- Step 1: Try a simple forward search first ---
    start_search_idx = int(np.searchsorted(lags, 0))
    first_idx = -1
    if config.use_simple_search:
        for idx in range(start_search_idx, len(magnitude)):
            if magnitude[idx] > detection_threshold:
                # Check if it's a local peak
                prev_val = magnitude[idx - 1] if idx > 0 else 0
                next_val = magnitude[idx + 1] if idx < len(magnitude) - 1 else 0
                if magnitude[idx] >= prev_val and magnitude[idx] >= next_val:
                    first_idx = idx
                    break  # Found it!

        # If the candidate lies outside the configured guard window, punt to the aperture search.
        if first_idx != -1 and guard_samples is not None:
            if peak_idx - first_idx > guard_samples:
                first_idx = -1

    # --- Step 2: If no simple path found, use the robust aperture search ---
    used_aperture = False
    if first_idx == -1:
        used_aperture = True
        window_samples = int(config.aperture_duration_ns * 1e-9 * sample_rate)
        start_window_idx = max(0, peak_idx - window_samples)

        # Search backwards from peak for the leading edge
        for idx in range(peak_idx, start_window_idx - 1, -1):
            if magnitude[idx] < detection_threshold:
                first_idx = idx + 1
                break
        else: # If loop finishes without finding an edge
            first_idx = start_window_idx

    # --- Refine and Return ---
    refined_lag = _refine_peak_location(lags, magnitude, first_idx)
    first_time = refined_lag / sample_rate
    first_amp = float(magnitude[first_idx])
    
    peak_lag = lags[peak_idx]
    peak_time = peak_lag / sample_rate

    return PathfinderResult(
        first_path_s=first_time,
        first_path_amplitude=first_amp,
        peak_path_s=peak_time,
        peak_path_amplitude=peak_amp,
        detection_threshold=float(detection_threshold),
        lag_samples_first=int(lags[first_idx]),
        lag_samples_peak=int(peak_lag),
        used_aperture_fallback=used_aperture,
    )


def _refine_peak_location(lags: NDArray, magnitude: NDArray, peak_idx: int) -> float:
    """Refine peak location using parabolic interpolation."""
    # Check if we can use 3 points around the peak
    if peak_idx <= 0 or peak_idx >= len(magnitude) - 1:
        return float(lags[peak_idx])

    # Use parabolic interpolation on 3 points around the peak
    try:
        # Get the three points: previous, current, next
        x1, x2, x3 = float(lags[peak_idx - 1]), float(lags[peak_idx]), float(lags[peak_idx + 1])
        y1, y2, y3 = magnitude[peak_idx - 1], magnitude[peak_idx], magnitude[peak_idx + 1]

        # Parabolic interpolation for sub-sample accuracy
        # Using the formula for a parabola y = ax^2 + bx + c passing through 3 points
        denom = (x1 - x2) * (x1 - x3) * (x2 - x3)
        if abs(denom) < 1e-12:
            return float(lags[peak_idx])

        # Calculate coefficients for the parabola y = ax^2 + bx + c
        a = (x3 * (y2 - y1) + x2 * (y1 - y3) + x1 * (y3 - y2)) / denom
        b = (x3**2 * (y1 - y2) + x2**2 * (y3 - y1) + x1**2 * (y2 - y3)) / denom
        c = (x2 * x3 * (x2 - x3) * y1 + x3 * x1 * (x3 - x1) * y2 + x1 * x2 * (x1 - x2) * y3) / denom

        # Find the vertex of the parabola (where dy/dx = 0)
        # dy/dx = 2ax + b = 0 => x = -b / (2a)
        refined_lag = -b / (2 * a) if abs(a) > 1e-12 else float(lags[peak_idx])
        return refined_lag
    except:
        return float(lags[peak_idx])
