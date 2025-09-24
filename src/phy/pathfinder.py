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
    """Detect the first significant peak in the channel impulse response."""
    if config is None:
        config = PathfinderConfig()

    if received.ndim != 1:
        raise ValueError('received must be 1-D array')

    corr = signal.convolve(received, preamble.matched_filter, mode='full')
    magnitude = np.abs(corr)
    if config.smoothing_kernel > 1 and config.smoothing_kernel % 2 == 1:
        magnitude = signal.medfilt(magnitude, kernel_size=config.smoothing_kernel)

    lags = np.arange(-(preamble.length - 1), received.size)
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

    rel_linear = 10.0 ** (config.relative_threshold_db / 20.0)
    rel_threshold = peak_amp * rel_linear

    noise_slice = magnitude[: max(preamble.length, 1)]
    noise_floor = float(np.median(noise_slice))
    guard_threshold = noise_floor * config.noise_guard_multiplier
    detection_threshold = max(rel_threshold, guard_threshold, _EPS)

    start_idx = int(np.searchsorted(lags, 0))
    first_idx = peak_idx
    for idx in range(start_idx, len(magnitude) - 1):
        sample = magnitude[idx]
        if sample < detection_threshold:
            continue
        prev_val = magnitude[idx - 1] if idx > 0 else sample
        next_val = magnitude[idx + 1]
        if sample >= prev_val and sample >= next_val:
            first_idx = idx
            break

    first_amp = float(magnitude[first_idx])
    first_lag = int(lags[first_idx])
    peak_lag = int(lags[peak_idx])

    first_time = first_lag / sample_rate
    peak_time = peak_lag / sample_rate

    return PathfinderResult(
        first_path_s=first_time,
        first_path_amplitude=first_amp,
        peak_path_s=peak_time,
        peak_path_amplitude=peak_amp,
        detection_threshold=float(detection_threshold),
        lag_samples_first=first_lag,
        lag_samples_peak=peak_lag,
    )
