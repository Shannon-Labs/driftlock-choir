"""Utility helpers for signal processing."""

from __future__ import annotations

import numpy as np


def apply_fractional_delay(signal: np.ndarray, delay_samples: float) -> np.ndarray:
    """Delay a complex signal by a non-integer number of samples."""
    if np.isclose(delay_samples, 0.0):
        return signal.copy()

    if delay_samples < 0:
        raise ValueError("Fractional delay must be non-negative")

    n_samples = len(signal)
    pad = int(np.ceil(delay_samples)) + 1
    padded = np.concatenate([signal, np.zeros(pad, dtype=signal.dtype)])

    spectrum = np.fft.fft(padded)
    freqs = np.fft.fftfreq(len(padded))
    phase_shift = np.exp(-1j * 2 * np.pi * freqs * delay_samples)
    shifted = np.fft.ifft(spectrum * phase_shift)

    return shifted[:n_samples]
