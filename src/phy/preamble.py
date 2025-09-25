"""Baseband preamble generation and delay estimation utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy import signal

from .formants import FormantSynthesisConfig, synthesize_formant_preamble


@dataclass(frozen=True)
class Preamble:
    """Container for a baseband preamble and its matched filter."""

    samples: NDArray[np.complex128]
    matched_filter: NDArray[np.complex128]
    metadata: Optional[Dict[str, object]] = None

    @property
    def length(self) -> int:
        return int(self.samples.size)


def generate_zadoff_chu(length: int, root: int = 1) -> Preamble:
    """Return a constant-modulus Zadoff–Chu sequence of given length."""
    if length <= 0:
        raise ValueError('Preamble length must be positive')
    if np.gcd(length, root) != 1:
        raise ValueError('Length and root must be coprime for Zadoff-Chu sequences')
    n = np.arange(length, dtype=float)
    seq = np.exp(-1j * np.pi * root * n * (n + 1) / length)
    matched = np.conj(seq[::-1])
    norm = np.linalg.norm(seq)
    if norm > 0:
        seq = seq / norm
        matched = matched / norm
    return Preamble(
        samples=seq.astype(np.complex128),
        matched_filter=matched.astype(np.complex128),
        metadata=None,
    )


def estimate_delay(
    received: NDArray[np.complex128],
    preamble: Preamble,
    sample_rate: float,
) -> float:
    """Estimate fractional delay between ``received`` and the known ``preamble``."""
    if received.ndim != 1:
        raise ValueError('received must be 1-D array')
    corr = signal.convolve(received, preamble.matched_filter, mode='full')
    abs_corr = np.abs(corr)
    peak_idx = int(np.argmax(abs_corr))
    lags = np.arange(-(preamble.length - 1), received.size)
    lag = float(lags[peak_idx])
    if 0 < peak_idx < len(abs_corr) - 1:
        y_prev = abs_corr[peak_idx - 1]
        y_curr = abs_corr[peak_idx]
        y_next = abs_corr[peak_idx + 1]
        denom = (y_prev - 2.0 * y_curr + y_next)
        if abs(denom) > 1e-12:
            offset = 0.5 * (y_prev - y_next) / denom
            lag += offset
    return lag / sample_rate


def build_preamble(
    length: int,
    sample_rate: float,
    bandwidth_hz: float,
    root: int = 1,
    *,
    mode: str = 'zadoff',
    formant_config: Optional[FormantSynthesisConfig] = None,
) -> Tuple[Preamble, NDArray[np.float64]]:
    """Generate a preamble and corresponding time axis (seconds)."""

    mode_key = mode.lower()
    time_axis = np.arange(length, dtype=float) / sample_rate

    if mode_key == 'formant':
        if formant_config is None:
            raise ValueError('formant_config must be provided when mode="formant"')
        waveform, library = synthesize_formant_preamble(length, sample_rate, formant_config)
        matched = np.conj(waveform[::-1])
        metadata: Dict[str, object] = {
            'formant_profile': formant_config.profile.upper(),
            'formant_library': library,
            'formant_descriptor': library[formant_config.profile.upper()],
            'formant_config': formant_config,
            'formant_analyze': True,
        }
        return Preamble(samples=waveform, matched_filter=matched, metadata=metadata), time_axis

    if mode_key != 'zadoff':
        raise ValueError(f'Unsupported preamble mode: {mode}')

    preamble = generate_zadoff_chu(length, root=root)
    window = signal.windows.kaiser(length, beta=6.0)
    shaped = preamble.samples * window
    norm = np.linalg.norm(shaped)
    if norm > 0:
        shaped = shaped / norm
    shaped_preamble = Preamble(
        samples=shaped.astype(np.complex128),
        matched_filter=np.conj(shaped[::-1]),
        metadata=None,
    )
    return shaped_preamble, time_axis
