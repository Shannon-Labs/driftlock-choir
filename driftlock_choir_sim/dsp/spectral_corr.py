from __future__ import annotations

import numpy as np


def simple_scf(x: np.ndarray, fs: float, alpha_hz: float, nfft: int = 1024) -> tuple[np.ndarray, np.ndarray]:
    """Very lightweight spectral correlation proxy at a single cycle frequency.
    Not a full SCF; uses product x(t)*conj(x(t) e^{-j2π α t})."""
    n = len(x)
    t = np.arange(n) / fs
    y = x * np.conj(x * np.exp(-1j * 2 * np.pi * alpha_hz * t))
    # FFT magnitude as proxy
    X = np.fft.rfft(y, n=nfft)
    f = np.fft.rfftfreq(nfft, 1 / fs)
    return f, np.abs(X)

