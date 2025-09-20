from __future__ import annotations

import numpy as np


def impose_fractional_delay_fft(x: np.ndarray, fs: float, tau_s: float) -> np.ndarray:
    """Impose continuous delay τ on complex x by spectral phase rotation.

    Uses complex FFT with a Hann window to limit circular artifacts, then
    approximately de-weights to restore amplitude.
    """
    if tau_s == 0.0:
        return x
    n = len(x)
    w = np.hanning(n).astype(x.dtype)
    X = np.fft.fft(x * w)
    f = np.fft.fftfreq(n, 1.0 / fs)
    phase = np.exp(-1j * 2.0 * np.pi * f * tau_s)
    Y = X * phase
    y = np.fft.ifft(Y)
    w_safe = np.maximum(w.real, 1e-3)
    y = (y / w_safe).astype(complex, copy=False)
    return y
