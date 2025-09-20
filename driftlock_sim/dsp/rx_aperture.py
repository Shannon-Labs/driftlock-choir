from __future__ import annotations

import numpy as np
from numpy.fft import rfft, rfftfreq


def envelope_spectrum(x: np.ndarray, fs: float, alpha: float = 2.0) -> tuple[np.ndarray, np.ndarray]:
    e = np.abs(x) ** alpha
    e = e - np.mean(e)
    E = np.abs(rfft(e))
    f = rfftfreq(len(e), 1 / fs)
    return f, E


def detect_df_peak(f: np.ndarray, E: np.ndarray, df_hz: float, tol: float = 0.02) -> tuple[float, float]:
    # Find peak near df_hz within tolerance fraction
    band = (f >= (1 - tol) * df_hz) & (f <= (1 + tol) * df_hz)
    if not np.any(band):
        return df_hz, -np.inf
    idx = np.argmax(E[band])
    f_band = f[band]
    E_band = E[band]
    return float(f_band[idx]), float(20 * np.log10(E_band[idx] / (np.mean(E) + 1e-12)))


def choir_health_index(f: np.ndarray, E: np.ndarray, df_hz: float, harmonics: int = 3, bw_bins: int = 2) -> float:
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
    return float(energy_peaks / (energy_total + 1e-12))

