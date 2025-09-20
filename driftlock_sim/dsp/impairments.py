from __future__ import annotations

import numpy as np
from scipy.signal import resample_poly


def apply_cfo(x: np.ndarray, fs: float, cfo_hz: float) -> np.ndarray:
    if cfo_hz == 0.0:
        return x
    n = len(x)
    t = np.arange(n) / fs
    return x * np.exp(1j * 2 * np.pi * cfo_hz * t)


def apply_phase_noise(x: np.ndarray, phase_noise_rad2: float, rng: np.random.Generator) -> np.ndarray:
    if phase_noise_rad2 <= 0.0:
        return x
    n = len(x)
    w = rng.normal(scale=np.sqrt(phase_noise_rad2), size=n)
    ph = np.cumsum(w)
    return x * np.exp(1j * ph)


def apply_sco(x: np.ndarray, sco_ppm: float) -> np.ndarray:
    if sco_ppm == 0.0:
        return x
    # Simple resampling by factor (1 + ppm*1e-6)
    factor = 1.0 + sco_ppm * 1e-6
    up = 100
    y = resample_poly(x, up=int(round(up * factor)), down=up)
    # Trim/pad to original length
    if len(y) >= len(x):
        return y[: len(x)]
    z = np.zeros_like(x)
    z[: len(y)] = y
    return z


def rapp_soft_clip(x: np.ndarray, p: float = 2.0, sat: float = 1.0) -> np.ndarray:
    # Smooth AM/AM nonlinearity
    a = np.abs(x)
    g = a / ((1 + (a / sat) ** (2 * p)) ** (1 / (2 * p)))
    return g * np.exp(1j * np.angle(x))


def aperture_branch(x: np.ndarray, alpha: float = 2.0, mix: float = 0.3) -> np.ndarray:
    if mix <= 0.0:
        return x
    env = np.abs(x) ** alpha
    env = env - np.mean(env)
    return (1 - mix) * x + mix * env


def cyclo_gate(x: np.ndarray, fs: float, rate_hz: float) -> np.ndarray:
    if rate_hz <= 0.0:
        return x
    n = len(x)
    t = np.arange(n) / fs
    gate = (0.5 * (1 + np.sign(np.sin(2 * np.pi * rate_hz * t)))).astype(float)
    return x * gate

