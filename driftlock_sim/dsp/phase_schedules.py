from __future__ import annotations

import numpy as np


def newman_phases(m: int) -> np.ndarray:
    # Newman phase sequence to reduce PAPR of multi-tone signals
    k = np.arange(m)
    return np.pi * k * (k - 1) / m


def amplitude_taper(m: int, mode: str = "equal") -> np.ndarray:
    if mode == "equal":
        return np.ones(m)
    if mode == "hann":
        return np.hanning(m)
    if mode == "taylor":
        # Simple Taylor-like taper (approx): use raised cosine
        n = np.arange(m)
        return 0.5 * (1 - np.cos(2 * np.pi * (n + 0.5) / m))
    raise ValueError(f"unknown amplitude mode: {mode}")

