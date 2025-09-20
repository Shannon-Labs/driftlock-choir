from __future__ import annotations

import numpy as np
from scipy.signal import lfilter


def tapped_delay_channel(x: np.ndarray, fs: float, taps: list[dict]) -> np.ndarray:
    """Apply LTI multipath: h(t)=sum alpha_i delta(t - tau_i)."""
    if not taps:
        return x
    n = len(x)
    h = np.zeros(n)
    for tp in taps:
        d = int(round(float(tp.get("delay_s", 0.0)) * fs))
        a = float(tp.get("gain", 1.0))
        if 0 <= d < n:
            h[d] += a
    y = lfilter(h, [1.0], x)
    return y


def awgn(x: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    if np.isinf(snr_db):
        return x
    p_sig = np.mean(np.abs(x)**2)
    snr = 10 ** (snr_db / 10.0)
    p_n = p_sig / snr
    n = (rng.normal(scale=np.sqrt(p_n/2), size=x.shape)
         + 1j * rng.normal(scale=np.sqrt(p_n/2), size=x.shape))
    return x + n

