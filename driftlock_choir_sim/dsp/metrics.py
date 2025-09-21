from __future__ import annotations

import numpy as np


def papr_db(x: np.ndarray) -> float:
    p = np.abs(x) ** 2
    return 10 * np.log10(np.max(p) / (np.mean(p) + 1e-12))


def ber_qpsk_awgn(esn0_db: float) -> float:
    from math import erfc, sqrt
    return 0.5 * erfc(sqrt(10 ** (esn0_db / 10.0)))


def rms_bandwidth_hz(fk: np.ndarray, weights: np.ndarray | None = None) -> float:
    import numpy as _np
    fk = _np.asarray(fk, float)
    if weights is None:
        weights = _np.ones_like(fk)
    w = weights / (float(_np.sum(weights)) + 1e-15)
    f0 = float(_np.sum(w * fk))
    return float(_np.sqrt(_np.sum(w * (fk - f0) ** 2)))
