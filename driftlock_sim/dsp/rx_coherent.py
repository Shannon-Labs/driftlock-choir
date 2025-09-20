from __future__ import annotations

import numpy as np


def estimate_tone_phasors(x: np.ndarray, fs: float, fk: np.ndarray) -> np.ndarray:
    """Return complex phasors for tones at frequencies ``fk``."""
    n = len(x)
    t = np.arange(n) / fs
    W = np.exp(-1j * 2 * np.pi * fk[:, None] * t[None, :])
    return (W @ x) / n


def unwrap_phase(ph: np.ndarray, fk: np.ndarray | None = None) -> np.ndarray:
    """Unwrap phases, optionally sorting by frequency first."""
    if fk is not None:
        sort_idx = np.argsort(fk)
        ph_sorted = ph[sort_idx]
        ph_unwrapped = np.unwrap(ph_sorted)
        unsort_idx = np.argsort(sort_idx)
        return ph_unwrapped[unsort_idx]
    return np.unwrap(ph)


def estimate_noise_power(
    x: np.ndarray,
    fs: float,
    fk: np.ndarray | None = None,
    phasors: np.ndarray | None = None,
) -> float:
    """Estimate time-domain noise power.

    When tone phasors are provided, we synthesise the deterministic comb and use
    the residual energy as the noise variance. Otherwise, fall back to a robust
    median-based estimate directly on ``x``.
    """
    if fk is not None and phasors is not None and len(fk):
        n = len(x)
        t = np.arange(n) / fs
        recon = np.sum(phasors[:, None] * np.exp(1j * 2 * np.pi * fk[:, None] * t[None, :]), axis=0)
        resid = x - recon
        return float(np.mean(np.abs(resid) ** 2) + 1e-18)

    # Fallback: robust median estimate of |x|^2
    return float(np.median(np.abs(x) ** 2) + 1e-18)


def per_tone_snr(
    phasors: np.ndarray,
    noise_power: float | None = None,
    n_samples: int | None = None,
) -> np.ndarray:
    """Estimate per-tone SNR from phasor magnitudes."""
    mag_sq = np.abs(phasors) ** 2
    if noise_power is not None and n_samples:
        # Correlating noise with a tone reduces the variance by N samples.
        noise_per_phasor = noise_power / max(n_samples, 1)
        snr = mag_sq / (noise_per_phasor + 1e-18)
    else:
        # Robust fallback using MAD around the median magnitude.
        med = np.median(mag_sq)
        mad = np.median(np.abs(mag_sq - med)) + 1e-18
        snr = mag_sq / mad
    return np.maximum(snr, 1e-9)


def wls_delay(
    fk: np.ndarray,
    ph: np.ndarray,
    snr: np.ndarray | None = None,
    return_stats: bool = False,
) -> tuple[float, float] | tuple[float, float, float]:
    """Weighted least-squares delay estimate using per-tone SNR weights."""
    fk = np.asarray(fk, dtype=float)
    ph = np.asarray(ph, dtype=float)
    if fk.shape != ph.shape:
        raise ValueError("fk and ph must have the same shape")

    if snr is not None:
        snr = np.maximum(np.asarray(snr, dtype=float), 1e-9)
        sigma_phi_sq = 1.0 / (2.0 * snr)
        weights = 1.0 / sigma_phi_sq
    else:
        weights = np.ones_like(fk)
        sigma_phi_sq = np.ones_like(fk)

    X = np.column_stack([-2 * np.pi * fk, np.ones_like(fk)])
    W = np.diag(weights)
    XtWX = X.T @ W @ X
    XtWy = X.T @ W @ ph

    beta = np.linalg.pinv(XtWX) @ XtWy
    tau = beta[0]

    cov = np.linalg.pinv(XtWX)
    var_tau = max(cov[0, 0], 0.0)
    ci = 1.96 * np.sqrt(var_tau)

    if return_stats:
        crlb_std = float(np.sqrt(var_tau))
        return float(tau), float(ci), crlb_std
    return float(tau), float(ci)

