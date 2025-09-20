from __future__ import annotations

import numpy as np


def estimate_tone_phasors(x: np.ndarray, fs: float, fk: np.ndarray) -> np.ndarray:
    # Vectorized correlation with complex exponentials at fk
    n = len(x)
    t = np.arange(n) / fs
    W = np.exp(-1j * 2 * np.pi * fk[:, None] * t[None, :])
    return W @ x


def unwrap_phase(ph: np.ndarray) -> np.ndarray:
    return np.unwrap(ph)


def estimate_noise_power(x: np.ndarray, fs: float, tone_mask: np.ndarray | None = None) -> float:
    """Estimate noise power from signal residuals.

    Args:
        x: Complex baseband signal
        fs: Sample rate
        tone_mask: Optional boolean mask of tone frequencies to exclude

    Returns:
        Estimated noise power (variance)
    """
    # Compute periodogram
    n = len(x)
    X = np.fft.fft(x)
    psd = np.abs(X)**2 / n

    if tone_mask is not None:
        # Create frequency mask to exclude tone regions
        f = np.fft.fftfreq(n, 1/fs)
        # Exclude ±5 bins around each tone
        exclude_bins = 5
        mask = np.ones(n, dtype=bool)
        for i, tone_freq in enumerate(f[tone_mask]):
            bin_idx = np.argmin(np.abs(f - tone_freq))
            start = max(0, bin_idx - exclude_bins)
            end = min(n, bin_idx + exclude_bins + 1)
            mask[start:end] = False
        noise_psd = psd[mask]
    else:
        # Use all frequency bins (less accurate but simpler)
        noise_psd = psd

    return float(np.median(noise_psd) + 1e-12)


def wls_delay(fk: np.ndarray, ph: np.ndarray, snr: np.ndarray | None = None,
              noise_power: float | None = None) -> tuple[float, float, float]:
    """Weighted least squares delay estimation with proper SNR weighting.

    Model: ph ≈ -2π fk τ + φ0

    Args:
        fk: Tone frequencies
        ph: Unwrapped phases
        snr: Per-tone SNR values (linear). If None, uses uniform weighting
        noise_power: Noise power estimate. If None and snr provided, uses 1/snr

    Returns:
        tau: Estimated delay
        ci: 95% confidence interval half-width
        crlb: Theoretical CRLB for this configuration
    """
    if snr is not None:
        # Use SNR-based weighting (optimal for WLS)
        w = np.maximum(snr, 1e-6)  # Avoid zero weights
    else:
        # Fall back to magnitude-based weighting
        w = np.ones_like(fk)

    # Model matrix: ph ≈ -2π fk τ + φ0
    X = np.column_stack([-2 * np.pi * fk, np.ones_like(fk)])
    W = np.diag(w)
    XtW = X.T @ W
    beta = np.linalg.pinv(XtW @ X) @ (XtW @ ph)
    tau = beta[0]

    # Covariance and confidence interval
    cov = np.linalg.pinv(XtW @ X)
    var_tau = cov[0, 0]
    ci = 1.96 * np.sqrt(max(var_tau, 0.0))

    # Theoretical CRLB (approximate)
    if noise_power is not None and snr is not None:
        # More accurate CRLB using per-tone SNR
        snr_sum = np.sum(snr)
        if snr_sum > 0:
            crlb = 1.0 / (2 * np.pi * np.sqrt(snr_sum))
        else:
            crlb = np.inf
    else:
        # Simplified CRLB using total SNR
        crlb = np.inf  # Will be computed externally

    return float(tau), float(ci), float(crlb)


def per_tone_snr(y: np.ndarray, n0: float | None = None) -> np.ndarray:
    """Estimate per-tone SNR using noise-ring method.

    For each tone, estimate signal power from the tone magnitude and noise
    power from a ring around the tone frequency. If n0 is provided, use it
    directly as noise power estimate.

    Args:
        y: Complex phasor estimates for each tone
        n0: Optional noise power estimate (if None, estimate from residuals)

    Returns:
        Array of SNR values (linear) for each tone
    """
    mag = np.abs(y)

    if n0 is not None:
        # Use provided noise power estimate
        noise_power = n0
    else:
        # Estimate noise power from residuals around tone magnitudes
        # Use median absolute deviation as robust noise estimate
        s = np.median(mag)
        noise_power = np.median(np.abs(mag - s))**2 + 1e-12

    # Signal power estimate (avoid extreme outliers)
    signal_power = np.maximum(mag**2, noise_power * 1e-6)

    # SNR in linear units
    snr_lin = signal_power / (noise_power + 1e-12)

    return snr_lin
