from __future__ import annotations

import numpy as np


def delay_crlb_std(beff_hz: float, snr_lin: float) -> float:
    """Order-of-magnitude CRLB for time delay in white noise.

    Approximate: sigma_tau ≈ 1 / (2π Beff sqrt(SNR)).
    """
    return 1.0 / (2 * np.pi * beff_hz * np.sqrt(max(snr_lin, 1e-12)))


def delay_crlb_rms_bandwidth(fk: np.ndarray, weights: np.ndarray | None = None,
                           snr_lin: float | None = None) -> float:
    """Compute CRLB using RMS bandwidth with proper tone weighting.

    Args:
        fk: Tone frequencies in Hz
        weights: Optional weights for each tone (e.g., amplitude^2 or SNR)
        snr_lin: Overall SNR (if None, assumes high SNR limit)

    Returns:
        CRLB standard deviation for delay estimate
    """
    if weights is None:
        weights = np.ones_like(fk)

    # Normalize weights
    weights = np.asarray(weights, dtype=float)
    weights = weights / (np.sum(weights) + 1e-15)

    # Compute RMS bandwidth
    f_mean = np.sum(weights * fk)
    b_rms = np.sqrt(np.sum(weights * (fk - f_mean)**2))

    if snr_lin is not None:
        # Full CRLB with SNR
        return 1.0 / (2 * np.pi * b_rms * np.sqrt(snr_lin))
    else:
        # High SNR limit (just depends on bandwidth)
        return 1.0 / (2 * np.pi * b_rms)


def delay_crlb_wls(fk: np.ndarray, snr_per_tone: np.ndarray | None = None,
                  noise_power: float | None = None) -> float:
    """Compute CRLB for WLS delay estimation with per-tone SNR.

    Args:
        fk: Tone frequencies in Hz
        snr_per_tone: Per-tone SNR values (linear)
        noise_power: Noise power estimate

    Returns:
        CRLB standard deviation for delay estimate
    """
    if snr_per_tone is None:
        # Fall back to RMS bandwidth method
        return delay_crlb_rms_bandwidth(fk)

    # For WLS with per-tone SNR, the CRLB is approximately
    # 1 / (2π sqrt(sum(SNR_k)))
    total_snr = np.sum(snr_per_tone)
    if total_snr > 0:
        return 1.0 / (2 * np.pi * np.sqrt(total_snr))
    else:
        return np.inf
