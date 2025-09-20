from __future__ import annotations

import numpy as np
from typing import Optional, Tuple


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


def traditional_wls_delay(
    fk: np.ndarray,
    ph: np.ndarray,
    snr: np.ndarray | None = None,
    return_stats: bool = False,
) -> tuple[float, float] | tuple[float, float, float]:
    """Traditional weighted least-squares delay estimate with basic SNR weighting."""
    fk = np.asarray(fk, dtype=float)
    ph = np.asarray(ph, dtype=float)
    if fk.shape != ph.shape:
        raise ValueError("fk and ph must have the same shape")

    # Traditional weighting: only SNR-based
    if snr is not None:
        snr = np.maximum(np.asarray(snr, dtype=float), 1e-9)
        sigma_phi_sq = 1.0 / (2.0 * snr)
        weights = 1.0 / sigma_phi_sq
    else:
        weights = np.ones_like(fk)

    X = np.column_stack([-2 * np.pi * fk, np.ones_like(fk)])
    W = np.diag(weights)
    XtWX = X.T @ W @ X
    XtWy = X.T @ W @ ph

    # Use SVD for numerical stability
    U, s, Vt = np.linalg.svd(XtWX, full_matrices=False)
    s_inv = np.where(s > 1e-12, 1.0 / s, 0.0)
    XtWX_inv = Vt.T @ np.diag(s_inv) @ U.T

    beta = XtWX_inv @ XtWy
    tau = beta[0]

    cov = XtWX_inv
    var_tau = max(cov[0, 0], 0.0)
    ci = 1.96 * np.sqrt(var_tau)

    if return_stats:
        crlb_std = float(np.sqrt(var_tau))
        return float(tau), float(ci), crlb_std
    return float(tau), float(ci)


def wls_delay(
    fk: np.ndarray,
    ph: np.ndarray,
    snr: np.ndarray | None = None,
    return_stats: bool = False,
) -> tuple[float, float] | tuple[float, float, float]:
    """Enhanced weighted least-squares delay estimate with optimal frequency weighting."""
    fk = np.asarray(fk, dtype=float)
    ph = np.asarray(ph, dtype=float)
    if fk.shape != ph.shape:
        raise ValueError("fk and ph must have the same shape")

    # Enhanced weighting strategy
    if snr is not None:
        snr = np.maximum(np.asarray(snr, dtype=float), 1e-9)
        # Base SNR weighting
        sigma_phi_sq = 1.0 / (2.0 * snr)
        weights = 1.0 / sigma_phi_sq

        # Optimal frequency weighting: higher frequencies provide better precision
        # Weight ∝ f² for delay estimation (since dφ/dτ = 2πf)
        freq_weights = (np.abs(fk) + 1e-6) ** 2
        freq_weights = freq_weights / np.max(freq_weights)  # Normalize

        # Robust outlier detection using phase residuals
        weights = _robust_weighting(fk, ph, weights, freq_weights, snr)
    else:
        # Default equal weighting with frequency optimization
        freq_weights = (np.abs(fk) + 1e-6) ** 2
        freq_weights = freq_weights / np.max(freq_weights)
        weights = freq_weights

    X = np.column_stack([-2 * np.pi * fk, np.ones_like(fk)])
    W = np.diag(weights)
    XtWX = X.T @ W @ X
    XtWy = X.T @ W @ ph

    # Use SVD for numerical stability
    U, s, Vt = np.linalg.svd(XtWX, full_matrices=False)
    s_inv = np.where(s > 1e-12, 1.0 / s, 0.0)
    XtWX_inv = Vt.T @ np.diag(s_inv) @ U.T

    beta = XtWX_inv @ XtWy
    tau = beta[0]

    cov = XtWX_inv
    var_tau = max(cov[0, 0], 0.0)
    ci = 1.96 * np.sqrt(var_tau)

    if return_stats:
        # Calculate CRLB using the Fisher Information Matrix approach
        # For timing estimation, CRLB = 1 / (2π * RMS_bandwidth * sqrt(SNR))
        if snr is not None:
            # Calculate effective RMS bandwidth considering frequency weighting
            freq_weights_normalized = freq_weights / np.max(freq_weights)
            b_rms = np.sqrt(np.sum(freq_weights_normalized * fk**2) / np.sum(freq_weights_normalized))

            # Calculate effective SNR considering weighting
            snr_effective = np.sum(snr * freq_weights_normalized) / np.sum(freq_weights_normalized)

            # CRLB for timing estimation
            fisher_info = (2 * np.pi * b_rms)**2 * snr_effective
            crlb_std = float(1.0 / np.sqrt(fisher_info))
        else:
            crlb_std = float(np.sqrt(var_tau))
        return float(tau), float(ci), crlb_std
    return float(tau), float(ci)


def adaptive_frequency_offset_selection(
    base_freq_hz: float,
    target_precision_ps: float = 10.0,
    bandwidth_hz: float = 20e6,
    max_offset_hz: float = 1e6,
    snr_db: float = 20.0,
) -> Tuple[float, float, dict]:
    """Adaptive frequency offset selection for optimal timing precision.

    Args:
        base_freq_hz: Base carrier frequency
        target_precision_ps: Target timing precision in picoseconds
        bandwidth_hz: Available bandwidth
        max_offset_hz: Maximum allowed frequency offset
        snr_db: Expected SNR

    Returns:
        Tuple of (optimal_offset_hz, expected_precision_ps, metadata_dict)
    """
    # Calculate CRLB for different frequency offsets
    offset_range = np.linspace(0, max_offset_hz, 100)
    precisions = []

    for offset in offset_range:
        # Effective frequency for beat signal
        f_beat = offset

        # RMS bandwidth calculation for multi-carrier case
        # This is a simplified model - in practice would use actual carrier frequencies
        b_rms = _calculate_rms_bandwidth(base_freq_hz, offset, bandwidth_hz)

        # CRLB calculation
        snr_linear = 10 ** (snr_db / 10.0)
        crlb_std = 1.0 / (2 * np.pi * b_rms * np.sqrt(snr_linear))

        precisions.append(crlb_std * 1e12)  # Convert to picoseconds

    precisions = np.array(precisions)

    # Find offset that achieves target precision
    target_mask = precisions <= target_precision_ps
    if np.any(target_mask):
        # Use the smallest offset that meets the target
        valid_indices = np.where(target_mask)[0]
        best_idx = valid_indices[0]
        optimal_offset = float(offset_range[best_idx])
        expected_precision = float(precisions[best_idx])
    else:
        # Use the offset that gives best possible precision
        best_idx = np.argmin(precisions)
        optimal_offset = float(offset_range[best_idx])
        expected_precision = float(precisions[best_idx])

    # Calculate additional metrics
    metadata = {
        'crlb_range_ps': precisions,
        'offset_range_hz': offset_range,
        'best_possible_precision_ps': float(np.min(precisions)),
        'precision_improvement_factor': float(precisions[0] / expected_precision) if offset_range[0] > 0 else 1.0,
        'bandwidth_utilization': optimal_offset / max_offset_hz,
    }

    return optimal_offset, expected_precision, metadata


def _calculate_rms_bandwidth(base_freq_hz: float, offset_hz: float, bandwidth_hz: float) -> float:
    """Calculate RMS bandwidth for timing estimation."""
    # Simplified RMS bandwidth calculation
    # In practice, this would account for the actual multi-carrier structure

    # For beat signal, the effective bandwidth is related to the offset
    # Higher offsets provide better timing precision but may be limited by bandwidth
    effective_bandwidth = min(offset_hz, bandwidth_hz / 4.0)  # Conservative estimate

    # RMS bandwidth for timing estimation
    # This is a theoretical approximation
    b_rms = effective_bandwidth / np.sqrt(12)  # For uniform distribution

    return max(b_rms, 1.0)  # Minimum bandwidth


def dynamic_offset_adaptation(
    x: np.ndarray,
    fs: float,
    fk: np.ndarray,
    current_snr_db: float,
    target_precision_ps: float = 10.0,
    adaptation_rate: float = 0.1,
) -> Tuple[float, float, dict]:
    """Dynamic frequency offset adaptation based on real-time conditions.

    Args:
        x: Received signal
        fs: Sample rate
        fk: Carrier frequencies
        current_snr_db: Current SNR estimate
        target_precision_ps: Target precision
        adaptation_rate: Rate of adaptation (0-1)

    Returns:
        Tuple of (recommended_offset_hz, confidence, metadata_dict)
    """
    # Estimate current performance
    Yk = estimate_tone_phasors(x, fs, fk)
    noise_power = estimate_noise_power(x, fs, fk, Yk)
    snr_per_tone = per_tone_snr(Yk, noise_power, len(x))

    # Calculate current precision
    ph = unwrap_phase(np.angle(Yk), fk)
    tau_current, _ = wls_delay(fk, ph, snr_per_tone)
    current_precision = abs(tau_current) * 1e12  # Convert to ps

    # Base carrier frequency (simplified - would be more sophisticated in practice)
    base_freq = np.mean(np.abs(fk))

    # Get optimal offset for current conditions
    optimal_offset, expected_precision, metadata = adaptive_frequency_offset_selection(
        base_freq_hz=base_freq,
        target_precision_ps=target_precision_ps,
        snr_db=current_snr_db,
    )

    # Adaptive combination of current and optimal
    # This provides smooth adaptation while responding to changing conditions
    if current_precision > target_precision_ps:
        # Current performance is worse than target, use optimal
        recommended_offset = optimal_offset
        confidence = min(1.0, expected_precision / current_precision)
    else:
        # Current performance is good, blend with optimal
        recommended_offset = (1 - adaptation_rate) * 0 + adaptation_rate * optimal_offset
        confidence = min(1.0, target_precision_ps / current_precision)

    # Update metadata
    metadata.update({
        'current_precision_ps': current_precision,
        'snr_per_tone_db': 10 * np.log10(snr_per_tone + 1e-12),
        'adaptation_confidence': confidence,
        'performance_ratio': current_precision / target_precision_ps,
    })

    return recommended_offset, confidence, metadata


def _robust_weighting(
    fk: np.ndarray,
    ph: np.ndarray,
    base_weights: np.ndarray,
    freq_weights: np.ndarray,
    snr: np.ndarray,
    max_iterations: int = 3,
    outlier_threshold: float = 2.0,
) -> np.ndarray:
    """Robust iterative weighting to handle outliers and improve estimation."""
    weights = base_weights * freq_weights

    for iteration in range(max_iterations):
        # Current estimate
        X = np.column_stack([-2 * np.pi * fk, np.ones_like(fk)])
        W = np.diag(weights)
        XtWX = X.T @ W @ X
        XtWy = X.T @ W @ ph

        U, s, Vt = np.linalg.svd(XtWX, full_matrices=False)
        s_inv = np.where(s > 1e-12, 1.0 / s, 0.0)
        XtWX_inv = Vt.T @ np.diag(s_inv) @ U.T
        beta = XtWX_inv @ XtWy

        # Compute residuals
        ph_pred = X @ beta
        residuals = ph - ph_pred
        residual_rms = np.sqrt(np.mean(residuals ** 2))

        # Robust weight adjustment based on residual magnitude
        # Downweight outliers using Huber-like weighting
        normalized_residuals = np.abs(residuals) / (residual_rms + 1e-12)
        robust_multipliers = np.where(
            normalized_residuals < outlier_threshold,
            1.0,
            outlier_threshold / (normalized_residuals + 1e-6)
        )

        # Combine with SNR and frequency weights
        weights = base_weights * freq_weights * robust_multipliers

        # Ensure minimum weight for numerical stability
        weights = np.maximum(weights, 1e-6 * np.max(weights))

    return weights

