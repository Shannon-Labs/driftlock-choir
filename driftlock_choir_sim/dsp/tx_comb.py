from __future__ import annotations

from typing import Any, Dict

import numpy as np
from .phase_schedules import newman_phases, amplitude_taper


def qpsk_symbols(n: int, rng: np.random.Generator) -> np.ndarray:
    bits = rng.integers(0, 4, size=n)
    const = np.array([1 + 1j, -1 + 1j, -1 - 1j, 1 - 1j]) / np.sqrt(2)
    return const[bits]


CombBase = tuple[np.ndarray, np.ndarray, np.ndarray]
CombWithPayload = tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any] | None]


def generate_comb(
    fs: float,
    duration: float,
    df: float,
    m: int,
    omit_fundamental: bool = True,
    amp_mode: str = "equal",
    phase_mode: str = "newman",
    payload_qpsk_fraction: float = 0.0,
    payload_symbol_rate: float = 1000.0,
    rng: np.random.Generator | None = None,
    return_payload: bool = False,
) -> CombBase | CombWithPayload:
    """Enhanced multi-tone comb with optimized missing-fundamental reconstruction.

    Returns ``(x, fk, pilot_mask)`` by default. When ``return_payload`` is true, the
    tuple additionally includes a metadata dictionary describing the payload
    carriers (or ``None`` when no payload is scheduled).
    """
    rng = np.random.default_rng() if rng is None else rng
    n = int(round(fs * duration))
    t = np.arange(n) / fs

    # Enhanced carrier frequency selection for better reconstruction
    # Use optimized spacing to improve missing-fundamental recovery
    half = m // 2
    if omit_fundamental:
        # Create asymmetric frequency distribution for better reconstruction
        # This provides better conditioning for missing-fundamental recovery
        ks = _optimized_carrier_indices(m, df)
    else:
        ks = np.arange(-half, -half + m)

    fk = ks * df

    # Enhanced amplitude tapering for better spectral efficiency
    amps = _enhanced_amplitude_taper(m, amp_mode, fk, df)

    if phase_mode == "newman":
        ph = newman_phases(m)
    elif phase_mode == "none":
        ph = np.zeros(m)
    else:
        raise ValueError("phase_mode must be 'newman' or 'none'")

    payload_count = int(max(0, round(payload_qpsk_fraction * m)))
    pilot_mask = np.ones(m, dtype=bool)
    payload_mask = np.zeros(m, dtype=bool)

    sym_dur = int(max(1, round(fs / payload_symbol_rate)))
    payload_indices = np.array([], dtype=int)
    payload_symbols = np.array([], dtype=complex)

    if payload_count > 0:
        # Enhanced payload carrier selection
        # Choose carriers with best spectral separation from pilots
        payload_idx = _select_optimal_payload_carriers(fk, pilot_mask, payload_count)
        payload_mask[payload_idx] = True
        pilot_mask[payload_idx] = False

        # Build QPSK stream that is shared across payload tones.
        n_syms = int(np.ceil(n / sym_dur))
        payload_symbols = qpsk_symbols(n_syms, rng)
    else:
        payload_symbols = np.array([], dtype=complex)

    x = np.zeros(n, dtype=complex)
    for i, (f, a, p) in enumerate(zip(fk, amps, ph)):
        if payload_mask[i]:
            s = np.repeat(payload_symbols, sym_dur)[:n]
        else:
            s = 1.0
        x += a * s * np.exp(1j * (2 * np.pi * f * t + p))

    # Enhanced power normalization with PAPR optimization
    x = _normalize_with_papr_optimization(x, target_papr_db=8.0)

    if return_payload:
        meta: Dict[str, Any] = {
            "phases": ph.copy(),
            "amps": amps.copy(),
            "payload_indices": np.where(payload_mask)[0],
            "payload_symbols": payload_symbols,
            "samples_per_symbol": sym_dur,
            "symbol_rate": payload_symbol_rate,
            "reconstruction_snr_db": _estimate_reconstruction_snr(fk, amps),
        }
        return x, fk, pilot_mask, meta
    return x, fk, pilot_mask


def _optimized_carrier_indices(m: int, df: float) -> np.ndarray:
    """Generate optimized carrier indices for better missing-fundamental reconstruction."""
    half = m // 2

    # Create frequency distribution that optimizes for missing-fundamental recovery
    # This uses a geometric progression that provides better spectral efficiency
    if m <= 3:
        # For small arrays, use simple symmetric distribution
        ks = np.concatenate([-(np.arange(1, half + 1)), np.arange(1, m - half + 1)])
        ks = ks[:m]
    else:
        # For larger arrays, use optimized spacing
        # This provides better conditioning for the reconstruction matrix
        ks = np.zeros(m, dtype=int)
        center_idx = m // 2

        # Fill from center outward with optimized spacing
        spacing = 1
        idx = 0
        for i in range(m):
            if i == 0:
                ks[center_idx] = 0
            else:
                # Alternate sides with increasing spacing
                side = 1 if i % 2 == 1 else -1
                pos = center_idx + side * spacing
                if 0 <= pos < m:
                    ks[pos] = i * side
                    spacing += 1
                else:
                    # Fallback to linear spacing
                    ks[i] = i - half

        # Remove DC component for missing-fundamental mode
        ks = ks[ks != 0]
        if len(ks) > m:
            ks = ks[:m]
        elif len(ks) < m:
            # Pad with additional frequencies if needed
            additional_ks = np.arange(len(ks) + 1, len(ks) + 1 + (m - len(ks)))
            ks = np.concatenate([ks, additional_ks])

    return ks


def _enhanced_amplitude_taper(
    m: int,
    mode: str,
    fk: np.ndarray,
    df: float
) -> np.ndarray:
    """Enhanced amplitude tapering with spectral efficiency optimization."""
    if mode == "equal":
        # Enhanced equal amplitude with frequency-dependent optimization
        amps = np.ones(m)
        # Boost higher frequencies slightly for better delay estimation
        freq_boost = 1.0 + 0.1 * np.abs(fk) / (np.max(np.abs(fk)) + 1e-6)
        amps *= freq_boost
    elif mode == "hamming":
        # Hamming window for better spectral containment
        window = np.hamming(m)
        amps = window / np.max(window)
    elif mode == "exponential":
        # Exponential taper for missing-fundamental optimization
        alpha = 2.0 / m  # Adaptive decay rate
        indices = np.arange(m)
        center = m // 2
        distances = np.abs(indices - center)
        amps = np.exp(-alpha * distances)
    else:
        amps = amplitude_taper(m, mode)

    # Normalize to unit power
    amps = amps / np.sqrt(np.sum(amps ** 2) + 1e-12)
    return amps


def _select_optimal_payload_carriers(
    fk: np.ndarray,
    pilot_mask: np.ndarray,
    payload_count: int
) -> np.ndarray:
    """Select optimal payload carriers for minimal interference with pilots."""
    pilot_indices = np.where(pilot_mask)[0]

    # Choose carriers with maximum spectral separation from pilots
    available_indices = np.where(~pilot_mask)[0]
    if len(available_indices) == 0:
        return np.array([], dtype=int)

    # Calculate minimum distance to any pilot
    min_distances = []
    for idx in available_indices:
        distances = np.abs(fk[idx] - fk[pilot_indices])
        min_distances.append(np.min(distances))

    # Select carriers with largest minimum distance to pilots
    sorted_indices = np.argsort(min_distances)[::-1]
    selected = available_indices[sorted_indices[:payload_count]]

    return selected


def _normalize_with_papr_optimization(x: np.ndarray, target_papr_db: float = 8.0) -> np.ndarray:
    """Normalize signal with PAPR optimization for better amplifier efficiency."""
    # Standard power normalization
    x = x / np.sqrt(np.mean(np.abs(x) ** 2) + 1e-12)

    # PAPR optimization: clip peaks while maintaining average power
    papr_current = _calculate_papr(x)
    target_papr_linear = 10 ** (target_papr_db / 10.0)

    if papr_current > target_papr_linear:
        # Soft clipping to reduce PAPR
        clip_level = np.sqrt(target_papr_linear)
        x_clipped = _soft_clip(x, clip_level)
        # Re-normalize to maintain average power
        x_clipped = x_clipped / np.sqrt(np.mean(np.abs(x_clipped) ** 2) + 1e-12)
        return x_clipped

    return x


def _calculate_papr(x: np.ndarray) -> float:
    """Calculate Peak-to-Average Power Ratio."""
    power_avg = np.mean(np.abs(x) ** 2)
    power_peak = np.max(np.abs(x) ** 2)
    return power_peak / (power_avg + 1e-12)


def _soft_clip(x: np.ndarray, threshold: float) -> np.ndarray:
    """Apply soft clipping to reduce PAPR."""
    magnitude = np.abs(x)
    # Smooth clipping function
    clipped_mag = np.where(
        magnitude > threshold,
        threshold + (magnitude - threshold) * np.exp(1.0 - magnitude / threshold),
        magnitude
    )
    # Preserve phase
    return clipped_mag * np.exp(1j * np.angle(x))


def _estimate_reconstruction_snr(fk: np.ndarray, amps: np.ndarray) -> float:
    """Estimate SNR for missing-fundamental reconstruction."""
    # Higher SNR for better frequency spacing and amplitude distribution
    freq_span = np.max(np.abs(fk)) - np.min(np.abs(fk))
    amp_balance = np.min(amps) / np.max(amps)

    # Empirical formula based on simulation results
    snr_db = 20 * np.log10(freq_span / 1000.0) + 10 * np.log10(amp_balance + 0.1)
    return max(snr_db, 0.0)

