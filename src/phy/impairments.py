from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from numpy.typing import NDArray

@dataclass
class ImpairmentConfig:
    """Configuration for physical layer hardware impairments."""
    # Amplifier non-linearity (3rd-order polynomial: y = c1*x + c3*x*|x|**2)
    # c1 is typically 1.0 (linear gain). c3 introduces distortion.
    # A complex c3 models both AM/AM and AM/PM distortion.
    amp_c1: complex = 1.0 + 0.0j
    amp_c3: complex = 0.0 + 0.0j

    # Oscillator phase noise profile (power-law model: S(f) = sum(h_i * f^i))
    # Coefficients for different noise types (h_alpha)
    # h_-2: Random Walk FM (rad^2/Hz^3)
    # h_-1: Flicker FM (rad^2/Hz^2)
    # h_0: White FM (rad^2/Hz)
    # h_1: Flicker PM (rad^2 * Hz)
    # h_2: White PM (rad^2 * Hz^2)
    phase_noise_h: dict[int, float] = None


def apply_amplifier_nonlinearity(
    signal_in: NDArray[np.complex128],
    c1: complex,
    c3: complex,
) -> NDArray[np.complex128]:
    """
    Applies amplifier non-linearity using a 3rd-order polynomial model.
    This models AM/AM and AM/PM conversion.
    """
    if np.isclose(c3, 0.0):
        return c1 * signal_in

    return c1 * signal_in + c3 * signal_in * np.abs(signal_in)**2


def generate_phase_noise(
    num_samples: int,
    sample_rate: float,
    h: dict[int, float],
    rng: np.random.Generator,
) -> NDArray[np.float64]:
    """
    Generates a phase noise process from a power-law spectral profile.
    S_phi(f) = sum( h_alpha * f^alpha ) for alpha in [-2, -1, 0, 1, 2]
    """
    if not h:
        return np.zeros(num_samples)

    # Frequency axis
    freqs = np.fft.fftfreq(num_samples, 1.0 / sample_rate)

    # Power Spectral Density (PSD) from profile
    psd = np.zeros_like(freqs)
    for alpha, h_val in h.items():
        # Avoid division by zero at DC for negative alphas
        f_safe = np.where(freqs == 0, 1e-12, freqs)
        psd += h_val * np.abs(f_safe)**alpha

    # Create frequency-domain random process with the desired PSD
    # Amplitude is sqrt(PSD), phase is random
    random_phases = np.exp(1j * rng.uniform(0, 2 * np.pi, size=psd.shape))
    fft_noise = np.sqrt(psd) * random_phases

    # IFFT to get time-domain phase noise process
    # The result is a sequence of phase errors in radians
    phase_noise_rad = np.fft.ifft(fft_noise).real * num_samples

    return phase_noise_rad
