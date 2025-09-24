"""Tapped-delay-line (TDL) baseband channel models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np
from numpy.typing import NDArray


def _ensure_array(values: Iterable[float | complex], dtype: str) -> NDArray[np.floating] | NDArray[np.complex128]:
    return np.asarray(list(values), dtype=dtype)


@dataclass
class TappedDelayLine:
    """Simple baseband TDL with optional Doppler and Rician components."""

    delays_s: NDArray[np.float64]
    gains_c: NDArray[np.complex128]
    k_factor_db: Optional[float] = None
    doppler_hz: float | NDArray[np.float64] = 0.0

    def __post_init__(self) -> None:
        self.delays_s = _ensure_array(self.delays_s, 'float64')
        self.gains_c = _ensure_array(self.gains_c, 'complex128')
        if self.delays_s.shape != self.gains_c.shape:
            raise ValueError('delays_s and gains_c must share shape')

    def sample(self, t: NDArray[np.float64]) -> NDArray[np.complex128]:
        grid = np.zeros_like(t, dtype=np.complex128)
        for delay, gain, doppler in zip(
            self.delays_s,
            self.gains_c,
            np.broadcast_to(self._doppler_array(), self.delays_s.shape),
            strict=False,
        ):
            idx = int(np.argmin(np.abs(t - delay)))
            phase = np.exp(1j * 2.0 * np.pi * doppler * (t[idx] - delay))
            grid[idx] += gain * phase
        return grid

    def apply_to_waveform(self, x: NDArray[np.complex128], fs: float) -> NDArray[np.complex128]:
        if np.allclose(self.gains_c, 0.0):
            return np.zeros_like(x)
        t_in = np.arange(len(x)) / fs
        t_out = t_in
        y = np.zeros_like(x, dtype=np.complex128)
        doppler_vals = np.broadcast_to(self._doppler_array(), self.delays_s.shape)
        for delay, gain, doppler in zip(self.delays_s, self.gains_c, doppler_vals, strict=False):
            phase_rotation = np.exp(1j * 2.0 * np.pi * doppler * (t_in - delay))
            rotated = x * phase_rotation
            real = np.interp(t_out - delay, t_in, rotated.real, left=0.0, right=0.0)
            imag = np.interp(t_out - delay, t_in, rotated.imag, left=0.0, right=0.0)
            y += gain * (real + 1j * imag)
        return y

    def narrowband_response(self, fc_hz: float) -> complex:
        omega = 2.0 * np.pi * fc_hz
        doppler_vals = np.broadcast_to(self._doppler_array(), self.delays_s.shape)
        response = np.sum(
            self.gains_c * np.exp(-1j * omega * self.delays_s) * np.exp(1j * 2.0 * np.pi * doppler_vals * 0.0)
        )
        return complex(response)

    def window(self, max_delay_s: float) -> TappedDelayLine:
        """Return a new TDL containing taps up to ``max_delay_s`` beyond the earliest path."""
        if max_delay_s < 0.0:
            max_delay_s = 0.0
        relative = self.delays_s - float(np.min(self.delays_s))
        mask = relative <= max_delay_s + 1e-15
        if not np.any(mask):
            idx = int(np.argmin(relative))
            mask[idx] = True
        delays = self.delays_s[mask]
        gains = self.gains_c[mask]
        doppler_vals = self._doppler_array()[mask]
        doppler_param: float | NDArray[np.float64]
        if np.isscalar(self.doppler_hz):
            doppler_param = float(self.doppler_hz)
        else:
            doppler_param = doppler_vals
        return TappedDelayLine(
            delays_s=delays,
            gains_c=gains,
            k_factor_db=self.k_factor_db,
            doppler_hz=doppler_param,
        )

    def _doppler_array(self) -> NDArray[np.float64]:
        return np.asarray(self.doppler_hz, dtype=float) if np.ndim(self.doppler_hz) else np.full_like(
            self.delays_s, float(self.doppler_hz), dtype=float
        )


def tdl_exponential(L: int, rms_delay_spread_s: float, k_factor_db: Optional[float] = None) -> TappedDelayLine:
    if L <= 0:
        raise ValueError('L must be positive')
    delays = np.linspace(0.0, rms_delay_spread_s * 6.0, L)
    powers = np.exp(-delays / max(rms_delay_spread_s, 1e-12))
    gains = np.sqrt(powers)
    gains /= np.linalg.norm(gains) + 1e-12
    return TappedDelayLine(delays_s=delays, gains_c=gains.astype(np.complex128), k_factor_db=k_factor_db)


def tdl_custom(delays_s: Iterable[float], gains_c: Iterable[complex], k_factor_db: Optional[float] = None) -> TappedDelayLine:
    return TappedDelayLine(delays_s=np.array(list(delays_s), dtype=float), gains_c=np.array(list(gains_c), dtype=complex), k_factor_db=k_factor_db)


# Standard channel profiles for realistic simulation
# Delays and powers are inspired by common academic/industry models.
TDL_PROFILES = {
    "IDEAL": {
        "delays_ns": [0.0],
        "powers_db": [0.0],
    },
    "INDOOR_OFFICE": {
        "delays_ns": [0.0, 20.0, 50.0, 80.0],
        "powers_db": [0.0, -3.0, -9.0, -15.0],
    },
    "URBAN_CANYON": {
        "delays_ns": [0.0, 50.0, 120.0, 200.0, 310.0, 450.0],
        "powers_db": [0.0, -2.5, -8.0, -13.2, -18.0, -24.0],
    },
}


def tdl_from_profile(profile_name: str, rng: np.random.Generator) -> TappedDelayLine:
    """
    Factory function to create a TappedDelayLine model from a standard profile.

    Args:
        profile_name: The name of the profile (e.g., "INDOOR_OFFICE").
        rng: A random number generator for creating random phase shifts.

    Returns:
        A TappedDelayLine object with a random phase realization.
    """
    profile = TDL_PROFILES.get(profile_name.upper())
    if profile is None:
        raise ValueError(f"Unknown TDL profile: {profile_name}")

    delays_s = np.array(profile["delays_ns"], dtype=float) * 1e-9
    powers_db = np.array(profile["powers_db"], dtype=float)

    # Convert powers from dB to linear scale
    linear_powers = 10 ** (powers_db / 10.0)

    # Generate random phases for each tap
    random_phases = rng.uniform(0, 2 * np.pi, size=len(delays_s))

    # Gains are complex: sqrt(power) * exp(j*phase)
    gains_c = np.sqrt(linear_powers) * np.exp(1j * random_phases)

    # Normalize total power to 1
    total_power = np.sum(np.abs(gains_c) ** 2)
    if total_power > 0:
        gains_c /= np.sqrt(total_power)

    return TappedDelayLine(delays_s=delays_s, gains_c=gains_c)
