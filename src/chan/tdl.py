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

