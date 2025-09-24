"""Noise primitives used across the synchronisation stack."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

_TWO_PI = 2.0 * np.pi


@dataclass
class NoiseParams:
    """Parameters for additive noise and legacy phase-noise paths."""

    snr_db: float
    phase_noise_psd: float
    jitter_rms: float


class PowerLawPhaseNoiseGenerator:
    """Synthesize oscillator phase noise from power-law ``h_α`` coefficients."""

    def __init__(
        self,
        coefficients: Dict[int, float],
        sample_rate: float,
        rng: Optional[np.random.Generator] = None,
        f_min_hz: Optional[float] = None,
    ) -> None:
        if sample_rate <= 0.0:
            raise ValueError("sample_rate must be positive")
        self._coefficients = {int(k): float(v) for k, v in (coefficients or {}).items() if v}
        self._sample_rate = float(sample_rate)
        self._rng = rng or np.random.default_rng()
        nyquist = self._sample_rate / 2.0
        default_fmin = self._sample_rate / 1_000.0
        self._f_min = float(f_min_hz) if f_min_hz else max(default_fmin, 1.0)
        if nyquist > 0:
            self._f_min = min(self._f_min, nyquist)

    def generate(self, n_samples: int) -> np.ndarray:
        if n_samples <= 0:
            return np.zeros(0, dtype=np.float64)
        if not self._coefficients:
            return np.zeros(n_samples, dtype=np.float64)

        freqs = np.fft.rfftfreq(n_samples, d=1.0 / self._sample_rate)
        safe_freqs = np.maximum(freqs, self._f_min)
        psd = np.zeros_like(freqs, dtype=np.float64)
        for alpha, coeff in self._coefficients.items():
            psd += coeff * np.power(safe_freqs, alpha, dtype=np.float64)
        psd = np.maximum(psd, 0.0)

        mag = np.sqrt(psd / 2.0)
        real = self._rng.standard_normal(size=freqs.shape)
        imag = self._rng.standard_normal(size=freqs.shape)
        spectrum = (real + 1j * imag) * mag
        spectrum[0] = 0.0
        if n_samples % 2 == 0 and spectrum.shape[0] > 1:
            spectrum[-1] = np.sqrt(psd[-1]) * self._rng.standard_normal()

        full = np.zeros(n_samples, dtype=np.complex128)
        full[: spectrum.shape[0]] = spectrum
        if n_samples > 1:
            mirror = np.conj(spectrum[1:-1][::-1])
            full[spectrum.shape[0]:] = mirror

        phase_noise = np.fft.ifft(full).real * np.sqrt(n_samples)
        return phase_noise.astype(np.float64)


class NoiseGenerator:
    """Generator for AWGN, phase noise, and timing jitter."""

    def __init__(
        self,
        params: NoiseParams,
        sample_rate: float = 1e6,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        self.params = params
        self.sample_rate = float(sample_rate)
        self._rng = rng or np.random.default_rng()

    def add_awgn(self, signal: np.ndarray, rng: Optional[np.random.Generator] = None) -> np.ndarray:
        signal_power = np.mean(np.abs(signal) ** 2)
        snr_linear = 10 ** (self.params.snr_db / 10.0)
        noise_power = signal_power / snr_linear
        generator = rng or self._rng

        if np.iscomplexobj(signal):
            noise = (
                generator.standard_normal(size=signal.shape)
                + 1j * generator.standard_normal(size=signal.shape)
            ) * np.sqrt(noise_power / 2.0)
        else:
            noise = generator.standard_normal(size=signal.shape) * np.sqrt(noise_power)
        return signal + noise

    def generate_phase_noise(
        self,
        n_samples: int,
        coefficients: Optional[Dict[int, float]] = None,
    ) -> np.ndarray:
        if coefficients:
            generator = PowerLawPhaseNoiseGenerator(coefficients, self.sample_rate, rng=self._rng)
            return generator.generate(n_samples)

        phase_noise_var = 10 ** (self.params.phase_noise_psd / 10.0)
        return np.cumsum(
            np.sqrt(phase_noise_var) * self._rng.standard_normal(n_samples)
        )

    def generate_timing_jitter(self, n_samples: int) -> np.ndarray:
        if n_samples <= 0:
            return np.zeros(0, dtype=np.float64)
        return self.params.jitter_rms * self._rng.standard_normal(n_samples)

    def integrated_phase_variance(self, beat_bw_hz: float, carrier_freq_hz: float) -> float:
        psd_linear = 10 ** (self.params.phase_noise_psd / 10.0)
        var_phase_noise = psd_linear * beat_bw_hz
        var_phase_jitter = (_TWO_PI * carrier_freq_hz * self.params.jitter_rms) ** 2
        return float(var_phase_noise + var_phase_jitter)

    def phase_noise_std(self, beat_bw_hz: float, carrier_freq_hz: float) -> float:
        return np.sqrt(self.integrated_phase_variance(beat_bw_hz, carrier_freq_hz))


__all__ = [
    'NoiseParams',
    'NoiseGenerator',
    'PowerLawPhaseNoiseGenerator',
]
