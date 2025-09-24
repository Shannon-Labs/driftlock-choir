"""Oscillator noise helpers with Allan-deviation inspired profiles."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np

from phy.noise import PowerLawPhaseNoiseGenerator


@dataclass
class OscillatorParams:
    """Parameters describing the oscillator noise profile."""

    allan_dev_1s: float = 1e-11
    drift_rate: float = 0.0
    flicker_corner: float = 1.0
    white_noise_level: float = 1.0
    phase_noise_h: Dict[int, float] = field(default_factory=dict)


class AllanDeviationGenerator:
    """Generate phase and frequency noise sequences for an LO."""

    def __init__(
        self,
        params: OscillatorParams,
        sample_rate: float = 1e6,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        self.params = params
        self.sample_rate = float(sample_rate)
        self._rng = rng or np.random.default_rng()
        self._phase_noise_gen = PowerLawPhaseNoiseGenerator(
            self.params.phase_noise_h,
            sample_rate=self.sample_rate,
            rng=self._rng,
        )

    def generate_phase_noise(self, duration: float) -> np.ndarray:
        n_samples = max(int(round(duration * self.sample_rate)), 0)
        if n_samples == 0:
            return np.zeros(0, dtype=np.float64)
        if not self.params.phase_noise_h:
            sigma = self.params.allan_dev_1s
            return np.cumsum(sigma * self._rng.standard_normal(n_samples))
        return self._phase_noise_gen.generate(n_samples)

    def generate_frequency_drift(self, duration: float) -> np.ndarray:
        n_samples = max(int(round(duration * self.sample_rate)), 0)
        if n_samples == 0:
            return np.zeros(0, dtype=np.float64)

        t = np.arange(n_samples, dtype=np.float64) / self.sample_rate
        linear = self.params.drift_rate * t
        flicker_scale = max(self.params.flicker_corner, 0.0)
        flicker = np.cumsum(self._rng.standard_normal(n_samples)) * flicker_scale / max(self.sample_rate, 1.0)
        white = self._rng.standard_normal(n_samples) * self.params.white_noise_level
        return linear + flicker + white
