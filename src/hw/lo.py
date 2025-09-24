"""Unlocked local oscillator (LO) drift model with thermal dynamics."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence, Tuple

import numpy as np

from phy.osc import AllanDeviationGenerator, OscillatorParams


@dataclass
class LOConfig:
    """Configuration for the nominal LO state and temperature sensitivity."""

    nominal_freq: float
    temp_coeff: float = 2.5  # Legacy linear ppm/°C coefficient
    aging_rate: float = 1e-7  # ppm/year
    initial_offset: float = 0.0  # ppm
    temp_poly_ppm: Tuple[float, float, float] = (2.5, 0.0, 0.0)
    reference_temp_c: float = 25.0

    def temperature_coefficients(self) -> Tuple[float, float, float]:
        """Return the polynomial coefficients for ΔT → ppm mapping."""

        if self.temp_poly_ppm:
            coeffs = tuple(self.temp_poly_ppm) + (0.0, 0.0, 0.0)
            return coeffs[:3]
        return (self.temp_coeff, 0.0, 0.0)


@dataclass
class ThermalConfig:
    """Parameters for a first-order thermal RC model."""

    ambient_c: float = 25.0
    initial_c: float = 25.0
    time_constant_s: float = 12.0
    steady_state_rise_c: float = 0.0  # Additional rise in steady state above ambient


class LocalOscillator:
    """Unlocked LO with temperature-driven drift and power-law phase noise."""

    def __init__(
        self,
        nominal_freq: float,
        osc_params: OscillatorParams,
        config: Optional[LOConfig] = None,
        thermal: Optional[ThermalConfig] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        self.nominal_freq = float(nominal_freq)
        self.osc_params = osc_params
        self.config = config or LOConfig(nominal_freq)
        self.thermal_config = thermal
        self._rng = rng or np.random.default_rng()

        self.noise_gen = AllanDeviationGenerator(osc_params, rng=self._rng)

        self.current_phase = 0.0
        self.current_freq_offset = self.config.initial_offset * 1e-6
        self.last_update_time = 0.0

        self._temperature_trace: Optional[Tuple[np.ndarray, np.ndarray]] = None
        if self.thermal_config is not None:
            self._temperature = float(self.thermal_config.initial_c)
        else:
            self._temperature = self.config.reference_temp_c

    # ------------------------------------------------------------------
    # Thermal helpers

    def set_temperature_trace(self, times_s: Sequence[float], temps_c: Sequence[float]) -> None:
        """Register an externally captured temperature trace (piecewise-linear)."""

        times = np.asarray(list(times_s), dtype=float)
        temps = np.asarray(list(temps_c), dtype=float)
        if times.ndim != 1 or temps.ndim != 1 or times.size != temps.size:
            raise ValueError("times_s and temps_c must be 1-D arrays of equal length")
        if times.size == 0:
            raise ValueError("temperature trace cannot be empty")
        order = np.argsort(times)
        self._temperature_trace = (times[order], temps[order])

    def clear_temperature_trace(self) -> None:
        self._temperature_trace = None

    def current_temperature(self) -> float:
        return float(self._temperature)

    def _target_temperature(self) -> float:
        if self.thermal_config is None:
            return self.config.reference_temp_c
        return self.thermal_config.ambient_c + self.thermal_config.steady_state_rise_c

    def _update_temperature(self, current_time: float, override_temp: Optional[float], dt: float) -> float:
        if override_temp is not None:
            self._temperature = float(override_temp)
            return self._temperature

        if self._temperature_trace is not None:
            times, temps = self._temperature_trace
            if current_time <= times[0]:
                self._temperature = float(temps[0])
            elif current_time >= times[-1]:
                self._temperature = float(temps[-1])
            else:
                self._temperature = float(np.interp(current_time, times, temps))
            return self._temperature

        if self.thermal_config is None:
            self._temperature = self.config.reference_temp_c
            return self._temperature

        tau = max(self.thermal_config.time_constant_s, 1e-6)
        target = self._target_temperature()
        alpha = np.exp(-dt / tau)
        self._temperature = target - (target - self._temperature) * alpha
        return self._temperature

    # ------------------------------------------------------------------

    def update_state(self, current_time: float, temperature: Optional[float] = None) -> None:
        """Advance the LO state to ``current_time`` with optional temperature override."""

        dt = current_time - self.last_update_time
        if dt > 0.0:
            temp = self._update_temperature(current_time, temperature, dt)
            coeffs = self.config.temperature_coefficients()
            delta_t = temp - self.config.reference_temp_c
            temp_ppm = 0.0
            for order, coeff in enumerate(coeffs, start=1):
                temp_ppm += coeff * (delta_t ** order)
            temp_drift = temp_ppm * 1e-6

            aging_drift = self.config.aging_rate * dt / (365.25 * 24 * 3600)

            phase_noise = self.noise_gen.generate_phase_noise(dt)
            freq_noise = self.noise_gen.generate_frequency_drift(dt)

            self.current_freq_offset += temp_drift + aging_drift + (freq_noise[-1] if freq_noise.size else 0.0)
            self.current_phase += 2 * np.pi * self.current_freq_offset * self.nominal_freq * dt
            if phase_noise.size:
                self.current_phase += phase_noise[-1]

        self.last_update_time = current_time

    def get_phase_at_time(self, timestamp: float, n_samples: int) -> np.ndarray:
        """Return the phase sequence (radians) over ``n_samples`` at ``timestamp``."""

        self.update_state(timestamp)
        sample_rate = 1e6
        dt = 1.0 / sample_rate
        t = np.arange(n_samples, dtype=float) * dt
        freq_offset_phase = 2 * np.pi * self.current_freq_offset * self.nominal_freq * t
        phase_noise = self.noise_gen.generate_phase_noise(n_samples / sample_rate)
        return self.current_phase + freq_offset_phase + phase_noise

    def get_frequency_offset(self) -> float:
        return self.current_freq_offset * self.nominal_freq

    def get_phase_offset(self) -> float:
        return self.current_phase
