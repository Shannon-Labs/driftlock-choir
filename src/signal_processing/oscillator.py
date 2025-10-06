"""
Oscillator models and signal generation for Driftlock Choir.

This module provides the heart of the chronometric symphony: the Oscillator.
It defines models for various oscillator types, from ideal, noiseless sources
to realistic, temperature-compensated (TCXO) and oven-controlled (OCXO)
crystal oscillators. Each oscillator can be imbued with phase noise,
temperature drift, and aging effects, allowing for the simulation of
real-world hardware with high fidelity.
"""

from typing import Optional, Tuple

import numpy as np

from ..core.constants import PhysicalConstants
from ..core.types import (Frequency, Hertz, MeasurementQuality,
                          OscillatorModel, Picoseconds, Seconds, Timestamp)
from .phase_noise import PhaseNoiseModel


class Oscillator:
    """
    A digital twin of a real-world oscillator, generating signals that form
    the basis of the chronometric interferometry measurement. This class can
    simulate various impairments like phase noise, temperature drift, and aging.
    """

    def __init__(self, model: OscillatorModel, initial_phase: float = 0.0):
        """
        Constructs an Oscillator based on a given model.

        Args:
            model: The OscillatorModel specifying the oscillator's characteristics.
            initial_phase: The starting phase of the oscillator in radians.
        """
        self.model = model
        self.initial_phase = initial_phase
        self._phase_noise_generator = None
        self._current_time = 0.0
        self._phase_noise_model = None

    def generate_signal(
        self,
        duration: Seconds,
        sampling_rate: Hertz,
        frequency_offset: Hertz = 0.0,
        phase_noise_enabled: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generates the complex baseband signal for this oscillator.

        The generated signal is a symphony of deterministic and stochastic
        processes, including the fundamental frequency, user-defined offsets,
        and the subtle whispers of phase noise.

        Args:
            duration: The length of the signal to generate, in seconds.
            sampling_rate: The rate at which to sample the signal, in Hertz.
            frequency_offset: An additional frequency offset to apply, in Hertz.
            phase_noise_enabled: If True, imbues the signal with phase noise.

        Returns:
            A tuple containing the time vector (in seconds) and the complex
            signal as a NumPy array.
        """
        n_samples = int(duration * sampling_rate)
        t = np.arange(n_samples) / sampling_rate

        # The fundamental frequency, including any offset.
        freq = self.model.nominal_frequency + frequency_offset

        # Generate the phase, either pristine or with noise.
        if phase_noise_enabled:
            phase = self._generate_phase_with_noise(t, freq)
        else:
            phase = 2 * np.pi * freq * t + self.initial_phase

        # Create the complex signal from the phase information.
        signal = np.exp(1j * phase)

        return t, signal

    def _generate_phase_with_noise(self, t: np.ndarray, freq: Hertz) -> np.ndarray:
        """
        Generates the phase of the signal, including realistic noise components.

        This is where the oscillator's true character is forged, combining the
        ideal phase evolution with the stochastic contributions of phase noise,
        temperature fluctuations, and the slow march of aging.

        Args:
            t: The time vector for which to generate the phase.
            freq: The base frequency of the oscillator.

        Returns:
            A NumPy array representing the phase of the signal at each point in time.
        """
        # Start with the ideal, linear phase evolution.
        phase = 2 * np.pi * freq * t + self.initial_phase

        # Add phase noise based on the oscillator's profile.
        if self.model.phase_noise_profile:
            phase_noise = self._generate_phase_noise_from_profile(t)
            phase += phase_noise

        # Add the subtle frequency drift caused by temperature changes.
        if self.model.temperature_coefficient != 0:
            temp_drift = self._generate_temperature_drift(t)
            phase += (
                2
                * np.pi
                * freq
                * temp_drift
                * self.model.temperature_coefficient
                * 1e-6
            )

        # Add the slow, inexorable frequency drift due to aging.
        if self.model.aging_rate != 0:
            aging_drift = self._generate_aging_drift(t)
            phase += 2 * np.pi * freq * aging_drift * 1e-9

        return phase

    def _generate_phase_noise_from_profile(self, t: np.ndarray) -> np.ndarray:
        """
        Generates a time-series of phase noise from the oscillator's profile.

        This method translates the frequency-domain phase noise specification
        into a time-domain signal, creating a realistic phase noise signature.

        Args:
            t: The time vector.

        Returns:
            A NumPy array representing the phase noise at each point in time.
        """
        # If a dedicated phase noise model exists, use it.
        if self._phase_noise_model:
            return self._phase_noise_model.generate_phase_noise_time_series(
                t[-1] - t[0], 1.0 / (t[1] - t[0]) if len(t) > 1 else 1.0
            )

        # Otherwise, generate from the profile.
        n_samples = len(t)
        dt = t[1] - t[0] if len(t) > 1 else 1.0
        fs = 1.0 / dt

        # Generate frequency domain noise.
        freqs = np.fft.fftfreq(n_samples, dt)

        # Create the phase noise spectrum.
        phase_noise_spectrum = np.zeros_like(freqs)

        for offset_freq, noise_level in self.model.phase_noise_profile.items():
            # Find the closest frequency bin.
            idx = np.argmin(np.abs(freqs - offset_freq))
            if idx < len(phase_noise_spectrum):
                # Convert dBc/Hz to linear power spectral density.
                psd = 10 ** (noise_level / 10)
                phase_noise_spectrum[idx] = psd

        # Interpolate across the frequency range using a log-log scale.
        positive_freqs = freqs[freqs > 0]
        if len(positive_freqs) > 1:
            positive_psd = phase_noise_spectrum[freqs > 0]

            # Remove zeros for log interpolation.
            mask = positive_psd > 0
            if np.any(mask):
                log_freqs = np.log10(positive_freqs[mask])
                log_psd = np.log10(positive_psd[mask])

                # Interpolate in log-log space.
                all_log_freqs = np.log10(np.abs(freqs[freqs > 0]))
                interpolated_log_psd = np.interp(
                    all_log_freqs,
                    log_freqs,
                    log_psd,
                    left=log_psd[0],
                    right=log_psd[-1],
                )

                phase_noise_spectrum[freqs > 0] = 10**interpolated_log_psd

                # Make the spectrum symmetric for negative frequencies.
                neg_freq_indices = freqs < 0
                pos_freq_values = phase_noise_spectrum[np.abs(freqs) > 0]
                phase_noise_spectrum[neg_freq_indices] = pos_freq_values[
                    : np.sum(neg_freq_indices)
                ]

        # Generate random phase noise in the frequency domain.
        noise_complex = np.random.normal(0, 1, n_samples) + 1j * np.random.normal(
            0, 1, n_samples
        )
        noise_freq = np.fft.fft(noise_complex)

        # Shape the noise with the phase noise spectrum.
        shaped_noise_freq = noise_freq * np.sqrt(phase_noise_spectrum * fs / 2)
        phase_noise = np.real(np.fft.ifft(shaped_noise_freq))

        return phase_noise

    def _generate_temperature_drift(self, t: np.ndarray) -> np.ndarray:
        """
        Generates a simulated temperature-induced frequency drift.

        This is a simplified model, representing the effect of temperature
        fluctuations on the oscillator's frequency.

        Args:
            t: The time vector.

        Returns:
            A NumPy array of the relative frequency change due to temperature.
        """
        # A simple sinusoidal model for temperature variation.
        temp_variation = 5.0 * np.sin(
            2 * np.pi * 1e-4 * t
        )  # 0.0001 Hz temperature variation
        return temp_variation

    def _generate_aging_drift(self, t: np.ndarray) -> np.ndarray:
        """
        Generates a simulated aging-induced frequency drift.

        This models the slow, long-term drift in the oscillator's frequency
        as it ages.

        Args:
            t: The time vector.

        Returns:
            A NumPy array of the relative frequency change due to aging.
        """
        # A linear model for aging.
        aging_rate_ppb_per_sec = self.model.aging_rate / 86400.0
        return aging_rate_ppb_per_sec * t

    def get_frequency_at_time(self, t: Seconds) -> Frequency:
        """
        Calculates the instantaneous frequency of the oscillator at a given time.

        Args:
            t: The time in seconds at which to calculate the frequency.

        Returns:
            A Frequency object, including the estimated frequency and its
            uncertainty.
        """
        # Start with the nominal frequency.
        freq = self.model.nominal_frequency

        # Add the effect of temperature.
        temp_effect = (
            freq
            * self.model.temperature_coefficient
            * 1e-6
            * np.sin(2 * np.pi * 1e-4 * t)
        )

        # Add the effect of aging.
        aging_effect = freq * self.model.aging_rate * 1e-9 * t / 86400.0

        actual_freq = freq + temp_effect + aging_effect

        # Estimate the uncertainty (simplified for this model).
        uncertainty = abs(temp_effect) * 0.1 + abs(aging_effect) * 0.1

        return Frequency(
            freq=Hertz(actual_freq),
            uncertainty=Hertz(uncertainty),
            quality=MeasurementQuality.GOOD,
        )

    @classmethod
    def create_tcxo_model(
        cls,
        nominal_freq: Hertz,
        temperature_coeff: float = 0.5,
        aging_rate: float = 1.0,
    ) -> OscillatorModel:
        """
        Creates a model for a Temperature-Compensated Crystal Oscillator (TCXO).

        TCXOs represent a common class of oscillators that offer good stability
        over a range of temperatures.

        Args:
            nominal_freq: The nominal frequency of the oscillator in Hertz.
            temperature_coeff: The temperature coefficient in parts-per-million
                per degree Celsius.
            aging_rate: The aging rate in parts-per-billion per day.

        Returns:
            An OscillatorModel for a TCXO.
        """
        # A typical phase noise profile for a TCXO.
        phase_noise_profile = {
            1.0: -80,
            10.0: -100,
            100.0: -120,
            1000.0: -130,
            10000.0: -140,
        }

        return OscillatorModel(
            nominal_frequency=nominal_freq,
            phase_noise_profile=phase_noise_profile,
            temperature_coefficient=temperature_coeff,
            aging_rate=aging_rate,
            current_temperature=25.0,
            current_age_days=0.0,
        )

    @classmethod
    def create_ocxo_model(
        cls,
        nominal_freq: Hertz,
        temperature_coeff: float = 0.01,
        aging_rate: float = 0.1,
    ) -> OscillatorModel:
        """
        Creates a model for an Oven-Controlled Crystal Oscillator (OCXO).

        OCXOs are a step up in performance from TCXOs, offering excellent
        stability by maintaining the crystal at a constant temperature.

        Args:
            nominal_freq: The nominal frequency of the oscillator in Hertz.
            temperature_coeff: The temperature coefficient in parts-per-million
                per degree Celsius.
            aging_rate: The aging rate in parts-per-billion per day.

        Returns:
            An OscillatorModel for an OCXO.
        """
        # A typical phase noise profile for an OCXO, superior to a TCXO.
        phase_noise_profile = {
            1.0: -90,
            10.0: -110,
            100.0: -130,
            1000.0: -140,
            10000.0: -150,
        }

        return OscillatorModel(
            nominal_frequency=nominal_freq,
            phase_noise_profile=phase_noise_profile,
            temperature_coefficient=temperature_coeff,
            aging_rate=aging_rate,
            current_temperature=25.0,
            current_age_days=0.0,
        )

    @classmethod
    def create_ideal_oscillator(cls, nominal_freq: Hertz) -> OscillatorModel:
        """
        Creates a model for an ideal, perfect oscillator.

        This oscillator is a purely theoretical construct, with no noise, drift,
        or other impairments. It serves as a perfect reference against which
        to compare more realistic models.

        Args:
            nominal_freq: The nominal frequency of the oscillator in Hertz.

        Returns:
            An OscillatorModel for an ideal oscillator.
        """
        return OscillatorModel(
            nominal_frequency=nominal_freq,
            phase_noise_profile={},  # No phase noise.
            temperature_coefficient=0.0,  # No temperature sensitivity.
            aging_rate=0.0,  # No aging.
            current_temperature=25.0,
            current_age_days=0.0,
        )

    def set_phase_noise_model(self, phase_noise_model: PhaseNoiseModel) -> None:
        """
        Applies a custom phase noise model to the oscillator.

        This allows for the use of more sophisticated, user-defined phase
        noise models.

        Args:
            phase_noise_model: The PhaseNoiseModel to apply.
        """
        self._phase_noise_model = phase_noise_model

    def get_phase_noise_model(self) -> Optional[PhaseNoiseModel]:
        """
        Retrieves the current phase noise model.

        Returns:
            The current PhaseNoiseModel, or None if one is not set.
        """
        return self._phase_noise_model

    def calculate_jitter(
        self,
        duration: Seconds = 1.0,
        integration_bandwidth: Tuple[Hertz, Hertz] = (1.0, 10000.0),
    ) -> Picoseconds:
        """
        Calculates the RMS jitter of the oscillator.

        Jitter is a measure of the timing variations in a signal, and is a
        critical parameter in high-precision timing applications.

        Args:
            duration: The duration over which to calculate the jitter.
            integration_bandwidth: The frequency range over which to integrate
                the phase noise.

        Returns:
            The RMS jitter in picoseconds.
        """
        if self._phase_noise_model:
            return self._phase_noise_model.calculate_jitter(
                duration, self.model.nominal_frequency, integration_bandwidth
            )
        else:
            # A simplified jitter calculation from the phase noise profile.
            if not self.model.phase_noise_profile:
                return Picoseconds(0.0)  # No jitter for an ideal oscillator.

            # Use the worst-case noise level as an approximation.
            worst_case_noise = max(self.model.phase_noise_profile.values())
            jitter_seconds = 10 ** (worst_case_noise / 20) / (
                2 * np.pi * self.model.nominal_frequency
            )
            return PhysicalConstants.seconds_to_ps(jitter_seconds)

    @classmethod
    def create_with_phase_noise_model(
        cls,
        nominal_freq: Hertz,
        phase_noise_model: PhaseNoiseModel,
        temperature_coeff: float = 0.0,
        aging_rate: float = 0.0,
    ) -> "Oscillator":
        """
        Creates an oscillator with a custom phase noise model.

        Args:
            nominal_freq: The nominal frequency of the oscillator.
            phase_noise_model: The custom phase noise model to use.
            temperature_coeff: The temperature coefficient.
            aging_rate: The aging rate.

        Returns:
            An Oscillator instance with the specified phase noise model.
        """
        # Create the oscillator model.
        model = OscillatorModel(
            nominal_frequency=nominal_freq,
            phase_noise_profile=phase_noise_model.get_phase_noise_profile(),
            temperature_coefficient=temperature_coeff,
            aging_rate=aging_rate,
            current_temperature=25.0,
            current_age_days=0.0,
        )

        # Create the oscillator and apply the phase noise model.
        oscillator = cls(model)
        oscillator.set_phase_noise_model(phase_noise_model)

        return oscillator
