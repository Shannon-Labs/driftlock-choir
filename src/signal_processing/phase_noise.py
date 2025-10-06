"""
Phase noise modeling and characterization for Driftlock Choir.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np

from ..core.constants import PhysicalConstants
from ..core.types import Decibels, Hertz, Picoseconds


class PhaseNoiseModel:
    """
    Comprehensive phase noise model for oscillator characterization.

    This class implements various phase noise models including white noise,
    flicker noise, and random walk, with configurable parameters for
    detailed characterization experiments.
    """

    def __init__(
        self,
        white_noise_level: Decibels = -140.0,
        flicker_noise_level: Decibels = -120.0,
        flicker_corner_freq: Hertz = 100.0,
        random_walk_level: Decibels = -100.0,
        random_walk_corner_freq: Hertz = 10.0,
    ):
        """
        Initialize phase noise model.

        Args:
            white_noise_level: White phase noise level at 1 Hz offset (dBc/Hz)
            flicker_noise_level: Flicker phase noise level at 1 Hz offset (dBc/Hz)
            flicker_corner_freq: Flicker noise corner frequency (Hz)
            random_walk_level: Random walk phase noise level at 1 Hz offset (dBc/Hz)
            random_walk_corner_freq: Random walk corner frequency (Hz)
        """
        self.white_noise_level = white_noise_level
        self.flicker_noise_level = flicker_noise_level
        self.flicker_corner_freq = flicker_corner_freq
        self.random_walk_level = random_walk_level
        self.random_walk_corner_freq = random_walk_corner_freq

    def get_phase_noise_at(self, offset_freq: Hertz) -> Decibels:
        """
        Get phase noise level at specific offset frequency.

        Args:
            offset_freq: Offset frequency from carrier (Hz)

        Returns:
            Phase noise level in dBc/Hz
        """
        if offset_freq <= 0:
            return Decibels(-100.0)  # Default value

        # White noise (frequency independent)
        white_noise = self.white_noise_level

        # Flicker noise (1/f)
        if offset_freq < self.flicker_corner_freq:
            flicker_noise = self.flicker_noise_level + 20 * np.log10(
                self.flicker_corner_freq / offset_freq
            )
        else:
            flicker_noise = self.flicker_noise_level

        # Random walk (1/f²)
        if offset_freq < self.random_walk_corner_freq:
            random_walk_noise = self.random_walk_level + 40 * np.log10(
                self.random_walk_corner_freq / offset_freq
            )
        else:
            random_walk_noise = self.random_walk_level

        # Total noise is sum of all components (in linear scale)
        total_noise_linear = (
            10 ** (white_noise / 10)
            + 10 ** (flicker_noise / 10)
            + 10 ** (random_walk_noise / 10)
        )

        return Decibels(10 * np.log10(total_noise_linear))

    def get_phase_noise_profile(
        self, freq_points: Optional[List[Hertz]] = None
    ) -> Dict[Hertz, Decibels]:
        """
        Get complete phase noise profile.

        Args:
            freq_points: List of frequencies to evaluate (auto-generated if None)

        Returns:
            Dictionary of offset frequencies to noise levels
        """
        if freq_points is None:
            # Generate logarithmic frequency points from 1 Hz to 100 kHz
            freq_points = np.logspace(0, 5, 50).tolist()

        return {freq: self.get_phase_noise_at(freq) for freq in freq_points}

    def generate_phase_noise_time_series(
        self, duration: float, sampling_rate: Hertz
    ) -> np.ndarray:
        """
        Generate phase noise time series.

        Args:
            duration: Duration in seconds
            sampling_rate: Sampling rate in Hz

        Returns:
            Phase noise time series (radians)
        """
        n_samples = int(duration * sampling_rate)
        dt = 1.0 / sampling_rate

        # Generate frequency domain noise
        freqs = np.fft.fftfreq(n_samples, dt)

        # Create phase noise spectrum
        phase_noise_spectrum = np.zeros_like(freqs)

        for i, freq in enumerate(freqs):
            if abs(freq) > 0:  # Skip DC
                noise_level = self.get_phase_noise_at(abs(freq))
                # Convert dBc/Hz to linear power spectral density
                psd = 10 ** (noise_level / 10)
                phase_noise_spectrum[i] = psd

        # Generate random phase noise
        # Create complex Gaussian noise with specified PSD
        noise_complex = np.random.normal(0, 1, n_samples) + 1j * np.random.normal(
            0, 1, n_samples
        )
        noise_freq = np.fft.fft(noise_complex)

        # Apply phase noise spectrum
        shaped_noise_freq = noise_freq * np.sqrt(
            phase_noise_spectrum * sampling_rate / 2
        )
        phase_noise = np.real(np.fft.ifft(shaped_noise_freq))

        return phase_noise

    def calculate_jitter(
        self,
        duration: float,
        sampling_rate: Hertz,
        integration_bandwidth: Tuple[Hertz, Hertz] = (1.0, 10000.0),
    ) -> Picoseconds:
        """
        Calculate RMS jitter from phase noise.

        Args:
            duration: Duration in seconds
            sampling_rate: Sampling rate in Hz
            integration_bandwidth: Frequency range for integration (Hz)

        Returns:
            RMS jitter in picoseconds
        """
        # Generate phase noise time series
        phase_noise = self.generate_phase_noise_time_series(duration, sampling_rate)

        # Calculate RMS phase deviation
        rms_phase = np.sqrt(np.mean(phase_noise**2))

        # Convert to time jitter
        # For a given frequency f, phase error φ corresponds to time error τ = φ/(2πf)
        # We'll use the carrier frequency for this calculation
        carrier_freq = 2.4e9  # Typical carrier frequency
        jitter_seconds = rms_phase / (2 * np.pi * carrier_freq)

        return PhysicalConstants.seconds_to_ps(jitter_seconds)

    @classmethod
    def create_tcxo_model(
        cls,
        white_noise: Decibels = -140.0,
        flicker_noise: Decibels = -120.0,
        random_walk: Decibels = -100.0,
    ) -> "PhaseNoiseModel":
        """
        Create TCXO-like phase noise model.

        Args:
            white_noise: White phase noise level
            flicker_noise: Flicker phase noise level
            random_walk: Random walk phase noise level

        Returns:
            TCXO phase noise model
        """
        return cls(
            white_noise_level=white_noise,
            flicker_noise_level=flicker_noise,
            flicker_corner_freq=Hertz(100.0),
            random_walk_level=random_walk,
            random_walk_corner_freq=Hertz(10.0),
        )

    @classmethod
    def create_ocxo_model(
        cls,
        white_noise: Decibels = -150.0,
        flicker_noise: Decibels = -130.0,
        random_walk: Decibels = -110.0,
    ) -> "PhaseNoiseModel":
        """
        Create OCXO-like phase noise model.

        Args:
            white_noise: White phase noise level
            flicker_noise: Flicker phase noise level
            random_walk: Random walk phase noise level

        Returns:
            OCXO phase noise model
        """
        return cls(
            white_noise_level=white_noise,
            flicker_noise_level=flicker_noise,
            flicker_corner_freq=Hertz(10.0),
            random_walk_level=random_walk,
            random_walk_corner_freq=Hertz(1.0),
        )

    @classmethod
    def create_custom_model(cls, noise_params: Dict[str, float]) -> "PhaseNoiseModel":
        """
        Create custom phase noise model from parameters.

        Args:
            noise_params: Dictionary of noise parameters

        Returns:
            Custom phase noise model
        """
        return cls(
            white_noise_level=Decibels(noise_params.get("white_noise", -140.0)),
            flicker_noise_level=Decibels(noise_params.get("flicker_noise", -120.0)),
            flicker_corner_freq=Hertz(noise_params.get("flicker_corner", 100.0)),
            random_walk_level=Decibels(noise_params.get("random_walk", -100.0)),
            random_walk_corner_freq=Hertz(noise_params.get("random_walk_corner", 10.0)),
        )


class PhaseNoiseSweep:
    """
    Utility class for sweeping phase noise parameters.
    """

    @staticmethod
    def generate_sweep_parameters(
        param_name: str, start_value: float, end_value: float, num_points: int
    ) -> List[float]:
        """
        Generate parameter sweep values.

        Args:
            param_name: Name of parameter to sweep
            start_value: Starting value
            end_value: Ending value
            num_points: Number of points

        Returns:
            List of parameter values
        """
        if param_name in ["white_noise", "flicker_noise", "random_walk"]:
            # Logarithmic sweep for noise levels
            return np.logspace(start_value, end_value, num_points).tolist()
        else:
            # Linear sweep for other parameters
            return np.linspace(start_value, end_value, num_points).tolist()

    @staticmethod
    def create_phase_noise_models_from_sweep(
        param_name: str, values: List[float], base_model: PhaseNoiseModel
    ) -> List[PhaseNoiseModel]:
        """
        Create phase noise models from parameter sweep.

        Args:
            param_name: Name of parameter to sweep
            values: List of parameter values
            base_model: Base phase noise model

        Returns:
            List of phase noise models
        """
        models = []

        for value in values:
            # Create a copy of the base model
            if param_name == "white_noise":
                model = PhaseNoiseModel(
                    white_noise_level=Decibels(value),
                    flicker_noise_level=base_model.flicker_noise_level,
                    flicker_corner_freq=base_model.flicker_corner_freq,
                    random_walk_level=base_model.random_walk_level,
                    random_walk_corner_freq=base_model.random_walk_corner_freq,
                )
            elif param_name == "flicker_noise":
                model = PhaseNoiseModel(
                    white_noise_level=base_model.white_noise_level,
                    flicker_noise_level=Decibels(value),
                    flicker_corner_freq=base_model.flicker_corner_freq,
                    random_walk_level=base_model.random_walk_level,
                    random_walk_corner_freq=base_model.random_walk_corner_freq,
                )
            elif param_name == "random_walk":
                model = PhaseNoiseModel(
                    white_noise_level=base_model.white_noise_level,
                    flicker_noise_level=base_model.flicker_noise_level,
                    flicker_corner_freq=base_model.flicker_corner_freq,
                    random_walk_level=Decibels(value),
                    random_walk_corner_freq=base_model.random_walk_corner_freq,
                )
            elif param_name == "flicker_corner":
                model = PhaseNoiseModel(
                    white_noise_level=base_model.white_noise_level,
                    flicker_noise_level=base_model.flicker_noise_level,
                    flicker_corner_freq=Hertz(value),
                    random_walk_level=base_model.random_walk_level,
                    random_walk_corner_freq=base_model.random_walk_corner_freq,
                )
            elif param_name == "random_walk_corner":
                model = PhaseNoiseModel(
                    white_noise_level=base_model.white_noise_level,
                    flicker_noise_level=base_model.flicker_noise_level,
                    flicker_corner_freq=base_model.flicker_corner_freq,
                    random_walk_level=base_model.random_walk_level,
                    random_walk_corner_freq=Hertz(value),
                )
            else:
                # Unknown parameter, use base model
                model = base_model

            models.append(model)

        return models
