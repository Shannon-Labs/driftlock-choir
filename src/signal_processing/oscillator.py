"""
Oscillator models and signal generation for Driftlock Choir.
"""

import numpy as np
from typing import Optional, Tuple
from ..core.types import (
    OscillatorModel, Hertz, Seconds, Picoseconds, Timestamp,
    MeasurementQuality, Frequency
)
from ..core.constants import PhysicalConstants
from .phase_noise import PhaseNoiseModel


class Oscillator:
    """
    Generates oscillator signals with realistic phase noise and impairments.
    
    This class implements various oscillator models including TCXO, OCXO,
    and custom phase noise profiles for chronometric interferometry experiments.
    """
    
    def __init__(self, model: OscillatorModel, initial_phase: float = 0.0):
        """
        Initialize oscillator with model parameters.
        
        Args:
            model: Oscillator model specification
            initial_phase: Initial phase in radians
        """
        self.model = model
        self.initial_phase = initial_phase
        self._phase_noise_generator = None
        self._current_time = 0.0
        self._phase_noise_model = None
        
    def generate_signal(self, 
                       duration: Seconds, 
                       sampling_rate: Hertz,
                       frequency_offset: Hertz = 0.0,
                       phase_noise_enabled: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate oscillator signal with optional phase noise.
        
        Args:
            duration: Signal duration in seconds
            sampling_rate: Sampling rate in Hz
            frequency_offset: Frequency offset from nominal in Hz
            phase_noise_enabled: Whether to add phase noise
            
        Returns:
            Tuple of (time_vector, complex_signal)
        """
        n_samples = int(duration * sampling_rate)
        t = np.arange(n_samples) / sampling_rate
        
        # Base frequency with offset
        freq = self.model.nominal_frequency + frequency_offset
        
        # Generate phase with noise if enabled
        if phase_noise_enabled:
            phase = self._generate_phase_with_noise(t, freq)
        else:
            phase = 2 * np.pi * freq * t + self.initial_phase
        
        # Generate complex signal
        signal = np.exp(1j * phase)
        
        return t, signal
    
    def _generate_phase_with_noise(self, t: np.ndarray, freq: Hertz) -> np.ndarray:
        """
        Generate phase with realistic phase noise.
        
        Args:
            t: Time vector
            freq: Base frequency
            
        Returns:
            Phase array with noise
        """
        # Base phase
        phase = 2 * np.pi * freq * t + self.initial_phase
        
        # Add phase noise based on profile
        if self.model.phase_noise_profile:
            phase_noise = self._generate_phase_noise_from_profile(t)
            phase += phase_noise
        
        # Add temperature effects
        if self.model.temperature_coefficient != 0:
            temp_drift = self._generate_temperature_drift(t)
            phase += 2 * np.pi * freq * temp_drift * self.model.temperature_coefficient * 1e-6
        
        # Add aging effects
        if self.model.aging_rate != 0:
            aging_drift = self._generate_aging_drift(t)
            phase += 2 * np.pi * freq * aging_drift * 1e-9
        
        return phase
    
    def _generate_phase_noise_from_profile(self, t: np.ndarray) -> np.ndarray:
        """
        Generate phase noise from oscillator profile.
        
        Args:
            t: Time vector
            
        Returns:
            Phase noise array
        """
        # If we have a phase noise model, use it
        if self._phase_noise_model:
            return self._phase_noise_model.generate_phase_noise_time_series(
                t[-1] - t[0], 1.0 / (t[1] - t[0]) if len(t) > 1 else 1.0
            )
        
        # Otherwise, use the original profile-based method
        n_samples = len(t)
        dt = t[1] - t[0] if len(t) > 1 else 1.0
        fs = 1.0 / dt
        
        # Generate frequency domain noise
        freqs = np.fft.fftfreq(n_samples, dt)
        
        # Create phase noise spectrum
        phase_noise_spectrum = np.zeros_like(freqs)
        
        for offset_freq, noise_level in self.model.phase_noise_profile.items():
            # Find closest frequency bin
            idx = np.argmin(np.abs(freqs - offset_freq))
            if idx < len(phase_noise_spectrum):
                # Convert dBc/Hz to linear power spectral density
                psd = 10 ** (noise_level / 10)
                phase_noise_spectrum[idx] = psd
        
        # Interpolate across frequency range
        # Use log-log interpolation for realistic phase noise
        positive_freqs = freqs[freqs > 0]
        if len(positive_freqs) > 1:
            positive_psd = phase_noise_spectrum[freqs > 0]
            
            # Remove zeros for log interpolation
            mask = positive_psd > 0
            if np.any(mask):
                log_freqs = np.log10(positive_freqs[mask])
                log_psd = np.log10(positive_psd[mask])
                
                # Interpolate in log-log space
                all_log_freqs = np.log10(np.abs(freqs[freqs > 0]))
                interpolated_log_psd = np.interp(all_log_freqs, log_freqs, log_psd,
                                               left=log_psd[0], right=log_psd[-1])
                
                phase_noise_spectrum[freqs > 0] = 10 ** interpolated_log_psd
                
                # Make symmetric for negative frequencies
                phase_noise_spectrum[freqs < 0] = phase_noise_spectrum[np.abs(freqs) > 0][::-1]
        
        # Generate random phase noise
        # Create complex Gaussian noise with specified PSD
        noise_complex = np.random.normal(0, 1, n_samples) + 1j * np.random.normal(0, 1, n_samples)
        noise_freq = np.fft.fft(noise_complex)
        
        # Apply phase noise spectrum
        shaped_noise_freq = noise_freq * np.sqrt(phase_noise_spectrum * fs / 2)
        phase_noise = np.real(np.fft.ifft(shaped_noise_freq))
        
        return phase_noise
    
    def _generate_temperature_drift(self, t: np.ndarray) -> np.ndarray:
        """
        Generate temperature-induced frequency drift.
        
        Args:
            t: Time vector
            
        Returns:
            Relative frequency change due to temperature
        """
        # Simple model: sinusoidal temperature variation
        # In reality, this would be more complex
        temp_variation = 5.0 * np.sin(2 * np.pi * 1e-4 * t)  # 0.0001 Hz temperature variation
        return temp_variation
    
    def _generate_aging_drift(self, t: np.ndarray) -> np.ndarray:
        """
        Generate aging-induced frequency drift.
        
        Args:
            t: Time vector
            
        Returns:
            Relative frequency change due to aging
        """
        # Linear aging model
        # Convert aging rate from ppb/day to ppb/second
        aging_rate_ppb_per_sec = self.model.aging_rate / 86400.0
        return aging_rate_ppb_per_sec * t
    
    def get_frequency_at_time(self, t: Seconds) -> Frequency:
        """
        Get instantaneous frequency at time t.
        
        Args:
            t: Time in seconds
            
        Returns:
            Frequency object with uncertainty
        """
        # Base frequency
        freq = self.model.nominal_frequency
        
        # Add temperature effect
        temp_effect = freq * self.model.temperature_coefficient * 1e-6 * \
                     np.sin(2 * np.pi * 1e-4 * t)
        
        # Add aging effect
        aging_effect = freq * self.model.aging_rate * 1e-9 * t / 86400.0
        
        actual_freq = freq + temp_effect + aging_effect
        
        # Estimate uncertainty (simplified)
        uncertainty = abs(temp_effect) * 0.1 + abs(aging_effect) * 0.1
        
        return Frequency(
            freq=Hertz(actual_freq),
            uncertainty=Hertz(uncertainty),
            quality=MeasurementQuality.GOOD
        )
    
    @classmethod
    def create_tcxo_model(cls, 
                         nominal_freq: Hertz,
                         temperature_coeff: float = 0.5,
                         aging_rate: float = 1.0) -> OscillatorModel:
        """
        Create a TCXO (Temperature-Compensated Crystal Oscillator) model.
        
        Args:
            nominal_freq: Nominal frequency in Hz
            temperature_coeff: Temperature coefficient in ppm/°C
            aging_rate: Aging rate in ppb/day
            
        Returns:
            TCXO oscillator model
        """
        # Typical TCXO phase noise profile
        phase_noise_profile = {
            1.0: -80,    # -80 dBc/Hz at 1 Hz offset
            10.0: -100,  # -100 dBc/Hz at 10 Hz offset
            100.0: -120, # -120 dBc/Hz at 100 Hz offset
            1000.0: -130, # -130 dBc/Hz at 1 kHz offset
            10000.0: -140, # -140 dBc/Hz at 10 kHz offset
        }
        
        return OscillatorModel(
            nominal_frequency=nominal_freq,
            phase_noise_profile=phase_noise_profile,
            temperature_coefficient=temperature_coeff,
            aging_rate=aging_rate,
            current_temperature=25.0,
            current_age_days=0.0
        )
    
    @classmethod
    def create_ocxo_model(cls, 
                         nominal_freq: Hertz,
                         temperature_coeff: float = 0.01,
                         aging_rate: float = 0.1) -> OscillatorModel:
        """
        Create an OCXO (Oven-Controlled Crystal Oscillator) model.
        
        Args:
            nominal_freq: Nominal frequency in Hz
            temperature_coeff: Temperature coefficient in ppm/°C
            aging_rate: Aging rate in ppb/day
            
        Returns:
            OCXO oscillator model
        """
        # Typical OCXO phase noise profile (better than TCXO)
        phase_noise_profile = {
            1.0: -90,    # -90 dBc/Hz at 1 Hz offset
            10.0: -110,  # -110 dBc/Hz at 10 Hz offset
            100.0: -130, # -130 dBc/Hz at 100 Hz offset
            1000.0: -140, # -140 dBc/Hz at 1 kHz offset
            10000.0: -150, # -150 dBc/Hz at 10 kHz offset
        }
        
        return OscillatorModel(
            nominal_frequency=nominal_freq,
            phase_noise_profile=phase_noise_profile,
            temperature_coefficient=temperature_coeff,
            aging_rate=aging_rate,
            current_temperature=25.0,
            current_age_days=0.0
        )
    
    @classmethod
    def create_ideal_oscillator(cls, nominal_freq: Hertz) -> OscillatorModel:
        """
        Create an ideal oscillator model (no noise, no drift).
        
        Args:
            nominal_freq: Nominal frequency in Hz
            
        Returns:
            Ideal oscillator model
        """
        return OscillatorModel(
            nominal_frequency=nominal_freq,
            phase_noise_profile={},  # No phase noise
            temperature_coefficient=0.0,  # No temperature sensitivity
            aging_rate=0.0,  # No aging
            current_temperature=25.0,
            current_age_days=0.0
        )
    
    def set_phase_noise_model(self, phase_noise_model: PhaseNoiseModel):
        """
        Set a custom phase noise model.
        
        Args:
            phase_noise_model: Phase noise model to use
        """
        self._phase_noise_model = phase_noise_model
    
    def get_phase_noise_model(self) -> Optional[PhaseNoiseModel]:
        """
        Get the current phase noise model.
        
        Returns:
            Current phase noise model or None
        """
        return self._phase_noise_model
    
    def calculate_jitter(self,
                         duration: Seconds = 1.0,
                         integration_bandwidth: Tuple[Hertz, Hertz] = (1.0, 10000.0)) -> Picoseconds:
        """
        Calculate RMS jitter for the oscillator.
        
        Args:
            duration: Duration for jitter calculation
            integration_bandwidth: Frequency range for integration
            
        Returns:
            RMS jitter in picoseconds
        """
        if self._phase_noise_model:
            return self._phase_noise_model.calculate_jitter(
                duration, self.model.nominal_frequency, integration_bandwidth
            )
        else:
            # Simplified jitter calculation from phase noise profile
            if not self.model.phase_noise_profile:
                return Picoseconds(0.0)  # No jitter for ideal oscillator
            
            # Use the worst-case noise level as approximation
            worst_case_noise = max(self.model.phase_noise_profile.values())
            jitter_seconds = 10 ** (worst_case_noise / 20) / (2 * np.pi * self.model.nominal_frequency)
            return PhysicalConstants.seconds_to_ps(jitter_seconds)
    
    @classmethod
    def create_with_phase_noise_model(cls,
                                     nominal_freq: Hertz,
                                     phase_noise_model: PhaseNoiseModel,
                                     temperature_coeff: float = 0.0,
                                     aging_rate: float = 0.0) -> 'Oscillator':
        """
        Create oscillator with custom phase noise model.
        
        Args:
            nominal_freq: Nominal frequency
            phase_noise_model: Phase noise model
            temperature_coeff: Temperature coefficient
            aging_rate: Aging rate
            
        Returns:
            Oscillator with custom phase noise model
        """
        # Create oscillator model
        model = OscillatorModel(
            nominal_frequency=nominal_freq,
            phase_noise_profile=phase_noise_model.get_phase_noise_profile(),
            temperature_coefficient=temperature_coeff,
            aging_rate=aging_rate,
            current_temperature=25.0,
            current_age_days=0.0
        )
        
        # Create oscillator
        oscillator = cls(model)
        oscillator.set_phase_noise_model(phase_noise_model)
        
        return oscillator