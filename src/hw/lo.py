"""
Unlocked Local Oscillator (LO) drift model.

This module implements realistic local oscillator behavior including
frequency drift, phase noise, and temperature-dependent variations.
"""

import numpy as np
from typing import Optional
from dataclasses import dataclass

from phy.osc import OscillatorParams, AllanDeviationGenerator


@dataclass
class LOConfig:
    """Configuration for local oscillator model."""
    nominal_freq: float      # Nominal LO frequency (Hz)
    temp_coeff: float        # Temperature coefficient (ppm/°C)
    aging_rate: float        # Aging rate (ppm/year)
    initial_offset: float    # Initial frequency offset (ppm)


class LocalOscillator:
    """Model for unlocked local oscillator with realistic drift characteristics."""
    
    def __init__(self, nominal_freq: float, osc_params: OscillatorParams, 
                 config: Optional[LOConfig] = None):
        self.nominal_freq = nominal_freq
        self.osc_params = osc_params
        self.config = config or LOConfig(nominal_freq, 2.5, 1e-7, 0.0)
        
        # Initialize oscillator noise generator
        self.noise_gen = AllanDeviationGenerator(osc_params)
        
        # State variables
        self.current_phase = 0.0
        self.current_freq_offset = self.config.initial_offset * 1e-6
        self.last_update_time = 0.0
        
    def update_state(self, current_time: float, temperature: float = 25.0):
        """Update oscillator state based on current time and temperature."""
        dt = current_time - self.last_update_time
        
        if dt > 0:
            # Apply temperature drift
            temp_drift = self.config.temp_coeff * 1e-6 * (temperature - 25.0)
            
            # Apply aging
            aging_drift = self.config.aging_rate * dt / (365.25 * 24 * 3600)
            
            # Add Allan deviation noise
            phase_noise = self.noise_gen.generate_phase_noise(dt)
            freq_noise = self.noise_gen.generate_frequency_drift(dt)
            
            # Update frequency offset
            self.current_freq_offset += temp_drift + aging_drift + freq_noise[-1]
            
            # Update phase (integrate frequency offset)
            self.current_phase += 2 * np.pi * self.current_freq_offset * self.nominal_freq * dt
            self.current_phase += phase_noise[-1]
            
        self.last_update_time = current_time
        
    def get_phase_at_time(self, timestamp: float, n_samples: int) -> np.ndarray:
        """Get phase sequence at specified timestamp."""
        self.update_state(timestamp)
        
        # Generate phase sequence for the duration
        sample_rate = 1e6  # Assume 1 MHz sample rate
        dt = 1.0 / sample_rate
        t = np.arange(n_samples) * dt
        
        # Phase evolution includes both frequency offset and phase noise
        freq_offset_phase = 2 * np.pi * self.current_freq_offset * self.nominal_freq * t
        phase_noise = self.noise_gen.generate_phase_noise(n_samples / sample_rate)
        
        return self.current_phase + freq_offset_phase + phase_noise
        
    def get_frequency_offset(self) -> float:
        """Get current frequency offset in Hz."""
        return self.current_freq_offset * self.nominal_freq
        
    def get_phase_offset(self) -> float:
        """Get current phase offset in radians."""
        return self.current_phase
