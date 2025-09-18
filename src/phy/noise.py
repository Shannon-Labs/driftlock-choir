"""
Noise models: AWGN, phase noise, and timing jitter.

This module provides various noise models commonly encountered in
wireless communication systems and timing synchronization.
"""

import numpy as np
from typing import Optional, Tuple
from dataclasses import dataclass


@dataclass
class NoiseParams:
    """Parameters for noise modeling."""
    snr_db: float           # Signal-to-noise ratio (dB)
    phase_noise_psd: float  # Phase noise power spectral density (dBc/Hz)
    jitter_rms: float       # RMS timing jitter (s)


class NoiseGenerator:
    """Generator for various types of noise in communication systems."""
    
    def __init__(self, params: NoiseParams, sample_rate: float = 1e6):
        self.params = params
        self.sample_rate = sample_rate
        
    def add_awgn(self, signal: np.ndarray) -> np.ndarray:
        """Add additive white Gaussian noise to signal."""
        signal_power = np.mean(np.abs(signal) ** 2)
        snr_linear = 10 ** (self.params.snr_db / 10)
        noise_power = signal_power / snr_linear
        
        if np.iscomplexobj(signal):
            noise = np.sqrt(noise_power / 2) * (np.random.randn(*signal.shape) + 
                                               1j * np.random.randn(*signal.shape))
        else:
            noise = np.sqrt(noise_power) * np.random.randn(*signal.shape)
            
        return signal + noise
        
    def generate_phase_noise(self, n_samples: int) -> np.ndarray:
        """Generate phase noise sequence."""
        # Simplified phase noise model
        phase_noise_var = 10 ** (self.params.phase_noise_psd / 10)
        return np.cumsum(np.sqrt(phase_noise_var) * np.random.randn(n_samples))
        
    def generate_timing_jitter(self, n_samples: int) -> np.ndarray:
        """Generate timing jitter sequence."""
        return self.params.jitter_rms * np.random.randn(n_samples)
