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
        
    def add_awgn(
        self,
        signal: np.ndarray,
        rng: Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        """Add additive white Gaussian noise to signal."""
        signal_power = np.mean(np.abs(signal) ** 2)
        snr_linear = 10 ** (self.params.snr_db / 10)
        noise_power = signal_power / snr_linear

        if rng is None:
            randn = np.random.randn
        else:
            randn = lambda *shape: rng.standard_normal(size=shape)

        if np.iscomplexobj(signal):
            noise_real = randn(*signal.shape)
            noise_imag = randn(*signal.shape)
            noise = np.sqrt(noise_power / 2) * (noise_real + 1j * noise_imag)
        else:
            noise = np.sqrt(noise_power) * randn(*signal.shape)

        return signal + noise
        
    def generate_phase_noise(self, n_samples: int) -> np.ndarray:
        """Generate phase noise sequence."""
        # Simplified phase noise model
        phase_noise_var = 10 ** (self.params.phase_noise_psd / 10)
        return np.cumsum(np.sqrt(phase_noise_var) * np.random.randn(n_samples))
        
    def generate_timing_jitter(self, n_samples: int) -> np.ndarray:
        """Generate timing jitter sequence."""
        return self.params.jitter_rms * np.random.randn(n_samples)
    
    def integrated_phase_variance(
        self,
        beat_bw_hz: float,
        carrier_freq_hz: float
    ) -> float:
        """
        Compute integrated phase variance over beat bandwidth.
        
        Includes white phase noise and timing jitter contributions.
        
        Args:
            beat_bw_hz: Beat signal bandwidth (Hz)
            carrier_freq_hz: Carrier frequency (Hz)
            
        Returns:
            Total phase variance σ_φ² (rad²)
        """
        # Phase noise PSD to linear (assuming dBc/Hz converted to rad²/Hz)
        # Note: dBc/Hz is for power, S_φ(f) ≈ 10^(PSD/10) for approximation
        psd_linear = 10 ** (self.params.phase_noise_psd / 10)
        
        # Integrated white phase noise variance: PSD * BW (for flat spectrum approximation)
        var_phase_noise = psd_linear * beat_bw_hz
        
        # Timing jitter contribution: σ_φ_jitter = 2π f_c σ_jitter
        var_phase_jitter = (2 * np.pi * carrier_freq_hz * self.params.jitter_rms) ** 2
        
        # Total phase variance
        total_var = var_phase_noise + var_phase_jitter
        
        return float(total_var)
    
    def phase_noise_std(self, beat_bw_hz: float, carrier_freq_hz: float) -> float:
        """Convenience: standard deviation of phase noise."""
        return np.sqrt(self.integrated_phase_variance(beat_bw_hz, carrier_freq_hz))
