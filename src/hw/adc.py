"""
Sub-sampling ADC model with aperture jitter and ENOB effects.

This module implements realistic ADC behavior including quantization noise,
aperture jitter, and effective number of bits (ENOB) degradation.
"""

import numpy as np
from typing import Optional, Tuple
from dataclasses import dataclass


@dataclass
class ADCParams:
    """Parameters for ADC modeling."""
    n_bits: int              # Nominal bit resolution
    enob: float              # Effective number of bits
    aperture_jitter: float   # Aperture jitter RMS (s)
    full_scale_range: float  # Full-scale input range (V)
    sample_rate: float       # ADC sampling rate (Hz)


class ADCModel:
    """Model for analog-to-digital converter with realistic imperfections."""
    
    def __init__(self, n_bits: int, sample_rate: float, 
                 aperture_jitter: float = 1e-12, enob: Optional[float] = None):
        self.n_bits = n_bits
        self.sample_rate = sample_rate
        self.aperture_jitter = aperture_jitter
        self.enob = enob if enob is not None else n_bits - 1.5  # Typical ENOB
        
        # ADC parameters
        self.full_scale = 2.0  # Assume ±1V full scale
        self.lsb = self.full_scale / (2 ** self.n_bits)
        
        # Quantization noise power based on ENOB
        self.quant_noise_power = (self.lsb ** 2) / 12 * (2 ** (2 * (self.n_bits - self.enob)))
        
    def digitize(self, analog_signal: np.ndarray) -> np.ndarray:
        """Convert analog signal to digital with ADC imperfections."""
        # Apply aperture jitter
        if self.aperture_jitter > 0:
            jittered_signal = self._apply_aperture_jitter(analog_signal)
        else:
            jittered_signal = analog_signal
            
        # Add quantization noise
        quantized_signal = self._quantize(jittered_signal)
        
        return quantized_signal
        
    def _apply_aperture_jitter(self, signal: np.ndarray) -> np.ndarray:
        """Apply aperture jitter effects to the signal."""
        # Simplified model: add phase noise proportional to signal derivative
        if len(signal) > 1:
            # Compute signal derivative (approximation)
            derivative = np.diff(signal, prepend=signal[0])
            
            # Generate jitter sequence
            jitter = self.aperture_jitter * np.random.randn(len(signal))
            
            # Apply jitter as phase modulation
            jitter_effect = 2 * np.pi * self.sample_rate * jitter * np.abs(derivative)
            
            if np.iscomplexobj(signal):
                return signal * np.exp(1j * jitter_effect)
            else:
                return signal + jitter_effect
        else:
            return signal
            
    def _quantize(self, signal: np.ndarray) -> np.ndarray:
        """Apply quantization with finite resolution and ENOB effects."""
        # Clip to full-scale range
        clipped_signal = np.clip(signal, -self.full_scale/2, self.full_scale/2)
        
        # Quantize to n_bits resolution
        quantized = np.round(clipped_signal / self.lsb) * self.lsb
        
        # Add quantization noise based on ENOB
        if self.quant_noise_power > 0:
            if np.iscomplexobj(signal):
                quant_noise = np.sqrt(self.quant_noise_power / 2) * \
                             (np.random.randn(*signal.shape) + 1j * np.random.randn(*signal.shape))
            else:
                quant_noise = np.sqrt(self.quant_noise_power) * np.random.randn(*signal.shape)
                
            quantized += quant_noise
            
        return quantized
        
    def get_snr_limit(self) -> float:
        """Get theoretical SNR limit based on ENOB."""
        return 6.02 * self.enob + 1.76  # dB
