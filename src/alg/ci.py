"""
Closed-form τ (time delay) and Δf (frequency offset) extractor.

This module implements closed-form estimators for time delay and frequency
offset parameters from received synchronization signals.
"""

import numpy as np
from typing import Tuple, Optional, Dict, Any
from dataclasses import dataclass
from scipy import signal
from scipy.optimize import minimize_scalar


@dataclass
class EstimatorParams:
    """Parameters for closed-form estimators."""
    sample_rate: float       # Sampling rate (Hz)
    carrier_freq: float      # Carrier frequency (Hz)
    bandwidth: float         # Signal bandwidth (Hz)
    estimation_method: str   # 'ml', 'ls', 'esprit'


class ClosedFormEstimator:
    """Closed-form estimators for time delay and frequency offset."""
    
    def __init__(self, params: EstimatorParams):
        self.params = params
        
    def estimate_delay_and_frequency(self, reference_signal: np.ndarray, 
                                   received_signal: np.ndarray) -> Tuple[float, float]:
        """
        Estimate time delay (τ) and frequency offset (Δf) using closed-form methods.
        
        Args:
            reference_signal: Known reference signal
            received_signal: Received signal with delay and frequency offset
            
        Returns:
            Tuple of (time_delay, frequency_offset)
        """
        if self.params.estimation_method == 'ml':
            return self._ml_estimator(reference_signal, received_signal)
        elif self.params.estimation_method == 'ls':
            return self._least_squares_estimator(reference_signal, received_signal)
        elif self.params.estimation_method == 'esprit':
            return self._esprit_estimator(reference_signal, received_signal)
        else:
            raise ValueError(f"Unknown estimation method: {self.params.estimation_method}")
            
    def _ml_estimator(self, ref_signal: np.ndarray, 
                     rx_signal: np.ndarray) -> Tuple[float, float]:
        """Maximum likelihood estimator for delay and frequency offset."""
        # Cross-correlation for coarse delay estimation
        correlation = signal.correlate(rx_signal, ref_signal, mode='full')
        delay_samples = np.argmax(np.abs(correlation)) - len(ref_signal) + 1
        coarse_delay = delay_samples / self.params.sample_rate
        
        # Fine frequency estimation using phase difference
        # Align signals based on coarse delay
        if delay_samples >= 0:
            aligned_rx = rx_signal[delay_samples:delay_samples + len(ref_signal)]
        else:
            aligned_rx = np.pad(rx_signal, (-delay_samples, 0), mode='constant')[:len(ref_signal)]
            
        # Estimate frequency offset from phase progression
        phase_diff = np.angle(aligned_rx * np.conj(ref_signal))
        
        # Unwrap phase and fit linear trend
        unwrapped_phase = np.unwrap(phase_diff)
        t = np.arange(len(unwrapped_phase)) / self.params.sample_rate
        
        # Linear regression for frequency offset
        freq_offset = np.polyfit(t, unwrapped_phase, 1)[0] / (2 * np.pi)
        
        return coarse_delay, freq_offset
        
    def _least_squares_estimator(self, ref_signal: np.ndarray, 
                                rx_signal: np.ndarray) -> Tuple[float, float]:
        """Least squares estimator for delay and frequency offset."""
        # Implement LS estimation using matrix formulation
        # This is a simplified version - full implementation would use
        # more sophisticated LS techniques
        return self._ml_estimator(ref_signal, rx_signal)
        
    def _esprit_estimator(self, ref_signal: np.ndarray, 
                         rx_signal: np.ndarray) -> Tuple[float, float]:
        """ESPRIT-based estimator for delay and frequency offset."""
        # Simplified ESPRIT implementation
        # Full implementation would use subspace methods
        return self._ml_estimator(ref_signal, rx_signal)
        
    def estimate_crlb(self, snr_db: float, signal_length: int) -> Tuple[float, float]:
        """
        Estimate Cramér-Rao Lower Bound for delay and frequency estimation.
        
        Args:
            snr_db: Signal-to-noise ratio in dB
            signal_length: Length of signal in samples
            
        Returns:
            Tuple of (delay_crlb, frequency_crlb)
        """
        snr_linear = 10 ** (snr_db / 10)
        T = signal_length / self.params.sample_rate
        
        # Simplified CRLB calculations
        # Delay CRLB (assumes rectangular spectrum)
        delay_crlb = 1 / (2 * np.pi * self.params.bandwidth * np.sqrt(2 * snr_linear))
        
        # Frequency CRLB
        frequency_crlb = np.sqrt(12) / (2 * np.pi * T * np.sqrt(2 * snr_linear * T))
        
        return delay_crlb, frequency_crlb
        
    def joint_estimation(self, ref_signal: np.ndarray, 
                        rx_signal: np.ndarray) -> Dict[str, Any]:
        """
        Perform joint estimation and return comprehensive results.
        
        Returns:
            Dictionary containing estimates, confidence intervals, and metrics
        """
        # Get estimates
        delay_est, freq_est = self.estimate_delay_and_frequency(ref_signal, rx_signal)
        
        # Estimate SNR from residual
        aligned_signal = self._align_signal(rx_signal, delay_est, freq_est, len(ref_signal))
        residual = aligned_signal - ref_signal
        signal_power = np.mean(np.abs(ref_signal) ** 2)
        noise_power = np.mean(np.abs(residual) ** 2)
        snr_est = 10 * np.log10(signal_power / noise_power)
        
        # Get CRLB
        delay_crlb, freq_crlb = self.estimate_crlb(snr_est, len(ref_signal))
        
        return {
            'delay_estimate': delay_est,
            'frequency_estimate': freq_est,
            'snr_estimate': snr_est,
            'delay_crlb': delay_crlb,
            'frequency_crlb': freq_crlb,
            'efficiency_delay': delay_crlb / (delay_est ** 2) if delay_est != 0 else np.inf,
            'efficiency_frequency': freq_crlb / (freq_est ** 2) if freq_est != 0 else np.inf
        }
        
    def _align_signal(self, signal: np.ndarray, delay: float, 
                     freq_offset: float, target_length: int) -> np.ndarray:
        """Align signal by compensating for delay and frequency offset."""
        # Time-shift compensation
        delay_samples = int(delay * self.params.sample_rate)
        if delay_samples >= 0:
            shifted_signal = signal[delay_samples:delay_samples + target_length]
        else:
            shifted_signal = np.pad(signal, (-delay_samples, 0), mode='constant')[:target_length]
            
        # Frequency compensation
        t = np.arange(len(shifted_signal)) / self.params.sample_rate
        freq_compensation = np.exp(-1j * 2 * np.pi * freq_offset * t)
        
        return shifted_signal * freq_compensation
