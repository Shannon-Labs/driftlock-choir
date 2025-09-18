"""
Joint Cramér-Rao Lower Bound (CRLB) for τ (delay) and Δf (frequency offset).

This module computes theoretical performance bounds for joint estimation
of time delay and frequency offset parameters.
"""

import numpy as np
from typing import Tuple, Dict, Any, Optional
from dataclasses import dataclass
from scipy.linalg import inv, det


@dataclass
class CRLBParams:
    """Parameters for CRLB computation."""
    snr_db: float            # Signal-to-noise ratio (dB)
    bandwidth: float         # Signal bandwidth (Hz)
    duration: float          # Signal duration (s)
    carrier_freq: float      # Carrier frequency (Hz)
    sample_rate: float       # Sampling rate (Hz)
    pulse_shape: str = 'rect'  # Pulse shape ('rect', 'rrc', 'gaussian')


class JointCRLBCalculator:
    """Calculator for joint CRLB of delay and frequency offset."""
    
    def __init__(self, params: CRLBParams):
        self.params = params
        
    def compute_joint_crlb(self) -> Dict[str, Any]:
        """
        Compute joint CRLB for delay and frequency offset estimation.
        
        Returns:
            Dictionary with CRLB values and Fisher Information Matrix
        """
        # Convert SNR to linear scale
        snr_linear = 10 ** (self.params.snr_db / 10)
        
        # Compute Fisher Information Matrix
        fim = self._compute_fisher_information_matrix(snr_linear)
        
        # CRLB is inverse of Fisher Information Matrix
        try:
            crlb_matrix = inv(fim)
            
            # Extract individual CRLBs
            delay_crlb = crlb_matrix[0, 0]
            freq_crlb = crlb_matrix[1, 1]
            cross_term = crlb_matrix[0, 1]
            
        except np.linalg.LinAlgError:
            # Singular matrix - parameters not jointly estimable
            delay_crlb = np.inf
            freq_crlb = np.inf
            cross_term = 0
            crlb_matrix = np.full((2, 2), np.inf)
            
        return {
            'delay_crlb_variance': delay_crlb,
            'frequency_crlb_variance': freq_crlb,
            'delay_crlb_std': np.sqrt(delay_crlb) if delay_crlb >= 0 else np.inf,
            'frequency_crlb_std': np.sqrt(freq_crlb) if freq_crlb >= 0 else np.inf,
            'cross_correlation': cross_term / np.sqrt(delay_crlb * freq_crlb) 
                               if delay_crlb > 0 and freq_crlb > 0 else 0,
            'fisher_information_matrix': fim,
            'crlb_matrix': crlb_matrix,
            'determinant_fim': det(fim) if not np.any(np.isinf(fim)) else 0
        }
        
    def _compute_fisher_information_matrix(self, snr_linear: float) -> np.ndarray:
        """Compute Fisher Information Matrix for joint estimation."""
        # Number of samples
        n_samples = int(self.params.duration * self.params.sample_rate)
        
        # Time vector
        t = np.arange(n_samples) / self.params.sample_rate
        
        # Signal model derivatives
        # For exponential signal: s(t) = exp(j*2*pi*f*t)
        # Partial derivatives w.r.t. delay (τ) and frequency offset (Δf)
        
        # Derivative w.r.t. delay: ∂s/∂τ = j*2*pi*f*s(t)
        ds_dtau = 1j * 2 * np.pi * self.params.carrier_freq
        
        # Derivative w.r.t. frequency: ∂s/∂Δf = j*2*pi*t*s(t)
        ds_dfreq = 1j * 2 * np.pi * t
        
        # Fisher Information Matrix elements
        # FIM[i,j] = 2*SNR*Re{∫ (∂s/∂θᵢ)* (∂s/∂θⱼ) dt}
        
        # FIM[0,0] - delay, delay
        fim_00 = 2 * snr_linear * np.real(np.conj(ds_dtau) * ds_dtau) * n_samples
        
        # FIM[1,1] - frequency, frequency  
        fim_11 = 2 * snr_linear * np.real(np.sum(np.conj(ds_dfreq) * ds_dfreq)) / self.params.sample_rate
        
        # FIM[0,1] = FIM[1,0] - delay, frequency cross-term
        fim_01 = 2 * snr_linear * np.real(np.sum(np.conj(ds_dtau) * ds_dfreq)) / self.params.sample_rate
        
        # Construct Fisher Information Matrix
        fim = np.array([
            [fim_00, fim_01],
            [fim_01, fim_11]
        ])
        
        return fim
        
    def compute_crlb_vs_snr(self, snr_range_db: np.ndarray) -> Dict[str, np.ndarray]:
        """Compute CRLB as function of SNR."""
        delay_crlb = np.zeros_like(snr_range_db)
        freq_crlb = np.zeros_like(snr_range_db)
        correlation = np.zeros_like(snr_range_db)
        
        original_snr = self.params.snr_db
        
        for i, snr_db in enumerate(snr_range_db):
            self.params.snr_db = snr_db
            results = self.compute_joint_crlb()
            
            delay_crlb[i] = results['delay_crlb_std']
            freq_crlb[i] = results['frequency_crlb_std']
            correlation[i] = results['cross_correlation']
            
        # Restore original SNR
        self.params.snr_db = original_snr
        
        return {
            'snr_db': snr_range_db,
            'delay_crlb_std': delay_crlb,
            'frequency_crlb_std': freq_crlb,
            'cross_correlation': correlation
        }
        
    def compute_crlb_vs_bandwidth(self, bandwidth_range: np.ndarray) -> Dict[str, np.ndarray]:
        """Compute CRLB as function of signal bandwidth."""
        delay_crlb = np.zeros_like(bandwidth_range)
        freq_crlb = np.zeros_like(bandwidth_range)
        
        original_bw = self.params.bandwidth
        
        for i, bw in enumerate(bandwidth_range):
            self.params.bandwidth = bw
            results = self.compute_joint_crlb()
            
            delay_crlb[i] = results['delay_crlb_std']
            freq_crlb[i] = results['frequency_crlb_std']
            
        # Restore original bandwidth
        self.params.bandwidth = original_bw
        
        return {
            'bandwidth': bandwidth_range,
            'delay_crlb_std': delay_crlb,
            'frequency_crlb_std': freq_crlb
        }
        
    def compute_crlb_vs_duration(self, duration_range: np.ndarray) -> Dict[str, np.ndarray]:
        """Compute CRLB as function of observation duration."""
        delay_crlb = np.zeros_like(duration_range)
        freq_crlb = np.zeros_like(duration_range)
        
        original_duration = self.params.duration
        
        for i, duration in enumerate(duration_range):
            self.params.duration = duration
            results = self.compute_joint_crlb()
            
            delay_crlb[i] = results['delay_crlb_std']
            freq_crlb[i] = results['frequency_crlb_std']
            
        # Restore original duration
        self.params.duration = original_duration
        
        return {
            'duration': duration_range,
            'delay_crlb_std': delay_crlb,
            'frequency_crlb_std': freq_crlb
        }


class AdvancedCRLBAnalysis:
    """Advanced CRLB analysis including multipath and interference effects."""
    
    def __init__(self, params: CRLBParams):
        self.params = params
        self.basic_calculator = JointCRLBCalculator(params)
        
    def compute_multipath_crlb(self, path_gains: np.ndarray, 
                              path_delays: np.ndarray) -> Dict[str, Any]:
        """
        Compute CRLB in multipath environment.
        
        Args:
            path_gains: Complex gains of multipath components
            path_delays: Delays of multipath components (s)
            
        Returns:
            CRLB results accounting for multipath
        """
        # Simplified multipath CRLB computation
        # In practice, this would involve more complex signal models
        
        # Effective SNR reduction due to multipath
        total_power = np.sum(np.abs(path_gains) ** 2)
        direct_power = np.abs(path_gains[0]) ** 2
        multipath_factor = direct_power / total_power
        
        # Adjust SNR
        effective_snr_db = self.params.snr_db + 10 * np.log10(multipath_factor)
        
        # Compute CRLB with adjusted SNR
        original_snr = self.params.snr_db
        self.params.snr_db = effective_snr_db
        
        results = self.basic_calculator.compute_joint_crlb()
        
        # Restore original SNR
        self.params.snr_db = original_snr
        
        # Add multipath-specific metrics
        results['multipath_factor'] = multipath_factor
        results['effective_snr_db'] = effective_snr_db
        results['rms_delay_spread'] = np.sqrt(np.sum(np.abs(path_gains) ** 2 * path_delays ** 2) / 
                                            np.sum(np.abs(path_gains) ** 2))
        
        return results
        
    def compute_interference_crlb(self, interference_power: float) -> Dict[str, Any]:
        """
        Compute CRLB in presence of interference.
        
        Args:
            interference_power: Interference power relative to signal power (linear)
            
        Returns:
            CRLB results accounting for interference
        """
        # Effective SINR
        snr_linear = 10 ** (self.params.snr_db / 10)
        sinr_linear = snr_linear / (1 + interference_power)
        sinr_db = 10 * np.log10(sinr_linear)
        
        # Compute CRLB with adjusted SINR
        original_snr = self.params.snr_db
        self.params.snr_db = sinr_db
        
        results = self.basic_calculator.compute_joint_crlb()
        
        # Restore original SNR
        self.params.snr_db = original_snr
        
        # Add interference-specific metrics
        results['interference_power'] = interference_power
        results['sinr_db'] = sinr_db
        results['sinr_degradation_db'] = self.params.snr_db - sinr_db
        
        return results
        
    def compute_efficiency_analysis(self, estimator_performance: Dict[str, float]) -> Dict[str, Any]:
        """
        Analyze estimator efficiency compared to CRLB.
        
        Args:
            estimator_performance: Dictionary with 'delay_mse' and 'frequency_mse'
            
        Returns:
            Efficiency analysis results
        """
        # Compute theoretical CRLB
        crlb_results = self.basic_calculator.compute_joint_crlb()
        
        # Calculate efficiencies
        delay_efficiency = (crlb_results['delay_crlb_variance'] / 
                          estimator_performance['delay_mse'] if estimator_performance['delay_mse'] > 0 else 0)
        
        freq_efficiency = (crlb_results['frequency_crlb_variance'] / 
                         estimator_performance['frequency_mse'] if estimator_performance['frequency_mse'] > 0 else 0)
        
        # Overall efficiency (geometric mean)
        overall_efficiency = np.sqrt(delay_efficiency * freq_efficiency)
        
        return {
            'crlb_results': crlb_results,
            'estimator_performance': estimator_performance,
            'delay_efficiency': delay_efficiency,
            'frequency_efficiency': freq_efficiency,
            'overall_efficiency': overall_efficiency,
            'delay_loss_db': -10 * np.log10(delay_efficiency) if delay_efficiency > 0 else np.inf,
            'frequency_loss_db': -10 * np.log10(freq_efficiency) if freq_efficiency > 0 else np.inf,
            'is_efficient': (delay_efficiency > 0.8 and freq_efficiency > 0.8),
            'performance_gap': {
                'delay_gap_db': 10 * np.log10(estimator_performance['delay_mse'] / 
                                             crlb_results['delay_crlb_variance'])
                               if crlb_results['delay_crlb_variance'] > 0 else np.inf,
                'frequency_gap_db': 10 * np.log10(estimator_performance['frequency_mse'] / 
                                                 crlb_results['frequency_crlb_variance'])
                                   if crlb_results['frequency_crlb_variance'] > 0 else np.inf
            }
        }
