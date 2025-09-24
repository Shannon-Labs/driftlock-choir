"""
Joint Cramér-Rao Lower Bound (CRLB) for τ (delay) and Δf (frequency offset).

This module computes theoretical performance bounds for joint estimation
of time delay and frequency offset parameters.
"""

import numpy as np
from typing import Tuple, Dict, Any, Optional, List
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
    carrier_frequencies: Optional[List[float]] = None  # For multi-frequency CRLB
    sigma_phase_rad: float = 0.1  # Phase noise std dev (rad)
    filter_bw_hz: Optional[float] = None  # Bandwidth of estimator's BPF


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
        
        # CRLB via closed-form inverse for 2×2 symmetric matrix [[a,b],[b,c]]
        a, b, c = float(fim[0, 0]), float(fim[0, 1]), float(fim[1, 1])
        determinant = a * c - b * b
        if determinant <= 0 or not np.isfinite(determinant):
            delay_crlb = np.inf
            freq_crlb = np.inf
            cross_term = 0.0
            crlb_matrix = np.full((2, 2), np.inf)
        else:
            inv_det = 1.0 / determinant
            crlb_matrix = np.array([[c * inv_det, -b * inv_det],
                                    [-b * inv_det, a * inv_det]], dtype=float)
            delay_crlb = crlb_matrix[0, 0]
            freq_crlb = crlb_matrix[1, 1]
            cross_term = crlb_matrix[0, 1]
            
        return {
            'delay_crlb_variance': delay_crlb,
            'frequency_crlb_variance': freq_crlb,
            'delay_crlb_std': np.sqrt(delay_crlb) if delay_crlb >= 0 else np.inf,
            'frequency_crlb_std': np.sqrt(freq_crlb) if freq_crlb >= 0 else np.inf,
            'cross_correlation': cross_term / np.sqrt(delay_crlb * freq_crlb) 
                               if delay_crlb > 0 and freq_crlb > 0 else 0,
            'fisher_information_matrix': fim,
            'crlb_matrix': crlb_matrix,
            'determinant_fim': float(determinant)
        }
        
    def _compute_fisher_information_matrix(self, snr_linear: float) -> np.ndarray:
        """Compute Fisher Information Matrix for joint estimation.

        Uses discrete-time FIM aligned to the estimator model:
        s[n] = exp(j*φ_n), φ_n = 2πΔf*t_n + θ - 2πfc*τ
        FIM = (2/σ²) Σ (∂φ/∂θ_i)(∂φ/∂θ_j)
        """
        # Number of samples
        n_samples = int(self.params.duration * self.params.sample_rate)

        # Time vector (discrete-time)
        t = np.arange(n_samples) / self.params.sample_rate

        # Signal model derivatives aligned to estimator
        # ∂φ/∂τ = -2πfc (from -2πfc*τ term)
        dphi_dtau = -2 * np.pi * self.params.carrier_freq

        # ∂φ/∂Δf = 2π*t (from 2πΔf*t term)
        dphi_ddeltaf = 2 * np.pi * t

        # For complex Gaussian noise with variance σ² = 1/(2*SNR_linear)
        # per sample (post-filtering), the FIM becomes:
        # FIM = (2/σ²) Σ (∂φ/∂θ_i)(∂φ/∂θ_j)
        # where σ² is the per-sample noise variance

        # RESEARCH-GRADE UPDATE: Calculate noise variance considering BPF
        # The original CRLB calculation was flawed because it used the wideband
        # SNR, while the estimator benefits from a narrow bandpass filter.
        # This corrected calculation provides a fair, "apples-to-apples" bound.
        wideband_noise_power = 1.0 / snr_linear
        if self.params.filter_bw_hz is not None and self.params.sample_rate > 0:
            # Noise Power Spectral Density (PSD) over the full sample rate
            noise_psd = wideband_noise_power / self.params.sample_rate
            # Noise power within the filter's bandwidth
            filtered_noise_power = noise_psd * self.params.filter_bw_hz
            noise_var = filtered_noise_power
        else:
            # Fallback to original wideband calculation if filter BW not provided
            noise_var = wideband_noise_power

        # Compute sums in float64
        S1 = float(np.sum(dphi_dtau ** 2))          # constant over n
        S2 = float(np.sum(dphi_ddeltaf ** 2))
        S12 = float(np.sum(dphi_dtau * dphi_ddeltaf))

        K = 2.0 / noise_var  # scaling factor

        a = K * S1
        b = K * S12
        c = K * S2

        # Return 2x2 matrix explicitly
        fim = np.array([[a, b], [b, c]], dtype=float)
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

            # Handle infinity values gracefully
            delay_val = results['delay_crlb_std']
            freq_val = results['frequency_crlb_std']
            corr_val = results['cross_correlation']

            delay_crlb[i] = delay_val if np.isfinite(delay_val) else 1e12  # Large finite value
            freq_crlb[i] = freq_val if np.isfinite(freq_val) else 1e12    # Large finite value
            correlation[i] = corr_val if np.isfinite(corr_val) else 0.0
            
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

    def compute_crlb_from_residuals(self, phase_residuals: np.ndarray,
                                   time_vector: np.ndarray,
                                   carrier_freq: float) -> Dict[str, Any]:
        """
        Compute CRLB from LS residuals (model-consistent bound).

        This uses the actual LS residuals to estimate noise variance,
        providing a practical bound aligned to the estimator.

        Args:
            phase_residuals: Phase residuals from LS fit (radians)
            time_vector: Time samples used in fit (seconds)
            carrier_freq: Carrier frequency (Hz)

        Returns:
            Dictionary with model-consistent CRLB bounds
        """
        # Estimate noise variance from residuals
        # For complex noise, σ² = E[|residuals|²]/2
        noise_var = np.mean(np.abs(phase_residuals) ** 2) / 2

        # If noise variance is too small, use a minimum threshold
        min_noise_var = 1e-12  # Minimum noise variance
        noise_var = max(noise_var, min_noise_var)

        # Compute discrete-time FIM with estimated noise variance
        n_samples = len(time_vector)
        t = time_vector

        # Signal model derivatives (same as in estimator)
        dphi_dtau = -2 * np.pi * carrier_freq
        dphi_ddeltaf = 2 * np.pi * t

        # FIM elements
        fim_00 = (2 / noise_var) * np.sum(dphi_dtau ** 2)
        fim_11 = (2 / noise_var) * np.sum(dphi_ddeltaf ** 2)
        fim_01 = (2 / noise_var) * np.sum(dphi_dtau * dphi_ddeltaf)

        fim = np.array([[fim_00, fim_01], [fim_01, fim_11]])

        # CRLB matrix
        try:
            crlb_matrix = inv(fim)
            delay_crlb_var = crlb_matrix[0, 0]
            freq_crlb_var = crlb_matrix[1, 1]
        except np.linalg.LinAlgError:
            delay_crlb_var = np.inf
            freq_crlb_var = np.inf

        return {
            'delay_crlb_variance': delay_crlb_var,
            'frequency_crlb_variance': freq_crlb_var,
            'delay_crlb_std': np.sqrt(delay_crlb_var) if delay_crlb_var >= 0 else np.inf,
            'frequency_crlb_std': np.sqrt(freq_crlb_var) if freq_crlb_var >= 0 else np.inf,
            'noise_variance_estimate': noise_var,
            'residual_rms': np.sqrt(np.mean(np.abs(phase_residuals) ** 2)),
            'fisher_information_matrix': fim,
            'crlb_matrix': crlb_matrix,
            'method': 'residual_based'
        }

    def verify_crlb_consistency(self, mc_rmse_delay: float, mc_rmse_freq: float,
                                ls_covariance: np.ndarray) -> Dict[str, Any]:
        """
        Verify CRLB consistency with Monte Carlo results and LS covariance.

        Args:
            mc_rmse_delay: Monte Carlo RMSE for delay (seconds)
            mc_rmse_freq: Monte Carlo RMSE for frequency (Hz)
            ls_covariance: 2x2 covariance matrix from LS fit

        Returns:
            Consistency analysis results
        """
        # Compute theoretical CRLB
        theoretical_crlb = self.compute_joint_crlb()

        # Extract bounds
        crlb_delay_var = theoretical_crlb['delay_crlb_variance']
        crlb_freq_var = theoretical_crlb['frequency_crlb_variance']

        # Compute efficiency ratios (should be O(1) for good estimators)
        delay_efficiency = crlb_delay_var / (mc_rmse_delay ** 2) if mc_rmse_delay > 0 else 0
        freq_efficiency = crlb_freq_var / (mc_rmse_freq ** 2) if mc_rmse_freq > 0 else 0

        # Compare to LS covariance (should be close to CRLB)
        ls_delay_var = ls_covariance[0, 0]
        ls_freq_var = ls_covariance[1, 1]

        crlb_vs_ls_delay = crlb_delay_var / ls_delay_var if ls_delay_var > 0 else np.inf
        crlb_vs_ls_freq = crlb_freq_var / ls_freq_var if ls_freq_var > 0 else np.inf

        return {
            'theoretical_crlb': theoretical_crlb,
            'mc_efficiency_delay': delay_efficiency,
            'mc_efficiency_freq': freq_efficiency,
            'crlb_vs_ls_delay_ratio': crlb_vs_ls_delay,
            'crlb_vs_ls_freq_ratio': crlb_vs_ls_freq,
            'crlb_consistent': (0.5 < crlb_vs_ls_delay < 2.0 and
                              0.5 < crlb_vs_ls_freq < 2.0),
            'mc_reasonable': (0.5 < delay_efficiency < 5.0 and
                            0.5 < freq_efficiency < 5.0),
            'overall_consistent': (0.5 < crlb_vs_ls_delay < 2.0 and
                                 0.5 < crlb_vs_ls_freq < 2.0 and
                                 0.5 < delay_efficiency < 5.0 and
                                 0.5 < freq_efficiency < 5.0)
        }


class MultiFrequencyCRLBCalculator:
    """Approximate CRLB for multi-carrier phase-slope delay estimation."""

    def __init__(self, params: CRLBParams):
        self.params = params

    def compute_crlb(self) -> Dict[str, Any]:
        carriers = self.params.carrier_frequencies or []
        if len(carriers) < 2:
            joint = JointCRLBCalculator(self.params)
            result = joint.compute_joint_crlb()
            result['model'] = 'single_carrier_fallback'
            result['carrier_count'] = len(carriers) or 1
            return result

        freqs = np.array(carriers, dtype=float)
        sigma_phase = float(self.params.sigma_phase_rad)
        if sigma_phase <= 0 or not np.isfinite(sigma_phase):
            raise ValueError('sigma_phase_rad must be positive for multi-frequency CRLB.')

        f_mean = float(np.mean(freqs))
        denom = float(np.sum((freqs - f_mean) ** 2))
        if denom <= 0:
            raise ValueError('Carrier frequencies must not be identical for multi-frequency CRLB.')

        slope_variance = sigma_phase ** 2 / denom
        tau_variance = slope_variance / ((2 * np.pi) ** 2)
        tau_std = float(np.sqrt(tau_variance)) if np.isfinite(tau_variance) else np.inf

        return {
            'model': 'multi_frequency_phase_slope',
            'carrier_count': len(freqs),
            'carrier_span_hz': float(np.max(freqs) - np.min(freqs)),
            'phase_variance_rad2': float(sigma_phase ** 2),
            'slope_crlb_variance': float(slope_variance),
            'delay_crlb_variance': float(tau_variance),
            'delay_crlb_std': tau_std,
            'reference_frequency_hz': f_mean,
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
