"""
Jacobian condition number analysis for parameter estimation robustness.

This module analyzes the conditioning of parameter estimation problems
through Jacobian analysis and sensitivity metrics.
"""

import numpy as np
from typing import Dict, List, Any, Optional, Callable, Tuple
from dataclasses import dataclass
from scipy.linalg import svd, norm, cond
from scipy.optimize import approx_fprime


@dataclass
class ConditioningParams:
    """Parameters for conditioning analysis."""
    perturbation_size: float = 1e-8    # Perturbation size for numerical derivatives
    condition_threshold: float = 1e12  # Threshold for ill-conditioning
    n_monte_carlo: int = 1000          # Monte Carlo samples for robustness analysis
    parameter_ranges: Optional[Dict[str, Tuple[float, float]]] = None


class JacobianAnalyzer:
    """Analyzer for Jacobian conditioning and parameter sensitivity."""
    
    def __init__(self, params: ConditioningParams):
        self.params = params
        
    def analyze_jacobian_conditioning(self, 
                                    measurement_function: Callable,
                                    parameter_values: np.ndarray,
                                    measurement_noise_std: float = 1e-9) -> Dict[str, Any]:
        """
        Analyze Jacobian conditioning for parameter estimation problem.
        
        Args:
            measurement_function: Function that computes measurements from parameters
            parameter_values: Current parameter values
            measurement_noise_std: Standard deviation of measurement noise
            
        Returns:
            Dictionary with conditioning analysis results
        """
        # Compute Jacobian matrix
        jacobian = self._compute_jacobian(measurement_function, parameter_values)
        
        # Singular Value Decomposition
        U, singular_values, Vt = svd(jacobian, full_matrices=False)
        
        # Condition number analysis
        condition_number = cond(jacobian)
        
        # Parameter sensitivity analysis
        sensitivity_analysis = self._analyze_parameter_sensitivity(
            jacobian, singular_values, Vt
        )
        
        # Noise sensitivity
        noise_sensitivity = self._analyze_noise_sensitivity(
            jacobian, measurement_noise_std
        )
        
        # Identifiability analysis
        identifiability = self._analyze_identifiability(
            singular_values, parameter_values.shape[0]
        )
        
        return {
            'jacobian_matrix': jacobian,
            'singular_values': singular_values,
            'condition_number': condition_number,
            'is_well_conditioned': condition_number < self.params.condition_threshold,
            'sensitivity_analysis': sensitivity_analysis,
            'noise_sensitivity': noise_sensitivity,
            'identifiability': identifiability,
            'svd_components': {
                'U': U,
                'singular_values': singular_values,
                'Vt': Vt
            }
        }
        
    def _compute_jacobian(self, measurement_function: Callable, 
                         parameter_values: np.ndarray) -> np.ndarray:
        """Compute Jacobian matrix using numerical differentiation."""
        # Get measurement dimension
        measurements = measurement_function(parameter_values)
        if np.isscalar(measurements):
            measurements = np.array([measurements])
            
        n_measurements = len(measurements)
        n_parameters = len(parameter_values)
        
        jacobian = np.zeros((n_measurements, n_parameters))
        
        for param_idx in range(n_parameters):
            # Compute partial derivative w.r.t. parameter param_idx
            def partial_func(param_val):
                params_perturbed = parameter_values.copy()
                params_perturbed[param_idx] = param_val
                result = measurement_function(params_perturbed)
                return result if hasattr(result, '__len__') else np.array([result])
                
            gradient = approx_fprime(
                parameter_values[param_idx], 
                partial_func, 
                self.params.perturbation_size
            )
            
            jacobian[:, param_idx] = gradient
            
        return jacobian
        
    def _analyze_parameter_sensitivity(self, jacobian: np.ndarray, 
                                     singular_values: np.ndarray,
                                     Vt: np.ndarray) -> Dict[str, Any]:
        """Analyze parameter sensitivity from Jacobian."""
        n_params = jacobian.shape[1]
        
        # Parameter sensitivity (column norms of Jacobian)
        param_sensitivities = np.array([norm(jacobian[:, i]) for i in range(n_params)])
        
        # Normalized sensitivities
        max_sensitivity = np.max(param_sensitivities)
        normalized_sensitivities = param_sensitivities / max_sensitivity if max_sensitivity > 0 else param_sensitivities
        
        # Most/least sensitive parameters
        most_sensitive_idx = np.argmax(param_sensitivities)
        least_sensitive_idx = np.argmin(param_sensitivities)
        
        # Parameter correlation (from V matrix)
        param_correlations = np.abs(Vt.T @ Vt)  # |V V^T|
        
        # Effective rank and parameter redundancy
        effective_rank = np.sum(singular_values > singular_values[0] * 1e-12)
        
        return {
            'parameter_sensitivities': param_sensitivities,
            'normalized_sensitivities': normalized_sensitivities,
            'most_sensitive_parameter': most_sensitive_idx,
            'least_sensitive_parameter': least_sensitive_idx,
            'sensitivity_ratio': (param_sensitivities[most_sensitive_idx] / 
                                param_sensitivities[least_sensitive_idx] 
                                if param_sensitivities[least_sensitive_idx] > 0 else np.inf),
            'parameter_correlations': param_correlations,
            'effective_rank': effective_rank,
            'parameter_redundancy': n_params - effective_rank
        }
        
    def _analyze_noise_sensitivity(self, jacobian: np.ndarray, 
                                 noise_std: float) -> Dict[str, Any]:
        """Analyze sensitivity to measurement noise."""
        # Compute pseudo-inverse for noise propagation analysis
        try:
            jacobian_pinv = np.linalg.pinv(jacobian)
            
            # Noise amplification factor
            noise_amplification = norm(jacobian_pinv, 'fro') * noise_std
            
            # Parameter-wise noise sensitivity
            param_noise_sensitivity = np.array([norm(jacobian_pinv[i, :]) * noise_std 
                                              for i in range(jacobian_pinv.shape[0])])
            
            # Noise-to-signal ratio for each parameter
            param_magnitudes = np.abs(np.diag(jacobian_pinv @ jacobian))
            noise_to_signal_ratio = param_noise_sensitivity / param_magnitudes
            
        except np.linalg.LinAlgError:
            # Singular matrix
            noise_amplification = np.inf
            param_noise_sensitivity = np.full(jacobian.shape[1], np.inf)
            noise_to_signal_ratio = np.full(jacobian.shape[1], np.inf)
            
        return {
            'noise_amplification_factor': noise_amplification,
            'parameter_noise_sensitivity': param_noise_sensitivity,
            'noise_to_signal_ratio': noise_to_signal_ratio,
            'max_noise_amplification': np.max(param_noise_sensitivity),
            'avg_noise_amplification': np.mean(param_noise_sensitivity)
        }
        
    def _analyze_identifiability(self, singular_values: np.ndarray, 
                               n_parameters: int) -> Dict[str, Any]:
        """Analyze parameter identifiability."""
        # Rank analysis
        tolerance = singular_values[0] * 1e-12 if len(singular_values) > 0 else 1e-12
        numerical_rank = np.sum(singular_values > tolerance)
        
        # Identifiability classification
        if numerical_rank == n_parameters:
            identifiability_status = "fully_identifiable"
        elif numerical_rank < n_parameters:
            identifiability_status = "partially_identifiable"
        else:
            identifiability_status = "over_determined"
            
        # Singular value ratios
        if len(singular_values) > 1:
            sv_ratios = singular_values[:-1] / singular_values[1:]
            max_sv_ratio = np.max(sv_ratios)
            min_sv_ratio = np.min(sv_ratios)
        else:
            sv_ratios = np.array([])
            max_sv_ratio = 1.0
            min_sv_ratio = 1.0
            
        return {
            'numerical_rank': numerical_rank,
            'theoretical_rank': n_parameters,
            'identifiability_status': identifiability_status,
            'singular_value_ratios': sv_ratios,
            'max_singular_value_ratio': max_sv_ratio,
            'min_singular_value_ratio': min_sv_ratio,
            'rank_deficiency': max(0, n_parameters - numerical_rank)
        }
        
    def monte_carlo_conditioning_analysis(self, 
                                        measurement_function: Callable,
                                        nominal_parameters: np.ndarray,
                                        parameter_uncertainties: np.ndarray) -> Dict[str, Any]:
        """
        Perform Monte Carlo analysis of conditioning over parameter space.
        
        Args:
            measurement_function: Measurement function
            nominal_parameters: Nominal parameter values
            parameter_uncertainties: Standard deviations for parameter perturbations
            
        Returns:
            Monte Carlo conditioning analysis results
        """
        condition_numbers = []
        singular_value_sets = []
        well_conditioned_count = 0
        
        for trial in range(self.params.n_monte_carlo):
            # Perturb parameters
            perturbed_params = nominal_parameters + np.random.normal(
                0, parameter_uncertainties, size=nominal_parameters.shape
            )
            
            # Compute Jacobian and condition number
            try:
                jacobian = self._compute_jacobian(measurement_function, perturbed_params)
                condition_num = cond(jacobian)
                
                if not np.isfinite(condition_num):
                    condition_num = 1e16  # Cap at large value
                    
                condition_numbers.append(condition_num)
                
                # Store singular values
                _, singular_vals, _ = svd(jacobian, full_matrices=False)
                singular_value_sets.append(singular_vals)
                
                if condition_num < self.params.condition_threshold:
                    well_conditioned_count += 1
                    
            except Exception:
                # Handle numerical issues
                condition_numbers.append(1e16)
                singular_value_sets.append(np.array([0]))
                
        condition_numbers = np.array(condition_numbers)
        
        # Statistical analysis
        return {
            'condition_number_statistics': {
                'mean': np.mean(condition_numbers),
                'std': np.std(condition_numbers),
                'median': np.median(condition_numbers),
                'min': np.min(condition_numbers),
                'max': np.max(condition_numbers),
                'percentiles': np.percentile(condition_numbers, [10, 25, 75, 90, 95, 99])
            },
            'well_conditioned_fraction': well_conditioned_count / self.params.n_monte_carlo,
            'condition_number_distribution': condition_numbers,
            'singular_value_statistics': self._analyze_singular_value_distribution(singular_value_sets),
            'robustness_metrics': {
                'condition_variability': np.std(condition_numbers) / np.mean(condition_numbers),
                'worst_case_amplification': np.max(condition_numbers),
                'reliable_operation_probability': well_conditioned_count / self.params.n_monte_carlo
            }
        }
        
    def _analyze_singular_value_distribution(self, singular_value_sets: List[np.ndarray]) -> Dict[str, Any]:
        """Analyze distribution of singular values across Monte Carlo trials."""
        # Find maximum number of singular values
        max_sv_count = max(len(sv_set) for sv_set in singular_value_sets)
        
        # Pad singular value sets and compute statistics
        sv_statistics = {}
        
        for sv_idx in range(max_sv_count):
            sv_values = []
            for sv_set in singular_value_sets:
                if sv_idx < len(sv_set):
                    sv_values.append(sv_set[sv_idx])
                else:
                    sv_values.append(0)  # Pad with zeros
                    
            sv_values = np.array(sv_values)
            
            sv_statistics[f'singular_value_{sv_idx}'] = {
                'mean': np.mean(sv_values),
                'std': np.std(sv_values),
                'min': np.min(sv_values),
                'max': np.max(sv_values),
                'zero_fraction': np.sum(sv_values == 0) / len(sv_values)
            }
            
        return sv_statistics
        
    def parameter_space_scan(self, 
                           measurement_function: Callable,
                           parameter_ranges: Dict[str, Tuple[float, float]],
                           scan_resolution: int = 20) -> Dict[str, Any]:
        """
        Scan parameter space to identify regions of good/poor conditioning.
        
        Args:
            measurement_function: Measurement function
            parameter_ranges: Dictionary mapping parameter names to (min, max) ranges
            scan_resolution: Number of points per parameter dimension
            
        Returns:
            Parameter space conditioning map
        """
        param_names = list(parameter_ranges.keys())
        n_params = len(param_names)
        
        # Create parameter grids
        param_grids = []
        for param_name in param_names:
            min_val, max_val = parameter_ranges[param_name]
            param_grids.append(np.linspace(min_val, max_val, scan_resolution))
            
        # Generate all combinations
        param_combinations = np.meshgrid(*param_grids, indexing='ij')
        param_points = np.stack([grid.flatten() for grid in param_combinations], axis=1)
        
        # Evaluate conditioning at each point
        condition_numbers = np.zeros(param_points.shape[0])
        
        for i, param_point in enumerate(param_points):
            try:
                jacobian = self._compute_jacobian(measurement_function, param_point)
                condition_numbers[i] = cond(jacobian)
                
                if not np.isfinite(condition_numbers[i]):
                    condition_numbers[i] = 1e16
                    
            except Exception:
                condition_numbers[i] = 1e16
                
        # Reshape results to grid format
        condition_grid = condition_numbers.reshape([scan_resolution] * n_params)
        
        # Find best and worst conditioning regions
        best_idx = np.unravel_index(np.argmin(condition_numbers), condition_grid.shape)
        worst_idx = np.unravel_index(np.argmax(condition_numbers), condition_grid.shape)
        
        best_params = {param_names[i]: param_grids[i][best_idx[i]] for i in range(n_params)}
        worst_params = {param_names[i]: param_grids[i][worst_idx[i]] for i in range(n_params)}
        
        return {
            'parameter_grids': {param_names[i]: param_grids[i] for i in range(n_params)},
            'condition_number_grid': condition_grid,
            'parameter_combinations': param_points,
            'condition_numbers': condition_numbers,
            'best_conditioning': {
                'condition_number': condition_numbers[np.argmin(condition_numbers)],
                'parameters': best_params
            },
            'worst_conditioning': {
                'condition_number': condition_numbers[np.argmax(condition_numbers)],
                'parameters': worst_params
            },
            'well_conditioned_fraction': np.sum(condition_numbers < self.params.condition_threshold) / len(condition_numbers)
        }
