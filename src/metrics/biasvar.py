"""
Bias-variance decomposition for synchronization parameter estimation.

This module provides tools for analyzing estimator performance through
bias-variance decomposition and related statistical metrics.
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from scipy import stats


@dataclass
class BiasVarianceParams:
    """Parameters for bias-variance analysis."""
    n_trials: int = 1000      # Number of Monte Carlo trials
    confidence_level: float = 0.95  # Confidence level for intervals
    bootstrap_samples: int = 500    # Bootstrap samples for uncertainty


class BiasVarianceAnalyzer:
    """Analyzer for bias-variance decomposition of estimators."""
    
    def __init__(self, params: BiasVarianceParams):
        self.params = params
        
    def analyze_estimator_performance(self, estimates: np.ndarray, 
                                    true_values: np.ndarray) -> Dict[str, Any]:
        """
        Perform comprehensive bias-variance analysis.
        
        Args:
            estimates: Estimator outputs [n_trials x n_parameters]
            true_values: True parameter values [n_parameters]
            
        Returns:
            Dictionary with bias-variance decomposition results
        """
        if estimates.ndim == 1:
            estimates = estimates.reshape(-1, 1)
            true_values = np.array([true_values])
            
        n_trials, n_params = estimates.shape
        
        results = {}
        
        for param_idx in range(n_params):
            param_estimates = estimates[:, param_idx]
            true_value = true_values[param_idx]
            
            # Basic statistics
            mean_estimate = np.mean(param_estimates)
            variance_estimate = np.var(param_estimates, ddof=1)
            std_estimate = np.sqrt(variance_estimate)
            
            # Bias calculation
            bias = mean_estimate - true_value
            bias_squared = bias ** 2
            
            # Mean Squared Error
            mse = np.mean((param_estimates - true_value) ** 2)
            
            # Bias-variance decomposition: MSE = Bias² + Variance + Noise
            # For noiseless case: MSE = Bias² + Variance
            noise_term = mse - bias_squared - variance_estimate
            
            # Confidence intervals
            ci_lower, ci_upper = self._compute_confidence_interval(
                param_estimates, self.params.confidence_level
            )
            
            # Bootstrap uncertainty estimates
            bootstrap_stats = self._bootstrap_analysis(param_estimates, true_value)
            
            # Distribution analysis
            dist_analysis = self._analyze_distribution(param_estimates)
            
            results[f'parameter_{param_idx}'] = {
                'basic_statistics': {
                    'mean': mean_estimate,
                    'variance': variance_estimate,
                    'std': std_estimate,
                    'min': np.min(param_estimates),
                    'max': np.max(param_estimates),
                    'median': np.median(param_estimates)
                },
                'bias_variance_decomposition': {
                    'bias': bias,
                    'bias_squared': bias_squared,
                    'variance': variance_estimate,
                    'mse': mse,
                    'noise_term': noise_term,
                    'bias_to_variance_ratio': bias_squared / variance_estimate if variance_estimate > 0 else np.inf
                },
                'confidence_intervals': {
                    'level': self.params.confidence_level,
                    'lower': ci_lower,
                    'upper': ci_upper,
                    'width': ci_upper - ci_lower,
                    'contains_true_value': ci_lower <= true_value <= ci_upper
                },
                'bootstrap_analysis': bootstrap_stats,
                'distribution_analysis': dist_analysis,
                'performance_metrics': {
                    'rmse': np.sqrt(mse),
                    'mae': np.mean(np.abs(param_estimates - true_value)),
                    'relative_bias': bias / true_value if true_value != 0 else np.inf,
                    'coefficient_of_variation': std_estimate / mean_estimate if mean_estimate != 0 else np.inf
                }
            }
            
        # Overall analysis
        results['overall_analysis'] = self._compute_overall_metrics(estimates, true_values)
        
        return results
        
    def _compute_confidence_interval(self, data: np.ndarray, 
                                   confidence_level: float) -> Tuple[float, float]:
        """Compute confidence interval for parameter estimates."""
        alpha = 1 - confidence_level
        lower_percentile = 100 * alpha / 2
        upper_percentile = 100 * (1 - alpha / 2)
        
        return np.percentile(data, [lower_percentile, upper_percentile])
        
    def _bootstrap_analysis(self, estimates: np.ndarray, 
                          true_value: float) -> Dict[str, Any]:
        """Perform bootstrap analysis for uncertainty estimation."""
        n_samples = len(estimates)
        bootstrap_means = []
        bootstrap_variances = []
        bootstrap_biases = []
        
        for _ in range(self.params.bootstrap_samples):
            # Bootstrap sample
            bootstrap_sample = np.random.choice(estimates, size=n_samples, replace=True)
            
            # Compute statistics
            bootstrap_means.append(np.mean(bootstrap_sample))
            bootstrap_variances.append(np.var(bootstrap_sample, ddof=1))
            bootstrap_biases.append(np.mean(bootstrap_sample) - true_value)
            
        return {
            'mean_of_means': np.mean(bootstrap_means),
            'std_of_means': np.std(bootstrap_means),
            'mean_of_variances': np.mean(bootstrap_variances),
            'std_of_variances': np.std(bootstrap_variances),
            'bias_distribution': {
                'mean': np.mean(bootstrap_biases),
                'std': np.std(bootstrap_biases),
                'percentiles': np.percentile(bootstrap_biases, [5, 25, 50, 75, 95])
            }
        }
        
    def _analyze_distribution(self, data: np.ndarray) -> Dict[str, Any]:
        """Analyze the distribution of estimates."""
        # Normality test
        shapiro_stat, shapiro_p = stats.shapiro(data)
        
        # Skewness and kurtosis
        skewness = stats.skew(data)
        kurtosis = stats.kurtosis(data)
        
        # Percentiles
        percentiles = np.percentile(data, [1, 5, 10, 25, 50, 75, 90, 95, 99])
        
        # Outlier detection (IQR method)
        q25, q75 = np.percentile(data, [25, 75])
        iqr = q75 - q25
        lower_fence = q25 - 1.5 * iqr
        upper_fence = q75 + 1.5 * iqr
        outliers = data[(data < lower_fence) | (data > upper_fence)]
        
        return {
            'normality_test': {
                'shapiro_wilk_statistic': shapiro_stat,
                'shapiro_wilk_p_value': shapiro_p,
                'is_normal': shapiro_p > 0.05
            },
            'moments': {
                'skewness': skewness,
                'kurtosis': kurtosis,
                'excess_kurtosis': kurtosis - 3
            },
            'percentiles': {
                'p1': percentiles[0], 'p5': percentiles[1], 'p10': percentiles[2],
                'p25': percentiles[3], 'p50': percentiles[4], 'p75': percentiles[5],
                'p90': percentiles[6], 'p95': percentiles[7], 'p99': percentiles[8]
            },
            'outliers': {
                'count': len(outliers),
                'fraction': len(outliers) / len(data),
                'values': outliers.tolist() if len(outliers) < 20 else outliers[:20].tolist()
            }
        }
        
    def _compute_overall_metrics(self, estimates: np.ndarray, 
                               true_values: np.ndarray) -> Dict[str, Any]:
        """Compute overall performance metrics across all parameters."""
        # Total MSE across all parameters
        total_mse = np.mean((estimates - true_values) ** 2)
        
        # Average bias and variance across parameters
        biases = np.mean(estimates, axis=0) - true_values
        variances = np.var(estimates, axis=0, ddof=1)
        
        avg_bias_magnitude = np.mean(np.abs(biases))
        avg_variance = np.mean(variances)
        
        # Correlation between parameters (if multi-dimensional)
        if estimates.shape[1] > 1:
            param_correlations = np.corrcoef(estimates.T)
        else:
            param_correlations = np.array([[1.0]])
            
        return {
            'total_mse': total_mse,
            'total_rmse': np.sqrt(total_mse),
            'average_bias_magnitude': avg_bias_magnitude,
            'average_variance': avg_variance,
            'parameter_correlations': param_correlations,
            'joint_performance': {
                'determinant_correlation': np.linalg.det(param_correlations),
                'condition_number': np.linalg.cond(param_correlations)
            }
        }
        
    def compare_estimators(self, estimator_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Compare multiple estimators using bias-variance analysis.
        
        Args:
            estimator_results: Dictionary of estimator names to their analysis results
            
        Returns:
            Comparison results
        """
        comparison = {}
        
        # Extract key metrics for comparison
        estimator_names = list(estimator_results.keys())
        n_params = len([k for k in estimator_results[estimator_names[0]].keys() 
                       if k.startswith('parameter_')])
        
        for param_idx in range(n_params):
            param_key = f'parameter_{param_idx}'
            param_comparison = {}
            
            for metric in ['bias', 'variance', 'mse']:
                param_comparison[metric] = {}
                for est_name in estimator_names:
                    if metric in ['bias', 'variance']:
                        param_comparison[metric][est_name] = \
                            estimator_results[est_name][param_key]['bias_variance_decomposition'][metric]
                    else:  # mse
                        param_comparison[metric][est_name] = \
                            estimator_results[est_name][param_key]['bias_variance_decomposition'][metric]
                            
                # Find best estimator for this metric
                if metric == 'bias':
                    best_est = min(param_comparison[metric].keys(), 
                                 key=lambda x: abs(param_comparison[metric][x]))
                else:  # variance, mse
                    best_est = min(param_comparison[metric].keys(), 
                                 key=lambda x: param_comparison[metric][x])
                                 
                param_comparison[f'best_{metric}'] = best_est
                
            comparison[param_key] = param_comparison
            
        # Overall comparison
        overall_mse = {}
        for est_name in estimator_names:
            overall_mse[est_name] = estimator_results[est_name]['overall_analysis']['total_mse']
            
        best_overall = min(overall_mse.keys(), key=lambda x: overall_mse[x])
        
        comparison['overall'] = {
            'mse_comparison': overall_mse,
            'best_overall': best_overall,
            'relative_performance': {
                est_name: overall_mse[best_overall] / overall_mse[est_name] 
                for est_name in estimator_names
            }
        }
        
        return comparison
        
    def generate_performance_report(self, analysis_results: Dict[str, Any]) -> str:
        """Generate a human-readable performance report."""
        report = ["=== Bias-Variance Analysis Report ===\n"]
        
        # Parameter-wise analysis
        param_keys = [k for k in analysis_results.keys() if k.startswith('parameter_')]
        
        for param_key in param_keys:
            param_results = analysis_results[param_key]
            param_idx = param_key.split('_')[1]
            
            report.append(f"Parameter {param_idx} Analysis:")
            report.append(f"  Mean Estimate: {param_results['basic_statistics']['mean']:.6f}")
            report.append(f"  Bias: {param_results['bias_variance_decomposition']['bias']:.6f}")
            report.append(f"  Variance: {param_results['bias_variance_decomposition']['variance']:.6e}")
            report.append(f"  MSE: {param_results['bias_variance_decomposition']['mse']:.6e}")
            report.append(f"  RMSE: {param_results['performance_metrics']['rmse']:.6f}")
            report.append(f"  Bias/Variance Ratio: {param_results['bias_variance_decomposition']['bias_to_variance_ratio']:.3f}")
            report.append(f"  Normal Distribution: {param_results['distribution_analysis']['normality_test']['is_normal']}")
            report.append("")
            
        # Overall analysis
        if 'overall_analysis' in analysis_results:
            overall = analysis_results['overall_analysis']
            report.append("Overall Performance:")
            report.append(f"  Total RMSE: {overall['total_rmse']:.6f}")
            report.append(f"  Average Bias Magnitude: {overall['average_bias_magnitude']:.6f}")
            report.append(f"  Average Variance: {overall['average_variance']:.6e}")
            
        return "\n".join(report)
