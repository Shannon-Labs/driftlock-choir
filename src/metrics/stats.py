"""
Statistical validation tools for DriftLock metrics.

This module provides confidence intervals (parametric and bootstrap), hypothesis
testing (t-tests and bootstrap), and effect size calculations for key performance
metrics such as RMSE, CRLB ratios, Δf SNR, and BER. Designed for integration
with simulation telemetry to quantify uncertainty and significance.
"""

import numpy as np
from typing import Dict, Tuple, Any, Optional, Callable
from dataclasses import dataclass
from scipy import stats
from scipy.stats import bootstrap as scipy_bootstrap

@dataclass
class StatsParams:
    """Parameters for statistical validation."""
    confidence_level: float = 0.95
    bootstrap_samples: int = 1000
    random_state: Optional[int] = None

class StatisticalValidator:
    """Validator for computing statistical metrics on simulation outputs."""
    
    def __init__(self, params: StatsParams):
        self.params = params
        self.rng = np.random.default_rng(params.random_state)
    
    def bootstrap_ci(
        self, 
        data: np.ndarray, 
        statistic: Callable[[np.ndarray], float], 
        ci_level: Optional[float] = None
    ) -> Tuple[float, float]:
        """
        Compute bootstrap confidence interval for a statistic.
        
        Args:
            data: Input data array
            statistic: Function to compute statistic from resampled data
            ci_level: Confidence level (default: params.confidence_level)
            
        Returns:
            (lower, upper) bounds of the CI
        """
        ci_level = ci_level or self.params.confidence_level
        alpha = 1 - ci_level
        
        try:
            # Use scipy's bootstrap for vectorized statistics
            res = scipy_bootstrap(
                (data,), 
                statistic, 
                n_resamples=self.params.bootstrap_samples,
                random_state=self.rng, 
                confidence_level=ci_level,
                method='percentile'
            )
            return res.confidence_interval.low, res.confidence_interval.high
        except Exception:
            # Fallback manual bootstrap for compatibility
            n = len(data)
            boot_stats = []
            for _ in range(self.params.bootstrap_samples):
                idx = self.rng.choice(n, size=n, replace=True)
                boot_data = data[idx]
                boot_stats.append(statistic(boot_data))
            boot_stats = np.array(boot_stats)
            lower = np.percentile(boot_stats, 100 * alpha / 2)
            upper = np.percentile(boot_stats, 100 * (1 - alpha / 2))
            return lower, upper
    
    def parametric_mean_ci(
        self, 
        data: np.ndarray, 
        ci_level: Optional[float] = None
    ) -> Tuple[float, float]:
        """
        Parametric CI for the mean using t-distribution.
        """
        ci_level = ci_level or self.params.confidence_level
        if len(data) < 2:
            return np.nan, np.nan
        
        mean = np.mean(data)
        sem = stats.sem(data)
        df = len(data) - 1
        t_crit = stats.t.ppf((1 + ci_level) / 2, df)
        margin = t_crit * sem
        return mean - margin, mean + margin
    
    def confidence_intervals_for_rmse(
        self, 
        errors: np.ndarray, 
        ci_level: Optional[float] = None,
        method: str = 'bootstrap'
    ) -> Dict[str, Any]:
        """
        Compute CI for RMSE from error samples.
        
        Supports 'bootstrap' (default) or 'parametric' (normal approximation on variance).
        """
        ci_level = ci_level or self.params.confidence_level
        point_rmse = float(np.sqrt(np.mean(errors**2)))
        
        if method == 'parametric':
            # Parametric: CI on MSE, then sqrt (approximate)
            mse = np.mean(errors**2)
            var_mse = np.var(errors**2, ddof=1) / len(errors)
            sem_mse = np.sqrt(var_mse)
            z_crit = stats.norm.ppf((1 + ci_level) / 2)
            margin_mse = z_crit * sem_mse
            ci_mse_lower, ci_mse_upper = mse - margin_mse, mse + margin_mse
            ci_rmse_lower = np.sqrt(max(0, ci_mse_lower))
            ci_rmse_upper = np.sqrt(ci_mse_upper)
        else:
            # Bootstrap
            def rmse_stat(x: np.ndarray) -> float:
                return np.sqrt(np.mean(x**2))
            lower, upper = self.bootstrap_ci(errors, rmse_stat, ci_level)
            ci_rmse_lower, ci_rmse_upper = float(lower), float(upper)
        
        return {
            'point_estimate': point_rmse,
            'ci_lower': ci_rmse_lower,
            'ci_upper': ci_rmse_upper,
            'ci_level': ci_level,
            'ci_width': ci_rmse_upper - ci_rmse_lower,
            'method': method,
            'n_samples': len(errors)
        }
    
    def confidence_intervals_for_crlb_ratio(
        self, 
        errors: np.ndarray, 
        predicted_stds: np.ndarray, 
        ci_level: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Bootstrap CI for CRLB ratio = RMSE(errors) / mean(predicted_stds).
        
        Assumes paired samples (same length arrays).
        """
        if len(errors) != len(predicted_stds):
            raise ValueError("Errors and predicted_stds must be same length")
        
        n = len(errors)
        def ratio_stat(indices):
            boot_errors = errors[indices]
            boot_stds = predicted_stds[indices]
            rmse = np.sqrt(np.mean(boot_errors**2))
            mean_std = np.mean(boot_stds)
            return rmse / mean_std if mean_std > 0 else np.inf
        
        ci_level = ci_level or self.params.confidence_level
        alpha = 1 - ci_level
        
        boot_ratios = []
        for _ in range(self.params.bootstrap_samples):
            idx = self.rng.choice(n, size=n, replace=True)
            r = ratio_stat(idx)
            boot_ratios.append(r)
        
        boot_ratios = np.array(boot_ratios)
        lower = np.percentile(boot_ratios, 100 * alpha / 2)
        upper = np.percentile(boot_ratios, 100 * (1 - alpha / 2))
        
        point_rmse = np.sqrt(np.mean(errors**2))
        point_mean_std = np.mean(predicted_stds)
        point_ratio = point_rmse / point_mean_std if point_mean_std > 0 else np.inf
        
        return {
            'point_estimate': float(point_ratio),
            'ci_lower': float(lower),
            'ci_upper': float(upper),
            'ci_level': ci_level,
            'ci_width': float(upper - lower),
            'method': 'bootstrap_paired',
            'n_samples': n,
            'point_rmse': float(point_rmse),
            'point_mean_std': float(point_mean_std)
        }
    
    def confidence_intervals_for_snr(
        self, 
        snr_values: np.ndarray, 
        ci_level: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        CI for mean SNR (dB) from per-sample SNR values.
        """
        ci_level = ci_level or self.params.confidence_level
        mean_snr = np.mean(snr_values)
        lower, upper = self.parametric_mean_ci(snr_values, ci_level)
        return {
            'point_estimate': float(mean_snr),
            'ci_lower': float(lower),
            'ci_upper': float(upper),
            'ci_level': ci_level,
            'ci_width': float(upper - lower),
            'method': 'parametric_t',
            'n_samples': len(snr_values)
        }
    
    def confidence_intervals_for_ber(
        self, 
        ber_values: np.ndarray, 
        ci_level: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Wilson score CI for BER proportions (if BER as success probabilities).
        Assumes ber_values are per-trial BER rates.
        """
        ci_level = ci_level or self.params.confidence_level
        n_trials = len(ber_values)
        total_errors = np.sum(ber_values * n_trials) if np.all(ber_values <= 1) else np.sum(ber_values)
        total_bits = n_trials * n_trials  # Approximate if per-trial fixed bits
        p = total_errors / total_bits
        z = stats.norm.ppf((1 + ci_level) / 2)
        center = (p + z**2 / (2 * total_bits)) / (1 + z**2 / total_bits)
        margin = z * np.sqrt(p * (1 - p) / total_bits + z**2 / (4 * total_bits**2)) / (1 + z**2 / total_bits)
        lower = max(0, center - margin)
        upper = min(1, center + margin)
        return {
            'point_estimate': float(p),
            'ci_lower': float(lower),
            'ci_upper': float(upper),
            'ci_level': ci_level,
            'method': 'wilson_score',
            'n_samples': n_trials
        }
    
    def paired_t_test(
        self, 
        driftlock choir_metrics: np.ndarray, 
        baseline_metrics: np.ndarray
    ) -> Dict[str, Any]:
        """
        Paired t-test comparing Driftlock vs baseline metrics.
        """
        if len(driftlock choir_metrics) != len(baseline_metrics):
            raise ValueError("Paired samples must have equal length")
        t_stat, p_value = stats.ttest_rel(driftlock choir_metrics, baseline_metrics)
        mean_diff = np.mean(driftlock choir_metrics - baseline_metrics)
        return {
            't_statistic': float(t_stat),
            'p_value': float(p_value),
            'mean_difference': float(mean_diff),
            'significant': p_value < (1 - self.params.confidence_level),
            'df': len(driftlock choir_metrics) - 1,
            'test_type': 'paired_ttest'
        }
    
    def independent_t_test(
        self, 
        driftlock choir_metrics: np.ndarray, 
        baseline_metrics: np.ndarray,
        equal_var: bool = False
    ) -> Dict[str, Any]:
        """
        Independent samples t-test (for unpaired runs).
        """
        t_stat, p_value = stats.ttest_ind(driftlock choir_metrics, baseline_metrics, equal_var=equal_var)
        mean_diff = np.mean(driftlock choir_metrics) - np.mean(baseline_metrics)
        return {
            't_statistic': float(t_stat),
            'p_value': float(p_value),
            'mean_difference': float(mean_diff),
            'significant': p_value < (1 - self.params.confidence_level),
            'test_type': 'independent_ttest',
            'equal_variance': equal_var
        }
    
    def bootstrap_hypothesis_test(
        self, 
        group1: np.ndarray, 
        group2: np.ndarray, 
        ci_level: Optional[float] = None,
        alternative: str = 'two-sided'
    ) -> Dict[str, Any]:
        """
        Bootstrap hypothesis test for difference between two groups.
        """
        ci_level = ci_level or self.params.confidence_level
        alpha = 1 - ci_level
        observed_diff = np.mean(group1) - np.mean(group2)
        
        combined = np.concatenate([group1, group2])
        n1, n2 = len(group1), len(group2)
        
        diffs = []
        for _ in range(self.params.bootstrap_samples):
            boot1 = self.rng.choice(combined, size=n1, replace=True)
            boot2 = self.rng.choice(combined, size=n2, replace=True)
            diff = np.mean(boot1) - np.mean(boot2)
            diffs.append(diff)
        
        diffs = np.array(diffs)
        lower, upper = np.percentile(diffs, [100 * alpha / 2, 100 * (1 - alpha / 2)])
        
        if alternative == 'two-sided':
            p_value = np.mean(np.abs(diffs - observed_diff) >= np.abs(observed_diff))
        elif alternative == 'less':
            p_value = np.mean(diffs >= observed_diff)
        else:  # greater
            p_value = np.mean(diffs <= observed_diff)
        
        significant = p_value < alpha
        
        return {
            'observed_difference': float(observed_diff),
            'bootstrap_ci_lower': float(lower),
            'bootstrap_ci_upper': float(upper),
            'p_value': float(p_value),
            'significant': significant,
            'ci_level': ci_level,
            'alternative': alternative,
            'method': 'bootstrap_permutation'
        }
    
    def effect_sizes(
        self, 
        driftlock choir: np.ndarray, 
        baseline: np.ndarray
    ) -> Dict[str, Any]:
        """
        Compute Cohen's d and relative improvement percentage.
        """
        mean_d = np.mean(driftlock choir)
        mean_b = np.mean(baseline)
        std_d = np.std(driftlock choir, ddof=1)
        std_b = np.std(baseline, ddof=1)
        n_d = len(driftlock choir)
        n_b = len(baseline)
        
        if n_d + n_b < 4:
            cohens_d = np.nan
        else:
            pooled_var = ((n_d - 1) * std_d**2 + (n_b - 1) * std_b**2) / (n_d + n_b - 2)
            pooled_std = np.sqrt(pooled_var)
            cohens_d = (mean_d - mean_b) / pooled_std if pooled_std > 0 else 0.0
        
        rel_improvement = ((mean_b - mean_d) / mean_b * 100) if mean_b != 0 else 0.0
        
        interpretation = self._cohens_d_interpretation(cohens_d)
        
        return {
            'cohens_d': float(cohens_d),
            'cohens_d_interpretation': interpretation,
            'relative_improvement_pct': float(rel_improvement),
            'mean_driftlock choir': float(mean_d),
            'mean_baseline': float(mean_b),
            'pooled_std': float(pooled_std) if 'pooled_std' in locals() else np.nan
        }
    
    def _cohens_d_interpretation(self, d: float) -> str:
        """Interpret Cohen's d magnitude."""
        abs_d = abs(d)
        if abs_d < 0.2:
            return "small"
        elif abs_d < 0.5:
            return "medium"
        elif abs_d < 0.8:
            return "large"
        else:
            return "very large"