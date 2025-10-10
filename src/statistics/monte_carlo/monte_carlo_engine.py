"""
Monte Carlo simulation engine for chronometric interferometry validation.

Provides comprehensive Monte Carlo validation with convergence diagnostics
and statistical analysis suitable for Nokia Bell Labs research standards.
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
import warnings
from dataclasses import dataclass, field
from enum import Enum
import time
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed


class ConvergenceCriterion(Enum):
    """Convergence criteria for Monte Carlo simulations."""
    EFFECTIVE_SAMPLE_SIZE = "effective_sample_size"
    GELMAN_RUBIN = "gelman_rubin"
    AUTOCORRELATION = "autocorrelation"
    STANDARD_ERROR = "standard_error"


@dataclass
class ConvergenceDiagnostics:
    """Diagnostics for Monte Carlo convergence assessment."""
    effective_sample_size: float
    gelman_rubin_statistic: float
    autocorrelation_time: float
    monte_carlo_standard_error: float
    convergence_status: bool
    recommendations: List[str]
    n_iterations: int


@dataclass
class MonteCarloResult:
    """Results from Monte Carlo simulation with full statistical analysis."""
    parameter_estimates: np.ndarray
    parameter_uncertainties: np.ndarray
    confidence_intervals: np.ndarray
    convergence_diagnostics: ConvergenceDiagnostics
    runtime_seconds: float
    n_iterations: int
    n_chains: int
    acceptance_rates: Optional[np.ndarray] = None
    posterior_samples: Optional[np.ndarray] = None


class MonteCarloEngine:
    """
    Advanced Monte Carlo simulation engine for chronometric interferometry validation.

    Provides comprehensive Monte Carlo validation with parallel processing,
    convergence diagnostics, and statistical analysis.
    """

    def __init__(self, n_iterations: int = 10000, n_chains: int = 4,
                 convergence_threshold: float = 1.1, parallel: bool = True):
        """
        Initialize Monte Carlo engine.

        Args:
            n_iterations: Number of iterations per chain
            n_chains: Number of parallel chains
            convergence_threshold: Threshold for convergence (Gelman-Rubin)
            parallel: Whether to use parallel processing
        """
        self.n_iterations = n_iterations
        self.n_chains = n_chains
        self.convergence_threshold = convergence_threshold
        self.parallel = parallel
        self.random_seed = 42  # For reproducibility

    def monte_carlo_parameter_estimation(self, simulation_model: Callable,
                                       parameter_ranges: Dict[str, Tuple[float, float]],
                                       true_parameters: Optional[Dict[str, float]] = None,
                                       convergence_criteria: List[ConvergenceCriterion] = None) -> MonteCarloResult:
        """
        Perform Monte Carlo parameter estimation with convergence diagnostics.

        Args:
            simulation_model: Function that simulates chronometric interferometry
            parameter_ranges: Dictionary of parameter ranges (min, max)
            true_parameters: True parameter values for validation
            convergence_criteria: List of convergence criteria to check

        Returns:
            MonteCarloResult with comprehensive analysis
        """
        start_time = time.time()

        if convergence_criteria is None:
            convergence_criteria = [
                ConvergenceCriterion.GELMAN_RUBIN,
                ConvergenceCriterion.EFFECTIVE_SAMPLE_SIZE
            ]

        # Initialize chains
        n_params = len(parameter_ranges)
        param_names = list(parameter_ranges.keys())

        # Generate random initial values for each chain
        np.random.seed(self.random_seed)
        initial_values = []
        for _ in range(self.n_chains):
            chain_init = []
            for param_name, (min_val, max_val) in parameter_ranges.items():
                init_val = np.random.uniform(min_val, max_val)
                chain_init.append(init_val)
            initial_values.append(chain_init)

        # Run parallel chains
        if self.parallel and self.n_chains > 1:
            chains_results = self._run_parallel_chains(
                simulation_model, initial_values, parameter_ranges
            )
        else:
            chains_results = self._run_single_chain(
                simulation_model, initial_values[0], parameter_ranges
            )
            chains_results = [chains_results]

        # Analyze convergence
        convergence_diagnostics = self._analyze_convergence(
            chains_results, convergence_criteria
        )

        # Combine results from all chains
        all_samples = np.concatenate(chains_results, axis=1)

        # Calculate parameter estimates and uncertainties
        parameter_estimates = np.mean(all_samples, axis=1)
        parameter_uncertainties = np.std(all_samples, axis=1, ddof=1)

        # Calculate confidence intervals
        confidence_intervals = np.percentile(all_samples, [2.5, 97.5], axis=1)

        runtime = time.time() - start_time

        return MonteCarloResult(
            parameter_estimates=parameter_estimates,
            parameter_uncertainties=parameter_uncertainties,
            confidence_intervals=confidence_intervals,
            convergence_diagnostics=convergence_diagnostics,
            runtime_seconds=runtime,
            n_iterations=self.n_iterations,
            n_chains=self.n_chains,
            posterior_samples=all_samples
        )

    def _run_single_chain(self, simulation_model: Callable,
                         initial_values: List[float],
                         parameter_ranges: Dict[str, Tuple[float, float]]) -> np.ndarray:
        """Run a single Monte Carlo chain."""
        n_params = len(initial_values)
        samples = np.zeros((n_params, self.n_iterations))
        current_params = np.array(initial_values)

        # Proposal distribution scales
        scales = [0.1 * (max_val - min_val)
                 for (min_val, max_val) in parameter_ranges.values()]

        for i in range(self.n_iterations):
            # Propose new parameters
            proposal = current_params + np.random.normal(0, scales, n_params)

            # Check bounds
            within_bounds = True
            for j, (param_name, (min_val, max_val)) in enumerate(parameter_ranges.items()):
                if not (min_val <= proposal[j] <= max_val):
                    within_bounds = False
                    break

            if within_bounds:
                # Calculate acceptance probability (simplified Metropolis-Hastings)
                try:
                    current_likelihood = simulation_model(*current_params)
                    proposal_likelihood = simulation_model(*proposal)

                    if proposal_likelihood > current_likelihood:
                        acceptance_prob = 1.0
                    else:
                        acceptance_prob = np.exp(proposal_likelihood - current_likelihood)

                    if np.random.random() < acceptance_prob:
                        current_params = proposal
                except:
                    # Handle simulation failures
                    pass

            samples[:, i] = current_params

        return samples

    def _run_parallel_chains(self, simulation_model: Callable,
                           initial_values: List[List[float]],
                           parameter_ranges: Dict[str, Tuple[float, float]]) -> List[np.ndarray]:
        """Run multiple chains in parallel."""
        with ProcessPoolExecutor(max_workers=self.n_chains) as executor:
            futures = []
            for init_values in initial_values:
                future = executor.submit(
                    self._run_single_chain,
                    simulation_model, init_values, parameter_ranges
                )
                futures.append(future)

            results = []
            for future in as_completed(futures):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    warnings.warn(f"Chain failed: {e}")

        return results

    def _analyze_convergence(self, chains_results: List[np.ndarray],
                           convergence_criteria: List[ConvergenceCriterion]) -> ConvergenceDiagnostics:
        """Analyze convergence using multiple diagnostics."""
        if len(chains_results) < 2:
            # Single chain analysis
            all_samples = chains_results[0]
            n_samples = all_samples.shape[1]

            # Effective sample size
            ess = self._calculate_effective_sample_size(all_samples)

            # Autocorrelation time
            autocorr_time = self._calculate_autocorrelation_time(all_samples)

            # Monte Carlo standard error
            mcse = np.std(all_samples, axis=1, ddof=1) / np.sqrt(ess)

            convergence_status = ess > 200  # Simple threshold for single chain

            recommendations = []
            if ess < 200:
                recommendations.append("Increase number of iterations for better effective sample size")
            if autocorr_time > 50:
                recommendations.append("High autocorrelation detected, consider thinning")

        else:
            # Multiple chains analysis
            # Stack chains for analysis
            stacked_chains = np.stack(chains_results, axis=2)  # params x iterations x chains

            # Gelman-Rubin statistic
            gelman_rubin = self._calculate_gelman_rubin(stacked_chains)

            # Effective sample size (across all chains)
            all_samples = np.concatenate(chains_results, axis=1)
            ess = self._calculate_effective_sample_size(all_samples)

            # Autocorrelation time
            autocorr_time = self._calculate_autocorrelation_time(all_samples)

            # Monte Carlo standard error
            mcse = np.std(all_samples, axis=1, ddof=1) / np.sqrt(ess)

            # Convergence status
            convergence_status = np.all(gelman_rubin < self.convergence_threshold)

            recommendations = []
            if np.any(gelman_rubin > self.convergence_threshold):
                recommendations.append("Chains have not converged, run more iterations")
            if ess < 400:
                recommendations.append("Low effective sample size, increase iterations")
            if autocorr_time > 50:
                recommendations.append("High autocorrelation, consider reparameterization")

        return ConvergenceDiagnostics(
            effective_sample_size=ess,
            gelman_rubin_statistic=gelman_rubin if len(chains_results) > 1 else np.nan,
            autocorrelation_time=autocorr_time,
            monte_carlo_standard_error=float(np.mean(mcse)),
            convergence_status=convergence_status,
            recommendations=recommendations,
            n_iterations=self.n_iterations
        )

    def _calculate_effective_sample_size(self, samples: np.ndarray) -> float:
        """Calculate effective sample size accounting for autocorrelation."""
        n_samples = samples.shape[1]
        ess_values = []

        for param_samples in samples:
            # Calculate autocorrelation
            autocorr = self._calculate_autocorrelation(param_samples)

            # Find where autocorrelation drops below 0.1
            lag_cutoff = np.where(np.abs(autocorr) < 0.1)[0]
            if len(lag_cutoff) > 0:
                lag = lag_cutoff[0]
            else:
                lag = len(autocorr) - 1

            # Effective sample size
            if lag > 0:
                ess = n_samples / (1 + 2 * np.sum(autocorr[1:lag+1]))
            else:
                ess = n_samples

            ess_values.append(ess)

        return float(np.mean(ess_values))

    def _calculate_autocorrelation(self, samples: np.ndarray) -> np.ndarray:
        """Calculate autocorrelation function."""
        n_samples = len(samples)
        if n_samples < 2:
            return np.array([1.0])

        # Remove mean
        centered_samples = samples - np.mean(samples)

        # Calculate autocorrelation
        autocorr = np.correlate(centered_samples, centered_samples, mode='full')
        autocorr = autocorr[n_samples-1:]  # Take second half
        autocorr = autocorr / autocorr[0]  # Normalize

        return autocorr[:min(100, len(autocorr))]  # Limit to 100 lags

    def _calculate_autocorrelation_time(self, samples: np.ndarray) -> float:
        """Calculate integrated autocorrelation time."""
        autocorr = self._calculate_autocorrelation(samples)

        # Find where autocorrelation drops below 0.05
        significant_lags = np.where(np.abs(autocorr) > 0.05)[0]
        if len(significant_lags) > 0:
            return float(significant_lags[-1])
        else:
            return 1.0

    def _calculate_gelman_rubin(self, chains: np.ndarray) -> np.ndarray:
        """Calculate Gelman-Rubin convergence statistic."""
        # chains shape: params x iterations x chains
        n_params, n_iterations, n_chains = chains.shape

        gr_values = []

        for param_idx in range(n_params):
            # Within-chain variance
            chain_means = np.mean(chains[param_idx], axis=1)
            chain_vars = np.var(chains[param_idx], axis=1, ddof=1)

            within_chain_var = np.mean(chain_vars)

            # Between-chain variance
            overall_mean = np.mean(chain_means)
            between_chain_var = n_iterations * np.var(chain_means, ddof=1)

            # Gelman-Rubin statistic
            total_var = ((n_iterations - 1) / n_iterations) * within_chain_var + \
                       (1 / n_iterations) * between_chain_var

            gr_stat = np.sqrt(total_var / within_chain_var)
            gr_values.append(gr_stat)

        return np.array(gr_values)

    def monte_carlo_power_analysis(self, effect_size: float, sample_size: int,
                                 alpha: float = 0.05, power: float = 0.8,
                                 n_simulations: int = 1000) -> Dict:
        """
        Perform Monte Carlo power analysis for experimental design.

        Args:
            effect_size: Expected effect size
            sample_size: Sample size per simulation
            alpha: Significance level
            power: Desired power
            n_simulations: Number of Monte Carlo simulations

        Returns:
            Dictionary with power analysis results
        """
        np.random.seed(self.random_seed)

        p_values = []

        for _ in range(n_simulations):
            # Generate null hypothesis data
            null_data = np.random.normal(0, 1, sample_size)

            # Generate alternative hypothesis data
            alt_data = np.random.normal(effect_size, 1, sample_size)

            # Perform t-test
            _, p_value = stats.ttest_ind(null_data, alt_data)
            p_values.append(p_value)

        # Calculate achieved power
        achieved_power = np.mean(np.array(p_values) < alpha)

        return {
            'effect_size': effect_size,
            'sample_size': sample_size,
            'alpha': alpha,
            'target_power': power,
            'achieved_power': achieved_power,
            'n_simulations': n_simulations,
            'recommendation': 'Increase sample size' if achieved_power < power else 'Sample size adequate'
        }