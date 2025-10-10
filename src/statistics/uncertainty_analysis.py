"""
Uncertainty analysis framework for chronometric interferometry research.

Provides comprehensive uncertainty quantification using Type A (statistical)
and Type B (systematic) analysis methods suitable for research publication.
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import erf
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
import warnings
from dataclasses import dataclass, field
from enum import Enum


class UncertaintyType(Enum):
    """Types of uncertainty analysis."""
    TYPE_A = "type_a"  # Statistical
    TYPE_B = "type_b"  # Systematic
    COMBINED = "combined"


class DistributionType(Enum):
    """Statistical distributions for uncertainty analysis."""
    GAUSSIAN = "gaussian"
    STUDENT_T = "student_t"
    CHI_SQUARE = "chi_square"
    UNIFORM = "uniform"
    TRIANGULAR = "triangular"


@dataclass
class UncertaintyComponent:
    """Individual uncertainty component with full characterization."""
    name: str
    value: float
    uncertainty: float
    distribution: DistributionType = DistributionType.GAUSSIAN
    degrees_of_freedom: Optional[int] = None
    correlation_coefficients: Dict[str, float] = field(default_factory=dict)
    sensitivity_coefficient: float = 1.0
    coverage_factor: float = 2.0  # k=2 for 95% coverage
    units: str = ""
    category: UncertaintyType = UncertaintyType.TYPE_A


@dataclass
class UncertaintyBudget:
    """Complete uncertainty budget for chronometric interferometry measurements."""
    components: List[UncertaintyComponent]
    combined_uncertainty: float
    expanded_uncertainty: float
    coverage_factor: float = 2.0
    confidence_level: float = 0.95
    effective_degrees_of_freedom: Optional[float] = None
    correlation_matrix: Optional[np.ndarray] = None


@dataclass
class BootstrapResult:
    """Results from bootstrap uncertainty analysis."""
    original_estimate: float
    bootstrap_mean: float
    bootstrap_std: float
    confidence_interval: Tuple[float, float]
    bias_corrected_estimate: float
    bias: float
    distribution_skewness: float
    distribution_kurtosis: float
    n_bootstrap: int


class UncertaintyAnalyzer:
    """
    Comprehensive uncertainty analysis for chronometric interferometry research.

    Implements Type A and Type B uncertainty analysis following
    ISO Guide to the Expression of Uncertainty in Measurement (GUM).
    """

    def __init__(self, confidence_level: float = 0.95):
        """
        Initialize uncertainty analyzer.

        Args:
            confidence_level: Desired confidence level for uncertainty intervals
        """
        self.confidence_level = confidence_level
        self.alpha = 1 - confidence_level
        self.coverage_factor = stats.norm.ppf(0.5 + confidence_level / 2)

    def analyze_type_a_uncertainty(self, measurements: np.ndarray,
                                  distribution: DistributionType = DistributionType.GAUSSIAN,
                                  outliers_removed: bool = False) -> UncertaintyComponent:
        """
        Perform Type A (statistical) uncertainty analysis.

        Args:
            measurements: Repeated measurements
            distribution: Assumed distribution
            outliers_removed: Whether outliers have been removed

        Returns:
            UncertaintyComponent with Type A analysis results
        """
        n = len(measurements)
        if n < 2:
            raise ValueError("Need at least 2 measurements for Type A uncertainty analysis")

        # Basic statistics
        mean_value = np.mean(measurements)
        std_dev = np.std(measurements, ddof=1)  # Sample standard deviation
        std_error = std_dev / np.sqrt(n)  # Standard error of the mean

        # Check for normality if sufficient data
        normality_p_value = None
        if n >= 8:
            _, normality_p_value = stats.shapiro(measurements)

        # Determine distribution and degrees of freedom
        if distribution == DistributionType.GAUSSIAN:
            if normality_p_value and normality_p_value < 0.05:
                warnings.warn("Data may not be normally distributed (p={:.3f})".format(normality_p_value))
            degrees_of_freedom = n - 1
        elif distribution == DistributionType.STUDENT_T:
            degrees_of_freedom = n - 1
        else:
            degrees_of_freedom = None

        return UncertaintyComponent(
            name="Type A - Statistical",
            value=mean_value,
            uncertainty=std_error,
            distribution=distribution,
            degrees_of_freedom=degrees_of_freedom,
            coverage_factor=self.coverage_factor,
            category=UncertaintyType.TYPE_A
        )

    def analyze_type_b_uncertainty(self, value: float, uncertainty_info: Dict,
                                  name: str = "Type B Component") -> UncertaintyComponent:
        """
        Perform Type B (systematic) uncertainty analysis.

        Args:
            value: Measured value
            uncertainty_info: Dictionary with uncertainty information
            name: Component name

        Returns:
            UncertaintyComponent with Type B analysis results
        """
        dist_type = uncertainty_info.get('distribution', DistributionType.GAUSSIAN)

        if dist_type == DistributionType.GAUSSIAN:
            # Standard deviation for normal distribution
            std_dev = uncertainty_info.get('std_dev')
            if std_dev is None:
                # If only standard uncertainty provided, assume it's 1σ
                std_dev = uncertainty_info.get('standard_uncertainty', 0)

        elif dist_type == DistributionType.UNIFORM:
            # Uniform distribution: σ = a/√3 where a is half-width
            half_width = uncertainty_info.get('half_width', 0)
            std_dev = half_width / np.sqrt(3)

        elif dist_type == DistributionType.TRIANGULAR:
            # Triangular distribution: σ = a/√6 where a is half-width
            half_width = uncertainty_info.get('half_width', 0)
            std_dev = half_width / np.sqrt(6)

        else:
            raise ValueError(f"Unsupported distribution: {dist_type}")

        # Apply sensitivity coefficient if provided
        sensitivity = uncertainty_info.get('sensitivity_coefficient', 1.0)
        std_dev *= sensitivity

        return UncertaintyComponent(
            name=name,
            value=value,
            uncertainty=std_dev,
            distribution=dist_type,
            sensitivity_coefficient=sensitivity,
            coverage_factor=self.coverage_factor,
            units=uncertainty_info.get('units', ''),
            category=UncertaintyType.TYPE_B
        )

    def combine_uncertainties(self, components: List[UncertaintyComponent],
                            correlation_matrix: Optional[np.ndarray] = None) -> UncertaintyBudget:
        """
        Combine uncertainty components using GUM methodology.

        Args:
            components: List of uncertainty components
            correlation_matrix: Correlation matrix between components

        Returns:
            Complete uncertainty budget
        """
        n_components = len(components)

        # Extract uncertainties and sensitivity coefficients
        uncertainties = np.array([comp.uncertainty for comp in components])
        sensitivities = np.array([comp.sensitivity_coefficient for comp in components])

        # Apply sensitivity coefficients
        weighted_uncertainties = uncertainties * sensitivities

        # Calculate combined uncertainty
        if correlation_matrix is not None:
            # Include correlations
            combined_var = np.dot(weighted_uncertainties.T,
                                np.dot(correlation_matrix, weighted_uncertainties))
        else:
            # Assume uncorrelated components
            combined_var = np.sum(weighted_uncertainties**2)

        combined_uncertainty = np.sqrt(combined_var)

        # Calculate effective degrees of freedom (Welch-Satterthwaite formula)
        effective_dof = self._calculate_effective_degrees_of_freedom(components)

        # Determine coverage factor based on effective degrees of freedom
        if effective_dof is not None and effective_dof < 30:
            # Use Student's t distribution for small samples
            coverage_factor = stats.t.ppf(0.5 + self.confidence_level / 2, effective_dof)
        else:
            coverage_factor = self.coverage_factor

        expanded_uncertainty = coverage_factor * combined_uncertainty

        return UncertaintyBudget(
            components=components,
            combined_uncertainty=combined_uncertainty,
            expanded_uncertainty=expanded_uncertainty,
            coverage_factor=coverage_factor,
            confidence_level=self.confidence_level,
            effective_degrees_of_freedom=effective_dof,
            correlation_matrix=correlation_matrix
        )

    def _calculate_effective_degrees_of_freedom(self, components: List[UncertaintyComponent]) -> Optional[float]:
        """Calculate effective degrees of freedom using Welch-Satterthwaite formula."""
        numerator = 0
        denominator = 0

        for comp in components:
            if comp.degrees_of_freedom is not None and comp.degrees_of_freedom > 0:
                contribution = (comp.uncertainty * comp.sensitivity_coefficient) ** 4
                numerator += contribution
                denominator += contribution / comp.degrees_of_freedom

        if denominator > 0:
            return numerator / denominator
        else:
            return None

    def calculate_uncertainty_budget(self, measurement_data: Dict,
                                   systematic_uncertainties: Dict = None) -> UncertaintyBudget:
        """
        Create complete uncertainty budget for chronometric interferometry measurement.

        Args:
            measurement_data: Dictionary with repeated measurements
            systematic_uncertainties: Dictionary of systematic uncertainty components

        Returns:
            Complete uncertainty budget
        """
        components = []

        # Analyze Type A uncertainties from measurements
        for param_name, measurements in measurement_data.items():
            type_a_component = self.analyze_type_a_uncertainty(
                measurements, name=f"Type A - {param_name}"
            )
            components.append(type_a_component)

        # Analyze Type B uncertainties
        if systematic_uncertainties:
            for param_name, uncertainty_info in systematic_uncertainties.items():
                value = measurement_data.get(param_name, [0])[0]  # Use mean if available
                type_b_component = self.analyze_type_b_uncertainty(
                    value, uncertainty_info, name=f"Type B - {param_name}"
                )
                components.append(type_b_component)

        # Combine all uncertainties
        return self.combine_uncertainties(components)


class BootstrapAnalyzer:
    """
    Bootstrap uncertainty analysis for robust uncertainty quantification.

    Provides non-parametric uncertainty analysis using resampling methods
    suitable for complex distributions and small sample sizes.
    """

    def __init__(self, n_bootstrap: int = 10000, confidence_level: float = 0.95):
        """
        Initialize bootstrap analyzer.

        Args:
            n_bootstrap: Number of bootstrap samples
            confidence_level: Confidence level for intervals
        """
        self.n_bootstrap = n_bootstrap
        self.confidence_level = confidence_level
        self.alpha = 1 - confidence_level

    def bootstrap_analysis(self, data: np.ndarray,
                          statistic: Callable = np.mean,
                          bias_correction: bool = True) -> BootstrapResult:
        """
        Perform bootstrap uncertainty analysis.

        Args:
            data: Input data array
            statistic: Statistic to bootstrap (default: mean)
            bias_correction: Whether to apply bias correction

        Returns:
            BootstrapResult with comprehensive analysis
        """
        n = len(data)
        original_estimate = statistic(data)

        # Generate bootstrap samples
        bootstrap_estimates = np.zeros(self.n_bootstrap)
        for i in range(self.n_bootstrap):
            # Resample with replacement
            bootstrap_sample = np.random.choice(data, size=n, replace=True)
            bootstrap_estimates[i] = statistic(bootstrap_sample)

        # Calculate bootstrap statistics
        bootstrap_mean = np.mean(bootstrap_estimates)
        bootstrap_std = np.std(bootstrap_estimates, ddof=1)

        # Calculate confidence intervals
        percentiles = [(self.alpha / 2) * 100, (1 - self.alpha / 2) * 100]
        confidence_interval = np.percentile(bootstrap_estimates, percentiles)

        # Calculate bias
        bias = bootstrap_mean - original_estimate
        bias_corrected_estimate = original_estimate - bias if bias_correction else original_estimate

        # Calculate distribution moments
        distribution_skewness = stats.skew(bootstrap_estimates)
        distribution_kurtosis = stats.kurtosis(bootstrap_estimates)

        return BootstrapResult(
            original_estimate=original_estimate,
            bootstrap_mean=bootstrap_mean,
            bootstrap_std=bootstrap_std,
            confidence_interval=confidence_interval,
            bias_corrected_estimate=bias_corrected_estimate,
            bias=bias,
            distribution_skewness=distribution_skewness,
            distribution_kurtosis=distribution_kurtosis,
            n_bootstrap=self.n_bootstrap
        )

    def bootstrap_parameter_estimation(self, x_data: np.ndarray, y_data: np.ndarray,
                                     estimator: Callable) -> BootstrapResult:
        """
        Bootstrap parameter estimation for regression problems.

        Args:
            x_data: Independent variable data
            y_data: Dependent variable data
            estimator: Parameter estimation function

        Returns:
            BootstrapResult for parameter estimation
        """
        n = len(x_data)
        original_params = estimator(x_data, y_data)

        # For simplicity, focus on first parameter
        original_estimate = original_params[0] if isinstance(original_params, (list, np.ndarray)) else original_params

        bootstrap_estimates = np.zeros(self.n_bootstrap)
        for i in range(self.n_bootstrap):
            # Resample with replacement
            indices = np.random.choice(n, size=n, replace=True)
            x_boot = x_data[indices]
            y_boot = y_data[indices]

            try:
                boot_params = estimator(x_boot, y_boot)
                bootstrap_estimates[i] = boot_params[0] if isinstance(boot_params, (list, np.ndarray)) else boot_params
            except:
                # Handle estimation failures
                bootstrap_estimates[i] = np.nan

        # Remove NaN values
        bootstrap_estimates = bootstrap_estimates[~np.isnan(bootstrap_estimates)]

        if len(bootstrap_estimates) == 0:
            raise ValueError("All bootstrap estimations failed")

        # Calculate bootstrap statistics
        bootstrap_mean = np.mean(bootstrap_estimates)
        bootstrap_std = np.std(bootstrap_estimates, ddof=1)

        # Calculate confidence intervals
        percentiles = [(self.alpha / 2) * 100, (1 - self.alpha / 2) * 100]
        confidence_interval = np.percentile(bootstrap_estimates, percentiles)

        # Calculate bias
        bias = bootstrap_mean - original_estimate
        bias_corrected_estimate = original_estimate - bias

        # Calculate distribution moments
        distribution_skewness = stats.skew(bootstrap_estimates)
        distribution_kurtosis = stats.kurtosis(bootstrap_estimates)

        return BootstrapResult(
            original_estimate=original_estimate,
            bootstrap_mean=bootstrap_mean,
            bootstrap_std=bootstrap_std,
            confidence_interval=confidence_interval,
            bias_corrected_estimate=bias_corrected_estimate,
            bias=bias,
            distribution_skewness=distribution_skewness,
            distribution_kurtosis=distribution_kurtosis,
            n_bootstrap=self.n_bootstrap
        )


class BayesianAnalyzer:
    """
    Bayesian uncertainty analysis for comprehensive uncertainty quantification.

    Provides Bayesian inference methods for uncertainty analysis with
    prior information and posterior distributions.
    """

    def __init__(self, confidence_level: float = 0.95):
        """
        Initialize Bayesian analyzer.

        Args:
            confidence_level: Confidence level for credible intervals
        """
        self.confidence_level = confidence_level

    def bayesian_parameter_estimation(self, data: np.ndarray,
                                    prior_mean: float, prior_std: float,
                                    likelihood_std: float) -> Dict:
        """
        Bayesian parameter estimation with conjugate priors.

        Args:
            data: Observed data
            prior_mean: Prior distribution mean
            prior_std: Prior distribution standard deviation
            likelihood_std: Likelihood standard deviation

        Returns:
            Dictionary with posterior analysis results
        """
        n = len(data)
        data_mean = np.mean(data)

        # Posterior parameters for normal-normal conjugate case
        prior_precision = 1 / prior_std**2
        likelihood_precision = n / likelihood_std**2

        posterior_precision = prior_precision + likelihood_precision
        posterior_std = 1 / np.sqrt(posterior_precision)
        posterior_mean = (prior_precision * prior_mean + likelihood_precision * data_mean) / posterior_precision

        # Calculate credible interval
        z_score = stats.norm.ppf(0.5 + self.confidence_level / 2)
        credible_interval = (
            posterior_mean - z_score * posterior_std,
            posterior_mean + z_score * posterior_std
        )

        return {
            'posterior_mean': posterior_mean,
            'posterior_std': posterior_std,
            'credible_interval': credible_interval,
            'prior_mean': prior_mean,
            'prior_std': prior_std,
            'data_mean': data_mean,
            'n_samples': n
        }

    def mcmc_uncertainty_analysis(self, log_likelihood: Callable,
                                 initial_params: np.ndarray,
                                 param_bounds: List[Tuple[float, float]],
                                 n_iterations: int = 10000,
                                 burn_in: int = 1000) -> Dict:
        """
        Markov Chain Monte Carlo uncertainty analysis.

        Args:
            log_likelihood: Log likelihood function
            initial_params: Initial parameter values
            param_bounds: Parameter bounds for sampling
            n_iterations: Number of MCMC iterations
            burn_in: Number of burn-in iterations

        Returns:
            Dictionary with MCMC analysis results
        """
        # Simplified Metropolis-Hastings MCMC
        n_params = len(initial_params)
        chains = np.zeros((n_iterations, n_params))
        current_params = initial_params.copy()
        current_log_likelihood = log_likelihood(current_params)

        # Proposal distribution scale
        proposal_scales = [0.1 * (bounds[1] - bounds[0]) for bounds in param_bounds]

        accepted = 0
        for i in range(n_iterations):
            # Propose new parameters
            proposal_params = current_params + np.random.normal(0, proposal_scales, n_params)

            # Check bounds
            within_bounds = all(param_bounds[j][0] <= proposal_params[j] <= param_bounds[j][1]
                              for j in range(n_params))

            if within_bounds:
                proposal_log_likelihood = log_likelihood(proposal_params)

                # Acceptance probability
                log_acceptance_prob = proposal_log_likelihood - current_log_likelihood
                accept_prob = min(1.0, np.exp(log_acceptance_prob))

                if np.random.random() < accept_prob:
                    current_params = proposal_params
                    current_log_likelihood = proposal_log_likelihood
                    accepted += 1

            chains[i] = current_params

        acceptance_rate = accepted / n_iterations

        # Remove burn-in
        posterior_samples = chains[burn_in:]

        # Calculate posterior statistics
        posterior_means = np.mean(posterior_samples, axis=0)
        posterior_stds = np.std(posterior_samples, axis=0)

        # Calculate credible intervals
        alpha = 1 - self.confidence_level
        credible_intervals = []
        for j in range(n_params):
            percentiles = [alpha / 2 * 100, (1 - alpha / 2) * 100]
            credible_intervals.append(np.percentile(posterior_samples[:, j], percentiles))

        return {
            'posterior_samples': posterior_samples,
            'posterior_means': posterior_means,
            'posterior_stds': posterior_stds,
            'credible_intervals': credible_intervals,
            'acceptance_rate': acceptance_rate,
            'n_iterations': n_iterations,
            'burn_in': burn_in
        }