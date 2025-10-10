"""
Uncertainty quantification visualization for chronometric interferometry research.

Provides sophisticated visualization of measurement uncertainties, confidence
intervals, error propagation, and statistical validation for research publications.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.special import erf
from typing import Dict, List, Tuple, Optional, Union, Any
import warnings
from dataclasses import dataclass

from ..base.figure_generator import FigureGenerator
from ..base.styles import IEEEStyle, NASAStyle
from ..base.utils import format_scientific_notation, format_uncertainty


@dataclass
class UncertaintyData:
    """Container for uncertainty analysis data."""
    nominal_values: np.ndarray
    uncertainties: np.ndarray
    confidence_intervals: np.ndarray  # Shape: (n, 2) for lower/upper bounds
    distribution_type: str = 'gaussian'
    correlation_matrix: Optional[np.ndarray] = None
    parameter_names: Optional[List[str]] = None


@dataclass
class ErrorBudgetData:
    """Container for error budget analysis data."""
    error_sources: List[str]
    contributions: np.ndarray  # Shape: (n_sources, n_measurements)
    total_uncertainty: np.ndarray
    correlation_matrix: Optional[np.ndarray] = None
    sensitivity_coefficients: Optional[np.ndarray] = None


class UncertaintyQuantificationPlotter:
    """
    Advanced uncertainty quantification visualization for chronometric interferometry.

    Creates publication-quality visualizations of measurement uncertainties,
    confidence intervals, error propagation, and statistical validation.
    """

    def __init__(self, figure_generator: Optional[FigureGenerator] = None,
                 style: str = 'ieee'):
        """
        Initialize uncertainty quantification plotter.

        Args:
            figure_generator: Figure generator instance
            style: Publication style ('ieee' or 'nasa')
        """
        self.fig_gen = figure_generator or FigureGenerator(default_style=style)
        self.style = style

    def plot_uncertainty_comparison(self, uncertainty_data: Dict[str, UncertaintyData],
                                  title: str = "Uncertainty Comparison",
                                  normalize: bool = False) -> plt.Figure:
        """
        Create comprehensive uncertainty comparison across multiple methods or conditions.

        Args:
            uncertainty_data: Dictionary of UncertaintyData objects
            title: Plot title
            normalize: Whether to normalize uncertainties

        Returns:
            Matplotlib figure
        """
        n_methods = len(uncertainty_data)
        method_names = list(uncertainty_data.keys())

        spec = FigureSpec(
            title=title,
            width=7.0,
            height=5.0,
            style=self.style
        )

        fig = self.fig_gen.create_figure(spec)

        # Create subplots
        ax1 = self.fig_gen.add_subplot(fig, (0, 0), title='(a) Uncertainty Magnitudes')
        ax2 = self.fig_gen.add_subplot(fig, (0, 1), title='(b) Relative Contributions')

        # Plot uncertainty magnitudes
        x_positions = np.arange(n_methods)
        uncertainties = [np.mean(unc_data.uncertainties) for unc_data in uncertainty_data.values()]
        error_bars = [np.std(unc_data.uncertainties) for unc_data in uncertainty_data.values()]

        if normalize:
            uncertainties = np.array(uncertainties) / np.min(uncertainties)
            error_bars = np.array(error_bars) / np.min(uncertainties)
            ax1.set_ylabel('Normalized Uncertainty')
        else:
            ax1.set_ylabel('Uncertainty (ps)')

        bars = ax1.bar(x_positions, uncertainties, yerr=error_bars,
                      capsize=5, alpha=0.7, color='#0072BD')
        ax1.set_xticks(x_positions)
        ax1.set_xticklabels(method_names, rotation=45, ha='right')
        ax1.grid(True, alpha=0.3)

        # Highlight best performing method
        best_idx = np.argmin(uncertainties)
        bars[best_idx].set_color('#D95319')

        # Plot uncertainty distributions
        for i, (name, unc_data) in enumerate(uncertainty_data.items()):
            if len(unc_data.nominal_values) > 10:  # Only plot if sufficient data
                self._plot_uncertainty_distribution(ax2, unc_data, x_positions[i], name)

        ax2.set_xlabel('Method/Condition')
        ax2.set_ylabel('Probability Density')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def plot_confidence_interval_analysis(self, measurements: np.ndarray,
                                        uncertainties: np.ndarray,
                                        confidence_levels: List[float] = [0.68, 0.95, 0.997],
                                        true_value: Optional[float] = None,
                                        title: str = "Confidence Interval Analysis") -> plt.Figure:
        """
        Create confidence interval analysis with multiple confidence levels.

        Args:
            measurements: Measured values
            uncertainties: Measurement uncertainties
            confidence_levels: List of confidence levels to plot
            true_value: True value for comparison
            title: Plot title

        Returns:
            Matplotlib figure
        """
        n_measurements = len(measurements)
        x_positions = np.arange(n_measurements)

        spec = FigureSpec(
            title=title,
            width=7.0,
            height=4.0,
            style=self.style
        )

        fig = self.fig_gen.create_figure(spec)
        ax = self.fig_gen.add_subplot(fig, 0)

        # Calculate confidence intervals for each level
        colors = ['#0072BD', '#D95319', '#77AC30', '#A2142F']
        alphas = [0.3, 0.5, 0.7, 0.9]

        for i, conf_level in enumerate(confidence_levels):
            if i >= len(colors):
                break

            # Calculate confidence interval multiplier
            if conf_level < 0.5:
                z_score = stats.norm.ppf(0.5 + conf_level/2)
            else:
                z_score = stats.norm.ppf(0.5 + conf_level/2)

            ci_width = z_score * uncertainties

            # Plot confidence intervals
            ax.fill_between(x_positions, measurements - ci_width, measurements + ci_width,
                          alpha=alphates[i], color=colors[i],
                          label=f'{conf_level*100:.1f}% CI')

        # Plot measurements
        ax.plot(x_positions, measurements, 'o', color='black', markersize=6,
               label='Measurements', zorder=5)

        # Plot true value if provided
        if true_value is not None:
            ax.axhline(y=true_value, color='red', linestyle='--', linewidth=2,
                      label=f'True Value ({format_scientific_notation(true_value)})')

        # Add coverage statistics
        coverage_stats = {}
        for conf_level in confidence_levels:
            if conf_level < 0.5:
                z_score = stats.norm.ppf(0.5 + conf_level/2)
            else:
                z_score = stats.norm.ppf(0.5 + conf_level/2)

            ci_width = z_score * uncertainties
            within_ci = np.abs(measurements - true_value) <= ci_width if true_value is not None else None
            coverage = np.sum(within_ci) / n_measurements if within_ci is not None else None
            coverage_stats[f'{conf_level*100:.0f}%'] = coverage

        # Add coverage statistics text
        if true_value is not None:
            coverage_text = "Coverage:\n"
            for level, coverage in coverage_stats.items():
                coverage_text += f"{level}: {coverage*100:.1f}%\n"
            ax.text(0.02, 0.98, coverage_text, transform=ax.transAxes,
                   verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

        ax.set_xlabel('Measurement Index')
        ax.set_ylabel('Value')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

        return fig

    def plot_error_budget_pie_chart(self, error_budget: ErrorBudgetData,
                                   title: str = "Error Budget Analysis") -> plt.Figure:
        """
        Create error budget pie chart showing contribution of each error source.

        Args:
            error_budget: ErrorBudgetData object
            title: Plot title

        Returns:
            Matplotlib figure
        """
        spec = FigureSpec(
            title=title,
            width=7.0,
            height=4.0,
            style=self.style
        )

        fig = self.fig_gen.create_figure(spec)

        # Create subplots
        ax1 = self.fig_gen.add_subplot(fig, (0, 0), title='(a) Error Source Contributions')
        ax2 = self.fig_gen.add_subplot(fig, (0, 1), title='(b) Contribution vs. Measurement')

        # Calculate average contributions
        avg_contributions = np.mean(np.abs(error_budget.contributions), axis=1)
        total_contribution = np.sum(avg_contributions)
        percentages = avg_contributions / total_contribution * 100

        # Create pie chart
        colors = plt.cm.Set3(np.linspace(0, 1, len(error_budget.error_sources)))
        wedges, texts, autotexts = ax1.pie(percentages, labels=error_budget.error_sources,
                                          autopct='%1.1f%%', colors=colors, startangle=90)

        # Format text
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')

        # Plot contribution trends
        x_positions = np.arange(len(error_budget.contributions[0]))
        bottom = np.zeros(len(x_positions))

        for i, (source, contributions) in enumerate(zip(error_budget.error_sources, error_budget.contributions)):
            ax2.bar(x_positions, contributions, bottom=bottom, label=source,
                   color=colors[i], alpha=0.8)
            bottom += contributions

        ax2.set_xlabel('Measurement Index')
        ax2.set_ylabel('Error Contribution')
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def plot_correlation_matrix(self, correlation_matrix: np.ndarray,
                               parameter_names: List[str],
                               title: str = "Parameter Correlation Matrix") -> plt.Figure:
        """
        Create correlation matrix heatmap.

        Args:
            correlation_matrix: Correlation matrix
            parameter_names: List of parameter names
            title: Plot title

        Returns:
            Matplotlib figure
        """
        spec = FigureSpec(
            title=title,
            width=6.0,
            height=5.0,
            style=self.style
        )

        fig = self.fig_gen.create_figure(spec)
        ax = self.fig_gen.add_subplot(fig, 0)

        # Create heatmap
        im = ax.imshow(correlation_matrix, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')

        # Add colorbar
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label('Correlation Coefficient')

        # Set ticks and labels
        ax.set_xticks(np.arange(len(parameter_names)))
        ax.set_yticks(np.arange(len(parameter_names)))
        ax.set_xticklabels(parameter_names, rotation=45, ha='right')
        ax.set_yticklabels(parameter_names)

        # Add correlation values as text
        for i in range(len(parameter_names)):
            for j in range(len(parameter_names)):
                text = ax.text(j, i, f'{correlation_matrix[i, j]:.2f}',
                             ha="center", va="center", color="black", fontweight='bold')

        ax.set_title(title)
        plt.tight_layout()
        return fig

    def plot_uncertainty_propagation(self, input_parameters: Dict[str, np.ndarray],
                                   output_values: np.ndarray,
                                   sensitivity_coefficients: np.ndarray,
                                   title: str = "Uncertainty Propagation Analysis") -> plt.Figure:
        """
        Create uncertainty propagation analysis showing how input uncertainties affect output.

        Args:
            input_parameters: Dictionary of input parameter arrays
            output_values: Output values
            sensitivity_coefficients: Sensitivity coefficients matrix
            title: Plot title

        Returns:
            Matplotlib figure
        """
        n_params = len(input_parameters)
        param_names = list(input_parameters.keys())

        spec = FigureSpec(
            title=title,
            width=7.0,
            height=5.0,
            style=self.style
        )

        fig = self.fig_gen.create_figure(spec)

        # Create subplots
        ax1 = self.fig_gen.add_subplot(fig, (0, 0), title='(a) Input Parameter Distributions')
        ax2 = self.fig_gen.add_subplot(fig, (0, 1), title='(b) Sensitivity Coefficients')

        # Plot input parameter distributions
        x_positions = np.arange(n_params)
        means = [np.mean(values) for values in input_parameters.values()]
        stds = [np.std(values) for values in input_parameters.values()]

        ax1.errorbar(x_positions, means, yerr=stds, fmt='o', capsize=5,
                    markersize=8, color='#0072BD')
        ax1.set_xticks(x_positions)
        ax1.set_xticklabels(param_names, rotation=45, ha='right')
        ax1.set_ylabel('Parameter Value')
        ax1.grid(True, alpha=0.3)

        # Plot sensitivity coefficients
        if sensitivity_coefficients.ndim == 1:
            sensitivities = sensitivity_coefficients
        else:
            # Use RMS of sensitivity coefficients for each parameter
            sensitivities = np.sqrt(np.mean(sensitivity_coefficients**2, axis=1))

        bars = ax2.bar(x_positions, np.abs(sensitivities), color='#D95319', alpha=0.7)
        ax2.set_xticks(x_positions)
        ax2.set_xticklabels(param_names, rotation=45, ha='right')
        ax2.set_ylabel('|Sensitivity Coefficient|')
        ax2.grid(True, alpha=0.3)

        # Highlight most sensitive parameter
        most_sensitive_idx = np.argmax(np.abs(sensitivities))
        bars[most_sensitive_idx].set_color('#A2142F')

        # Add contribution analysis
        contributions = stds * np.abs(sensitivities)
        total_uncertainty = np.sqrt(np.sum(contributions**2))
        contribution_percentages = contributions**2 / total_uncertainty**2 * 100

        # Add contribution text
        contrib_text = "Uncertainty Contributions:\n"
        for i, (name, contrib_pct) in enumerate(zip(param_names, contribution_percentages)):
            contrib_text += f"{name}: {contrib_pct:.1f}%\n"
        ax2.text(0.02, 0.98, contrib_text, transform=ax2.transAxes,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

        plt.tight_layout()
        return fig

    def plot_monte_carlo_validation(self, true_values: np.ndarray,
                                   estimated_values: np.ndarray,
                                   uncertainties: np.ndarray,
                                   n_iterations: int = 10000,
                                   title: str = "Monte Carlo Validation") -> plt.Figure:
        """
        Create Monte Carlo validation plot comparing estimated distributions to true values.

        Args:
            true_values: True parameter values
            estimated_values: Estimated parameter values
            uncertainties: Estimation uncertainties
            n_iterations: Number of Monte Carlo iterations
            title: Plot title

        Returns:
            Matplotlib figure
        """
        spec = FigureSpec(
            title=title,
            width=7.0,
            height=4.0,
            style=self.style
        )

        fig = self.fig_gen.create_figure(spec)

        # Create subplots
        ax1 = self.fig_gen.add_subplot(fig, (0, 0), title='(a) Validation Results')
        ax2 = self.fig_gen.add_subplot(fig, (0, 1), title='(b) Coverage Analysis')

        # Plot estimation vs. true values
        n_params = len(true_values)
        x_positions = np.arange(n_params)

        ax1.errorbar(x_positions, estimated_values, yerr=uncertainties,
                    fmt='o', capsize=5, markersize=8, color='#0072BD',
                    label='Estimated ± σ')
        ax1.plot(x_positions, true_values, 's', markersize=8, color='#D95319',
                label='True Value')

        ax1.set_xlabel('Parameter Index')
        ax1.set_ylabel('Parameter Value')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Generate Monte Carlo samples and calculate coverage
        n_covered = np.zeros(n_params)
        for i in range(n_iterations):
            samples = np.random.normal(estimated_values, uncertainties)
            within_uncertainty = np.abs(samples - true_values) <= uncertainties
            n_covered += np.sum(within_uncertainty, axis=0) / n_iterations

        coverage_percentages = n_covered / n_iterations * 100
        expected_coverage = 68.3  # 1σ coverage for normal distribution

        # Plot coverage analysis
        bars = ax2.bar(x_positions, coverage_percentages, color='#77AC30', alpha=0.7)
        ax2.axhline(y=expected_coverage, color='red', linestyle='--', linewidth=2,
                   label=f'Expected ({expected_coverage}%)')
        ax2.set_xlabel('Parameter Index')
        ax2.set_ylabel('Coverage (%)')
        ax2.set_ylim([0, 100])
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Highlight poor coverage
        poor_coverage_idx = np.where(coverage_percentages < expected_coverage * 0.9)[0]
        for idx in poor_coverage_idx:
            bars[idx].set_color('#A2142F')

        plt.tight_layout()
        return fig

    def plot_bootstrap_uncertainty(self, measurements: np.ndarray,
                                 bootstrap_results: np.ndarray,
                                 confidence_levels: List[float] = [0.68, 0.95, 0.99],
                                 title: str = "Bootstrap Uncertainty Analysis") -> plt.Figure:
        """
        Create bootstrap uncertainty analysis plot.

        Args:
            measurements: Original measurements
            bootstrap_results: Bootstrap resampling results (n_iterations x n_measurements)
            confidence_levels: Confidence levels to compute
            title: Plot title

        Returns:
            Matplotlib figure
        """
        spec = FigureSpec(
            title=title,
            width=7.0,
            height=4.0,
            style=self.style
        )

        fig = self.fig_gen.create_figure(spec)

        # Create subplots
        ax1 = self.fig_gen.add_subplot(fig, (0, 0), title='(a) Bootstrap Distribution')
        ax2 = self.fig_gen.add_subplot(fig, (0, 1), title='(b) Confidence Intervals')

        # Plot bootstrap distribution for first measurement
        first_measurement_bootstrap = bootstrap_results[:, 0]
        ax1.hist(first_measurement_bootstrap, bins=50, alpha=0.7, color='#0072BD',
                density=True, label='Bootstrap Distribution')
        ax1.axvline(x=np.mean(first_measurement_bootstrap), color='red',
                   linestyle='-', linewidth=2, label='Bootstrap Mean')
        ax1.axvline(x=measurements[0], color='green', linestyle='--',
                   linewidth=2, label='Original Measurement')
        ax1.set_xlabel('Value')
        ax1.set_ylabel('Probability Density')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Calculate confidence intervals for all measurements
        n_measurements = measurements.shape[0]
        x_positions = np.arange(n_measurements)

        colors = ['#D95319', '#77AC30', '#A2142F']
        for i, conf_level in enumerate(confidence_levels):
            if i >= len(colors):
                break

            percentiles = [(100 - conf_level * 100) / 2, 50, 100 - (100 - conf_level * 100) / 2]
            ci_values = np.percentile(bootstrap_results, percentiles, axis=0)

            ax2.fill_between(x_positions, ci_values[0], ci_values[2],
                          alpha=0.3, color=colors[i],
                          label=f'{conf_level*100:.0f}% CI')

        # Plot measurements
        ax2.plot(x_positions, measurements, 'o', color='black', markersize=6,
                label='Measurements', zorder=5)

        ax2.set_xlabel('Measurement Index')
        ax2.set_ylabel('Value')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def _plot_uncertainty_distribution(self, ax: plt.Axes, unc_data: UncertaintyData,
                                     x_position: float, label: str) -> None:
        """Plot uncertainty distribution as a small distribution."""
        if unc_data.distribution_type == 'gaussian':
            x_range = np.linspace(unc_data.nominal_values.mean() - 3 * unc_data.uncertainties.mean(),
                                 unc_data.nominal_values.mean() + 3 * unc_data.uncertainties.mean(),
                                 100)
            pdf = stats.norm.pdf(x_range, unc_data.nominal_values.mean(),
                                unc_data.uncertainties.mean())

            # Scale and offset for visualization
            pdf_scaled = pdf * 0.8  # Scale for visibility
            ax.fill_between(x_range, x_position - pdf_scaled/2, x_position + pdf_scaled/2,
                          alpha=0.6, label=label)


# Import FigureSpec for use in this module
from dataclasses import dataclass
from typing import Optional, List, Dict

@dataclass
class FigureSpec:
    """Figure specification for uncertainty quantification plots."""
    title: str = ""
    width: float = 7.0
    height: Optional[float] = None
    style: str = 'ieee'