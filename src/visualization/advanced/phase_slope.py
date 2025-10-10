"""
Advanced phase-slope analysis visualization for chronometric interferometry.

Provides sophisticated visualization and analysis of phase-slope estimation
with confidence intervals, uncertainty quantification, and statistical validation.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import curve_fit
from typing import Dict, Tuple, Optional, List, Union
import warnings
from dataclasses import dataclass

from ..base.figure_generator import FigureGenerator
from ..base.styles import IEEEStyle, NASAStyle
from ..base.utils import format_scientific_notation, format_uncertainty


@dataclass
class PhaseSlopeResult:
    """Results from phase-slope analysis with uncertainty quantification."""
    slope: float
    slope_uncertainty: float
    intercept: float
    intercept_uncertainty: float
    r_squared: float
    rmse: float
    timing_estimate: float  # τ = Δφ/(2π·Δf)
    timing_uncertainty: float
    frequency_offset: float  # Δf from slope
    frequency_uncertainty: float
    confidence_intervals: Dict[str, Tuple[float, float]]
    fit_quality: Dict[str, float]
    residuals: np.ndarray
    fit_line: np.ndarray


class PhaseSlopeAnalyzer:
    """
    Advanced phase-slope analysis for chronometric interferometry.

    Implements sophisticated linear regression with uncertainty quantification,
    outlier detection, and statistical validation for τ/Δf extraction.
    """

    def __init__(self, confidence_level: float = 0.95):
        """
        Initialize phase-slope analyzer.

        Args:
            confidence_level: Confidence level for intervals (0.0-1.0)
        """
        self.confidence_level = confidence_level
        self.alpha = 1 - confidence_level

    def linear_fit_with_uncertainty(self, x_data: np.ndarray, y_data: np.ndarray,
                                  y_uncertainty: Optional[np.ndarray] = None,
                                  outlier_method: str = 'none',
                                  robust_fitting: bool = False) -> PhaseSlopeResult:
        """
        Perform linear regression with comprehensive uncertainty analysis.

        Args:
            x_data: Independent variable (time)
            y_data: Dependent variable (phase)
            y_uncertainty: Measurement uncertainties
            outlier_method: Outlier detection method ('none', 'sigma', 'iqr', 'cook')
            robust_fitting: Whether to use robust regression

        Returns:
            PhaseSlopeResult with comprehensive analysis
        """
        if len(x_data) < 3:
            raise ValueError("Need at least 3 data points for linear regression")

        # Remove outliers if requested
        x_clean, y_clean, y_unc_clean = self._remove_outliers(
            x_data, y_data, y_uncertainty, method=outlier_method
        )

        # Perform linear regression
        if robust_fitting:
            slope, intercept, slope_unc, intercept_unc = self._robust_linear_fit(
                x_clean, y_clean, y_unc_clean
            )
        else:
            slope, intercept, slope_unc, intercept_unc = self._weighted_linear_fit(
                x_clean, y_clean, y_unc_clean
            )

        # Calculate fit statistics
        y_pred = slope * x_clean + intercept
        residuals = y_clean - y_pred

        # R-squared
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((y_clean - np.mean(y_clean))**2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        # RMSE
        rmse = np.sqrt(np.mean(residuals**2))

        # Calculate confidence intervals
        n = len(x_clean)
        x_mean = np.mean(x_clean)
        s_xx = np.sum((x_clean - x_mean)**2)

        # Critical t-value
        t_crit = stats.t.ppf(1 - self.alpha/2, n - 2)

        # Confidence intervals for predictions
        confidence_intervals = self._calculate_confidence_intervals(
            x_clean, slope, intercept, slope_unc, intercept_unc,
            s_xx, t_crit, residuals
        )

        # Extract timing and frequency parameters
        frequency_offset = slope / (2 * np.pi)  # Δf in Hz
        frequency_uncertainty = slope_unc / (2 * np.pi)

        # Timing estimate τ = Δφ/(2π·Δf)
        if abs(frequency_offset) > 1e-10:
            timing_estimate = intercept / (2 * np.pi * frequency_offset)
            timing_uncertainty = self._propagate_timing_uncertainty(
                intercept, intercept_unc, frequency_offset, frequency_uncertainty
            )
        else:
            timing_estimate = np.nan
            timing_uncertainty = np.nan

        # Additional fit quality metrics
        fit_quality = {
            'aic': self._calculate_aic(residuals, 2),  # 2 parameters
            'bic': self._calculate_bic(residuals, 2, n),
            'jarque_bera': stats.jarque_bera(residuals)[1],  # p-value
            'durbin_watson': self._durbin_watson(residuals),
            'effective_sample_size': n
        }

        return PhaseSlopeResult(
            slope=slope,
            slope_uncertainty=slope_unc,
            intercept=intercept,
            intercept_uncertainty=intercept_unc,
            r_squared=r_squared,
            rmse=rmse,
            timing_estimate=timing_estimate,
            timing_uncertainty=timing_uncertainty,
            frequency_offset=frequency_offset,
            frequency_uncertainty=frequency_uncertainty,
            confidence_intervals=confidence_intervals,
            fit_quality=fit_quality,
            residuals=residuals,
            fit_line=y_pred
        )

    def _remove_outliers(self, x_data: np.ndarray, y_data: np.ndarray,
                        y_uncertainty: Optional[np.ndarray],
                        method: str) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """Remove outliers using specified method."""
        if method == 'none':
            return x_data, y_data, y_uncertainty

        # Initial linear fit
        slope, intercept = np.polyfit(x_data, y_data, 1)
        residuals = y_data - (slope * x_data + intercept)

        if method == 'sigma':
            # Remove points > 3 sigma from fit
            std_residual = np.std(residuals)
            mask = np.abs(residuals) < 3 * std_residual
        elif method == 'iqr':
            # Remove points beyond 1.5 * IQR
            q75, q25 = np.percentile(residuals, [75, 25])
            iqr = q75 - q25
            mask = np.abs(residuals) < 1.5 * iqr
        elif method == 'cook':
            # Cook's distance outlier detection
            mask = self._cook_distance_outliers(x_data, y_data)
        else:
            raise ValueError(f"Unknown outlier method: {method}")

        return x_data[mask], y_data[mask], y_uncertainty[mask] if y_uncertainty is not None else None

    def _weighted_linear_fit(self, x_data: np.ndarray, y_data: np.ndarray,
                             y_uncertainty: Optional[np.ndarray]) -> Tuple[float, float, float, float]:
        """Perform weighted linear regression."""
        if y_uncertainty is None:
            # Ordinary least squares
            coeffs = np.polyfit(x_data, y_data, 1, cov=True)
            slope, intercept = coeffs[0]
            cov_matrix = coeffs[1]
            slope_unc = np.sqrt(cov_matrix[0, 0])
            intercept_unc = np.sqrt(cov_matrix[1, 1])
        else:
            # Weighted least squares
            weights = 1.0 / (y_uncertainty**2)
            w = np.sum(weights)
            wx = np.sum(weights * x_data)
            wy = np.sum(weights * y_data)
            wxy = np.sum(weights * x_data * y_data)
            wxx = np.sum(weights * x_data**2)

            denominator = w * wxx - wx**2
            slope = (w * wxy - wx * wy) / denominator
            intercept = (wxx * wy - wx * wxy) / denominator

            # Uncertainty estimates
            y_pred = slope * x_data + intercept
            residuals = y_data - y_pred
            sigma_squared = np.sum(weights * residuals**2) / (len(x_data) - 2)

            slope_unc = np.sqrt(w * sigma_squared / denominator)
            intercept_unc = np.sqrt(wxx * sigma_squared / denominator)

        return slope, intercept, slope_unc, intercept_unc

    def _robust_linear_fit(self, x_data: np.ndarray, y_data: np.ndarray,
                          y_uncertainty: Optional[np.ndarray]) -> Tuple[float, float, float, float]:
        """Perform robust linear regression using Huber loss."""
        try:
            from sklearn.linear_model import HuberRegressor
        except ImportError:
            warnings.warn("scikit-learn not available, falling back to OLS")
            return self._weighted_linear_fit(x_data, y_data, y_uncertainty)

        # Robust regression
        huber = HuberRegressor(epsilon=1.35)
        huber.fit(x_data.reshape(-1, 1), y_data)

        slope = huber.coef_[0]
        intercept = huber.intercept_

        # Estimate uncertainties using bootstrap
        n_bootstrap = 1000
        bootstrap_slopes = []
        bootstrap_intercepts = []

        for _ in range(n_bootstrap):
            # Resample with replacement
            indices = np.random.choice(len(x_data), len(x_data), replace=True)
            x_boot = x_data[indices]
            y_boot = y_data[indices]

            huber_boot = HuberRegressor(epsilon=1.35)
            huber_boot.fit(x_boot.reshape(-1, 1), y_boot)

            bootstrap_slopes.append(huber_boot.coef_[0])
            bootstrap_intercepts.append(huber_boot.intercept_)

        slope_unc = np.std(bootstrap_slopes)
        intercept_unc = np.std(bootstrap_intercepts)

        return slope, intercept, slope_unc, intercept_unc

    def _calculate_confidence_intervals(self, x_data: np.ndarray, slope: float,
                                      intercept: float, slope_unc: float,
                                      intercept_unc: float, s_xx: float,
                                      t_crit: float, residuals: np.ndarray) -> Dict[str, Tuple[float, float]]:
        """Calculate confidence intervals for fit parameters and predictions."""
        n = len(x_data)
        x_mean = np.mean(x_data)
        mse = np.sum(residuals**2) / (n - 2)

        # Parameter confidence intervals
        slope_ci = (
            slope - t_crit * slope_unc,
            slope + t_crit * slope_unc
        )
        intercept_ci = (
            intercept - t_crit * intercept_unc,
            intercept + t_crit * intercept_unc
        )

        # Prediction confidence intervals (at mean x)
        pred_unc = np.sqrt(mse * (1/n + (x_mean - x_mean)**2 / s_xx))
        pred_y = slope * x_mean + intercept
        prediction_ci = (
            pred_y - t_crit * pred_unc,
            pred_y + t_crit * pred_unc
        )

        return {
            'slope': slope_ci,
            'intercept': intercept_ci,
            'prediction': prediction_ci
        }

    def _propagate_timing_uncertainty(self, intercept: float, intercept_unc: float,
                                    frequency_offset: float, frequency_unc: float) -> float:
        """Propagate uncertainty for timing estimate τ = intercept/(2π·frequency_offset)."""
        # Partial derivatives
        dtau_dintercept = 1 / (2 * np.pi * frequency_offset)
        dtau_dfreq = -intercept / (2 * np.pi * frequency_offset**2)

        # Uncertainty propagation (assuming uncorrelated)
        timing_unc = np.sqrt(
            (dtau_dintercept * intercept_unc)**2 + (dtau_dfreq * frequency_unc)**2
        )

        return timing_unc

    def _cook_distance_outliers(self, x_data: np.ndarray, y_data: np.ndarray) -> np.ndarray:
        """Detect outliers using Cook's distance."""
        # Simple implementation - in practice, use statsmodels for full analysis
        leverage = 1/len(x_data) + (x_data - np.mean(x_data))**2 / np.sum((x_data - np.mean(x_data))**2)
        residuals = y_data - np.polyval(np.polyfit(x_data, y_data, 1), x_data)
        mse = np.mean(residuals**2)
        cooks_d = leverage * residuals**2 / (2 * mse * (1 - leverage)**2)

        # Threshold: 4/n (common rule of thumb)
        threshold = 4 / len(x_data)
        return cooks_d <= threshold

    def _calculate_aic(self, residuals: np.ndarray, n_params: int) -> float:
        """Calculate Akaike Information Criterion."""
        n = len(residuals)
        mse = np.mean(residuals**2)
        log_likelihood = -n/2 * (np.log(2*np.pi*mse) + 1)
        return 2 * n_params - 2 * log_likelihood

    def _calculate_bic(self, residuals: np.ndarray, n_params: int, n_samples: int) -> float:
        """Calculate Bayesian Information Criterion."""
        n = len(residuals)
        mse = np.mean(residuals**2)
        log_likelihood = -n/2 * (np.log(2*np.pi*mse) + 1)
        return n_params * np.log(n) - 2 * log_likelihood

    def _durbin_watson(self, residuals: np.ndarray) -> float:
        """Calculate Durbin-Watson statistic for autocorrelation."""
        diff = np.diff(residuals)
        return np.sum(diff**2) / np.sum(residuals**2)


class PhaseSlopePlotter:
    """
    Advanced plotting for phase-slope analysis with publication-quality visualizations.
    """

    def __init__(self, figure_generator: Optional[FigureGenerator] = None,
                 style: str = 'ieee'):
        """
        Initialize phase-slope plotter.

        Args:
            figure_generator: Figure generator instance
            style: Publication style ('ieee' or 'nasa')
        """
        self.fig_gen = figure_generator or FigureGenerator(default_style=style)
        self.style = style

    def plot_phase_slope_analysis(self, x_data: np.ndarray, y_data: np.ndarray,
                                 analysis_result: PhaseSlopeResult,
                                 show_residuals: bool = True,
                                 show_confidence_bands: bool = True,
                                 show_equation: bool = True,
                                 title: str = "Phase-Slope Analysis") -> plt.Figure:
        """
        Create comprehensive phase-slope analysis plot.

        Args:
            x_data: Time data
            y_data: Phase data
            analysis_result: Results from PhaseSlopeAnalyzer
            show_residuals: Whether to show residuals subplot
            show_confidence_bands: Whether to show confidence bands
            show_equation: Whether to show fit equation
            title: Plot title

        Returns:
            Matplotlib figure
        """
        if show_residuals:
            # Two-panel figure
            panel_configs = [
                {
                    'type': 'phase_slope_main',
                    'title': '(a) Phase-Slope Analysis',
                    'data': {
                        'x_data': x_data,
                        'y_data': y_data,
                        'result': analysis_result,
                        'show_confidence_bands': show_confidence_bands,
                        'show_equation': show_equation
                    }
                },
                {
                    'type': 'residuals',
                    'title': '(b) Fit Residuals',
                    'data': {
                        'x_data': x_data,
                        'residuals': analysis_result.residuals
                    }
                }
            ]

            spec = self.fig_gen.get_figure_template('phase_slope_analysis')
            fig = self.fig_gen.create_multi_panel_figure(panel_configs, FigureSpec(**spec))

        else:
            # Single panel figure
            spec = FigureSpec(title=title, style=self.style)
            fig = self.fig_gen.create_figure(spec)
            ax = self.fig_gen.add_subplot(fig, 0)

            self._plot_phase_slope_main(ax, x_data, y_data, analysis_result,
                                       show_confidence_bands, show_equation)

        return fig

    def _plot_phase_slope_main(self, ax: plt.Axes, x_data: np.ndarray,
                               y_data: np.ndarray, result: PhaseSlopeResult,
                               show_confidence_bands: bool, show_equation: bool) -> None:
        """Plot main phase-slope analysis."""
        # Plot raw data with error bars if available
        ax.plot(x_data, y_data, 'o', markersize=4, alpha=0.7,
               label='Phase measurements', color='#0072BD')

        # Plot fit line
        x_fit = np.linspace(x_data.min(), x_data.max(), 100)
        y_fit = result.slope * x_fit + result.intercept
        ax.plot(x_fit, y_fit, '-', linewidth=2, color='#D95319',
               label='Linear fit')

        # Show confidence bands
        if show_confidence_bands and 'prediction' in result.confidence_intervals:
            ci_lower, ci_upper = result.confidence_intervals['prediction']
            ax.fill_between(x_fit, ci_lower, ci_upper, alpha=0.3,
                          color='#D95319', label='95% CI')

        # Show equation and statistics
        if show_equation:
            equation_text = (
                f"φ = {format_scientific_notation(result.slope)}·t + "
                f"{format_scientific_notation(result.intercept)}\n"
                f"R² = {result.r_squared:.3f}, RMSE = {format_scientific_notation(result.rmse)}"
            )

            ax.text(0.05, 0.95, equation_text, transform=ax.transAxes,
                   verticalalignment='top', fontsize=9,
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

        # Extracted parameters box
        params_text = (
            f"Δf = {format_uncertainty(result.frequency_offset, result.frequency_uncertainty)} Hz\n"
            f"τ = {format_uncertainty(result.timing_estimate, result.timing_uncertainty)} ps"
        )

        ax.text(0.95, 0.95, params_text, transform=ax.transAxes,
               verticalalignment='top', horizontalalignment='right', fontsize=9,
               bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.9))

        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Phase (rad)')
        ax.set_title('Phase-Slope Analysis')
        ax.legend()
        ax.grid(True, alpha=0.3)

    def plot_residuals(self, ax: plt.Axes, x_data: np.ndarray,
                      residuals: np.ndarray) -> None:
        """Plot fit residuals."""
        ax.plot(x_data, residuals, 'o-', markersize=3, linewidth=1,
               color='#77AC30', alpha=0.7)

        # Zero line
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)

        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Residuals (rad)')
        ax.set_title('Fit Residuals')
        ax.grid(True, alpha=0.3)

    def plot_uncertainty_analysis(self, analysis_results: List[PhaseSlopeResult],
                                 parameter_names: List[str] = None,
                                 title: str = "Uncertainty Analysis") -> plt.Figure:
        """
        Plot uncertainty analysis across multiple measurements.

        Args:
            analysis_results: List of phase-slope analysis results
            parameter_names: Names for each measurement
            title: Plot title

        Returns:
            Matplotlib figure
        """
        n_results = len(analysis_results)
        if parameter_names is None:
            parameter_names = [f"Measurement {i+1}" for i in range(n_results)]

        spec = FigureSpec(title=title, width=7.0, height=4.0, style=self.style)
        fig = self.fig_gen.create_figure(spec)

        # Create subplots
        ax1 = self.fig_gen.add_subplot(fig, (0, 0), title='Frequency Offset')
        ax2 = self.fig_gen.add_subplot(fig, (0, 1), title='Timing Estimate')

        # Extract parameters and uncertainties
        freq_offsets = [r.frequency_offset for r in analysis_results]
        freq_uncs = [r.frequency_uncertainty for r in analysis_results]
        timing_estimates = [r.timing_estimate for r in analysis_results]
        timing_uncs = [r.timing_uncertainty for r in analysis_results]

        # Plot with error bars
        x_pos = np.arange(n_results)

        ax1.errorbar(x_pos, freq_offsets, yerr=freq_uncs, fmt='o',
                    capsize=5, capthick=1, color='#0072BD')
        ax1.set_ylabel('Frequency Offset (Hz)')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(parameter_names, rotation=45, ha='right')

        ax2.errorbar(x_pos, timing_estimates, yerr=timing_uncs, fmt='o',
                    capsize=5, capthick=1, color='#D95319')
        ax2.set_ylabel('Timing Estimate (ps)')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(parameter_names, rotation=45, ha='right')

        plt.tight_layout()
        return fig


# Import FigureSpec for use in this module
from dataclasses import dataclass
from typing import Optional, List

@dataclass
class FigureSpec:
    """Figure specification for phase-slope plots."""
    title: str = ""
    width: float = 3.5
    height: Optional[float] = None
    style: str = 'ieee'