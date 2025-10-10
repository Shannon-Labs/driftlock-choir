"""
3D performance surface visualization for chronometric interferometry.

Provides sophisticated multi-dimensional visualization of system performance
across SNR, frequency offset (Δf), and timing (τ) parameter spaces with
interactive capabilities and publication-quality rendering.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.colors import LogNorm, Normalize
from scipy.interpolate import griddata, interp2d
from scipy.ndimage import gaussian_filter
from typing import Dict, List, Tuple, Optional, Union, Any
import warnings

from ..base.figure_generator import FigureGenerator
from ..base.styles import IEEEStyle, NASAStyle
from ..base.utils import format_scientific_notation, add_colorbar_with_label


class PerformanceSurfacePlotter:
    """
    Advanced 3D performance surface visualization for chronometric interferometry.

    Creates publication-quality multi-dimensional visualizations of system
    performance across parameter spaces with sophisticated rendering and
    analysis capabilities.
    """

    def __init__(self, figure_generator: Optional[FigureGenerator] = None,
                 style: str = 'ieee'):
        """
        Initialize performance surface plotter.

        Args:
            figure_generator: Figure generator instance
            style: Publication style ('ieee' or 'nasa')
        """
        self.fig_gen = figure_generator or FigureGenerator(default_style=style)
        self.style = style

    def plot_3d_performance_surface(self, x_param: np.ndarray, y_param: np.ndarray,
                                   performance: np.ndarray,
                                   x_label: str = None, y_label: str = None,
                                   z_label: str = None,
                                   title: str = "Performance Surface",
                                   colormap: str = 'viridis',
                                   log_scale: bool = False,
                                   interpolate: bool = True,
                                   show_contours: bool = True) -> plt.Figure:
        """
        Create 3D surface plot of performance metrics.

        Args:
            x_param: X-axis parameter values (e.g., SNR)
            y_param: Y-axis parameter values (e.g., Δf)
            performance: Performance metric values (2D array)
            x_label: X-axis label
            y_label: Y-axis label
            z_label: Z-axis label
            title: Plot title
            colormap: Colormap name
            log_scale: Whether to use log scale for performance
            interpolate: Whether to interpolate surface
            show_contours: Whether to show contour lines

        Returns:
            Matplotlib figure
        """
        spec = self.fig_gen.get_figure_template('performance_characterization')
        spec.update({
            'title': title,
            'width': 7.0,
            'height': 5.5,
            'style': self.style
        })

        fig = self.fig_gen.create_figure(FigureSpec(**spec))
        ax = fig.add_subplot(111, projection='3d')

        # Create mesh grid
        X, Y = np.meshgrid(x_param, y_param)

        # Interpolate if requested
        if interpolate and performance.shape[0] < 50:
            X_fine, Y_fine, performance_fine = self._interpolate_surface(
                X, Y, performance, factor=2
            )
        else:
            X_fine, Y_fine, performance_fine = X, Y, performance

        # Apply log scale if requested
        if log_scale:
            performance_plot = np.log10(performance_fine)
            z_label = f"log₁₀({z_label})" if z_label else "log₁₀(Performance)"
        else:
            performance_plot = performance_fine

        # Create surface plot
        surf = ax.plot_surface(X_fine, Y_fine, performance_plot,
                              cmap=colormap, alpha=0.9,
                              linewidth=0.1, antialiased=True,
                              edgecolor='none')

        # Add contour lines if requested
        if show_contours:
            # Project contours onto XY plane
            ax.contour(X_fine, Y_fine, performance_plot,
                      zdir='z', offset=performance_plot.min(),
                      cmap=colormap, alpha=0.5, linewidths=0.5)

            # Project contours onto bottom
            ax.contour(X_fine, Y_fine, performance_plot,
                      zdir='z', offset=performance_plot.min() * 1.1,
                      cmap=colormap, alpha=0.3, linewidths=0.3)

        # Add colorbar
        cbar = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5, pad=0.1)
        cbar.set_label(z_label, rotation=270, labelpad=15)

        # Set labels
        if x_label:
            ax.set_xlabel(x_label)
        if y_label:
            ax.set_ylabel(y_label)
        if z_label:
            ax.set_zlabel(z_label)

        ax.set_title(title)
        ax.grid(True, alpha=0.3)

        # Adjust viewing angle for best visualization
        ax.view_init(elev=30, azim=45)

        return fig

    def plot_multi_surface_comparison(self, x_param: np.ndarray, y_param: np.ndarray,
                                     performance_data: Dict[str, np.ndarray],
                                     x_label: str = None, y_label: str = None,
                                     title: str = "Performance Comparison",
                                     colormaps: List[str] = None) -> plt.Figure:
        """
        Create multiple surface plots for comparison.

        Args:
            x_param: X-axis parameter values
            y_param: Y-axis parameter values
            performance_data: Dictionary of performance arrays
            x_label: X-axis label
            y_label: Y-axis label
            title: Plot title
            colormaps: List of colormaps for each surface

        Returns:
            Matplotlib figure with subplots
        """
        n_surfaces = len(performance_data)
        if colormaps is None:
            colormaps = ['viridis', 'plasma', 'inferno', 'magma'][:n_surfaces]

        # Calculate subplot grid
        n_cols = min(2, n_surfaces)
        n_rows = int(np.ceil(n_surfaces / n_cols))

        spec = FigureSpec(
            title=title,
            width=7.0,
            height=3.5 * n_rows,
            style=self.style
        )

        fig = self.fig_gen.create_figure(spec)

        for i, (name, performance) in enumerate(performance_data.items()):
            ax = fig.add_subplot(n_rows, n_cols, i + 1, projection='3d')

            # Create mesh grid
            X, Y = np.meshgrid(x_param, y_param)

            # Create surface plot
            surf = ax.plot_surface(X, Y, performance,
                                  cmap=colormaps[i], alpha=0.9,
                                  linewidth=0.1, antialiased=True)

            # Add colorbar
            cbar = fig.colorbar(surf, ax=ax, shrink=0.6, aspect=10, pad=0.1)
            cbar.set_label(name, rotation=270, labelpad=12)

            # Set labels
            if i == 0 or i == n_cols:
                ax.set_ylabel(y_label)
            if i >= n_surfaces - n_cols:
                ax.set_xlabel(x_label)

            ax.set_title(f'({chr(97+i)}) {name}')
            ax.grid(True, alpha=0.3)

            # Consistent viewing angle
            ax.view_init(elev=30, azim=45)

        plt.tight_layout()
        return fig

    def plot_parameter_sweep_analysis(self, sweep_results: Dict[str, Dict],
                                     primary_metric: str = 'timing_rmse',
                                     title: str = "Parameter Sweep Analysis") -> plt.Figure:
        """
        Create comprehensive parameter sweep analysis visualization.

        Args:
            sweep_results: Results from parameter sweep experiments
            primary_metric: Primary metric to highlight
            title: Plot title

        Returns:
            Matplotlib figure
        """
        # Create 2x2 subplot layout
        panel_configs = [
            {
                'type': 'contour_heatmap',
                'title': '(a) Timing Precision (ps)',
                'data': sweep_results.get('timing_rmse', {})
            },
            {
                'type': 'contour_heatmap',
                'title': '(b) Frequency Accuracy (ppb)',
                'data': sweep_results.get('frequency_rmse', {})
            },
            {
                'type': 'convergence_plot',
                'title': '(c) Convergence Time (ms)',
                'data': sweep_results.get('convergence_time', {})
            },
            {
                'type': 'efficiency_plot',
                'title': '(d) Computational Efficiency',
                'data': sweep_results.get('cpu_time', {})
            }
        ]

        spec = FigureSpec(
            title=title,
            width=7.0,
            style=self.style
        )

        fig = self.fig_gen.create_multi_panel_figure(panel_configs, spec)

        return fig

    def plot_performance_optimization(self, x_param: np.ndarray, y_param: np.ndarray,
                                     performance: np.ndarray,
                                     optimal_point: Tuple[float, float] = None,
                                     constraint_regions: List[Dict] = None,
                                     title: str = "Performance Optimization") -> plt.Figure:
        """
        Create performance optimization visualization with optimal operating points.

        Args:
            x_param: X-axis parameter values
            y_param: Y-axis parameter values
            performance: Performance metric values
            optimal_point: (x, y) coordinates of optimal point
            constraint_regions: List of constraint region dictionaries
            title: Plot title

        Returns:
            Matplotlib figure
        """
        spec = FigureSpec(
            title=title,
            width=7.0,
            height=5.0,
            style=self.style
        )

        fig = self.fig_gen.create_figure(spec)
        ax = self.fig_gen.add_subplot(fig, 0)

        # Create contour plot
        X, Y = np.meshgrid(x_param, y_param)
        contour = ax.contourf(X, Y, performance, levels=20, cmap='RdYlBu_r')
        ax.contour(X, Y, performance, levels=10, colors='black',
                  alpha=0.3, linewidths=0.5)

        # Add constraint regions
        if constraint_regions:
            for region in constraint_regions:
                self._add_constraint_region(ax, region)

        # Mark optimal point
        if optimal_point:
            ax.plot(optimal_point[0], optimal_point[1], 'r*',
                   markersize=15, markeredgewidth=2, label='Optimal Point')

            # Add confidence region around optimal point
            self._add_optimal_confidence_region(ax, optimal_point, X, Y, performance)

        # Add colorbar
        cbar = fig.colorbar(contour, ax=ax)
        cbar.set_label('Performance Metric')

        ax.set_xlabel('Parameter 1')
        ax.set_ylabel('Parameter 2')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

        return fig

    def plot_robustness_surface(self, x_param: np.ndarray, y_param: np.ndarray,
                               performance_nominal: np.ndarray,
                               performance_degraded: np.ndarray,
                               degradation_type: str = "Interference",
                               title: str = "Robustness Analysis") -> plt.Figure:
        """
        Create robustness analysis comparing nominal vs. degraded performance.

        Args:
            x_param: X-axis parameter values
            y_param: Y-axis parameter values
            performance_nominal: Performance under nominal conditions
            performance_degraded: Performance under degraded conditions
            degradation_type: Type of degradation
            title: Plot title

        Returns:
            Matplotlib figure
        """
        spec = FigureSpec(
            title=title,
            width=7.0,
            style=self.style
        )

        fig = self.fig_gen.create_figure(spec)

        # Create subplots
        ax1 = self.fig_gen.add_subplot(fig, (0, 0), title='(a) Nominal Performance')
        ax2 = self.fig_gen.add_subplot(fig, (0, 1), title='(b) Degraded Performance')

        # Create mesh grid
        X, Y = np.meshgrid(x_param, y_param)

        # Plot nominal performance
        contour1 = ax1.contourf(X, Y, performance_nominal, levels=20, cmap='viridis')
        fig.colorbar(contour1, ax=ax1)
        ax1.set_xlabel('Parameter 1')
        ax1.set_ylabel('Parameter 2')

        # Plot degraded performance
        contour2 = ax2.contourf(X, Y, performance_degraded, levels=20, cmap='viridis')
        fig.colorbar(contour2, ax=ax2)
        ax2.set_xlabel('Parameter 1')

        # Calculate and show performance degradation
        performance_diff = performance_degraded - performance_nominal
        ax3 = self.fig_gen.add_subplot(fig, (1, 0), colspan=2, title='(c) Performance Degradation')
        contour3 = ax3.contourf(X, Y, performance_diff, levels=20,
                               cmap='RdYlBu_r', center=0)
        fig.colorbar(contour3, ax=ax3, label='Performance Difference')
        ax3.set_xlabel('Parameter 1')
        ax3.set_ylabel('Parameter 2')

        # Add statistics text
        mean_degradation = np.mean(performance_diff)
        max_degradation = np.max(np.abs(performance_diff))
        stats_text = f"Mean degradation: {format_scientific_notation(mean_degradation)}\n"
        stats_text += f"Maximum degradation: {format_scientific_notation(max_degradation)}"
        ax3.text(0.02, 0.98, stats_text, transform=ax3.transAxes,
                 verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

        plt.tight_layout()
        return fig

    def plot_sensitivity_analysis(self, parameter_names: List[str],
                                 sensitivity_coefficients: np.ndarray,
                                 uncertainty_contributions: np.ndarray = None,
                                 title: str = "Sensitivity Analysis") -> plt.Figure:
        """
        Create sensitivity analysis visualization.

        Args:
            parameter_names: List of parameter names
            sensitivity_coefficients: Sensitivity coefficients
            uncertainty_contributions: Contribution to total uncertainty
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
        if uncertainty_contributions is not None:
            ax1 = self.fig_gen.add_subplot(fig, (0, 0), title='(a) Sensitivity Coefficients')
            ax2 = self.fig_gen.add_subplot(fig, (0, 1), title='(b) Uncertainty Contributions')
        else:
            ax1 = self.fig_gen.add_subplot(fig, (0, 0), title='Sensitivity Coefficients')

        # Plot sensitivity coefficients
        y_pos = np.arange(len(parameter_names))
        bars = ax1.barh(y_pos, np.abs(sensitivity_coefficients),
                       color='#0072BD', alpha=0.7)
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels(parameter_names)
        ax1.set_xlabel('|Sensitivity Coefficient|')
        ax1.grid(True, alpha=0.3)

        # Highlight most sensitive parameters
        max_sensitivity_idx = np.argmax(np.abs(sensitivity_coefficients))
        bars[max_sensitivity_idx].set_color('#D95319')

        # Plot uncertainty contributions if available
        if uncertainty_contributions is not None:
            bars2 = ax2.barh(y_pos, uncertainty_contributions,
                            color='#77AC30', alpha=0.7)
            ax2.set_yticks(y_pos)
            ax2.set_yticklabels(parameter_names)
            ax2.set_xlabel('Uncertainty Contribution (%)')
            ax2.grid(True, alpha=0.3)

            # Highlight largest contributors
            max_contrib_idx = np.argmax(uncertainty_contributions)
            bars2[max_contrib_idx].set_color('#A2142F')

        plt.tight_layout()
        return fig

    def _interpolate_surface(self, X: np.ndarray, Y: np.ndarray,
                            Z: np.ndarray, factor: int = 2) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Interpolate surface for smoother visualization."""
        # Create finer grid
        x_fine = np.linspace(X.min(), X.max(), X.shape[1] * factor)
        y_fine = np.linspace(Y.min(), Y.max(), Y.shape[0] * factor)
        X_fine, Y_fine = np.meshgrid(x_fine, y_fine)

        # Interpolate Z values
        points = np.column_stack([X.ravel(), Y.ravel()])
        Z_fine = griddata(points, Z.ravel(), (X_fine, Y_fine), method='cubic')

        # Handle NaN values with nearest neighbor interpolation
        mask = np.isnan(Z_fine)
        if np.any(mask):
            Z_fine[mask] = griddata(points, Z.ravel(), (X_fine[mask], Y_fine[mask]), method='nearest')

        return X_fine, Y_fine, Z_fine

    def _add_constraint_region(self, ax: plt.Axes, region: Dict) -> None:
        """Add constraint region to plot."""
        if region['type'] == 'rectangle':
            rect = plt.Rectangle((region['x_min'], region['y_min']),
                                region['x_max'] - region['x_min'],
                                region['y_max'] - region['y_min'],
                                fill=True, alpha=0.3, color=region.get('color', 'red'),
                                label=region.get('label', 'Constraint'))
            ax.add_patch(rect)
        elif region['type'] == 'circle':
            circle = plt.Circle((region['x_center'], region['y_center']),
                               region['radius'], fill=True, alpha=0.3,
                               color=region.get('color', 'red'),
                               label=region.get('label', 'Constraint'))
            ax.add_patch(circle)

    def _add_optimal_confidence_region(self, ax: plt.Axes, optimal_point: Tuple[float, float],
                                      X: np.ndarray, Y: np.ndarray, Z: np.ndarray) -> None:
        """Add confidence region around optimal point."""
        # Find performance value at optimal point
        x_idx = np.argmin(np.abs(X[0, :] - optimal_point[0]))
        y_idx = np.argmin(np.abs(Y[:, 0] - optimal_point[1]))
        optimal_value = Z[y_idx, x_idx]

        # Define confidence threshold (e.g., within 5% of optimal)
        confidence_threshold = optimal_value * 1.05

        # Create confidence region
        confidence_region = Z <= confidence_threshold
        ax.contour(X, Y, confidence_region.astype(float), levels=[0.5],
                  colors='red', linewidths=2, linestyles='--',
                  label='95% Confidence Region')


# Import FigureSpec for use in this module
from dataclasses import dataclass
from typing import Optional, List, Dict

@dataclass
class FigureSpec:
    """Figure specification for performance surface plots."""
    title: str = ""
    width: float = 7.0
    height: Optional[float] = None
    style: str = 'ieee'