"""
Base figure generation framework for chronometric interferometry research.

Provides a high-level interface for creating publication-quality figures with
consistent styling, formatting, and export capabilities.
"""

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from dataclasses import dataclass
import warnings

from .styles import IEEEStyle, NASAStyle, PublicationStyle
from .utils import (
    export_publication_figure, format_scientific_notation,
    create_subplot_grid, add_subplot_labels, apply_journal_style,
    create_figure_caption, validate_figure_quality
)


@dataclass
class FigureSpec:
    """Specification for creating publication-quality figures."""
    title: str
    width: float = 3.5  # Single column width
    height: Optional[float] = None
    style: str = 'ieee'
    dpi: int = 300
    formats: List[str] = None
    export_path: Optional[Path] = None
    subplot_grid: Optional[Tuple[int, int]] = None
    journal: str = 'ieee'

    def __post_init__(self):
        if self.formats is None:
            self.formats = ['pdf', 'png']
        if self.height is None:
            # Golden ratio for aesthetic proportions
            self.height = self.width / 1.618
        if self.export_path is None:
            self.export_path = Path('./figures')


class FigureGenerator:
    """
    High-level figure generation framework for chronometric interferometry research.

    Provides consistent styling, automated export, and publication-quality formatting
    for all research figures.
    """

    def __init__(self, default_style: str = 'ieee', default_dpi: int = 300):
        """
        Initialize figure generator.

        Args:
            default_style: Default publication style ('ieee' or 'nasa')
            default_dpi: Default DPI for figure generation
        """
        self.default_style = default_style
        self.default_dpi = default_dpi
        self.figure_cache = {}
        self.style_configs = {
            'ieee': IEEEStyle(),
            'nasa': NASAStyle()
        }

    def create_figure(self, spec: FigureSpec) -> plt.Figure:
        """
        Create a new figure with specified specifications.

        Args:
            spec: Figure specification

        Returns:
            Matplotlib figure object
        """
        # Apply publication style
        style = self.style_configs.get(spec.style, IEEEStyle())
        style_params = style.apply_style(fig_width=spec.width, fig_height=spec.height)

        # Create figure
        if spec.subplot_grid:
            n_rows, n_cols = spec.subplot_grid
            fig = plt.figure(figsize=style_params['fig_width'],
                           height=style_params['fig_height'])
            gs = gridspec.GridSpec(n_rows, n_cols, figure=fig)
        else:
            fig, ax = plt.subplots(figsize=style_params['fig_width'],
                                  height=style_params['fig_height'])
            gs = None

        # Apply journal-specific styling
        apply_journal_style(fig, spec.journal)

        # Store figure metadata
        self.figure_cache[id(fig)] = {
            'spec': spec,
            'style_params': style_params,
            'gridspec': gs
        }

        return fig

    def add_subplot(self, fig: plt.Figure, position: Union[int, Tuple[int, int]],
                   title: str = None, xlabel: str = None, ylabel: str = None,
                   style_overrides: Dict = None) -> plt.Axes:
        """
        Add subplot to figure with consistent formatting.

        Args:
            fig: Figure to add subplot to
            position: Subplot position (index or row, col tuple)
            title: Subplot title
            xlabel: X-axis label
            ylabel: Y-axis label
            style_overrides: Override default styling

        Returns:
            Matplotlib axes object
        """
        fig_data = self.figure_cache.get(id(fig))
        if fig_data is None:
            raise ValueError("Figure not created with this generator")

        spec = fig_data['spec']
        gs = fig_data['gridspec']

        if gs is None:
            # Single subplot figure
            ax = fig.axes[0] if fig.axes else plt.gca()
        else:
            # Multi-subplot figure
            if isinstance(position, int):
                row, col = divmod(position, gs.ncols)
            else:
                row, col = position
            ax = fig.add_subplot(gs[row, col])

        # Apply style formatting
        style = self.style_configs.get(spec.style, IEEEStyle())
        if style_overrides:
            # Apply any style overrides
            pass

        style.format_axes(ax, xlabel=xlabel, ylabel=ylabel, title=title)

        return ax

    def plot_with_uncertainty(self, ax: plt.Axes, x_data: np.ndarray,
                            y_data: np.ndarray, y_error: np.ndarray = None,
                            x_error: np.ndarray = None, label: str = None,
                            color: str = None, linestyle: str = '-',
                            marker: str = None, alpha: float = 0.8) -> None:
        """
        Plot data with uncertainty bars.

        Args:
            ax: Matplotlib axes
            x_data: X-coordinate data
            y_data: Y-coordinate data
            y_error: Y-direction uncertainty
            x_error: X-direction uncertainty
            label: Data label for legend
            color: Line color
            linestyle: Line style
            marker: Marker style
            alpha: Transparency
        """
        # Plot main data
        ax.plot(x_data, y_data, linestyle=linestyle, marker=marker,
               color=color, label=label, alpha=alpha)

        # Add error bars if provided
        if y_error is not None or x_error is not None:
            ax.errorbar(x_data, y_data, yerr=y_error, xerr=x_error,
                       fmt='none', ecolor=color, alpha=alpha * 0.7,
                       capsize=3, capthick=1)

    def plot_phase_slope(self, ax: plt.Axes, time_data: np.ndarray,
                        phase_data: np.ndarray, fit_result: Dict = None,
                        show_equation: bool = True, confidence_interval: bool = True) -> None:
        """
        Plot phase-slope analysis with linear fit.

        Args:
            ax: Matplotlib axes
            time_data: Time points
            phase_data: Phase measurements
            fit_result: Linear fit results dict
            show_equation: Whether to show fit equation
            confidence_interval: Whether to show confidence intervals
        """
        # Plot raw phase data
        ax.plot(time_data, phase_data, 'o', markersize=4, alpha=0.7,
               label='Phase measurements', color='#0072BD')

        if fit_result is not None:
            # Plot fitted line
            time_fit = np.linspace(time_data.min(), time_data.max(), 100)
            phase_fit = fit_result['slope'] * time_fit + fit_result['intercept']
            ax.plot(time_fit, phase_fit, '-', linewidth=2, color='#D95319',
                   label='Linear fit')

            # Show equation
            if show_equation:
                slope_text = f"Slope: {format_scientific_notation(fit_result['slope'])} rad/s"
                r2_text = f"R² = {fit_result.get('r_squared', 0):.3f}"
                ax.text(0.05, 0.95, f"{slope_text}\n{r2_text}",
                       transform=ax.transAxes, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

            # Show confidence intervals
            if confidence_interval and 'confidence_interval' in fit_result:
                ci_lower, ci_upper = fit_result['confidence_interval']
                ax.fill_between(time_fit, ci_lower, ci_upper, alpha=0.3,
                              color='#D95319', label='95% CI')

        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Phase (rad)')
        ax.set_title('Phase-Slope Analysis')
        ax.legend()
        ax.grid(True, alpha=0.3)

    def plot_performance_surface(self, ax: plt.Axes, x_param: np.ndarray,
                                y_param: np.ndarray, performance: np.ndarray,
                                x_label: str = None, y_label: str = None,
                                z_label: str = None, colormap: str = 'viridis') -> None:
        """
        Plot 2D performance surface as contour plot.

        Args:
            ax: Matplotlib axes
            x_param: X-axis parameter values
            y_param: Y-axis parameter values
            performance: Performance metric values (2D array)
            x_label: X-axis label
            y_label: Y-axis label
            z_label: Z-axis (color) label
            colormap: Colormap name
        """
        X, Y = np.meshgrid(x_param, y_param)

        # Create contour plot
        contour = ax.contourf(X, Y, performance, levels=20, cmap=colormap)
        ax.contour(X, Y, performance, levels=10, colors='black',
                  alpha=0.3, linewidths=0.5)

        # Add colorbar
        cbar = plt.colorbar(contour, ax=ax)
        if z_label:
            cbar.set_label(z_label)

        # Set labels
        if x_label:
            ax.set_xlabel(x_label)
        if y_label:
            ax.set_ylabel(y_label)

        ax.set_title('Performance Surface')

    def export_figure(self, fig: plt.Figure, filename: str,
                     formats: List[str] = None, validate: bool = True) -> Dict:
        """
        Export figure in publication-ready formats.

        Args:
            fig: Figure to export
            filename: Base filename
            formats: Export formats
            validate: Whether to validate figure quality

        Returns:
            Export results dictionary
        """
        fig_data = self.figure_cache.get(id(fig))
        if fig_data is None:
            raise ValueError("Figure not created with this generator")

        spec = fig_data['spec']

        if formats is None:
            formats = spec.formats

        # Validate figure quality if requested
        validation_results = {}
        if validate:
            validation_results = validate_figure_quality(fig, spec.dpi)

        # Export figure
        export_path = spec.export_path / filename
        export_publication_figure(fig, export_path, formats=formats, dpi=spec.dpi)

        return {
            'export_path': export_path,
            'formats': formats,
            'validation': validation_results,
            'spec': spec
        }

    def create_multi_panel_figure(self, panel_configs: List[Dict],
                                 spec: FigureSpec = None) -> plt.Figure:
        """
        Create multi-panel figure with consistent styling.

        Args:
            panel_configs: List of panel configuration dictionaries
            spec: Figure specification

        Returns:
            Matplotlib figure with multiple panels
        """
        if spec is None:
            spec = FigureSpec(
                title="Multi-panel Research Figure",
                width=7.0,  # Double column for multi-panel
                style=self.default_style
            )

        # Calculate subplot grid
        n_panels = len(panel_configs)
        n_rows, n_cols = create_subplot_grid(n_panels, max_cols=2)
        spec.subplot_grid = (n_rows, n_cols)

        # Create figure
        fig = self.create_figure(spec)

        # Add panels
        for i, panel_config in enumerate(panel_configs, 1):
            row, col = divmod(i - 1, n_cols)
            ax = self.add_subplot(fig, (row, col))

            # Configure panel based on config
            panel_type = panel_config.get('type', 'generic')
            if panel_type == 'phase_slope':
                self.plot_phase_slope(ax, **panel_config.get('data', {}))
            elif panel_type == 'performance_surface':
                self.plot_performance_surface(ax, **panel_config.get('data', {}))
            else:
                # Generic plot
                ax.plot(**panel_config.get('data', {}))

            # Set panel title
            if 'title' in panel_config:
                ax.set_title(panel_config['title'])

        # Add subplot labels
        labels = [chr(65 + i) for i in range(n_panels)]  # A, B, C, ...
        add_subplot_labels(fig, labels)

        return fig

    def close_all_figures(self) -> None:
        """Close all figures and clear cache."""
        plt.close('all')
        self.figure_cache.clear()

    def get_figure_template(self, figure_type: str) -> Dict:
        """
        Get predefined figure template for common plot types.

        Args:
            figure_type: Type of figure template

        Returns:
            Template configuration dictionary
        """
        templates = {
            'phase_slope_analysis': {
                'panels': [
                    {
                        'type': 'phase_slope',
                        'title': '(a) Phase-Slope Analysis',
                        'data': {}
                    },
                    {
                        'type': 'residuals',
                        'title': '(b) Fit Residuals',
                        'data': {}
                    }
                ],
                'width': 7.0,
                'style': 'ieee'
            },
            'performance_characterization': {
                'panels': [
                    {
                        'type': 'performance_surface',
                        'title': '(a) Timing Precision',
                        'data': {}
                    },
                    {
                        'type': 'performance_surface',
                        'title': '(b) Frequency Accuracy',
                        'data': {}
                    }
                ],
                'width': 7.0,
                'style': 'ieee'
            },
            'algorithm_comparison': {
                'panels': [
                    {
                        'type': 'comparison',
                        'title': '(a) Timing Performance',
                        'data': {}
                    },
                    {
                        'type': 'comparison',
                        'title': '(b) Convergence Analysis',
                        'data': {}
                    }
                ],
                'width': 7.0,
                'style': 'ieee'
            }
        }

        return templates.get(figure_type, {})