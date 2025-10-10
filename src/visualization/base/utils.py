"""
Utility functions for publication-quality chronometric interferometry visualizations.

Provides common formatting, export, and processing utilities for creating
professional research figures.
"""

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from pathlib import Path
from typing import Union, Optional, Tuple, List, Dict, Any
import warnings


def format_scientific_notation(value: float, precision: int = 2,
                             use_math_mode: bool = True) -> str:
    """
    Format a number in scientific notation for publication.

    Args:
        value: Number to format
        precision: Number of decimal places
        use_math_mode: Whether to use LaTeX math mode

    Returns:
        Formatted string
    """
    if value == 0:
        return "0"

    exponent = int(np.floor(np.log10(abs(value))))
    mantissa = value / (10 ** exponent)

    if use_math_mode:
        return f"{mantissa:.{precision}f} × 10^{{{exponent}}}"
    else:
        return f"{mantissa:.{precision}f}e{exponent}"


def format_uncertainty(value: float, uncertainty: float,
                      precision: int = 2, use_pm: bool = True) -> str:
    """
    Format measurement with uncertainty for publication.

    Args:
        value: Measured value
        uncertainty: Measurement uncertainty
        precision: Number of significant figures in uncertainty
        use_pm: Whether to use ± symbol

    Returns:
        Formatted string with uncertainty
    """
    if uncertainty == 0:
        return f"{value:.{precision}f}"

    # Round uncertainty to specified precision
    unc_rounded = round(uncertainty, precision)

    # Round value to same decimal place as uncertainty
    if unc_rounded < 1:
        value_rounded = round(value, precision)
    else:
        decimal_places = -int(np.floor(np.log10(unc_rounded))) + precision - 1
        value_rounded = round(value, max(0, decimal_places))

    if use_pm:
        return f"{value_rounded:.{precision}f} ± {unc_rounded:.{precision}f}"
    else:
        return f"({value_rounded:.{precision}f} ± {unc_rounded:.{precision}f})"


def set_figure_dpi(dpi: int = 300) -> None:
    """
    Set figure DPI for publication quality.

    Args:
        dpi: Dots per inch for figure rendering
    """
    plt.rcParams.update({
        'figure.dpi': dpi,
        'savefig.dpi': dpi,
        'figure.facecolor': 'white',
        'savefig.facecolor': 'white',
        'savefig.edgecolor': 'none'
    })


def export_publication_figure(fig: plt.Figure, filename: Union[str, Path],
                            formats: List[str] = ['pdf', 'png'],
                            dpi: int = 300, transparent: bool = False,
                            bbox_inches: str = 'tight', pad_inches: float = 0.1) -> None:
    """
    Export figure in publication-ready formats.

    Args:
        fig: Matplotlib figure to export
        filename: Base filename (without extension)
        formats: List of formats to export ('pdf', 'png', 'eps', 'svg')
        dpi: Resolution for raster formats
        transparent: Whether to use transparent background
        bbox_inches: Bounding box calculation
        pad_inches: Padding around figure
    """
    filename = Path(filename)

    export_kwargs = {
        'dpi': dpi,
        'transparent': transparent,
        'bbox_inches': bbox_inches,
        'pad_inches': pad_inches,
        'metadata': {
            'Title': 'Chronometric Interferometry Research',
            'Author': 'Driftlock Choir Research Team',
            'Creator': 'Matplotlib'
        }
    }

    for fmt in formats:
        output_path = filename.with_suffix(f'.{fmt}')

        if fmt.lower() == 'pdf':
            # PDF with vector graphics
            fig.savefig(output_path, format='pdf', **export_kwargs)
        elif fmt.lower() == 'png':
            # PNG with high resolution
            fig.savefig(output_path, format='png', **export_kwargs)
        elif fmt.lower() == 'eps':
            # EPS for journal submission
            fig.savefig(output_path, format='eps', **export_kwargs)
        elif fmt.lower() == 'svg':
            # SVG for web/presentation
            fig.savefig(output_path, format='svg', **export_kwargs)
        else:
            warnings.warn(f"Unsupported format: {fmt}")


def create_subplot_grid(n_plots: int, max_cols: int = 3,
                       aspect_ratio: float = 1.618) -> Tuple[int, int]:
    """
    Calculate optimal subplot grid layout.

    Args:
        n_plots: Number of subplots needed
        max_cols: Maximum number of columns
        aspect_ratio: Desired height/width ratio

    Returns:
        Tuple of (n_rows, n_cols)
    """
    if n_plots == 1:
        return 1, 1

    # Calculate optimal columns
    n_cols = min(max_cols, int(np.ceil(np.sqrt(n_plots * aspect_ratio))))
    n_rows = int(np.ceil(n_plots / n_cols))

    return n_rows, n_cols


def add_subplot_labels(fig: plt.Figure, labels: List[str],
                      style: str = 'uppercase', offset: float = 0.02) -> None:
    """
    Add subplot labels (A, B, C, etc.) to figure.

    Args:
        fig: Matplotlib figure
        labels: List of labels for each subplot
        style: Label style ('uppercase', 'lowercase', 'numbers')
        offset: Offset from subplot corner (fraction of axes width)
    """
    if style == 'uppercase':
        formatted_labels = [f"({label.upper()})" for label in labels]
    elif style == 'lowercase':
        formatted_labels = [f"({label.lower()})" for label in labels]
    elif style == 'numbers':
        formatted_labels = [f"({label})" for label in labels]
    else:
        formatted_labels = labels

    for i, (ax, label) in enumerate(zip(fig.axes, formatted_labels)):
        # Position label in upper-left corner
        ax.text(offset, 1 - offset, label, transform=ax.transAxes,
               fontsize=12, fontweight='bold', va='top', ha='left')


def format_axis_scientific(ax: plt.Axes, axis: str = 'both',
                         use_offset: bool = True) -> None:
    """
    Format axis tick labels in scientific notation.

    Args:
        ax: Matplotlib axes
        axis: Which axis to format ('x', 'y', 'both')
        use_offset: Whether to use offset notation
    """
    tick_formatter = mpl.ticker.ScalarFormatter(useOffset=use_offset)
    tick_formatter.set_scientific(True)
    tick_formatter.set_powerlimits((-2, 3))

    if axis in ['x', 'both']:
        ax.xaxis.set_major_formatter(tick_formatter)
    if axis in ['y', 'both']:
        ax.yaxis.set_major_formatter(tick_formatter)


def add_colorbar_with_label(cbar, label: str, rotation: float = 270,
                          labelpad: float = 20) -> None:
    """
    Add properly formatted colorbar with label.

    Args:
        cbar: Colorbar object
        label: Colorbar label
        rotation: Label rotation angle
        labelpad: Label padding
    """
    cbar.set_label(label, rotation=rotation, labelpad=labelpad, fontsize=10)
    cbar.ax.tick_params(labelsize=9)


def calculate_error_bars(data: np.ndarray, confidence: float = 0.95,
                        method: str = 'std') -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate error bars for data visualization.

    Args:
        data: Input data array (2D: samples x measurements)
        confidence: Confidence level (0.0-1.0)
        method: Error calculation method ('std', 'sem', 'percentile')

    Returns:
        Tuple of (lower_error, upper_error)
    """
    if method == 'std':
        # Standard deviation
        std = np.std(data, axis=0)
        return std, std
    elif method == 'sem':
        # Standard error of the mean
        sem = np.std(data, axis=0) / np.sqrt(len(data))
        return sem, sem
    elif method == 'percentile':
        # Percentile-based confidence intervals
        alpha = 1 - confidence
        lower = np.percentile(data, 100 * alpha/2, axis=0)
        upper = np.percentile(data, 100 * (1 - alpha/2), axis=0)
        mean = np.mean(data, axis=0)
        return mean - lower, upper - mean
    else:
        raise ValueError(f"Unknown method: {method}")


def apply_journal_style(fig: plt.Figure, journal: str = 'ieee') -> None:
    """
    Apply journal-specific style to figure.

    Args:
        fig: Matplotlib figure
        journal: Journal name ('ieee', 'nature', 'science', 'aps')
    """
    if journal.lower() == 'ieee':
        # IEEE transactions style
        for ax in fig.axes:
            ax.spines['top'].set_visible(True)
            ax.spines['right'].set_visible(True)
            ax.grid(True, alpha=0.3)
    elif journal.lower() == 'nature':
        # Nature style (cleaner)
        for ax in fig.axes:
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.grid(False)
    elif journal.lower() == 'science':
        # Science style
        for ax in fig.axes:
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.grid(True, alpha=0.2)
    elif journal.lower() == 'aps':
        # APS (Physical Review) style
        for ax in fig.axes:
            ax.spines['top'].set_visible(True)
            ax.spines['right'].set_visible(True)
            ax.grid(True, alpha=0.25)


def create_figure_caption(title: str, description: str,
                         units: Dict[str, str] = None) -> str:
    """
    Create publication-ready figure caption.

    Args:
        title: Figure title
        description: Detailed description
        units: Dictionary of units used in figure

    Returns:
        Formatted caption string
    """
    caption = f"**{title}.** {description}"

    if units:
        unit_list = [f"{symbol}: {unit}" for symbol, unit in units.items()]
        caption += f" Units: {', '.join(unit_list)}."

    return caption


def validate_figure_quality(fig: plt.Figure, min_dpi: int = 300,
                          check_colors: bool = True) -> Dict[str, Any]:
    """
    Validate figure quality for publication.

    Args:
        fig: Matplotlib figure to validate
        min_dpi: Minimum required DPI
        check_colors: Whether to check color accessibility

    Returns:
        Dictionary of validation results
    """
    validation_results = {
        'dpi_check': fig.dpi >= min_dpi,
        'figure_size_check': True,  # Add size validation if needed
        'color_check': True,
        'warnings': [],
        'errors': []
    }

    if fig.dpi < min_dpi:
        validation_results['errors'].append(f"DPI too low: {fig.dpi} < {min_dpi}")

    if check_colors:
        # Add color accessibility checks
        pass

    return validation_results


def apply_ieee_colors(fig: plt.Figure) -> None:
    """Apply IEEE color palette to all lines in figure."""
    from .styles import IEEEStyle
    ieee_style = IEEEStyle()
    colors = ieee_style.colors.get_colors(8)

    for i, ax in enumerate(fig.axes):
        for j, line in enumerate(ax.lines):
            line.set_color(colors[j % len(colors)])

    fig.canvas.draw_idle()


def apply_nasa_colors(fig: plt.Figure) -> None:
    """Apply NASA color palette to all lines in figure."""
    from .styles import NASAStyle
    nasa_style = NASAStyle()
    colors = nasa_style.colors.get_colors(8)

    for i, ax in enumerate(fig.axes):
        for j, line in enumerate(ax.lines):
            line.set_color(colors[j % len(colors)])

    fig.canvas.draw_idle()