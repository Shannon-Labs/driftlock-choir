"""
Publication-quality styling system for chronometric interferometry visualizations.

Implements IEEE and NASA publication standards with color-blind accessible palettes,
professional typography, and journal-ready formatting.
"""

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
import warnings


@dataclass
class ColorPalette:
    """Color-blind accessible color palette for publications."""
    # IEEE color-blind accessible palette
    blue: str = '#0072BD'      # Primary blue
    orange: str = '#D95319'    # Secondary orange
    yellow: str = '#EDB120'    # Accent yellow
    purple: str = '#7E2F8E'    # Secondary purple
    green: str = '#77AC30'     # Primary green
    cyan: str = '#4DBEEE'      # Accent cyan
    red: str = '#A2142F'       # Alert red
    gray: str = '#7F7F7F'      # Neutral gray

    def get_colors(self, n: int) -> List[str]:
        """Get first n colors from palette."""
        color_list = [
            self.blue, self.orange, self.yellow, self.purple,
            self.green, self.cyan, self.red, self.gray
        ]
        return color_list[:n] if n <= len(color_list) else color_list + color_list[:n-len(color_list)]


@dataclass
class TypographySettings:
    """Typography settings matching IEEE/NASA publication standards."""
    font_family: str = 'DejaVu Sans'
    font_size: int = 10
    title_size: int = 12
    label_size: int = 10
    tick_size: int = 9
    legend_size: int = 9
    math_fontsize: int = 10

    def apply_to_rcParams(self):
        """Apply typography settings to matplotlib rcParams."""
        mpl.rcParams.update({
            'font.family': self.font_family,
            'font.size': self.font_size,
            'axes.titlesize': self.title_size,
            'axes.labelsize': self.label_size,
            'xtick.labelsize': self.tick_size,
            'ytick.labelsize': self.tick_size,
            'legend.fontsize': self.legend_size,
            'mathtext.fontset': 'dejavusans',
            'mathtext.default': 'regular'
        })


class IEEEStyle:
    """IEEE publication style for chronometric interferometry research."""

    def __init__(self):
        self.colors = ColorPalette()
        self.typography = TypographySettings()
        self.dpi = 300
        self.figure_format = 'pdf'

        # IEEE figure size specifications (column width = 3.5 inches, 2-column = 7.0 inches)
        self.single_column_width = 3.5
        self.double_column_width = 7.0
        self.max_height = 8.0  # Maximum figure height

        # Line weights and markers
        self.line_width = 1.0
        self.marker_size = 4
        self.error_bar_width = 1.0
        self.error_bar_cap_size = 3

    def apply_style(self, fig_width: Optional[float] = None,
                   fig_height: Optional[float] = None) -> Dict:
        """Apply IEEE style to matplotlib figure.

        Args:
            fig_width: Figure width in inches (default: single column)
            fig_height: Figure height in inches (default: golden ratio)

        Returns:
            Dictionary of style parameters
        """
        # Set figure size
        if fig_width is None:
            fig_width = self.single_column_width
        if fig_height is None:
            # Golden ratio for aesthetic proportions
            fig_height = fig_width / 1.618

        # Apply typography
        self.typography.apply_to_rcParams()

        # Set figure parameters
        plt.rcParams.update({
            'figure.figsize': [fig_width, fig_height],
            'figure.dpi': self.dpi,
            'savefig.dpi': self.dpi,
            'savefig.format': self.figure_format,
            'savefig.bbox': 'tight',
            'savefig.pad_inches': 0.1,

            # Line and marker styles
            'lines.linewidth': self.line_width,
            'lines.markersize': self.marker_size,
            'axes.linewidth': 0.8,

            # Grid and spines
            'axes.grid': True,
            'grid.alpha': 0.3,
            'grid.linewidth': 0.5,
            'axes.axisbelow': True,

            # Error bars
            'errorbar.capsize': self.error_bar_cap_size,
            'errorbar.capthick': self.error_bar_width,

            # Color cycle
            'axes.prop_cycle': plt.cycler('color', self.colors.get_colors(8))
        })

        return {
            'fig_width': fig_width,
            'fig_height': fig_height,
            'dpi': self.dpi,
            'colors': self.colors
        }

    def format_axes(self, ax: plt.Axes,
                   xlabel: str = None, ylabel: str = None,
                   title: str = None, legend: bool = False) -> None:
        """Format axes according to IEEE standards.

        Args:
            ax: Matplotlib axes object
            xlabel: X-axis label with units
            ylabel: Y-axis label with units
            title: Plot title
            legend: Whether to add legend
        """
        if xlabel:
            ax.set_xlabel(xlabel, fontsize=self.typography.label_size)
        if ylabel:
            ax.set_ylabel(ylabel, fontsize=self.typography.label_size)
        if title:
            ax.set_title(title, fontsize=self.typography.title_size, pad=10)

        # Tick formatting
        ax.tick_params(axis='both', which='major',
                      labelsize=self.typography.tick_size)

        # Legend formatting
        if legend:
            ax.legend(frameon=True, fancybox=True, shadow=False,
                     framealpha=0.9, edgecolor='gray',
                     fontsize=self.typography.legend_size)

        # Spine formatting
        for spine in ax.spines.values():
            spine.set_linewidth(0.8)
            spine.set_color('black')


class NASAStyle:
    """NASA publication style with enhanced visual clarity."""

    def __init__(self):
        self.colors = ColorPalette()
        self.typography = TypographySettings()
        self.typography.font_family = 'Arial'  # NASA preferred
        self.dpi = 300
        self.figure_format = 'pdf'

        # NASA figure specifications
        self.single_column_width = 3.3
        self.double_column_width = 6.8
        self.max_height = 9.0

        # Enhanced visual settings
        self.line_width = 1.2
        self.marker_size = 5
        self.grid_alpha = 0.4

    def apply_style(self, fig_width: Optional[float] = None,
                   fig_height: Optional[float] = None) -> Dict:
        """Apply NASA style to matplotlib figure."""
        if fig_width is None:
            fig_width = self.single_column_width
        if fig_height is None:
            fig_height = fig_width / 1.414  # sqrt(2) ratio

        # Apply typography
        self.typography.apply_to_rcParams()

        # Set figure parameters
        plt.rcParams.update({
            'figure.figsize': [fig_width, fig_height],
            'figure.dpi': self.dpi,
            'savefig.dpi': self.dpi,
            'savefig.format': self.figure_format,
            'savefig.bbox': 'tight',
            'savefig.pad_inches': 0.15,

            # Enhanced line and marker styles
            'lines.linewidth': self.line_width,
            'lines.markersize': self.marker_size,
            'lines.markeredgewidth': 0.8,
            'axes.linewidth': 1.0,

            # Enhanced grid
            'axes.grid': True,
            'grid.alpha': self.grid_alpha,
            'grid.linewidth': 0.6,
            'grid.linestyle': '--',
            'axes.axisbelow': True,

            # Color cycle
            'axes.prop_cycle': plt.cycler('color', self.colors.get_colors(8))
        })

        return {
            'fig_width': fig_width,
            'fig_height': fig_height,
            'dpi': self.dpi,
            'colors': self.colors
        }

    def format_axes(self, ax: plt.Axes,
                   xlabel: str = None, ylabel: str = None,
                   title: str = None, legend: bool = False) -> None:
        """Format axes according to NASA standards."""
        if xlabel:
            ax.set_xlabel(xlabel, fontsize=self.typography.label_size, fontweight='bold')
        if ylabel:
            ax.set_ylabel(ylabel, fontsize=self.typography.label_size, fontweight='bold')
        if title:
            ax.set_title(title, fontsize=self.typography.title_size,
                        fontweight='bold', pad=12)

        # Enhanced tick formatting
        ax.tick_params(axis='both', which='major',
                      labelsize=self.typography.tick_size, width=1.0)
        ax.tick_params(axis='both', which='minor', width=0.5)

        # Enhanced legend
        if legend:
            ax.legend(frameon=True, fancybox=True, shadow=False,
                     framealpha=0.95, edgecolor='black',
                     fontsize=self.typography.legend_size,
                     title_fontsize=self.typography.legend_size + 1)

        # Enhanced spines
        for spine in ax.spines.values():
            spine.set_linewidth(1.0)
            spine.set_color('black')


class PublicationStyle:
    """Unified publication style supporting both IEEE and NASA standards."""

    def __init__(self, style: str = 'ieee'):
        """
        Initialize publication style.

        Args:
            style: Publication style ('ieee' or 'nasa')
        """
        if style.lower() == 'ieee':
            self.style = IEEEStyle()
        elif style.lower() == 'nasa':
            self.style = NASAStyle()
        else:
            raise ValueError(f"Unknown style: {style}. Use 'ieee' or 'nasa'")

    def apply_style(self, **kwargs) -> Dict:
        """Apply the selected publication style."""
        return self.style.apply_style(**kwargs)

    def format_axes(self, ax: plt.Axes, **kwargs) -> None:
        """Format axes according to the selected publication style."""
        return self.style.format_axes(ax, **kwargs)


def apply_ieee_colors(fig: plt.Figure) -> None:
    """Apply IEEE color palette to all lines in figure."""
    ieee_style = IEEEStyle()
    colors = ieee_style.colors.get_colors(8)

    for i, ax in enumerate(fig.axes):
        for j, line in enumerate(ax.lines):
            line.set_color(colors[j % len(colors)])

    fig.canvas.draw_idle()


def apply_nasa_colors(fig: plt.Figure) -> None:
    """Apply NASA color palette to all lines in figure."""
    nasa_style = NASAStyle()
    colors = nasa_style.colors.get_colors(8)

    for i, ax in enumerate(fig.axes):
        for j, line in enumerate(ax.lines):
            line.set_color(colors[j % len(colors)])

    fig.canvas.draw_idle()


# Predefined style configurations
IEEE_CONFIG = {
    'single_column': {'fig_width': 3.5, 'style': 'ieee'},
    'double_column': {'fig_width': 7.0, 'style': 'ieee'},
    'square': {'fig_width': 3.5, 'fig_height': 3.5, 'style': 'ieee'}
}

NASA_CONFIG = {
    'single_column': {'fig_width': 3.3, 'style': 'nasa'},
    'double_column': {'fig_width': 6.8, 'style': 'nasa'},
    'square': {'fig_width': 3.3, 'fig_height': 3.3, 'style': 'nasa'}
}