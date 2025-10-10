"""
Base visualization module for chronometric interferometry research.

This module provides publication-quality visualization capabilities with
IEEE/NASA standards for research presentations and publications.
"""

from .figure_generator import FigureGenerator
from .styles import IEEEStyle, NASAStyle, PublicationStyle
from .utils import (
    apply_ieee_colors,
    apply_nasa_colors,
    format_scientific_notation,
    set_figure_dpi,
    export_publication_figure
)

__all__ = [
    'FigureGenerator',
    'IEEEStyle',
    'NASAStyle',
    'PublicationStyle',
    'apply_ieee_colors',
    'apply_nasa_colors',
    'format_scientific_notation',
    'set_figure_dpi',
    'export_publication_figure'
]