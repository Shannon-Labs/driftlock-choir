"""
Statistical validation framework for chronometric interferometry research.

This module provides rigorous statistical analysis, uncertainty quantification,
and validation capabilities suitable for Nokia Bell Labs research standards.
"""

from .uncertainty_analysis import UncertaintyAnalyzer, BootstrapAnalyzer, BayesianAnalyzer
from .validation.hypothesis_testing import HypothesisTestSuite, StatisticalValidator
from .validation.model_validation import ModelValidator, GoodnessOfFit
from .monte_carlo.monte_carlo_engine import MonteCarloEngine, ConvergenceDiagnostics

__all__ = [
    'UncertaintyAnalyzer',
    'BootstrapAnalyzer',
    'BayesianAnalyzer',
    'HypothesisTestSuite',
    'StatisticalValidator',
    'ModelValidator',
    'GoodnessOfFit',
    'MonteCarloEngine',
    'ConvergenceDiagnostics'
]