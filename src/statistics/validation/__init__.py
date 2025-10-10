"""
Statistical validation module for chronometric interferometry research.

This module provides hypothesis testing, model validation, and statistical
validation capabilities suitable for research publication standards.
"""

from .hypothesis_testing import HypothesisTestSuite, StatisticalValidator
from .model_validation import ModelValidator, GoodnessOfFit

__all__ = [
    'HypothesisTestSuite',
    'StatisticalValidator',
    'ModelValidator',
    'GoodnessOfFit'
]