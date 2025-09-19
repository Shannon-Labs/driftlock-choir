"""
DriftLock: Physical-layer synchronization framework.

This package provides comprehensive simulation tools for analyzing
synchronization performance in wireless networks with realistic
hardware imperfections and channel conditions.
"""

__version__ = "1.0.0"
__author__ = "Shannon Labs"
__email__ = "contact@shannonlabs.com"

# Package-level imports for convenience
from . import phy
from . import hw
from . import alg
from . import metrics
from . import mac

__all__ = ['phy', 'hw', 'alg', 'metrics', 'mac']
