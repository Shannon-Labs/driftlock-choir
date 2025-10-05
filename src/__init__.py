"""
Driftlock Choir: Ultra-Precise Distributed Clock Synchronization

A research framework for exploring chronometric interferometry-based
clock synchronization in wireless sensor networks.
"""

__version__ = "0.1.0"
__author__ = "Driftlock Choir Team"

from .core.types import *
from .signal_processing.oscillator import Oscillator
from .signal_processing.beat_note import BeatNoteProcessor
from .algorithms.estimator import EstimatorFactory
from .algorithms.consensus import MetropolisConsensus, ConsensusSimulator

__all__ = [
    "Oscillator",
    "BeatNoteProcessor",
    "EstimatorFactory",
    "MetropolisConsensus",
    "ConsensusSimulator",
]