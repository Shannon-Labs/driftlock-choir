"""
Signal processing components for Driftlock Choir.
"""

from .beat_note import BeatNoteProcessor
from .channel import ChannelSimulator
from .oscillator import Oscillator

__all__ = [
    "Oscillator",
    "BeatNoteProcessor",
    "ChannelSimulator",
]
