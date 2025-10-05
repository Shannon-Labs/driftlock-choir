"""
Signal processing components for Driftlock Choir.
"""

from .oscillator import Oscillator
from .beat_note import BeatNoteProcessor
from .channel import ChannelSimulator

__all__ = [
    "Oscillator",
    "BeatNoteProcessor", 
    "ChannelSimulator",
]