"""Offline helpers to reuse the chronometric interferometry estimator with recorded IQ captures."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.algorithms.estimator import EstimatorFactory
from src.core.types import BeatNoteData, Hertz, Seconds, Timestamp
from src.signal_processing.beat_note import BeatNoteProcessor


def load_complex_iq(path: Path, dtype: np.dtype = np.complex64) -> np.ndarray:
    """Load complex IQ samples saved as .npy or raw binary."""
    if path.suffix == ".npy":
        data = np.load(path)
    else:
        data = np.fromfile(path, dtype=dtype)
    if data.ndim != 1:
        raise ValueError("IQ capture must be a 1-D array")
    return data.astype(np.complex128)


def build_beat_note(
    tx_capture: np.ndarray,
    rx_capture: np.ndarray,
    sampling_rate_hz: float,
    tx_frequency_hz: float,
    rx_frequency_hz: float,
    snr_db: float = 30.0,
) -> BeatNoteData:
    if len(tx_capture) != len(rx_capture):
        raise ValueError("TX and RX captures must have the same length")

    processor = BeatNoteProcessor(Hertz(sampling_rate_hz))
    timestamp = Timestamp.from_ps(0.0)

    return processor.generate_beat_note(
        tx_signal=tx_capture,
        rx_signal=rx_capture,
        tx_frequency=Hertz(tx_frequency_hz),
        rx_frequency=Hertz(rx_frequency_hz),
        duration=Seconds(len(tx_capture) / sampling_rate_hz),
        timestamp=timestamp,
        add_noise=False,
        snr_db=snr_db,
    )


def estimate_from_capture(
    tx_capture: np.ndarray,
    rx_capture: np.ndarray,
    sampling_rate_hz: float,
    tx_frequency_hz: float,
    rx_frequency_hz: float,
    estimator: str = "phase_slope",
    snr_db: float = 30.0,
):
    beat_note = build_beat_note(
        tx_capture=tx_capture,
        rx_capture=rx_capture,
        sampling_rate_hz=sampling_rate_hz,
        tx_frequency_hz=tx_frequency_hz,
        rx_frequency_hz=rx_frequency_hz,
        snr_db=snr_db,
    )

    estimator_impl = EstimatorFactory.create_estimator(estimator)
    result = estimator_impl.estimate(beat_note)

    metadata = {
        "samples": len(tx_capture),
        "sampling_rate_hz": sampling_rate_hz,
        "snr_db": snr_db,
        "quality": beat_note.quality.value,
    }

    return result, metadata
