#!/usr/bin/env python3
"""Generate the 2.4 GHz beat-note hero figure highlighting 13.5 ps recovery."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from src.algorithms.estimator import EstimatorFactory
from src.core.types import Hertz, Picoseconds, Seconds, Timestamp
from src.signal_processing.beat_note import BeatNoteProcessor
from src.signal_processing.channel import ChannelSimulator
from src.signal_processing.oscillator import Oscillator
from src.visualization.base.utils import set_figure_dpi


def generate_hero_signals() -> tuple:
    np.random.seed(42)

    tx_frequency = Hertz(2.4e9)
    rx_frequency = Hertz(2.4e9 + 150.0)
    sampling_rate = Hertz(40e6)
    duration = Seconds(0.002)
    true_tau_ps = 13.5
    snr_db = 35.0

    tx_model = Oscillator.create_tcxo_model(tx_frequency)
    rx_model = Oscillator.create_tcxo_model(rx_frequency)

    tx = Oscillator(tx_model)
    rx = Oscillator(rx_model)

    _, tx_signal = tx.generate_signal(
        duration=duration,
        sampling_rate=sampling_rate,
        frequency_offset=Hertz(0.0),
        phase_noise_enabled=False,
    )
    _, rx_signal = rx.generate_signal(
        duration=duration,
        sampling_rate=sampling_rate,
        frequency_offset=Hertz(0.0),
        phase_noise_enabled=False,
    )

    channel_sim = ChannelSimulator(sampling_rate)
    los_channel = channel_sim.create_multipath_channel(
        delays=[Picoseconds(true_tau_ps)],
        powers=[1.0],
    )
    rx_signal = channel_sim.apply_channel(rx_signal, los_channel, rx_frequency)
    rx_signal = channel_sim.add_thermal_noise(rx_signal, snr_db=snr_db)

    processor = BeatNoteProcessor(sampling_rate)
    beat_note = processor.generate_beat_note(
        tx_signal=tx_signal,
        rx_signal=rx_signal,
        tx_frequency=tx_frequency,
        rx_frequency=rx_frequency,
        duration=duration,
        timestamp=Timestamp.from_ps(0.0),
        add_noise=False,
        snr_db=snr_db,
    )

    return (
        beat_note,
        processor,
        tx_frequency,
        rx_frequency,
        sampling_rate,
        true_tau_ps,
    )


def build_hero_figure(
    beat_note,
    processor,
    tx_frequency,
    rx_frequency,
    sampling_rate,
    true_tau_ps,
) -> plt.Figure:
    estimator = EstimatorFactory.create_estimator("phase_slope")
    estimation = estimator.estimate(beat_note)

    time_vector = beat_note.get_time_vector()
    time_vector = time_vector - time_vector[0]
    time_us = time_vector * 1e6

    waveform = beat_note.waveform.real
    window_samples = int(0.0004 * float(sampling_rate))
    window_samples = max(window_samples, 1)

    phase_time, phase = processor.extract_instantaneous_phase(beat_note)
    phase_time = phase_time - phase_time[0]
    design_matrix = np.column_stack(
        [phase_time, np.ones_like(phase_time)]
    )
    slope, intercept = np.linalg.lstsq(design_matrix, phase, rcond=None)[0]
    phase_fit = design_matrix @ np.array([slope, intercept])

    carrier_frequency = float(rx_frequency)
    intercept_wrapped = np.arctan2(np.sin(intercept), np.cos(intercept))
    tau_estimate_ps = abs(intercept_wrapped) / (2 * np.pi * carrier_frequency) * 1e12
    tau_error_ps = tau_estimate_ps - true_tau_ps

    phase_residual = phase - phase_fit
    residual_tau_ps = phase_residual / (2 * np.pi * carrier_frequency) * 1e12

    set_figure_dpi(300)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), constrained_layout=True)

    # Panel A: beat note waveform
    slice_end = min(window_samples, len(waveform))
    axes[0].plot(time_us[:slice_end], waveform[:slice_end], color="#0066cc", linewidth=1.2)
    axes[0].set_title("Beat Note (2.4 GHz)")
    axes[0].set_xlabel("Time (µs)")
    axes[0].set_ylabel("Amplitude")
    axes[0].grid(alpha=0.3)

    # Panel B: phase unwrapping with fit
    axes[1].plot(time_us, phase, color="#1a936f", linewidth=1.0, label="Unwrapped phase")
    axes[1].plot(
        time_us,
        phase_fit,
        color="#ef476f",
        linewidth=1.2,
        linestyle="--",
        label="Phase slope fit",
    )
    axes[1].set_title("Phase Slope → τ")
    axes[1].set_xlabel("Time (µs)")
    axes[1].set_ylabel("Phase (rad)")
    axes[1].legend(loc="lower right")
    axes[1].grid(alpha=0.3)
    axes[1].text(
        0.02,
        0.05,
        f"τ̂ = {tau_estimate_ps:.2f} ps\nτ = {true_tau_ps:.2f} ps",
        transform=axes[1].transAxes,
        fontsize=10,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.7, edgecolor="none"),
    )

    # Panel C: τ residuals
    axes[2].plot(time_us, residual_tau_ps, color="#073b4c", linewidth=1.0)
    axes[2].axhline(0.0, color="#ef476f", linewidth=1.0, linestyle=":")
    axes[2].set_title("τ Residuals (ps)")
    axes[2].set_xlabel("Time (µs)")
    axes[2].set_ylabel("Δτ (ps)")
    axes[2].grid(alpha=0.3)
    axes[2].text(
        0.02,
        0.92,
        f"|τ̂ − τ| = {abs(tau_error_ps):.3f} ps",
        transform=axes[2].transAxes,
        fontsize=10,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.7, edgecolor="none"),
    )

    fig.suptitle(
        "Chronometric Interferometry Beat-Note Analysis — 13.5 ps Recovery",
        fontsize=14,
        fontweight="bold",
    )

    return fig


def main() -> None:
    (
        beat_note,
        processor,
        tx_frequency,
        rx_frequency,
        sampling_rate,
        true_tau_ps,
    ) = generate_hero_signals()

    fig = build_hero_figure(
        beat_note,
        processor,
        tx_frequency,
        rx_frequency,
        sampling_rate,
        true_tau_ps,
    )

    output_dir = PROJECT_ROOT / "docs" / "assets" / "images"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "hero_beat_note_tau13p5ps.png"
    fig.savefig(output_path, dpi=250, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved beat-note hero image to {output_path}")


if __name__ == "__main__":
    main()
