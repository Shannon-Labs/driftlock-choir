import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# --- Add project root to sys.path to allow importing from src ---
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.algorithms.estimator import EstimatorFactory
from src.core.types import (
    BeatNoteData,
    Hertz,
    MeasurementQuality,
    Picoseconds,
    Seconds,
    Timestamp,
)
from src.signal_processing.beat_note import BeatNoteProcessor

def generate_narrative_visualization():
    """Generate an authentic, data-driven, three-panel narrative visualization."""

    # --- 1. Setup Simulation Parameters (Ground Truth) ---
    SAMPLING_RATE_HZ = 40e6
    DURATION_S = 0.5
    TAU_PS = 13.5
    DELTA_F_HZ = 150.0
    SNR_DB = 40.0

    print("Step 1: Generating base signal data from simulation...")
    
    num_samples = int(DURATION_S * SAMPLING_RATE_HZ)
    t = np.arange(num_samples) / SAMPLING_RATE_HZ
    
    tx_signal = np.exp(1j * 2 * np.pi * (1e6 + DELTA_F_HZ) * t)
    rx_signal = np.exp(1j * 2 * np.pi * 1e6 * (t - TAU_PS * 1e-12))

    processor = BeatNoteProcessor(Hertz(SAMPLING_RATE_HZ))
    beat_note_data = processor.generate_beat_note(
        tx_signal=tx_signal,
        rx_signal=rx_signal,
        tx_frequency=Hertz(1e6 + DELTA_F_HZ),
        rx_frequency=Hertz(1e6),
        duration=Seconds(DURATION_S),
        timestamp=Timestamp(
            time=Seconds(0.0), 
            uncertainty=Picoseconds(0.0), 
            quality=MeasurementQuality.EXCELLENT
        ),
        add_noise=True,
        snr_db=SNR_DB,
    )

    # --- 2. Iterative Analysis for Convergence Plot (Panel 2) ---
    print("Step 2: Performing iterative analysis to show convergence...")
    estimator = EstimatorFactory.create_estimator("phase_slope")
    num_steps = 100
    min_samples = int(0.01 * SAMPLING_RATE_HZ)
    
    convergence_time = []
    timing_errors = []

    for i in range(1, num_steps + 1):
        chunk_end = min_samples + (i * (len(beat_note_data.waveform) - min_samples) // num_steps)
        if chunk_end > len(beat_note_data.waveform):
            continue

        chunk_waveform = beat_note_data.waveform[:chunk_end]
        sample_rate_hz = float(beat_note_data.sampling_rate)
        chunk_samples = len(chunk_waveform)
        chunk_duration = Seconds((chunk_samples + 1e-6) / sample_rate_hz)
        
        # This constructor needs all the fields from the original BeatNoteData
        chunk_data = BeatNoteData(
            waveform=chunk_waveform,
            sampling_rate=beat_note_data.sampling_rate,
            duration=chunk_duration,
            tx_frequency=beat_note_data.tx_frequency,
            rx_frequency=beat_note_data.rx_frequency,
            timestamp=beat_note_data.timestamp,
            snr=beat_note_data.snr,
            quality=beat_note_data.quality
        )
        
        result = estimator.estimate(chunk_data)
        
        convergence_time.append(chunk_duration)
        timing_errors.append(abs(result.tau - TAU_PS))

    # --- 3. Plotting ---
    print("Step 3: Creating the three-panel visualization...")
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(3, 1, figsize=(12, 16), facecolor='#ffffff')
    fig.suptitle("Driftlock Choir: The Synchronization Process", fontsize=20, fontweight='bold', color='#1a252f')

    # --- Panel 1: Before Synchronization ---
    t_panel1 = beat_note_data.get_time_vector()[:int(0.0001 * SAMPLING_RATE_HZ)]
    osc_a = np.cos(2 * np.pi * 25000 * t_panel1)
    osc_b = np.cos(2 * np.pi * (25000 + DELTA_F_HZ) * (t_panel1 - TAU_PS * 1e-12))
    
    axes[0].plot(t_panel1 * 1e6, osc_a, color='#4285F4', alpha=0.9, label='Oscillator A (Reference)')
    axes[0].plot(t_panel1 * 1e6, osc_b, color='#DB4437', alpha=0.9, label='Oscillator B (Drifting)')
    axes[0].set_title("BEFORE: Unsynchronized Signals", fontsize=14, fontweight='bold')
    axes[0].text(0.03, 0.9, "Signals have different frequencies and a phase offset", 
                 transform=axes[0].transAxes, verticalalignment='top', fontsize=11, bbox=dict(boxstyle='round,pad=0.5', fc='white', alpha=0.7))
    axes[0].set_xlabel("Time (microseconds)")
    axes[0].set_ylabel("Signal Strength")
    axes[0].legend()

    # --- Panel 2: The Convergence Process ---
    axes[1].plot(convergence_time, timing_errors, color='#F4B400', linewidth=2.5, label='Timing Error')
    axes[1].set_title("DURING: The Algorithm Locks On", fontsize=14, fontweight='bold')
    axes[1].set_xlabel("Signal Processing Time (seconds)")
    axes[1].set_ylabel("Timing Error (picoseconds)")
    axes[1].set_yscale('log')
    axes[1].grid(True, which="both", ls="--", c='0.7')
    axes[1].text(0.97, 0.9, "The estimator iteratively refines its measurement,\nrapidly reducing the timing error.", 
                 transform=axes[1].transAxes, verticalalignment='top', horizontalalignment='right', fontsize=11, bbox=dict(boxstyle='round,pad=0.5', fc='white', alpha=0.7))

    # --- Panel 3: After Synchronization ---
    final_error_ps = timing_errors[-1]
    t_after = beat_note_data.get_time_vector()[int(0.1*SAMPLING_RATE_HZ):int(0.1001*SAMPLING_RATE_HZ)]
    osc_a_after = np.cos(2 * np.pi * 25000 * t_after)
    osc_b_corrected = np.cos(2 * np.pi * 25000 * (t_after - final_error_ps * 1e-12))

    axes[2].plot(t_after * 1e6, osc_a_after, color='#4285F4', label='Oscillator A')
    axes[2].plot(t_after * 1e6, osc_b_corrected, '--', color='#0F9D58', dashes=(5, 5), linewidth=2.5, label='Oscillator B (Corrected)')
    axes[2].set_title(f"AFTER: Stable Lock Achieved ({final_error_ps:.2f} ps Residual Error)", fontsize=14, fontweight='bold')
    axes[2].text(0.03, 0.9, "The remaining error is measured, and the signals are now effectively synchronized.", 
                 transform=axes[2].transAxes, verticalalignment='top', fontsize=11, bbox=dict(boxstyle='round,pad=0.5', fc='white', alpha=0.7))
    axes[2].set_xlabel("Time (microseconds)")
    axes[2].set_ylabel("Signal Strength")
    axes[2].legend(loc='lower left')

    # --- Final Touches & Save ---
    fig.tight_layout(rect=[0, 0.03, 1, 0.96])
    output_dir = "docs/assets/images"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "narrative_synchronization_13p5ps.png")
    plt.savefig(output_path, dpi=300)
    print(f"\nNew visualization saved to: {output_path}")

if __name__ == '__main__':
    generate_narrative_visualization()
