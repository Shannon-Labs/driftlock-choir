import argparse
import json
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from pathlib import Path
import sys

# Ensure the source directory is in the path to import driftlock modules
sys.path.append(str(Path(__file__).resolve().parents[2]))

def main():
    """Generate an animation visualizing the Phase 1 handshake process."""
    parser = argparse.ArgumentParser(description='Generate Phase 1 Handshake Animation')
    parser.add_argument('--results_path', type=str,
                        default='results/phase1/phase1_results.json',
                        help='Path to the phase1_results.json file with a trace.')
    parser.add_argument('--output_path', type=str,
                        default='driftlock_choir_sim/outputs/movies/phase1_handshake_demo.mp4',
                        help='Path to save the output MP4 video.')
    args = parser.parse_args()

    results_path = Path(args.results_path)
    output_path = Path(args.output_path)

    if not results_path.exists():
        print(f"Error: Results file not found at {results_path}")
        print("Please run `PYTHONPATH=. python sim/phase1.py --run-legacy-snr-sweep --capture-trace-snr-db 20` first.")
        sys.exit(1)

    with open(results_path, 'r') as f:
        results = json.load(f)

    if 'example_trace' not in results or results['example_trace'] is None:
        print(f"Error: No 'example_trace' found in {results_path}.")
        print("Please re-run `sim/phase1.py` with the `--capture-trace-snr-db 20` argument.")
        sys.exit(1)

    trace = results['example_trace']
    snr_db = results['config'].get('capture_trace_snr_db', 'N/A')
    tof_rmse_ps = 0.0
    try:
        snr_index = results['snr_sweep']['snr_db'].index(snr_db)
        tof_rmse_ps = results['snr_sweep']['tof_rmse_ps'][snr_index]
    except (ValueError, IndexError, KeyError):
        print("Warning: Could not find exact SNR in results to display RMSE.")

    # Setup the plot
    fig, axes = plt.subplots(3, 1, figsize=(8, 8), constrained_layout=True)
    fig.suptitle(f'Driftlock: Core Insight (at {snr_db} dB SNR)\nAchieved Precision: {tof_rmse_ps:.2f} ps', fontsize=16, fontweight='bold')

    time_us = np.array(trace['time_us'])
    beat_raw_real = np.array(trace['beat_raw_real'])
    beat_filtered_real = np.array(trace['beat_filtered_real'])
    adc_time_us = np.array(trace['adc_time_us'])
    unwrapped_phase = np.array(trace['unwrapped_phase'])
    phase_fit = np.array(trace['phase_fit'])

    # Plot 1: Raw Beat Signal
    axes[0].set_title("(A) Measure Noisy Beat Signal")
    axes[0].set_xlabel("Time (μs)")
    axes[0].set_ylabel("Amplitude")
    axes[0].set_xlim(time_us.min(), time_us.max())
    axes[0].set_ylim(np.min(beat_raw_real) * 1.2, np.max(beat_raw_real) * 1.2)
    axes[0].grid(True, alpha=0.3)
    line1, = axes[0].plot([], [], lw=1.5, color='blue')

    # Plot 2: Filtered Beat Signal
    axes[1].set_title("(B) Filter to Isolate Beat")
    axes[1].set_xlabel("Time (μs)")
    axes[1].set_ylabel("Amplitude")
    axes[1].set_xlim(time_us.min(), time_us.max())
    axes[1].set_ylim(np.min(beat_filtered_real) * 1.2, np.max(beat_filtered_real) * 1.2)
    axes[1].grid(True, alpha=0.3)
    line2, = axes[1].plot([], [], lw=2, color='green')

    # Plot 3: Phase Extraction
    axes[2].set_title("(C) Extract Phase & Fit Linear Model")
    axes[2].set_xlabel("Time (μs)")
    axes[2].set_ylabel("Phase (rad)")
    axes[2].set_xlim(adc_time_us.min(), adc_time_us.max())
    axes[2].set_ylim(np.min(unwrapped_phase) - 0.5, np.max(unwrapped_phase) + 0.5)
    axes[2].grid(True, alpha=0.3)
    line3, = axes[2].plot([], [], lw=2, color='red', label='Measured Phase')
    line4, = axes[2].plot([], [], lw=2.5, color='black', linestyle='--', label='Linear Fit (The Timing Info)')
    axes[2].legend(loc='upper left')

    total_frames = 300 # 15 seconds at 20 fps

    def animate(i):
        fraction = i / total_frames
        end_idx12 = int(len(time_us) * fraction)
        line1.set_data(time_us[:end_idx12], beat_raw_real[:end_idx12])
        line2.set_data(time_us[:end_idx12], beat_filtered_real[:end_idx12])

        end_idx3 = int(len(adc_time_us) * fraction)
        line3.set_data(adc_time_us[:end_idx3], unwrapped_phase[:end_idx3])
        if i > total_frames * 0.5: # Start drawing the fit line halfway through
            fit_fraction = (i - total_frames * 0.5) / (total_frames * 0.5)
            fit_idx = int(len(adc_time_us) * fit_fraction)
            line4.set_data(adc_time_us[:fit_idx], phase_fit[:fit_idx])

        return line1, line2, line3, line4

    # Create and save the animation
    ani = animation.FuncAnimation(fig, animate, frames=total_frames, blit=False, interval=50)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Saving 15-second animation to {output_path}...")
    ani.save(str(output_path), writer='ffmpeg', fps=20, dpi=120)
    print("Animation saved successfully.")
    plt.close()

if __name__ == "__main__":
    main()