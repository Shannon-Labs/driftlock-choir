"""
Script to run the RF Chronometric Interferometry demonstration
and generate visualizations for the OSS release.
"""

#!/usr/bin/env python3

import os
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_PATH = PROJECT_ROOT / "src"

if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))


def run_demo():
    """Run the RF Chronometric Interferometry demonstration."""
    print("Running RF Chronometric Interferometry Demonstration")
    print("=" * 60)

    # Try importing and running the experiment
    try:
        from src.experiments.e1_basic_beat_note import ExperimentE1
        from src.experiments.runner import ExperimentContext

        # Create experiment
        experiment = ExperimentE1()

        # Get the default configuration
        config = experiment.create_default_config()

        # Update parameters for the demo
        config.parameters["plot_results"] = True
        config.parameters["save_plots"] = True
        config.parameters["duration_seconds"] = 0.01  # 10 ms for better visualization
        config.parameters["sampling_rate_hz"] = 20e6  # 20 MS/s
        config.parameters["true_tau_ps"] = 100.0  # 100 ps delay
        config.parameters["true_delta_f_hz"] = 50.0  # 50 Hz offset
        config.parameters["snr_db"] = 35.0  # Good SNR
        config.parameters["add_phase_noise"] = True  # Include realistic phase noise

        # Create context
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"results/demo_run_{timestamp}"
        os.makedirs(output_dir, exist_ok=True)

        context = ExperimentContext(
            config=config, output_dir=output_dir, random_seed=42, verbose=True
        )

        print(f"Running experiment with parameters:")
        print(f"  - TX Frequency: {config.parameters['tx_frequency_hz']/1e6:.1f} MHz")
        print(f"  - RX Frequency: {config.parameters['rx_frequency_hz']/1e6:.1f} MHz")
        print(f"  - Duration: {config.parameters['duration_seconds']*1000:.1f} ms")
        print(f"  - Time-of-flight: {config.parameters['true_tau_ps']:.1f} ps")
        print(f"  - Frequency offset: {config.parameters['true_delta_f_hz']:.1f} Hz")
        print()

        # Run the experiment
        result = experiment.run_experiment(context, config.parameters)

        print(f"Experiment completed with success: {result.success}")
        print(f"Timing RMSE: {result.metrics.rmse_timing:.2f} ps")
        print(f"Frequency RMSE: {result.metrics.rmse_frequency:.2f} ppb")
        print()

        if result.success:
            print(
                "✓ RF Chronometric Interferometry demonstration completed successfully!"
            )
            print(f"✓ Visualization saved to: {output_dir}")
        else:
            print("⚠ Experiment completed but with accuracy below threshold.")
            print("  This is expected for demonstration with challenging conditions.")

        return result, output_dir

    except Exception as e:
        print(f"❌ Error running experiment: {e}")
        import traceback

        traceback.print_exc()
        return None, None


def create_educational_plots():
    """Create educational visualizations explaining the concept."""
    print("\nCreating educational visualizations...")

    # Create a concept diagram
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Example parameters
    t = np.linspace(0, 0.01, 1000)  # 10 ms time vector
    f_tx = 2442e6  # 2442 MHz
    f_rx = 2442e6 + 100  # 100 Hz offset
    tau = 100e-12  # 100 ps delay

    # Signal 1: TX signal (with delay)
    sig_tx = np.cos(2 * np.pi * f_tx * (t - tau))
    # Signal 2: RX signal
    sig_rx = np.cos(2 * np.pi * f_rx * t)

    # Beat note (mixing of TX and RX)
    beat_note = sig_tx * sig_rx

    # Plot 1: Individual signals
    axes[0, 0].plot(t * 1e3, sig_tx, "b-", linewidth=1, label="TX Signal (Delayed)")
    axes[0, 0].plot(t * 1e3, sig_rx, "r-", linewidth=1, label="RX Signal")
    axes[0, 0].set_title("RF Signals in Chronometric Interferometry", fontweight="bold")
    axes[0, 0].set_xlabel("Time (ms)")
    axes[0, 0].set_ylabel("Amplitude")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Plot 2: Beat note
    axes[0, 1].plot(t * 1e3, beat_note, "g-", linewidth=1)
    axes[0, 1].set_title("Beat Note Formation (TX × RX)", fontweight="bold")
    axes[0, 1].set_xlabel("Time (ms)")
    axes[0, 1].set_ylabel("Amplitude")
    axes[0, 1].grid(True, alpha=0.3)

    # Plot 3: Concept diagram
    ax = axes[1, 0]
    ax.arrow(
        0.2,
        0.5,
        0.3,
        0,
        head_width=0.05,
        head_length=0.05,
        fc="blue",
        ec="blue",
        linewidth=3,
    )
    ax.arrow(
        0.7,
        0.5,
        -0.3,
        0,
        head_width=0.05,
        head_length=0.05,
        fc="red",
        ec="red",
        linewidth=3,
    )
    ax.text(0.1, 0.6, "Node A\n(Transmitter)", fontsize=12, ha="center")
    ax.text(0.9, 0.6, "Node B\n(Receiver)", fontsize=12, ha="center")
    ax.text(
        0.5, 0.7, f"τ = {tau*1e12:.0f} ps", fontsize=12, ha="center", style="italic"
    )
    ax.text(
        0.5, 0.3, f"Δf = {f_rx-f_tx:.0f} Hz", fontsize=12, ha="center", style="italic"
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("Concept: Two-Way Signal Exchange", fontweight="bold")
    ax.axis("off")

    # Plot 4: Phase evolution
    instantaneous_phase = np.arctan2(np.imag(beat_note), np.real(beat_note))
    # Unwrap the phase to show continuous evolution
    unwrapped_phase = np.unwrap(instantaneous_phase)
    axes[1, 1].plot(t * 1e3, unwrapped_phase, "m-", linewidth=1)
    axes[1, 1].set_title(
        "Phase Evolution Contains Delay Information", fontweight="bold"
    )
    axes[1, 1].set_xlabel("Time (ms)")
    axes[1, 1].set_ylabel("Phase (radians)")
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    ed_viz_path = "results/educational_visualization.png"
    os.makedirs("results", exist_ok=True)
    plt.savefig(ed_viz_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ Educational visualization saved to: {ed_viz_path}")

    return ed_viz_path


def main():
    """Main execution function."""
    print("Driftlock Choir: RF Chronometric Interferometry OSS Demo")
    print("Generating educational content and visualizations...")
    print()

    # Create the educational visualization
    ed_viz_path = create_educational_plots()

    # Run the actual experiment
    result, output_dir = run_demo()

    # Create a summary
    print("\n" + "=" * 60)
    print("RF CHRONOMETRIC INTERFEROMETRY DEMO SUMMARY")
    print("=" * 60)
    print("Files generated:")
    print(f"  - Educational writeup: CHRONOMETRIC_INTERFEROMETRY_EXPLAINED.md")
    print(f"  - Educational visualization: {ed_viz_path}")
    if output_dir:
        print(f"  - Experiment results: {output_dir}/")
        print(
            f"  - Main visualization: {output_dir}/E1_Basic_Beat_Note_visualization.png"
        )
    print()
    print("To understand the technique, please read:")
    print("  1. CHRONOMETRIC_INTERFEROMETRY_EXPLAINED.md")
    print("  2. The generated visualizations")
    print("  3. The source code in src/experiments/e1_basic_beat_note.py")
    print()
    print("The demonstration shows how RF signals can be used to measure")
    print("picosecond-level time-of-flight delays through beat-note analysis.")

    return 0 if result else 1


if __name__ == "__main__":
    sys.exit(main())
