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
        config.parameters["plot_results"] = False
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





def main():


    """Main execution function."""


    print("Driftlock Choir: RF Chronometric Interferometry OSS Demo")


    print("Generating educational content and visualizations...")


    print()





    # Run the actual experiment


    result, output_dir = run_demo()





    # Create a summary


    print("\n" + "=" * 60)


    print("RF CHRONOMETRIC INTERFEROMETRY DEMO SUMMARY")


    print("=" * 60)


    print("Files generated:")


    print(f"  - Educational writeup: CHRONOMETRIC_INTERFEROMETRY_EXPLAINED.md")


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
