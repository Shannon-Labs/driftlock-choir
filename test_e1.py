"""
Test script for Experiment E1: Basic Beat-Note Formation and Analysis.

This script runs a basic test of our chronometric interferometry framework
to validate that the core components are working correctly.
"""

#!/usr/bin/env python3

import os
import sys

import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from src.core.types import ExperimentConfig, Timestamp
from src.experiments.e1_basic_beat_note import ExperimentE1
from src.experiments.runner import ExperimentContext


def test_basic_functionality():
    """Test basic functionality of the framework."""
    print("=== Testing Basic Functionality ===")

    try:
        # Create experiment
        experiment = ExperimentE1()
        print("✓ Experiment E1 created successfully")

        # Create configuration
        config = experiment.create_default_config()
        print("✓ Default configuration created successfully")

        # Create context
        context = ExperimentContext(
            config=config,
            output_dir="test_results",
            random_seed=42,
            verbose=False,  # Reduce output for testing
        )
        print("✓ Experiment context created successfully")

        # Run experiment with minimal parameters
        test_params = config.parameters.copy()
        test_params["plot_results"] = False  # Don't show plots during testing
        test_params["save_plots"] = False  # Don't save plots during testing
        test_params["duration_seconds"] = 0.01  # Shorter duration for testing

        result = experiment.run_experiment(context, test_params)
        print("✓ Experiment executed successfully")

        # Check results
        print(f"✓ Success: {result.success}")
        print(f"✓ Timing RMSE: {result.metrics.rmse_timing:.1f} ps")
        print(f"✓ Frequency RMSE: {result.metrics.rmse_frequency:.1f} ppb")

        if result.success:
            print("✓ Experiment completed successfully!")
            return True
        else:
            print(f"✗ Experiment failed: {result.error_message}")
            return False

    except Exception as e:
        print(f"✗ Test failed with exception: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_signal_generation():
    """Test signal generation components."""
    print("\n=== Testing Signal Generation ===")

    try:
        from src.core.types import Hertz, Seconds
        from src.signal_processing.oscillator import Oscillator

        # Create ideal oscillator
        oscillator_model = Oscillator.create_ideal_oscillator(Hertz(2.4e9))
        oscillator = Oscillator(oscillator_model)
        print("✓ Ideal oscillator created successfully")

        # Generate signal
        duration = Seconds(0.001)  # 1 ms
        sampling_rate = Hertz(1e6)  # 1 MHz
        time, signal = oscillator.generate_signal(duration, sampling_rate)
        print("✓ Signal generated successfully")
        print(f"✓ Signal length: {len(signal)} samples")
        print(f"✓ Signal duration: {time[-1]:.6f} seconds")

        # Verify signal properties
        expected_samples = int(duration * sampling_rate)
        if len(signal) == expected_samples:
            print("✓ Signal length matches expected")
        else:
            print(
                f"✗ Signal length mismatch: expected {expected_samples}, got {len(signal)}"
            )
            return False

        # Check signal magnitude (should be approximately 1 for complex exponential)
        magnitude = np.mean(np.abs(signal))
        if 0.9 < magnitude < 1.1:
            print(f"✓ Signal magnitude is correct: {magnitude:.3f}")
        else:
            print(f"✗ Signal magnitude is incorrect: {magnitude:.3f}")
            return False

        return True

    except Exception as e:
        print(f"✗ Signal generation test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_beat_note_processing():
    """Test beat note processing."""
    print("\n=== Testing Beat Note Processing ===")

    try:
        from src.core.types import Hertz, Seconds, Timestamp
        from src.signal_processing.beat_note import BeatNoteProcessor
        from src.signal_processing.oscillator import Oscillator

        # Create beat note for more realistic frequencies
        tx_oscillator = Oscillator(
            Oscillator.create_ideal_oscillator(Hertz(2.4e9))
        )  # 2.4 GHz
        rx_oscillator = Oscillator(
            Oscillator.create_ideal_oscillator(Hertz(2.4e9 + 100.0))
        )  # 2.4 GHz + 100 Hz
        print("✓ Oscillators created successfully")

        # Generate signals
        duration = Seconds(0.001)
        sampling_rate = Hertz(1e6)
        _, tx_signal = tx_oscillator.generate_signal(duration, sampling_rate)
        _, rx_signal = rx_oscillator.generate_signal(
            duration, sampling_rate, Hertz(0.0)
        )  # No additional offset
        print("✓ Signals generated successfully")

        # Create beat note processor
        processor = BeatNoteProcessor(sampling_rate)
        print("✓ Beat note processor created successfully")

        # Generate beat note
        timestamp = Timestamp.from_ps(0.0)
        beat_note = processor.generate_beat_note(
            tx_signal=tx_signal,
            rx_signal=rx_signal,
            tx_frequency=Hertz(2.4e9),  # 2.4 GHz
            rx_frequency=Hertz(2.4e9 + 100.0),  # 2.4 GHz + 100 Hz
            duration=duration,
            timestamp=timestamp,
            add_noise=False,
            snr_db=40.0,
        )
        print("✓ Beat note generated successfully")

        # Extract beat frequency
        beat_freq, freq_uncertainty = processor.extract_beat_frequency(beat_note)
        print(
            f"✓ Beat frequency extracted: {beat_freq:.1f} ± {freq_uncertainty:.1f} Hz"
        )

        # Expected beat frequency is 100 Hz (the nominal difference)
        # The 50 Hz offset is applied during signal generation, not reflected in beat_note.get_beat_frequency()
        expected_beat_freq = 100.0
        if abs(beat_freq - expected_beat_freq) < 10.0:
            print("✓ Beat frequency is correct")
        else:
            print(
                f"✗ Beat frequency mismatch: expected ~{expected_beat_freq}, got {beat_freq}"
            )
            return False

        # Estimate τ and Δf
        estimation_result = processor.estimate_tau_delta_f(beat_note)
        print(
            f"✓ τ estimated: {estimation_result.tau:.1f} ± {estimation_result.tau_uncertainty:.1f} ps"
        )
        print(
            f"✓ Δf estimated: {estimation_result.delta_f:.1f} ± {estimation_result.delta_f_uncertainty:.1f} Hz"
        )

        return True

    except Exception as e:
        print(f"✗ Beat note processing test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_algorithms():
    """Test estimation algorithms."""
    print("\n=== Testing Estimation Algorithms ===")

    try:
        from src.algorithms.estimator import EstimatorFactory
        from src.core.types import Hertz, Seconds, Timestamp
        from src.signal_processing.beat_note import BeatNoteProcessor
        from src.signal_processing.oscillator import Oscillator

        # Create beat note with realistic frequencies
        tx_oscillator = Oscillator(
            Oscillator.create_ideal_oscillator(Hertz(2.4e9))
        )  # 2.4 GHz
        rx_oscillator = Oscillator(
            Oscillator.create_ideal_oscillator(Hertz(2.4e9 + 100.0))
        )  # 2.4 GHz + 100 Hz

        duration = Seconds(0.001)
        sampling_rate = Hertz(1e6)
        _, tx_signal = tx_oscillator.generate_signal(duration, sampling_rate)
        _, rx_signal = rx_oscillator.generate_signal(
            duration, sampling_rate, Hertz(0.0)
        )  # No additional offset

        processor = BeatNoteProcessor(sampling_rate)
        timestamp = Timestamp.from_ps(0.0)
        beat_note = processor.generate_beat_note(
            tx_signal=tx_signal,
            rx_signal=rx_signal,
            tx_frequency=Hertz(2.4e9),  # 2.4 GHz
            rx_frequency=Hertz(2.4e9 + 100.0),  # 2.4 GHz + 100 Hz
            duration=duration,
            timestamp=timestamp,
            add_noise=False,
            snr_db=40.0,
        )

        # Test different estimators
        available_estimators = EstimatorFactory.get_available_estimators()
        print(f"✓ Available estimators: {available_estimators}")

        for estimator_type in available_estimators:
            estimator = EstimatorFactory.create_estimator(estimator_type)
            result = estimator.estimate(beat_note)
            print(
                f"✓ {estimator_type} estimator: τ={result.tau:.1f}ps, Δf={result.delta_f:.1f}Hz"
            )

        return True

    except Exception as e:
        print(f"✗ Algorithm test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("Driftlock Choir - Experiment E1 Test Suite")
    print("=" * 50)

    # Run tests
    tests = [
        test_basic_functionality,
        test_signal_generation,
        test_beat_note_processing,
        test_algorithms,
    ]

    results = []
    for test in tests:
        results.append(test())

    # Summary
    print("\n" + "=" * 50)
    print("Test Summary:")
    passed = sum(results)
    total = len(results)
    print(f"Passed: {passed}/{total}")

    if passed == total:
        print("✓ All tests passed! The framework is working correctly.")
        return 0
    else:
        print("✗ Some tests failed. Please check the implementation.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
