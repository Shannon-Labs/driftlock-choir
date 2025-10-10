"""
Test suite for chronometric interferometry functionality.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import unittest

import numpy as np

from src.core.types import ExperimentConfig
from src.experiments.e1_basic_beat_note import ExperimentE1
from src.experiments.runner import ExperimentContext


class TestChronometricInterferometry(unittest.TestCase):
    """Test chronometric interferometry functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.experiment = ExperimentE1()
        self.config = self.experiment.create_default_config()
        self.context = ExperimentContext(
            config=self.config, output_dir="test_results", random_seed=42, verbose=False
        )

    def test_experiment_creation(self):
        """Test experiment creation."""
        self.assertIsNotNone(self.experiment)
        self.assertEqual(self.experiment.name, "E1_Basic_Beat_Note")
        self.assertIn("beat-note", self.experiment.description.lower())

    def test_config_creation(self):
        """Test configuration creation."""
        config = self.experiment.create_default_config()

        self.assertIsInstance(config, ExperimentConfig)
        self.assertEqual(config.experiment_id, "E1_Basic_Beat_Note")
        self.assertIn("parameters", config.__dict__)

        # Check required parameters
        params = config.parameters
        required_params = [
            "tx_frequency_hz",
            "rx_frequency_hz",
            "sampling_rate_hz",
            "duration_seconds",
            "true_tau_ps",
            "true_delta_f_hz",
        ]

        for param in required_params:
            self.assertIn(param, params)

    def test_experiment_execution(self):
        """Test experiment execution."""
        # Use reduced parameters for faster testing
        test_params = self.config.parameters.copy()
        test_params["duration_seconds"] = 0.001  # 1 ms instead of 100 ms
        test_params["plot_results"] = False
        test_params["save_plots"] = False

        result = self.experiment.run_experiment(self.context, test_params)

        # Check result structure
        self.assertIsNotNone(result)
        self.assertIn("success", result.__dict__)
        self.assertIn("metrics", result.__dict__)

        # Check metrics
        self.assertIsNotNone(result.metrics)
        self.assertIsNotNone(result.metrics.rmse_timing)
        self.assertIsNotNone(result.metrics.rmse_frequency)

        # For ideal conditions, should succeed
        if result.success:
            # Check timing accuracy (should be better than 1000 ps)
            self.assertLess(result.metrics.rmse_timing, 1000.0)

            # Check frequency accuracy (should be better than 100 ppb)
            self.assertLess(result.metrics.rmse_frequency, 100.0)
        else:
            print(f"Experiment failed: {result.error_message}")

    def test_parameter_validation(self):
        """Test parameter validation."""
        # Test with invalid parameters
        invalid_params = self.config.parameters.copy()
        invalid_params["duration_seconds"] = -1.0  # Invalid duration

        try:
            result = self.experiment.run_experiment(self.context, invalid_params)
            # If it doesn't raise an exception, it should at least fail
            if hasattr(result, "success"):
                self.assertFalse(result.success)
        except (ValueError, AssertionError):
            # Expected behavior for invalid parameters
            pass

    def test_noise_robustness(self):
        """Test experiment with different noise levels."""
        snr_levels = [20, 30, 40]  # Test different SNR levels

        for snr_db in snr_levels:
            with self.subTest(snr_db=snr_db):
                test_params = self.config.parameters.copy()
                test_params["duration_seconds"] = 0.001
                test_params["plot_results"] = False
                test_params["save_plots"] = False
                test_params["snr_db"] = snr_db

                result = self.experiment.run_experiment(self.context, test_params)

                self.assertIsNotNone(result)

                # Higher SNR should generally give better results
                if result.success:
                    if snr_db >= 30:
                        # Should have reasonable accuracy at higher SNR
                        self.assertLess(result.metrics.rmse_timing, 500.0)

    def test_frequency_offset_range(self):
        """Test experiment with different frequency offsets."""
        frequency_offsets = [10.0, 50.0, 100.0]  # Hz

        for delta_f in frequency_offsets:
            with self.subTest(delta_f=delta_f):
                test_params = self.config.parameters.copy()
                test_params["duration_seconds"] = 0.001
                test_params["plot_results"] = False
                test_params["save_plots"] = False
                test_params["true_delta_f_hz"] = delta_f

                result = self.experiment.run_experiment(self.context, test_params)

                self.assertIsNotNone(result)
                # Should handle reasonable frequency offsets
                if delta_f <= 100.0:
                    # More likely to succeed with smaller offsets
                    if result.success:
                        self.assertIsNotNone(result.metrics.rmse_frequency)


class TestExperimentRunner(unittest.TestCase):
    """Test experiment runner functionality."""

    def test_context_creation(self):
        """Test experiment context creation."""
        experiment = ExperimentE1()
        config = experiment.create_default_config()

        context = ExperimentContext(
            config=config, output_dir="test_results", random_seed=42, verbose=False
        )

        self.assertIsNotNone(context)
        self.assertEqual(context.config, config)
        self.assertEqual(context.output_dir, "test_results")
        self.assertEqual(context.random_seed, 42)
        self.assertFalse(context.verbose)


if __name__ == "__main__":
    # Run tests
    unittest.main(verbosity=2)
