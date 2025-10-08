"""
Test suite for the basic beat note processing functionality.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import unittest

import numpy as np

from src.algorithms.estimator import EstimatorFactory
from src.core.constants import PhysicalConstants
from src.core.types import ChannelModel, Hertz, Picoseconds, Seconds, Timestamp
from src.signal_processing.beat_note import BeatNoteProcessor
from src.signal_processing.channel import ChannelSimulator
from src.signal_processing.oscillator import Oscillator
from src.signal_processing.utils import apply_fractional_delay


class TestBasicBeatNote(unittest.TestCase):
    """Test basic beat note functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.tx_frequency = Hertz(2.4e9)
        self.rx_frequency = Hertz(2.4e9 + 100.0)
        self.sampling_rate = Hertz(1e6)
        self.duration = Seconds(0.001)
        self.true_tau = Picoseconds(1000.0)
        self.true_delta_f = Hertz(50.0)

    def test_oscillator_creation(self):
        """Test oscillator creation."""
        # Test ideal oscillator
        ideal_model = Oscillator.create_ideal_oscillator(self.tx_frequency)
        self.assertIsNotNone(ideal_model)
        self.assertEqual(ideal_model.frequency, self.tx_frequency)
        self.assertFalse(ideal_model.phase_noise_enabled)

        # Test TCXO oscillator
        tcxo_model = Oscillator.create_tcxo_model(self.tx_frequency)
        self.assertIsNotNone(tcxo_model)
        self.assertEqual(tcxo_model.frequency, self.tx_frequency)
        self.assertTrue(tcxo_model.phase_noise_enabled)

    def test_signal_generation(self):
        """Test signal generation."""
        oscillator_model = Oscillator.create_ideal_oscillator(self.tx_frequency)
        oscillator = Oscillator(oscillator_model)

        time, signal = oscillator.generate_signal(
            duration=self.duration,
            sampling_rate=self.sampling_rate,
            frequency_offset=Hertz(0.0),
            phase_noise_enabled=False,
        )

        # Check signal properties
        expected_samples = int(self.duration * self.sampling_rate)
        self.assertEqual(len(signal), expected_samples)
        self.assertEqual(len(time), expected_samples)

        # Check signal magnitude (should be approximately 1)
        magnitude = np.mean(np.abs(signal))
        self.assertAlmostEqual(magnitude, 1.0, places=1)

    def test_fractional_delay_accuracy(self):
        """Fractional delay should closely match analytic shift."""
        sampling_rate = float(self.sampling_rate)
        n_samples = 2048
        time = np.arange(n_samples) / sampling_rate
        tone_freq = 50e3
        signal = np.exp(1j * 2 * np.pi * tone_freq * time)

        delay_samples = 0.37
        delayed = apply_fractional_delay(signal, delay_samples)

        expected = np.exp(
            1j * 2 * np.pi * tone_freq * (time - delay_samples / sampling_rate)
        )

        error = np.mean(np.abs(delayed - expected))
        self.assertLess(error, 5e-3)

    def test_channel_fractional_delay(self):
        """Channel simulator should apply fractional delays accurately."""
        channel_sim = ChannelSimulator(self.sampling_rate)
        base_signal = np.exp(1j * np.linspace(0, 2 * np.pi, 4000, endpoint=False))

        delay_ps = Picoseconds(275.0)
        channel_model = ChannelModel(
            delay_spread=Picoseconds(0.0),
            path_delays=[delay_ps],
            path_gains=[1.0],
            doppler_shift=Hertz(0.0),
            temperature=25.0,
            humidity=50.0,
        )

        output = channel_sim.apply_channel(base_signal, channel_model, self.tx_frequency)

        fractional_delay = float(delay_ps) * float(self.sampling_rate) * 1e-12
        expected = apply_fractional_delay(base_signal, fractional_delay)

        distance_m = PhysicalConstants.ps_to_meters(float(delay_ps))
        path_loss_db = channel_sim._calculate_free_space_path_loss(
            distance_m, float(self.tx_frequency)
        )
        gain = 10 ** (-path_loss_db / 20)
        expected *= gain

        self.assertLess(np.mean(np.abs(output - expected)), 1e-3)

    def test_beat_note_generation(self):
        """Test beat note generation."""
        # Create oscillators
        tx_oscillator = Oscillator(
            Oscillator.create_ideal_oscillator(self.tx_frequency)
        )
        rx_oscillator = Oscillator(
            Oscillator.create_ideal_oscillator(self.rx_frequency)
        )

        # Generate signals
        _, tx_signal = tx_oscillator.generate_signal(
            duration=self.duration,
            sampling_rate=self.sampling_rate,
            frequency_offset=Hertz(0.0),
            phase_noise_enabled=False,
        )

        _, rx_signal = rx_oscillator.generate_signal(
            duration=self.duration,
            sampling_rate=self.sampling_rate,
            frequency_offset=self.true_delta_f,
            phase_noise_enabled=False,
        )

        # Create beat note processor
        processor = BeatNoteProcessor(self.sampling_rate)

        # Generate beat note
        timestamp = Timestamp.from_ps(0.0)
        beat_note = processor.generate_beat_note(
            tx_signal=tx_signal,
            rx_signal=rx_signal,
            tx_frequency=self.tx_frequency,
            rx_frequency=self.rx_frequency,
            duration=self.duration,
            timestamp=timestamp,
            add_noise=False,
            snr_db=40.0,
        )

        # Check beat note properties
        self.assertIsNotNone(beat_note)
        self.assertEqual(len(beat_note.waveform), len(tx_signal))
        self.assertEqual(beat_note.sampling_rate, self.sampling_rate)

        # Check beat frequency
        expected_beat_freq = (
            abs(self.rx_frequency - self.tx_frequency) + self.true_delta_f
        )
        measured_beat_freq = beat_note.get_beat_frequency()
        self.assertAlmostEqual(measured_beat_freq, expected_beat_freq, places=0)

    def test_beat_frequency_extraction(self):
        """Test beat frequency extraction."""
        # Create simple beat note
        tx_oscillator = Oscillator(
            Oscillator.create_ideal_oscillator(self.tx_frequency)
        )
        rx_oscillator = Oscillator(
            Oscillator.create_ideal_oscillator(self.rx_frequency)
        )

        _, tx_signal = tx_oscillator.generate_signal(
            duration=self.duration, sampling_rate=self.sampling_rate
        )

        _, rx_signal = rx_oscillator.generate_signal(
            duration=self.duration,
            sampling_rate=self.sampling_rate,
            frequency_offset=self.true_delta_f,
        )

        processor = BeatNoteProcessor(self.sampling_rate)
        timestamp = Timestamp.from_ps(0.0)
        beat_note = processor.generate_beat_note(
            tx_signal=tx_signal,
            rx_signal=rx_signal,
            tx_frequency=self.tx_frequency,
            rx_frequency=self.rx_frequency,
            duration=self.duration,
            timestamp=timestamp,
            add_noise=False,
            snr_db=40.0,
        )

        # Extract beat frequency
        beat_freq, freq_uncertainty = processor.extract_beat_frequency(beat_note)

        # Check results
        self.assertIsInstance(beat_freq, float)
        self.assertIsInstance(freq_uncertainty, float)
        self.assertGreater(beat_freq, 0)
        self.assertGreater(freq_uncertainty, 0)

    def test_channel_simulator(self):
        """Test channel simulator."""
        channel_sim = ChannelSimulator(self.sampling_rate)

        # Create test signal
        oscillator = Oscillator(Oscillator.create_ideal_oscillator(self.tx_frequency))
        _, signal = oscillator.generate_signal(
            duration=self.duration, sampling_rate=self.sampling_rate
        )

        # Add thermal noise
        snr_db = 30.0
        noisy_signal = channel_sim.add_thermal_noise(signal, snr_db=snr_db)

        # Check that noise was added
        self.assertEqual(len(noisy_signal), len(signal))
        self.assertFalse(np.array_equal(signal, noisy_signal))

        # Check SNR is approximately correct
        signal_power = np.mean(np.abs(signal) ** 2)
        noise_power = np.mean(np.abs(noisy_signal - signal) ** 2)
        measured_snr = 10 * np.log10(signal_power / noise_power)

        # Allow for some tolerance due to random noise
        self.assertAlmostEqual(measured_snr, snr_db, places=0)

    def test_estimator_factory(self):
        """Test estimator factory."""
        # Test available estimators
        available = EstimatorFactory.get_available_estimators()
        self.assertIsInstance(available, list)
        self.assertGreater(len(available), 0)
        self.assertIn("phase_slope", available)

        # Test estimator creation
        estimator = EstimatorFactory.create_estimator("phase_slope")
        self.assertIsNotNone(estimator)

        # Test invalid estimator
        with self.assertRaises(ValueError):
            EstimatorFactory.create_estimator("invalid_estimator")


class TestBasicConsensus(unittest.TestCase):
    """Test basic consensus functionality."""

    def test_metropolis_consensus_import(self):
        """Test that consensus modules can be imported."""
        try:
            from src.algorithms.consensus import (ConsensusSimulator,
                                                  MetropolisConsensus)

            self.assertTrue(True)  # If we get here, imports worked
        except ImportError as e:
            self.fail(f"Failed to import consensus modules: {e}")

    def test_network_topology_creation(self):
        """Test network topology creation."""
        import numpy as np

        from src.core.types import NetworkTopology

        # Create simple 2-node topology
        adjacency_matrix = np.array([[0, 1], [1, 0]])
        node_ids = [0, 1]
        laplacian = np.array([[1, -1], [-1, 1]])
        spectral_gap = 2.0

        topology = NetworkTopology(
            adjacency_matrix=adjacency_matrix,
            node_ids=node_ids,
            laplacian=laplacian,
            spectral_gap=spectral_gap,
            is_connected=True,
        )

        self.assertIsNotNone(topology)
        self.assertEqual(len(topology.node_ids), 2)
        self.assertTrue(topology.is_connected)
        self.assertEqual(topology.spectral_gap, 2.0)


if __name__ == "__main__":
    # Run tests
    unittest.main(verbosity=2)
