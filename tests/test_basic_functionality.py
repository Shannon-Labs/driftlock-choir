"""
Test suite for the basic beat note processing functionality.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import unittest
import numpy as np

from src.signal_processing.oscillator import Oscillator
from src.signal_processing.beat_note import BeatNoteProcessor
from src.signal_processing.channel import ChannelSimulator
from src.algorithms.estimator import EstimatorFactory
from src.core.types import Hertz, Seconds, Picoseconds, Timestamp


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
            phase_noise_enabled=False
        )
        
        # Check signal properties
        expected_samples = int(self.duration * self.sampling_rate)
        self.assertEqual(len(signal), expected_samples)
        self.assertEqual(len(time), expected_samples)
        
        # Check signal magnitude (should be approximately 1)
        magnitude = np.mean(np.abs(signal))
        self.assertAlmostEqual(magnitude, 1.0, places=1)
    
    def test_beat_note_generation(self):
        """Test beat note generation."""
        # Create oscillators
        tx_oscillator = Oscillator(Oscillator.create_ideal_oscillator(self.tx_frequency))
        rx_oscillator = Oscillator(Oscillator.create_ideal_oscillator(self.rx_frequency))
        
        # Generate signals
        _, tx_signal = tx_oscillator.generate_signal(
            duration=self.duration,
            sampling_rate=self.sampling_rate,
            frequency_offset=Hertz(0.0),
            phase_noise_enabled=False
        )
        
        _, rx_signal = rx_oscillator.generate_signal(
            duration=self.duration,
            sampling_rate=self.sampling_rate,
            frequency_offset=self.true_delta_f,
            phase_noise_enabled=False
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
            snr_db=40.0
        )
        
        # Check beat note properties
        self.assertIsNotNone(beat_note)
        self.assertEqual(len(beat_note.waveform), len(tx_signal))
        self.assertEqual(beat_note.sampling_rate, self.sampling_rate)
        
        # Check beat frequency
        expected_beat_freq = abs(self.rx_frequency - self.tx_frequency) + self.true_delta_f
        measured_beat_freq = beat_note.get_beat_frequency()
        self.assertAlmostEqual(measured_beat_freq, expected_beat_freq, places=0)
    
    def test_beat_frequency_extraction(self):
        """Test beat frequency extraction."""
        # Create simple beat note
        tx_oscillator = Oscillator(Oscillator.create_ideal_oscillator(self.tx_frequency))
        rx_oscillator = Oscillator(Oscillator.create_ideal_oscillator(self.rx_frequency))
        
        _, tx_signal = tx_oscillator.generate_signal(
            duration=self.duration,
            sampling_rate=self.sampling_rate
        )
        
        _, rx_signal = rx_oscillator.generate_signal(
            duration=self.duration,
            sampling_rate=self.sampling_rate,
            frequency_offset=self.true_delta_f
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
            snr_db=40.0
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
            duration=self.duration,
            sampling_rate=self.sampling_rate
        )
        
        # Add thermal noise
        snr_db = 30.0
        noisy_signal = channel_sim.add_thermal_noise(signal, snr_db=snr_db)
        
        # Check that noise was added
        self.assertEqual(len(noisy_signal), len(signal))
        self.assertFalse(np.array_equal(signal, noisy_signal))
        
        # Check SNR is approximately correct
        signal_power = np.mean(np.abs(signal)**2)
        noise_power = np.mean(np.abs(noisy_signal - signal)**2)
        measured_snr = 10 * np.log10(signal_power / noise_power)
        
        # Allow for some tolerance due to random noise
        self.assertAlmostEqual(measured_snr, snr_db, places=0)
    
    def test_estimator_factory(self):
        """Test estimator factory."""
        # Test available estimators
        available = EstimatorFactory.get_available_estimators()
        self.assertIsInstance(available, list)
        self.assertGreater(len(available), 0)
        self.assertIn('phase_slope', available)
        
        # Test estimator creation
        estimator = EstimatorFactory.create_estimator('phase_slope')
        self.assertIsNotNone(estimator)
        
        # Test invalid estimator
        with self.assertRaises(ValueError):
            EstimatorFactory.create_estimator('invalid_estimator')


class TestBasicConsensus(unittest.TestCase):
    """Test basic consensus functionality."""
    
    def test_metropolis_consensus_import(self):
        """Test that consensus modules can be imported."""
        try:
            from src.algorithms.consensus import MetropolisConsensus, ConsensusSimulator
            self.assertTrue(True)  # If we get here, imports worked
        except ImportError as e:
            self.fail(f"Failed to import consensus modules: {e}")
    
    def test_network_topology_creation(self):
        """Test network topology creation."""
        from src.core.types import NetworkTopology
        import numpy as np
        
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
            is_connected=True
        )
        
        self.assertIsNotNone(topology)
        self.assertEqual(len(topology.node_ids), 2)
        self.assertTrue(topology.is_connected)
        self.assertEqual(topology.spectral_gap, 2.0)


if __name__ == '__main__':
    # Run tests
    unittest.main(verbosity=2)