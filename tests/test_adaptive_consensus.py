"""
Unit tests for the adaptive consensus algorithm.
"""

import unittest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from typing import List

from src.algorithms.adaptive_consensus import AdaptiveConsensus, AdaptiveTrackingResult, AdaptiveConsensusState
from src.algorithms.consensus import ConsensusMessage
from src.core.types import (
    NodeState, NetworkTopology, ConsensusState, Picoseconds, PPB, Hertz, Timestamp,
    MeasurementQuality
)
from src.core.constants import PhysicalConstants


class TestAdaptiveConsensus(unittest.TestCase):
    """Test cases for the AdaptiveConsensus class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create test topology
        n_nodes = 4
        adjacency_matrix = np.array([
            [0, 1, 1, 0],
            [1, 0, 1, 1],
            [1, 1, 0, 1],
            [0, 1, 1, 0]
        ])
        laplacian = np.diag(np.sum(adjacency_matrix, axis=1)) - adjacency_matrix
        
        self.topology = NetworkTopology(
            adjacency_matrix=adjacency_matrix,
            node_ids=list(range(n_nodes)),
            laplacian=laplacian,
            spectral_gap=0.5,  # Example value
            is_connected=True
        )
        
        # Create test node states
        self.node_states = [
            NodeState(
                node_id=i,
                clock_bias=Picoseconds(i * 1000.0),  # Different biases
                clock_bias_uncertainty=Picoseconds(100.0),
                frequency_offset=PPB(i * 10.0),  # Different offsets
                frequency_offset_uncertainty=PPB(1.0),
                last_update=Timestamp.from_ps(0.0),
                quality=MeasurementQuality.GOOD
            )
            for i in range(n_nodes)
        ]
        
        # Create test messages
        self.messages = [
            ConsensusMessage(
                sender_id=0,
                receiver_id=1,
                iteration=0,
                tau_estimate=Picoseconds(1000.0),
                tau_uncertainty=Picoseconds(100.0),
                delta_f_estimate=Hertz(10.0),
                delta_f_uncertainty=Hertz(1.0),
                timestamp=Timestamp.from_ps(0.0)
            ),
            ConsensusMessage(
                sender_id=1,
                receiver_id=0,
                iteration=0,
                tau_estimate=Picoseconds(2000.0),
                tau_uncertainty=Picoseconds(100.0),
                delta_f_estimate=Hertz(20.0),
                delta_f_uncertainty=Hertz(1.0),
                timestamp=Timestamp.from_ps(0.0)
            )
        ]
        
        # Create adaptive consensus instance
        self.adaptive_consensus = AdaptiveConsensus(
            convergence_threshold=1e-3,
            max_iterations=100,
            adaptation_enabled=True
        )
    
    def test_initialization(self):
        """Test AdaptiveConsensus initialization."""
        # Test with adaptation enabled
        consensus = AdaptiveConsensus(
            convergence_threshold=1e-4,
            max_iterations=200,
            adaptation_enabled=True
        )
        
        self.assertEqual(consensus.convergence_threshold, 1e-4)
        self.assertEqual(consensus.max_iterations, 200)
        self.assertTrue(consensus.adaptation_enabled)
        self.assertIsNotNone(consensus.spectral_estimator)
        self.assertIsNotNone(consensus.step_size_tuner)
        self.assertIsNotNone(consensus.variance_safeguard)
        self.assertIsNotNone(consensus.ml_hooks)
        
        # Test with adaptation disabled
        consensus_disabled = AdaptiveConsensus(adaptation_enabled=False)
        self.assertFalse(consensus_disabled.adaptation_enabled)
    
    def test_initialize(self):
        """Test consensus initialization."""
        # Initialize consensus
        initial_state = self.adaptive_consensus.initialize(self.node_states, self.topology)
        
        # Check initial state
        self.assertIsInstance(initial_state, ConsensusState)
        self.assertEqual(initial_state.iteration, 0)
        self.assertEqual(len(initial_state.node_states), len(self.node_states))
        self.assertEqual(initial_state.topology, self.topology)
        self.assertIsNotNone(initial_state.weight_matrix)
        self.assertGreater(initial_state.convergence_metric, 0.0)
        
        # Check adaptive state
        self.assertIsNotNone(self.adaptive_consensus.adaptive_state)
        self.assertIsInstance(self.adaptive_consensus.adaptive_state, AdaptiveConsensusState)
        self.assertEqual(self.adaptive_consensus.adaptive_state.adaptation_enabled, True)
        self.assertEqual(len(self.adaptive_consensus.consensus_history), 1)
    
    def test_step_with_adaptation_enabled(self):
        """Test consensus step with adaptation enabled."""
        # Initialize consensus
        initial_state = self.adaptive_consensus.initialize(self.node_states, self.topology)
        
        # Perform step
        new_state = self.adaptive_consensus.step(initial_state, self.messages)
        
        # Check new state
        self.assertIsInstance(new_state, ConsensusState)
        self.assertEqual(new_state.iteration, 1)
        self.assertEqual(len(new_state.node_states), len(self.node_states))
        self.assertEqual(new_state.topology, self.topology)
        
        # Check adaptive state was updated
        self.assertIsNotNone(self.adaptive_consensus.adaptive_state)
        self.assertEqual(self.adaptive_consensus.adaptive_state.consensus_state, new_state)
        self.assertGreater(len(self.adaptive_consensus.consensus_history), 1)
        
        # Check tracking history
        self.assertGreater(len(self.adaptive_consensus.adaptive_state.tracking_history), 0)
    
    def test_step_with_adaptation_disabled(self):
        """Test consensus step with adaptation disabled."""
        # Create consensus with adaptation disabled
        consensus_disabled = AdaptiveConsensus(adaptation_enabled=False)
        
        # Initialize consensus
        initial_state = consensus_disabled.initialize(self.node_states, self.topology)
        
        # Perform step
        new_state = consensus_disabled.step(initial_state, self.messages)
        
        # Check new state
        self.assertIsInstance(new_state, ConsensusState)
        self.assertEqual(new_state.iteration, 1)
        self.assertEqual(len(new_state.node_states), len(self.node_states))
    
    def test_adapt_parameters(self):
        """Test parameter adaptation."""
        # Initialize consensus
        initial_state = self.adaptive_consensus.initialize(self.node_states, self.topology)
        
        # Get initial step size
        initial_step_size = self.adaptive_consensus.adaptive_state.current_step_size
        
        # Adapt parameters
        self.adaptive_consensus.adapt_parameters(initial_state)
        
        # Check that parameters were adapted
        self.assertIsNotNone(self.adaptive_consensus.adaptive_state.spectral_result)
        self.assertIsNotNone(self.adaptive_consensus.adaptive_state.step_size_result)
        
        # Step size may have changed
        new_step_size = self.adaptive_consensus.adaptive_state.current_step_size
        self.assertIsInstance(new_step_size, float)
        self.assertGreater(new_step_size, 0.0)
        self.assertLessEqual(new_step_size, 1.0)
    
    def test_has_converged(self):
        """Test convergence detection."""
        # Initialize consensus
        initial_state = self.adaptive_consensus.initialize(self.node_states, self.topology)
        
        # Test with non-converged state
        self.assertFalse(self.adaptive_consensus.has_converged(initial_state))
        
        # Test with converged state (low convergence metric)
        converged_state = ConsensusState(
            iteration=1,
            node_states=self.node_states,
            topology=self.topology,
            weight_matrix=initial_state.weight_matrix,
            convergence_metric=1e-4,  # Below threshold
            timestamp=Timestamp.from_ps(0.0)
        )
        self.assertTrue(self.adaptive_consensus.has_converged(converged_state))
        
        # Test with max iterations reached
        max_iter_state = ConsensusState(
            iteration=100,
            node_states=self.node_states,
            topology=self.topology,
            weight_matrix=initial_state.weight_matrix,
            convergence_metric=0.1,  # Above threshold
            timestamp=Timestamp.from_ps(0.0)
        )
        self.assertTrue(self.adaptive_consensus.has_converged(max_iter_state))
    
    def test_variance_safeguards(self):
        """Test variance safeguard application."""
        # Initialize consensus
        initial_state = self.adaptive_consensus.initialize(self.node_states, self.topology)
        
        # Create node states with high variance
        high_variance_states = [
            NodeState(
                node_id=i,
                clock_bias=Picoseconds(i * 10000.0),  # High variance
                clock_bias_uncertainty=Picoseconds(10000.0),  # High uncertainty
                frequency_offset=PPB(i * 100.0),  # High variance
                frequency_offset_uncertainty=PPB(100.0),  # High uncertainty
                last_update=Timestamp.from_ps(0.0),
                quality=MeasurementQuality.POOR
            )
            for i in range(4)
        ]
        
        # Create state with high variance
        high_variance_state = ConsensusState(
            iteration=1,
            node_states=high_variance_states,
            topology=self.topology,
            weight_matrix=initial_state.weight_matrix,
            convergence_metric=0.1,
            timestamp=Timestamp.from_ps(0.0)
        )
        
        # Apply variance safeguards
        safeguarded_state = self.adaptive_consensus._apply_variance_safeguards(high_variance_state)
        
        # Check that variance monitoring was performed
        self.assertIsNotNone(self.adaptive_consensus.adaptive_state.variance_result)
        
        # Check that reweighting or saturation was applied if needed
        if self.adaptive_consensus.adaptive_state.variance_result.warning_level in ["warning", "critical"]:
            self.assertTrue(
                self.adaptive_consensus.adaptive_state.reweighting_result is not None or
                self.adaptive_consensus.adaptive_state.saturation_result is not None
            )
    
    def test_ml_predictions(self):
        """Test ML prediction integration."""
        # Initialize consensus
        initial_state = self.adaptive_consensus.initialize(self.node_states, self.topology)
        
        # Mock ML hooks to return predictions
        with patch.object(self.adaptive_consensus.ml_hooks, 'predict_weights') as mock_weights, \
             patch.object(self.adaptive_consensus.ml_hooks, 'predict_step_size') as mock_step_size, \
             patch.object(self.adaptive_consensus.ml_hooks, 'predict_stability') as mock_stability:
            
            # Set up mock return values
            mock_weights.return_value = np.eye(4)  # Identity matrix
            mock_step_size.return_value = 0.2
            mock_stability.return_value = 0.8
            
            # Try ML predictions
            self.adaptive_consensus._try_ml_predictions(initial_state)
            
            # Check that predictions were attempted
            mock_weights.assert_called_once()
            mock_step_size.assert_called_once()
            mock_stability.assert_called_once()
            
            # Check that ML predictions were recorded
            self.assertTrue(self.adaptive_consensus.adaptive_state.ml_predictions_used.get('weights', False))
            self.assertTrue(self.adaptive_consensus.adaptive_state.ml_predictions_used.get('step_size', False))
            self.assertTrue(self.adaptive_consensus.adaptive_state.ml_predictions_used.get('stability', False))
    
    def test_adaptive_parameter_tracking(self):
        """Test adaptive parameter tracking."""
        # Initialize consensus
        initial_state = self.adaptive_consensus.initialize(self.node_states, self.topology)
        
        # Perform step
        new_state = self.adaptive_consensus.step(initial_state, self.messages)
        
        # Check tracking history
        self.assertGreater(len(self.adaptive_consensus.adaptive_state.tracking_history), 0)
        
        # Get latest tracking result
        latest_tracking = self.adaptive_consensus.adaptive_state.tracking_history[-1]
        self.assertIsInstance(latest_tracking, AdaptiveTrackingResult)
        self.assertEqual(latest_tracking.iteration, new_state.iteration)
        self.assertIsInstance(latest_tracking.spectral_gap, float)
        self.assertIsInstance(latest_tracking.step_size, float)
        self.assertIsInstance(latest_tracking.variance_level, float)
        self.assertIsInstance(latest_tracking.ml_prediction_used, bool)
        self.assertIsInstance(latest_tracking.adaptation_rationale, str)
    
    def test_get_adaptive_statistics(self):
        """Test getting adaptive statistics."""
        # Initialize consensus
        initial_state = self.adaptive_consensus.initialize(self.node_states, self.topology)
        
        # Get statistics
        stats = self.adaptive_consensus.get_adaptive_statistics()
        
        # Check statistics
        self.assertIsInstance(stats, dict)
        self.assertIn('adaptation_enabled', stats)
        self.assertIn('current_step_size', stats)
        self.assertIn('tracking_history_length', stats)
        self.assertIn('ml_predictions_used', stats)
        self.assertIn('spectral_estimator_stats', stats)
        self.assertIn('step_size_tuner_stats', stats)
        self.assertIn('variance_safeguard_stats', stats)
        self.assertIn('ml_hooks_stats', stats)
    
    def test_reset_adaptive_state(self):
        """Test resetting adaptive state."""
        # Initialize consensus
        initial_state = self.adaptive_consensus.initialize(self.node_states, self.topology)
        
        # Perform step to create history
        new_state = self.adaptive_consensus.step(initial_state, self.messages)
        
        # Verify state exists
        self.assertIsNotNone(self.adaptive_consensus.adaptive_state)
        self.assertGreater(len(self.adaptive_consensus.consensus_history), 0)
        
        # Reset state
        self.adaptive_consensus.reset_adaptive_state()
        
        # Verify state was reset
        self.assertIsNone(self.adaptive_consensus.adaptive_state)
        self.assertEqual(len(self.adaptive_consensus.consensus_history), 0)
    
    def test_backward_compatibility(self):
        """Test backward compatibility with existing consensus algorithms."""
        # Create consensus with adaptation disabled
        consensus_disabled = AdaptiveConsensus(adaptation_enabled=False)
        
        # Initialize consensus
        initial_state = consensus_disabled.initialize(self.node_states, self.topology)
        
        # Perform multiple steps
        current_state = initial_state
        for _ in range(5):
            current_state = consensus_disabled.step(current_state, self.messages)
        
        # Check that consensus progresses normally
        self.assertGreater(current_state.iteration, 0)
        self.assertEqual(len(current_state.node_states), len(self.node_states))
        
        # Check convergence detection
        converged = consensus_disabled.has_converged(current_state)
        self.assertIsInstance(converged, bool)
    
    def test_fallback_mechanism(self):
        """Test fallback mechanism when adaptive components fail."""
        # Initialize consensus
        initial_state = self.adaptive_consensus.initialize(self.node_states, self.topology)
        
        # Mock step method to raise an exception
        with patch.object(self.adaptive_consensus, '_perform_adaptive_step') as mock_step:
            mock_step.side_effect = Exception("Adaptive step failed")
            
            # Perform step (should fall back to traditional consensus)
            new_state = self.adaptive_consensus.step(initial_state, self.messages)
            
            # Check that fallback was used
            self.assertIsInstance(new_state, ConsensusState)
            self.assertEqual(new_state.iteration, 1)
    
    def test_convergence_with_various_conditions(self):
        """Test convergence under various conditions."""
        # Test with different initial states
        for i in range(3):
            # Create different initial states
            node_states = [
                NodeState(
                    node_id=j,
                    clock_bias=Picoseconds((i + j) * 1000.0),
                    clock_bias_uncertainty=Picoseconds(100.0),
                    frequency_offset=PPB((i + j) * 10.0),
                    frequency_offset_uncertainty=PPB(1.0),
                    last_update=Timestamp.from_ps(0.0),
                    quality=MeasurementQuality.GOOD
                )
                for j in range(4)
            ]
            
            # Create consensus
            consensus = AdaptiveConsensus(
                convergence_threshold=1e-3,
                max_iterations=50,
                adaptation_enabled=True
            )
            
            # Initialize consensus
            initial_state = consensus.initialize(node_states, self.topology)
            
            # Run consensus until convergence or max iterations
            current_state = initial_state
            for _ in range(50):
                if consensus.has_converged(current_state):
                    break
                
                # Generate messages for this iteration
                messages = []
                for sender_state in current_state.node_states:
                    neighbors = current_state.topology.get_neighbors(sender_state.node_id)
                    for neighbor_id in neighbors:
                        message = ConsensusMessage(
                            sender_id=sender_state.node_id,
                            receiver_id=neighbor_id,
                            iteration=current_state.iteration,
                            tau_estimate=sender_state.clock_bias,
                            tau_uncertainty=sender_state.clock_bias_uncertainty,
                            delta_f_estimate=Hertz(sender_state.frequency_offset),
                            delta_f_uncertainty=Hertz(sender_state.frequency_offset_uncertainty),
                            timestamp=sender_state.last_update
                        )
                        messages.append(message)
                
                # Perform step
                current_state = consensus.step(current_state, messages)
            
            # Check that consensus either converged or reached max iterations
            self.assertTrue(
                consensus.has_converged(current_state) or 
                current_state.iteration >= 50
            )


if __name__ == '__main__':
    unittest.main()