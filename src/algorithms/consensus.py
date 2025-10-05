"""
Consensus algorithms for distributed clock synchronization.
"""

import numpy as np
import time
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field
from abc import ABC, abstractmethod

from ..core.types import (
    EstimationResult, NodeState, Picoseconds, PPB, Hertz, Seconds,
    Timestamp, MeasurementQuality, ConsensusState, NetworkTopology
)
from ..core.constants import PhysicalConstants


@dataclass
class ConsensusMessage:
    """Message exchanged between nodes during consensus."""
    sender_id: int
    receiver_id: int
    iteration: int
    tau_estimate: Picoseconds
    tau_uncertainty: Picoseconds
    delta_f_estimate: Hertz
    delta_f_uncertainty: Hertz
    timestamp: Timestamp
    # Extended fields for adaptive consensus
    spectral_gap: Optional[float] = None
    step_size: Optional[float] = None
    variance_warning: Optional[Dict[str, Any]] = None
    ml_prediction: Optional[Dict[str, Any]] = None
    
    def get_weight(self, method: str = "inverse_variance") -> float:
        """
        Get weight for this message based on uncertainty.
        
        Args:
            method: Weighting method ("inverse_variance", "uniform")
            
        Returns:
            Weight value
        """
        if method == "inverse_variance":
            # Inverse variance weighting
            tau_var = self.tau_uncertainty ** 2
            delta_f_var = self.delta_f_uncertainty ** 2
            
            # Normalize by nominal values to make them comparable
            nominal_tau = 1000.0  # 1 ns
            nominal_delta_f = 100.0  # 100 Hz
            
            tau_weight = 1.0 / (tau_var / nominal_tau**2) if tau_var > 0 else 0.0
            delta_f_weight = 1.0 / (delta_f_var / nominal_delta_f**2) if delta_f_var > 0 else 0.0
            
            # Combined weight (geometric mean)
            return np.sqrt(tau_weight * delta_f_weight)
        
        elif method == "uniform":
            return 1.0
        
        else:
            raise ValueError(f"Unknown weighting method: {method}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'sender_id': self.sender_id,
            'receiver_id': self.receiver_id,
            'iteration': self.iteration,
            'tau_estimate': self.tau_estimate,
            'tau_uncertainty': self.tau_uncertainty,
            'delta_f_estimate': self.delta_f_estimate,
            'delta_f_uncertainty': self.delta_f_uncertainty,
            'timestamp': self._timestamp_to_dict(self.timestamp),
            'spectral_gap': self.spectral_gap,
            'step_size': self.step_size,
            'variance_warning': self.variance_warning,
            'ml_prediction': self.ml_prediction
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ConsensusMessage':
        """Create from dictionary for deserialization."""
        # Handle backward compatibility
        spectral_gap = data.get('spectral_gap')
        step_size = data.get('step_size')
        variance_warning = data.get('variance_warning')
        ml_prediction = data.get('ml_prediction')
        
        return cls(
            sender_id=data['sender_id'],
            receiver_id=data['receiver_id'],
            iteration=data['iteration'],
            tau_estimate=Picoseconds(data['tau_estimate']),
            tau_uncertainty=Picoseconds(data['tau_uncertainty']),
            delta_f_estimate=Hertz(data['delta_f_estimate']),
            delta_f_uncertainty=Hertz(data['delta_f_uncertainty']),
            timestamp=cls._timestamp_from_dict(data['timestamp']),
            spectral_gap=spectral_gap,
            step_size=step_size,
            variance_warning=variance_warning,
            ml_prediction=ml_prediction
        )
    
    @staticmethod
    def _timestamp_to_dict(timestamp: 'Timestamp') -> Dict[str, Any]:
        """Convert Timestamp to dictionary."""
        return {
            'time': timestamp.time,
            'uncertainty': timestamp.uncertainty,
            'quality': timestamp.quality.value
        }
    
    @staticmethod
    def _timestamp_from_dict(data: Dict[str, Any]) -> 'Timestamp':
        """Create Timestamp from dictionary."""
        return Timestamp(
            time=Seconds(data['time']),
            uncertainty=Picoseconds(data['uncertainty']),
            quality=MeasurementQuality(data['quality'])
        )


class ConsensusAlgorithm(ABC):
    """
    Abstract base class for consensus algorithms.
    """
    
    @abstractmethod
    def initialize(self, 
                  node_states: List[NodeState],
                  topology: NetworkTopology) -> ConsensusState:
        """
        Initialize consensus algorithm.
        
        Args:
            node_states: Initial states of all nodes
            topology: Network topology
            
        Returns:
            Initial consensus state
        """
        pass
    
    @abstractmethod
    def step(self, 
            consensus_state: ConsensusState,
            messages: List[ConsensusMessage]) -> ConsensusState:
        """
        Perform one step of consensus algorithm.
        
        Args:
            consensus_state: Current consensus state
            messages: Messages exchanged in this iteration
            
        Returns:
            Updated consensus state
        """
        pass
    
    @abstractmethod
    def has_converged(self, consensus_state: ConsensusState) -> bool:
        """
        Check if consensus has converged.
        
        Args:
            consensus_state: Current consensus state
            
        Returns:
            True if converged
        """
        pass


class MetropolisConsensus(ConsensusAlgorithm):
    """
    Metropolis averaging consensus algorithm.
    
    This algorithm implements the classic Metropolis-Hastings approach
    to distributed averaging, where each node averages its estimate
    with its neighbors using Metropolis weights.
    """
    
    def __init__(self, 
                 convergence_threshold: float = 1e-3,
                 max_iterations: int = 100):
        """
        Initialize Metropolis consensus algorithm.
        
        Args:
            convergence_threshold: Threshold for convergence detection
            max_iterations: Maximum number of iterations
        """
        self.convergence_threshold = convergence_threshold
        self.max_iterations = max_iterations
        self._metropolis_weights = None
    
    def initialize(self, 
                  node_states: List[NodeState],
                  topology: NetworkTopology) -> ConsensusState:
        """
        Initialize Metropolis consensus algorithm.
        
        Args:
            node_states: Initial states of all nodes
            topology: Network topology
            
        Returns:
            Initial consensus state
        """
        # Calculate Metropolis weights
        self._metropolis_weights = self._calculate_metropolis_weights(topology)
        
        # Create initial consensus state
        return ConsensusState(
            iteration=0,
            node_states=node_states.copy(),
            topology=topology,
            weight_matrix=self._metropolis_weights,
            convergence_metric=self._calculate_convergence_metric(node_states),
            timestamp=Timestamp.from_ps(PhysicalConstants.seconds_to_ps(time.time()))
        )
    
    def step(self, 
            consensus_state: ConsensusState,
            messages: List[ConsensusMessage]) -> ConsensusState:
        """
        Perform one step of Metropolis consensus.
        
        Args:
            consensus_state: Current consensus state
            messages: Messages exchanged in this iteration
            
        Returns:
            Updated consensus state
        """
        # Create new node states
        new_node_states = []
        
        for node_state in consensus_state.node_states:
            # Find messages for this node
            node_messages = [m for m in messages if m.receiver_id == node_state.node_id]
            
            if not node_messages:
                # No messages, keep current state
                new_node_states.append(node_state)
                continue
            
            # Calculate weighted average
            tau_sum = node_state.clock_bias
            delta_f_sum = node_state.frequency_offset
            weight_sum = 1.0  # Weight for current state
            
            for message in node_messages:
                weight = self._metropolis_weights[node_state.node_id, message.sender_id]
                tau_sum += weight * message.tau_estimate
                delta_f_sum += weight * message.delta_f_estimate
                weight_sum += weight
            
            # Normalize
            new_tau = tau_sum / weight_sum
            new_delta_f = delta_f_sum / weight_sum
            
            # Update uncertainty (simplified)
            new_tau_uncertainty = node_state.clock_bias_uncertainty * 0.9  # Assume some improvement
            new_delta_f_uncertainty = node_state.frequency_offset_uncertainty * 0.9
            
            # Create new node state
            new_node_state = NodeState(
                node_id=node_state.node_id,
                clock_bias=Picoseconds(new_tau),
                clock_bias_uncertainty=Picoseconds(new_tau_uncertainty),
                frequency_offset=PPB(new_delta_f),
                frequency_offset_uncertainty=PPB(new_delta_f_uncertainty),
                last_update=Timestamp.from_ps(PhysicalConstants.seconds_to_ps(time.time())),
                quality=node_state.quality
            )
            
            new_node_states.append(new_node_state)
        
        # Create new consensus state
        new_consensus_state = ConsensusState(
            iteration=consensus_state.iteration + 1,
            node_states=new_node_states,
            topology=consensus_state.topology,
            weight_matrix=consensus_state.weight_matrix,
            convergence_metric=self._calculate_convergence_metric(new_node_states),
            timestamp=Timestamp.from_ps(PhysicalConstants.seconds_to_ps(time.time()))
        )
        
        return new_consensus_state
    
    def has_converged(self, consensus_state: ConsensusState) -> bool:
        """
        Check if Metropolis consensus has converged.
        
        Args:
            consensus_state: Current consensus state
            
        Returns:
            True if converged
        """
        # Check convergence metric
        if consensus_state.convergence_metric < self.convergence_threshold:
            return True
        
        # Check maximum iterations
        if consensus_state.iteration >= self.max_iterations:
            return True
        
        return False
    
    def _calculate_metropolis_weights(self, topology: NetworkTopology) -> np.ndarray:
        """
        Calculate Metropolis weights for the topology.
        
        Args:
            topology: Network topology
            
        Returns:
            Weight matrix
        """
        n = len(topology.node_ids)
        weights = np.zeros((n, n))
        
        # Get node degrees
        degrees = [topology.get_degree(node_id) for node_id in topology.node_ids]
        
        # Calculate Metropolis weights
        for i, node_i in enumerate(topology.node_ids):
            for j, node_j in enumerate(topology.node_ids):
                if i == j:
                    weights[i, j] = 1.0 - sum([
                        1.0 / max(1, max(degrees[i], degrees[k]))
                        for k in range(n) 
                        if k != i and topology.adjacency_matrix[i, k] > 0
                    ])
                elif topology.adjacency_matrix[i, j] > 0:
                    weights[i, j] = 1.0 / max(1, max(degrees[i], degrees[j]))
                else:
                    weights[i, j] = 0.0
        
        return weights
    
    def _calculate_convergence_metric(self, node_states: List[NodeState]) -> float:
        """
        Calculate convergence metric from node states.
        
        Args:
            node_states: Current node states
            
        Returns:
            Convergence metric (lower is better)
        """
        if len(node_states) < 2:
            return 0.0
        
        # Calculate standard deviation of timing and frequency errors
        timing_errors = [float(state.clock_bias) for state in node_states]
        frequency_errors = [float(state.frequency_offset) for state in node_states]
        
        timing_std = np.std(timing_errors)
        frequency_std = np.std(frequency_errors)
        
        # Normalize and combine
        timing_metric = timing_std / 1000.0  # Normalize by 1 ns
        frequency_metric = frequency_std / 100.0  # Normalize by 100 ppb
        
        return np.sqrt(timing_metric**2 + frequency_metric**2)


class InverseVarianceConsensus(ConsensusAlgorithm):
    """
    Inverse variance weighting consensus algorithm.
    
    This algorithm uses inverse variance weighting to combine estimates,
    giving more weight to measurements with lower uncertainty.
    """
    
    def __init__(self, 
                 convergence_threshold: float = 1e-3,
                 max_iterations: int = 100):
        """
        Initialize inverse variance consensus algorithm.
        
        Args:
            convergence_threshold: Threshold for convergence detection
            max_iterations: Maximum number of iterations
        """
        self.convergence_threshold = convergence_threshold
        self.max_iterations = max_iterations
    
    def initialize(self, 
                  node_states: List[NodeState],
                  topology: NetworkTopology) -> ConsensusState:
        """
        Initialize inverse variance consensus algorithm.
        
        Args:
            node_states: Initial states of all nodes
            topology: Network topology
            
        Returns:
            Initial consensus state
        """
        # Create initial consensus state
        return ConsensusState(
            iteration=0,
            node_states=node_states.copy(),
            topology=topology,
            weight_matrix=self._create_uniform_weights(topology),
            convergence_metric=self._calculate_convergence_metric(node_states),
            timestamp=Timestamp.from_ps(PhysicalConstants.seconds_to_ps(time.time()))
        )
    
    def step(self, 
            consensus_state: ConsensusState,
            messages: List[ConsensusMessage]) -> ConsensusState:
        """
        Perform one step of inverse variance consensus.
        
        Args:
            consensus_state: Current consensus state
            messages: Messages exchanged in this iteration
            
        Returns:
            Updated consensus state
        """
        # Create new node states
        new_node_states = []
        
        for node_state in consensus_state.node_states:
            # Find messages for this node
            node_messages = [m for m in messages if m.receiver_id == node_state.node_id]
            
            if not node_messages:
                # No messages, keep current state
                new_node_states.append(node_state)
                continue
            
            # Calculate inverse variance weighted average
            tau_sum = node_state.clock_bias / (node_state.clock_bias_uncertainty ** 2)
            delta_f_sum = node_state.frequency_offset / (node_state.frequency_offset_uncertainty ** 2)
            weight_sum_tau = 1.0 / (node_state.clock_bias_uncertainty ** 2)
            weight_sum_delta_f = 1.0 / (node_state.frequency_offset_uncertainty ** 2)
            
            for message in node_messages:
                if message.tau_uncertainty > 0:
                    tau_sum += message.tau_estimate / (message.tau_uncertainty ** 2)
                    weight_sum_tau += 1.0 / (message.tau_uncertainty ** 2)
                
                if message.delta_f_uncertainty > 0:
                    delta_f_sum += message.delta_f_estimate / (message.delta_f_uncertainty ** 2)
                    weight_sum_delta_f += 1.0 / (message.delta_f_uncertainty ** 2)
            
            # Normalize
            new_tau = tau_sum / weight_sum_tau if weight_sum_tau > 0 else node_state.clock_bias
            new_delta_f = delta_f_sum / weight_sum_delta_f if weight_sum_delta_f > 0 else node_state.frequency_offset
            
            # Update uncertainty (inverse of sum of inverse variances)
            new_tau_uncertainty = np.sqrt(1.0 / weight_sum_tau) if weight_sum_tau > 0 else node_state.clock_bias_uncertainty
            new_delta_f_uncertainty = np.sqrt(1.0 / weight_sum_delta_f) if weight_sum_delta_f > 0 else node_state.frequency_offset_uncertainty
            
            # Create new node state
            new_node_state = NodeState(
                node_id=node_state.node_id,
                clock_bias=Picoseconds(new_tau),
                clock_bias_uncertainty=Picoseconds(new_tau_uncertainty),
                frequency_offset=PPB(new_delta_f),
                frequency_offset_uncertainty=PPB(new_delta_f_uncertainty),
                last_update=Timestamp.from_ps(PhysicalConstants.seconds_to_ps(time.time())),
                quality=node_state.quality
            )
            
            new_node_states.append(new_node_state)
        
        # Create new consensus state
        new_consensus_state = ConsensusState(
            iteration=consensus_state.iteration + 1,
            node_states=new_node_states,
            topology=consensus_state.topology,
            weight_matrix=consensus_state.weight_matrix,
            convergence_metric=self._calculate_convergence_metric(new_node_states),
            timestamp=Timestamp.from_ps(PhysicalConstants.seconds_to_ps(time.time()))
        )
        
        return new_consensus_state
    
    def has_converged(self, consensus_state: ConsensusState) -> bool:
        """
        Check if inverse variance consensus has converged.
        
        Args:
            consensus_state: Current consensus state
            
        Returns:
            True if converged
        """
        # Check convergence metric
        if consensus_state.convergence_metric < self.convergence_threshold:
            return True
        
        # Check maximum iterations
        if consensus_state.iteration >= self.max_iterations:
            return True
        
        return False
    
    def _create_uniform_weights(self, topology: NetworkTopology) -> np.ndarray:
        """
        Create uniform weight matrix for the topology.
        
        Args:
            topology: Network topology
            
        Returns:
            Weight matrix
        """
        n = len(topology.node_ids)
        weights = np.zeros((n, n))
        
        # Set uniform weights for connected nodes
        for i in range(n):
            for j in range(n):
                if i == j:
                    weights[i, j] = 1.0
                elif topology.adjacency_matrix[i, j] > 0:
                    weights[i, j] = 1.0 / topology.get_degree(topology.node_ids[i])
        
        return weights
    
    def _calculate_convergence_metric(self, node_states: List[NodeState]) -> float:
        """
        Calculate convergence metric from node states.
        
        Args:
            node_states: Current node states
            
        Returns:
            Convergence metric (lower is better)
        """
        if len(node_states) < 2:
            return 0.0
        
        # Calculate weighted standard deviation of timing and frequency errors
        timing_errors = [float(state.clock_bias) for state in node_states]
        frequency_errors = [float(state.frequency_offset) for state in node_states]
        
        timing_weights = [1.0 / (float(state.clock_bias_uncertainty) ** 2) 
                         if state.clock_bias_uncertainty > 0 else 1.0 
                         for state in node_states]
        frequency_weights = [1.0 / (float(state.frequency_offset_uncertainty) ** 2) 
                            if state.frequency_offset_uncertainty > 0 else 1.0 
                            for state in node_states]
        
        # Weighted standard deviation
        timing_mean = np.average(timing_errors, weights=timing_weights)
        frequency_mean = np.average(frequency_errors, weights=frequency_weights)
        
        timing_var = np.average([(x - timing_mean) ** 2 for x in timing_errors], weights=timing_weights)
        frequency_var = np.average([(x - frequency_mean) ** 2 for x in frequency_errors], weights=frequency_weights)
        
        # Normalize and combine
        timing_metric = np.sqrt(timing_var) / 1000.0  # Normalize by 1 ns
        frequency_metric = np.sqrt(frequency_var) / 100.0  # Normalize by 100 ppb
        
        return np.sqrt(timing_metric**2 + frequency_metric**2)


class ConsensusSimulator:
    """
    Simulator for consensus algorithms.
    """
    
    def __init__(self, algorithm: ConsensusAlgorithm):
        """
        Initialize consensus simulator.
        
        Args:
            algorithm: Consensus algorithm to simulate
        """
        self.algorithm = algorithm
        self.consensus_history = []
    
    def run_simulation(self, 
                      initial_states: List[NodeState],
                      topology: NetworkTopology,
                      max_iterations: Optional[int] = None) -> List[ConsensusState]:
        """
        Run consensus simulation.
        
        Args:
            initial_states: Initial node states
            topology: Network topology
            max_iterations: Maximum iterations (overrides algorithm setting)
            
        Returns:
            List of consensus states for each iteration
        """
        # Initialize consensus
        consensus_state = self.algorithm.initialize(initial_states, topology)
        self.consensus_history = [consensus_state]
        
        # Override max iterations if specified
        if max_iterations is not None:
            original_max_iterations = self.algorithm.max_iterations
            self.algorithm.max_iterations = max_iterations
        
        # Run consensus iterations
        while not self.algorithm.has_converged(consensus_state):
            # Generate messages for this iteration
            messages = self._generate_messages(consensus_state)
            
            # Perform consensus step
            consensus_state = self.algorithm.step(consensus_state, messages)
            self.consensus_history.append(consensus_state)
        
        # Restore original max iterations
        if max_iterations is not None:
            self.algorithm.max_iterations = original_max_iterations
        
        return self.consensus_history
    
    def _generate_messages(self, consensus_state: ConsensusState) -> List[ConsensusMessage]:
        """
        Generate messages for consensus iteration.
        
        Args:
            consensus_state: Current consensus state
            
        Returns:
            List of messages to exchange
        """
        messages = []
        
        for i, sender_state in enumerate(consensus_state.node_states):
            # Find neighbors
            neighbors = consensus_state.topology.get_neighbors(sender_state.node_id)
            
            for neighbor_id in neighbors:
                # Find receiver state
                receiver_state = consensus_state.get_state_by_id(neighbor_id)
                if receiver_state is None:
                    continue
                
                # Create message
                message = ConsensusMessage(
                    sender_id=sender_state.node_id,
                    receiver_id=neighbor_id,
                    iteration=consensus_state.iteration,
                    tau_estimate=sender_state.clock_bias,
                    tau_uncertainty=sender_state.clock_bias_uncertainty,
                    delta_f_estimate=Hertz(sender_state.frequency_offset * sender_state.get_frequency_hz(2.4e9) * 1e-9),
                    delta_f_uncertainty=Hertz(sender_state.frequency_offset_uncertainty * sender_state.get_frequency_hz(2.4e9) * 1e-9),
                    timestamp=sender_state.last_update
                )
                
                messages.append(message)
        
        return messages
    
    def get_convergence_metrics(self) -> Dict[str, List[float]]:
        """
        Get convergence metrics from simulation history.
        
        Returns:
            Dictionary of metric names to values over iterations
        """
        metrics = {
            'convergence_metric': [],
            'max_timing_error': [],
            'max_frequency_error': [],
            'avg_timing_error': [],
            'avg_frequency_error': []
        }
        
        for consensus_state in self.consensus_history:
            metrics['convergence_metric'].append(consensus_state.convergence_metric)
            metrics['max_timing_error'].append(float(consensus_state.get_max_timing_error()))
            metrics['max_frequency_error'].append(float(consensus_state.get_max_frequency_error()))
            
            # Calculate average errors
            timing_errors = [float(state.clock_bias) for state in consensus_state.node_states]
            frequency_errors = [float(state.frequency_offset) for state in consensus_state.node_states]
            
            metrics['avg_timing_error'].append(np.mean(np.abs(timing_errors)))
            metrics['avg_frequency_error'].append(np.mean(np.abs(frequency_errors)))
        
        return metrics