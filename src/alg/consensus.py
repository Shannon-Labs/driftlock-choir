"""
Consensus algorithms: Vanilla and Chebyshev accelerated variants.

This module implements distributed consensus algorithms for synchronization
parameter estimation across multiple nodes in a network.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh


@dataclass
class ConsensusParams:
    """Parameters for consensus algorithms."""
    max_iterations: int = 1000
    tolerance: float = 1e-6
    step_size: float = 0.01
    acceleration_factor: float = 0.9
    network_topology: str = 'random'  # 'random', 'ring', 'complete'


class NetworkTopology:
    """Network topology management for consensus algorithms."""
    
    @staticmethod
    def create_adjacency_matrix(n_nodes: int, topology: str, 
                               connectivity: float = 0.3) -> np.ndarray:
        """Create adjacency matrix for given topology."""
        if topology == 'complete':
            return np.ones((n_nodes, n_nodes)) - np.eye(n_nodes)
        elif topology == 'ring':
            adj = np.zeros((n_nodes, n_nodes))
            for i in range(n_nodes):
                adj[i, (i + 1) % n_nodes] = 1
                adj[i, (i - 1) % n_nodes] = 1
            return adj
        elif topology == 'random':
            adj = np.random.rand(n_nodes, n_nodes) < connectivity
            adj = adj.astype(float)
            np.fill_diagonal(adj, 0)
            # Make symmetric
            adj = (adj + adj.T) / 2
            adj = (adj > 0).astype(float)
            return adj
        else:
            raise ValueError(f"Unknown topology: {topology}")
    
    @staticmethod
    def create_laplacian_matrix(adjacency: np.ndarray) -> np.ndarray:
        """Create Laplacian matrix from adjacency matrix."""
        degree = np.sum(adjacency, axis=1)
        return np.diag(degree) - adjacency


class VanillaConsensus:
    """Standard consensus algorithm for distributed parameter estimation."""
    
    def __init__(self, params: ConsensusParams):
        self.params = params
        
    def run_consensus(self, initial_estimates: np.ndarray, 
                     adjacency_matrix: np.ndarray,
                     measurements: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Run vanilla consensus algorithm.
        
        Args:
            initial_estimates: Initial parameter estimates for each node [n_nodes x n_params]
            adjacency_matrix: Network adjacency matrix [n_nodes x n_nodes]
            measurements: Optional measurement updates [n_nodes x n_params]
            
        Returns:
            Dictionary with consensus results and convergence metrics
        """
        n_nodes, n_params = initial_estimates.shape
        
        # Create mixing matrix (Metropolis-Hastings weights)
        mixing_matrix = self._create_mixing_matrix(adjacency_matrix)
        
        # Initialize consensus variables
        estimates = initial_estimates.copy()
        convergence_history = []
        
        for iteration in range(self.params.max_iterations):
            # Store previous estimates
            prev_estimates = estimates.copy()
            
            # Consensus update
            estimates = mixing_matrix @ estimates
            
            # Optional measurement incorporation
            if measurements is not None:
                estimates += self.params.step_size * measurements
                
            # Check convergence
            consensus_error = np.max(np.std(estimates, axis=0))
            convergence_history.append(consensus_error)
            
            if consensus_error < self.params.tolerance:
                break
                
        return {
            'final_estimates': estimates,
            'consensus_value': np.mean(estimates, axis=0),
            'convergence_history': convergence_history,
            'iterations': iteration + 1,
            'converged': consensus_error < self.params.tolerance
        }
        
    def _create_mixing_matrix(self, adjacency: np.ndarray) -> np.ndarray:
        """Create mixing matrix using Metropolis-Hastings weights."""
        n_nodes = adjacency.shape[0]
        mixing = np.zeros_like(adjacency)
        
        for i in range(n_nodes):
            for j in range(n_nodes):
                if i == j:
                    # Self-weight
                    mixing[i, i] = 1 - np.sum(adjacency[i, :]) * self.params.step_size
                elif adjacency[i, j] > 0:
                    # Neighbor weight (Metropolis-Hastings)
                    degree_i = np.sum(adjacency[i, :])
                    degree_j = np.sum(adjacency[j, :])
                    mixing[i, j] = self.params.step_size / max(degree_i, degree_j)
                    
        return mixing


class ChebyshevAcceleratedConsensus:
    """Chebyshev accelerated consensus algorithm for faster convergence."""
    
    def __init__(self, params: ConsensusParams):
        self.params = params
        
    def run_consensus(self, initial_estimates: np.ndarray, 
                     adjacency_matrix: np.ndarray,
                     measurements: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Run Chebyshev accelerated consensus algorithm.
        
        Args:
            initial_estimates: Initial parameter estimates for each node
            adjacency_matrix: Network adjacency matrix
            measurements: Optional measurement updates
            
        Returns:
            Dictionary with consensus results and convergence metrics
        """
        n_nodes, n_params = initial_estimates.shape
        
        # Create mixing matrix
        mixing_matrix = self._create_optimal_mixing_matrix(adjacency_matrix)
        
        # Get spectral properties for acceleration
        eigenvals = np.linalg.eigvals(mixing_matrix)
        lambda_2 = np.sort(np.abs(eigenvals))[-2]  # Second largest eigenvalue
        
        # Initialize variables
        estimates = initial_estimates.copy()
        prev_estimates = estimates.copy()
        convergence_history = []
        
        # Chebyshev acceleration parameters
        alpha = (1 - lambda_2) / (1 + lambda_2)
        
        for iteration in range(self.params.max_iterations):
            # Store for acceleration
            old_estimates = estimates.copy()
            
            # Standard consensus step
            estimates = mixing_matrix @ estimates
            
            # Optional measurement incorporation
            if measurements is not None:
                estimates += self.params.step_size * measurements
                
            # Chebyshev acceleration
            if iteration > 0:
                estimates = estimates + alpha * (estimates - prev_estimates)
                
            prev_estimates = old_estimates
            
            # Check convergence
            consensus_error = np.max(np.std(estimates, axis=0))
            convergence_history.append(consensus_error)
            
            if consensus_error < self.params.tolerance:
                break
                
        return {
            'final_estimates': estimates,
            'consensus_value': np.mean(estimates, axis=0),
            'convergence_history': convergence_history,
            'iterations': iteration + 1,
            'converged': consensus_error < self.params.tolerance,
            'acceleration_factor': alpha
        }
        
    def _create_optimal_mixing_matrix(self, adjacency: np.ndarray) -> np.ndarray:
        """Create optimal mixing matrix for fastest convergence."""
        # Compute Laplacian
        laplacian = NetworkTopology.create_laplacian_matrix(adjacency)
        
        # Get second smallest eigenvalue
        eigenvals, eigenvecs = eigsh(laplacian, k=2, which='SM')
        lambda_2 = eigenvals[1]  # Algebraic connectivity
        
        # Optimal step size for fastest convergence
        optimal_step = 2 / (eigenvals[-1] + lambda_2)
        
        # Create mixing matrix
        n_nodes = adjacency.shape[0]
        mixing = np.eye(n_nodes) - optimal_step * laplacian
        
        return mixing


class DistributedSynchronization:
    """Complete distributed synchronization framework."""
    
    def __init__(self, params: ConsensusParams, use_acceleration: bool = True):
        self.params = params
        self.consensus_alg = (ChebyshevAcceleratedConsensus(params) 
                            if use_acceleration else VanillaConsensus(params))
        
    def synchronize_network(self, node_estimates: Dict[int, Tuple[float, float]], 
                           network_topology: np.ndarray) -> Dict[str, Any]:
        """
        Perform distributed network synchronization.
        
        Args:
            node_estimates: Dictionary of {node_id: (delay_estimate, freq_estimate)}
            network_topology: Adjacency matrix for network
            
        Returns:
            Synchronization results with consensus values
        """
        # Convert estimates to matrix format
        node_ids = sorted(node_estimates.keys())
        n_nodes = len(node_ids)
        initial_matrix = np.zeros((n_nodes, 2))  # [delay, frequency]
        
        for i, node_id in enumerate(node_ids):
            initial_matrix[i, :] = node_estimates[node_id]
            
        # Run consensus
        results = self.consensus_alg.run_consensus(initial_matrix, network_topology)
        
        # Convert back to node-specific results
        final_sync_params = {}
        for i, node_id in enumerate(node_ids):
            final_sync_params[node_id] = {
                'delay': results['final_estimates'][i, 0],
                'frequency': results['final_estimates'][i, 1]
            }
            
        return {
            'node_sync_params': final_sync_params,
            'global_consensus': {
                'delay': results['consensus_value'][0],
                'frequency': results['consensus_value'][1]
            },
            'convergence_metrics': {
                'iterations': results['iterations'],
                'converged': results['converged'],
                'history': results['convergence_history']
            }
        }
