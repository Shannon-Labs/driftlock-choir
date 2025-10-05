"""
Basic Consensus Demo

This example demonstrates a simple two-node consensus algorithm 
for distributed clock synchronization.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib.pyplot as plt

from src.algorithms.consensus import MetropolisConsensus, ConsensusSimulator
from src.core.types import NodeState, NetworkTopology, Picoseconds, PPB, Timestamp


def create_two_node_topology() -> NetworkTopology:
    """Create a simple two-node topology."""
    adjacency_matrix = np.array([
        [0, 1],
        [1, 0]
    ])
    
    # Calculate Laplacian matrix
    degree_matrix = np.diag(np.sum(adjacency_matrix, axis=1))
    laplacian = degree_matrix - adjacency_matrix
    
    # For two-node topology, spectral gap is 2
    spectral_gap = 2.0
    
    return NetworkTopology(
        adjacency_matrix=adjacency_matrix,
        node_ids=[0, 1],
        laplacian=laplacian,
        spectral_gap=spectral_gap,
        is_connected=True
    )


def create_initial_node_states() -> list:
    """Create initial node states with different clock biases."""
    # Node 0: Reference node (zero bias)
    node0 = NodeState(
        node_id=0,
        clock_bias=Picoseconds(0.0),
        clock_bias_uncertainty=Picoseconds(50.0),
        frequency_offset=PPB(0.0),
        frequency_offset_uncertainty=PPB(5.0),
        last_update=Timestamp.from_ps(0.0),
        quality="good"
    )
    
    # Node 1: Has significant clock bias
    node1 = NodeState(
        node_id=1,
        clock_bias=Picoseconds(1000.0),  # 1 ns offset
        clock_bias_uncertainty=Picoseconds(50.0),
        frequency_offset=PPB(100.0),  # 100 ppb frequency offset
        frequency_offset_uncertainty=PPB(5.0),
        last_update=Timestamp.from_ps(0.0),
        quality="good"
    )
    
    return [node0, node1]


def run_consensus_demo():
    """Run the basic consensus demonstration."""
    print("Basic Two-Node Consensus Demo")
    print("=" * 40)
    
    # Create network topology
    print("Creating two-node topology...")
    topology = create_two_node_topology()
    print(f"Spectral gap: {topology.spectral_gap}")
    
    # Create initial node states
    print("\nCreating initial node states...")
    initial_states = create_initial_node_states()
    
    print("Initial states:")
    for state in initial_states:
        print(f"  Node {state.node_id}: "
              f"bias={state.clock_bias:.1f}ps, "
              f"freq_offset={state.frequency_offset:.1f}ppb")
    
    # Create consensus algorithm
    print("\nCreating Metropolis consensus algorithm...")
    consensus_algorithm = MetropolisConsensus(
        convergence_threshold=1e-3,  # 1 ps convergence threshold
        max_iterations=20
    )
    
    # Create consensus simulator
    simulator = ConsensusSimulator(consensus_algorithm)
    
    # Run consensus simulation
    print("\nRunning consensus simulation...")
    consensus_history = simulator.run_simulation(
        initial_states=initial_states,
        topology=topology,
        max_iterations=20
    )
    
    print(f"Consensus completed in {len(consensus_history)} iterations")
    
    # Print convergence history
    print("\nConvergence History:")
    print("-" * 60)
    print("Iter |  Node 0 Bias  |  Node 1 Bias  | Convergence Metric")
    print("-" * 60)
    
    for i, state in enumerate(consensus_history):
        # Extract node states
        node0_bias = state.node_states[0].clock_bias
        node1_bias = state.node_states[1].clock_bias
        convergence_metric = state.convergence_metric
        
        print(f"{i:4d} | {node0_bias:10.1f}ps | {node1_bias:10.1f}ps | {convergence_metric:.6f}")
    
    # Check final convergence
    final_state = consensus_history[-1]
    final_bias_0 = final_state.node_states[0].clock_bias
    final_bias_1 = final_state.node_states[1].clock_bias
    final_difference = abs(final_bias_0 - final_bias_1)
    
    print(f"\nFinal Results:")
    print(f"Node 0 final bias: {final_bias_0:.1f} ps")
    print(f"Node 1 final bias: {final_bias_1:.1f} ps")
    print(f"Final difference: {final_difference:.1f} ps")
    print(f"Converged: {'✓' if final_difference < 10.0 else '✗'}")
    
    return consensus_history


def plot_consensus_evolution(consensus_history):
    """Plot the evolution of consensus over iterations."""
    try:
        iterations = list(range(len(consensus_history)))
        node0_biases = [state.node_states[0].clock_bias for state in consensus_history]
        node1_biases = [state.node_states[1].clock_bias for state in consensus_history]
        convergence_metrics = [state.convergence_metric for state in consensus_history]
        
        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        # Plot clock biases
        ax1.plot(iterations, node0_biases, 'b-o', label='Node 0', linewidth=2)
        ax1.plot(iterations, node1_biases, 'r-o', label='Node 1', linewidth=2)
        ax1.set_title('Clock Bias Evolution During Consensus')
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Clock Bias (ps)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot convergence metric
        ax2.semilogy(iterations, convergence_metrics, 'g-o', linewidth=2)
        ax2.set_title('Convergence Metric Evolution')
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Convergence Metric (log scale)')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plt.savefig('consensus_demo.png', dpi=300, bbox_inches='tight')
        print("\nPlot saved as 'consensus_demo.png'")
        
        # Show plot if in interactive environment
        plt.show()
        
    except ImportError:
        print("\nMatplotlib not available. Install with: pip install matplotlib")
    except Exception as e:
        print(f"Error generating plots: {e}")


def main():
    """Main demonstration function."""
    # Run consensus demo
    consensus_history = run_consensus_demo()
    
    # Plot results
    plot_consensus_evolution(consensus_history)
    
    print("\nConsensus demo completed!")
    print("\nKey Takeaways:")
    print("- Two nodes with different initial clock biases")
    print("- Metropolis consensus algorithm drives them toward agreement") 
    print("- Convergence typically achieved in 5-10 iterations")
    print("- Forms the foundation for larger network synchronization")


if __name__ == "__main__":
    main()