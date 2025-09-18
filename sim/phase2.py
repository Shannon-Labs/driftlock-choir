"""
Phase 2 Simulation: 50-node network convergence analysis.

This simulation analyzes distributed synchronization convergence in
a 50-node network using consensus algorithms.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, List, Tuple
import json
import os
from dataclasses import dataclass
import time

# Add src to path for imports
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from net.topo import RandomGeometricGraph, TopologyParams
from alg.consensus import (VanillaConsensus, ChebyshevAcceleratedConsensus, 
                          ConsensusParams, DistributedSynchronization)
from alg.ci import ClosedFormEstimator, EstimatorParams
from metrics.biasvar import BiasVarianceAnalyzer, BiasVarianceParams


@dataclass
class Phase2Config:
    """Configuration for Phase 2 simulation."""
    # Network parameters
    n_nodes: int = 50
    area_size: float = 1000.0  # meters
    comm_range: float = 200.0  # meters
    
    # Consensus parameters
    max_iterations: int = 1000
    tolerance: float = 1e-6
    
    # Simulation parameters
    n_monte_carlo: int = 100
    n_topologies: int = 20
    
    # Synchronization parameters
    carrier_freq: float = 2.4e9
    sample_rate: float = 1e6
    snr_db: float = 15.0
    
    # Output configuration
    save_results: bool = True
    plot_results: bool = True
    results_dir: str = "results/phase2"


class Phase2Simulator:
    """50-node network convergence simulator."""
    
    def __init__(self, config: Phase2Config):
        self.config = config
        
        # Create results directory
        if self.config.save_results:
            os.makedirs(self.config.results_dir, exist_ok=True)
            
    def run_full_simulation(self) -> Dict[str, Any]:
        """Run complete Phase 2 simulation suite."""
        print("Starting Phase 2 Simulation: 50-node convergence analysis")
        
        results = {
            'topology_analysis': self.analyze_network_topologies(),
            'consensus_comparison': self.compare_consensus_algorithms(),
            'scalability_analysis': self.analyze_scalability(),
            'robustness_analysis': self.analyze_robustness()
        }
        
        if self.config.save_results:
            self._save_results(results)
            
        if self.config.plot_results:
            self._generate_plots(results)
            
        return results
        
    def analyze_network_topologies(self) -> Dict[str, Any]:
        """Analyze different network topologies."""
        print("Analyzing network topologies...")
        
        topologies = ['rgg', 'clustered']  # Random geometric, clustered
        connectivity_levels = [0.2, 0.3, 0.4, 0.5]
        
        topology_results = {
            'topologies': topologies,
            'connectivity_levels': connectivity_levels,
            'convergence_metrics': {},
            'topology_properties': {}
        }
        
        for topo_type in topologies:
            topology_results['convergence_metrics'][topo_type] = {}
            topology_results['topology_properties'][topo_type] = {}
            
            for connectivity in connectivity_levels:
                print(f"  Processing {topo_type} topology, connectivity={connectivity}")
                
                # Generate multiple topology instances
                convergence_times = []
                final_errors = []
                topology_metrics = []
                
                for trial in range(self.config.n_topologies):
                    # Generate topology
                    topo_params = TopologyParams(
                        n_nodes=self.config.n_nodes,
                        area_size=self.config.area_size,
                        comm_range=self.config.comm_range,
                        connectivity_prob=connectivity,
                        topology_type=topo_type
                    )
                    
                    topo_gen = RandomGeometricGraph(topo_params)
                    topo_results = topo_gen.generate_topology(seed=trial)
                    
                    # Store topology metrics
                    topology_metrics.append(topo_results['topology_metrics'])
                    
                    # Run consensus simulation
                    consensus_results = self._run_consensus_simulation(
                        topo_results['adjacency_matrix']
                    )
                    
                    convergence_times.append(consensus_results['convergence_time'])
                    final_errors.append(consensus_results['final_error'])
                    
                # Store results
                key = f"connectivity_{connectivity}"
                topology_results['convergence_metrics'][topo_type][key] = {
                    'convergence_times': convergence_times,
                    'final_errors': final_errors,
                    'avg_convergence_time': np.mean(convergence_times),
                    'std_convergence_time': np.std(convergence_times),
                    'avg_final_error': np.mean(final_errors),
                    'success_rate': np.sum(np.array(final_errors) < self.config.tolerance) / len(final_errors)
                }
                
                topology_results['topology_properties'][topo_type][key] = {
                    'avg_degree': np.mean([m['avg_degree'] for m in topology_metrics]),
                    'avg_clustering': np.mean([m['clustering_coefficient'] for m in topology_metrics]),
                    'avg_path_length': np.mean([m['avg_path_length'] for m in topology_metrics if np.isfinite(m['avg_path_length'])]),
                    'connectivity_fraction': np.mean([m['is_connected'] for m in topology_metrics])
                }
                
        return topology_results
        
    def compare_consensus_algorithms(self) -> Dict[str, Any]:
        """Compare vanilla and accelerated consensus algorithms."""
        print("Comparing consensus algorithms...")
        
        algorithms = ['vanilla', 'accelerated']
        
        comparison_results = {
            'algorithms': algorithms,
            'performance_metrics': {}
        }
        
        # Generate a reference topology
        topo_params = TopologyParams(
            n_nodes=self.config.n_nodes,
            area_size=self.config.area_size,
            comm_range=self.config.comm_range,
            connectivity_prob=0.3,
            topology_type='rgg'
        )
        
        topo_gen = RandomGeometricGraph(topo_params)
        topo_results = topo_gen.generate_topology(seed=42)
        adjacency_matrix = topo_results['adjacency_matrix']
        
        for algorithm in algorithms:
            print(f"  Testing {algorithm} consensus...")
            
            convergence_histories = []
            convergence_times = []
            final_errors = []
            computation_times = []
            
            for trial in range(self.config.n_monte_carlo):
                # Generate initial estimates
                initial_estimates = self._generate_initial_estimates()
                
                # Run consensus
                start_time = time.time()
                
                if algorithm == 'vanilla':
                    consensus_alg = VanillaConsensus(
                        ConsensusParams(
                            max_iterations=self.config.max_iterations,
                            tolerance=self.config.tolerance
                        )
                    )
                else:  # accelerated
                    consensus_alg = ChebyshevAcceleratedConsensus(
                        ConsensusParams(
                            max_iterations=self.config.max_iterations,
                            tolerance=self.config.tolerance
                        )
                    )
                    
                results = consensus_alg.run_consensus(initial_estimates, adjacency_matrix)
                
                computation_time = time.time() - start_time
                computation_times.append(computation_time)
                
                # Store results
                convergence_histories.append(results['convergence_history'])
                convergence_times.append(results['iterations'])
                final_errors.append(results['convergence_history'][-1] if results['convergence_history'] else np.inf)
                
            comparison_results['performance_metrics'][algorithm] = {
                'avg_convergence_time': np.mean(convergence_times),
                'std_convergence_time': np.std(convergence_times),
                'avg_final_error': np.mean(final_errors),
                'avg_computation_time': np.mean(computation_times),
                'success_rate': np.sum(np.array(final_errors) < self.config.tolerance) / len(final_errors),
                'convergence_histories': convergence_histories[:10]  # Store first 10 for plotting
            }
            
        return comparison_results
        
    def analyze_scalability(self) -> Dict[str, Any]:
        """Analyze scalability with different network sizes."""
        print("Analyzing scalability...")
        
        network_sizes = [10, 20, 30, 40, 50, 75, 100]
        
        scalability_results = {
            'network_sizes': network_sizes,
            'convergence_metrics': {},
            'computational_complexity': {}
        }
        
        for n_nodes in network_sizes:
            print(f"  Testing network size: {n_nodes} nodes")
            
            # Generate topology
            topo_params = TopologyParams(
                n_nodes=n_nodes,
                area_size=self.config.area_size,
                comm_range=self.config.comm_range,
                connectivity_prob=0.3,
                topology_type='rgg'
            )
            
            topo_gen = RandomGeometricGraph(topo_params)
            topo_results = topo_gen.generate_topology()
            
            # Run consensus trials
            convergence_times = []
            computation_times = []
            final_errors = []
            
            n_trials = min(50, self.config.n_monte_carlo)  # Reduce trials for large networks
            
            for trial in range(n_trials):
                initial_estimates = np.random.randn(n_nodes, 2) * 1e-6  # Small random estimates
                
                start_time = time.time()
                
                consensus_alg = ChebyshevAcceleratedConsensus(
                    ConsensusParams(
                        max_iterations=self.config.max_iterations,
                        tolerance=self.config.tolerance
                    )
                )
                
                results = consensus_alg.run_consensus(
                    initial_estimates, 
                    topo_results['adjacency_matrix']
                )
                
                computation_time = time.time() - start_time
                
                convergence_times.append(results['iterations'])
                computation_times.append(computation_time)
                final_errors.append(results['convergence_history'][-1] if results['convergence_history'] else np.inf)
                
            scalability_results['convergence_metrics'][n_nodes] = {
                'avg_convergence_time': np.mean(convergence_times),
                'avg_final_error': np.mean(final_errors),
                'success_rate': np.sum(np.array(final_errors) < self.config.tolerance) / len(final_errors)
            }
            
            scalability_results['computational_complexity'][n_nodes] = {
                'avg_computation_time': np.mean(computation_times),
                'std_computation_time': np.std(computation_times),
                'time_per_iteration': np.mean(computation_times) / np.mean(convergence_times) if np.mean(convergence_times) > 0 else 0
            }
            
        return scalability_results
        
    def analyze_robustness(self) -> Dict[str, Any]:
        """Analyze robustness to node failures and measurement errors."""
        print("Analyzing robustness...")
        
        failure_rates = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25]
        measurement_error_levels = [1e-9, 1e-8, 1e-7, 1e-6, 1e-5]
        
        robustness_results = {
            'node_failure_analysis': {},
            'measurement_error_analysis': {}
        }
        
        # Generate reference topology
        topo_params = TopologyParams(
            n_nodes=self.config.n_nodes,
            area_size=self.config.area_size,
            comm_range=self.config.comm_range,
            connectivity_prob=0.3,
            topology_type='rgg'
        )
        
        topo_gen = RandomGeometricGraph(topo_params)
        topo_results = topo_gen.generate_topology(seed=42)
        
        # Node failure analysis
        for failure_rate in failure_rates:
            print(f"  Testing node failure rate: {failure_rate*100:.1f}%")
            
            convergence_times = []
            final_errors = []
            
            for trial in range(self.config.n_monte_carlo):
                # Create failed node mask
                n_failed = int(failure_rate * self.config.n_nodes)
                failed_nodes = np.random.choice(self.config.n_nodes, n_failed, replace=False)
                
                # Modify adjacency matrix to remove failed nodes
                adj_modified = topo_results['adjacency_matrix'].copy()
                adj_modified[failed_nodes, :] = 0
                adj_modified[:, failed_nodes] = 0
                
                # Generate initial estimates (only for active nodes)
                initial_estimates = self._generate_initial_estimates()
                initial_estimates[failed_nodes, :] = 0  # Failed nodes have no estimates
                
                # Run consensus
                consensus_alg = ChebyshevAcceleratedConsensus(
                    ConsensusParams(
                        max_iterations=self.config.max_iterations,
                        tolerance=self.config.tolerance
                    )
                )
                
                results = consensus_alg.run_consensus(initial_estimates, adj_modified)
                
                convergence_times.append(results['iterations'])
                final_errors.append(results['convergence_history'][-1] if results['convergence_history'] else np.inf)
                
            robustness_results['node_failure_analysis'][failure_rate] = {
                'avg_convergence_time': np.mean(convergence_times),
                'avg_final_error': np.mean(final_errors),
                'success_rate': np.sum(np.array(final_errors) < self.config.tolerance) / len(final_errors)
            }
            
        # Measurement error analysis
        for error_level in measurement_error_levels:
            print(f"  Testing measurement error level: {error_level:.0e}")
            
            convergence_times = []
            final_errors = []
            
            for trial in range(self.config.n_monte_carlo):
                # Generate initial estimates with measurement errors
                initial_estimates = self._generate_initial_estimates()
                measurement_noise = np.random.normal(0, error_level, initial_estimates.shape)
                initial_estimates += measurement_noise
                
                # Run consensus
                consensus_alg = ChebyshevAcceleratedConsensus(
                    ConsensusParams(
                        max_iterations=self.config.max_iterations,
                        tolerance=self.config.tolerance
                    )
                )
                
                results = consensus_alg.run_consensus(
                    initial_estimates, 
                    topo_results['adjacency_matrix']
                )
                
                convergence_times.append(results['iterations'])
                final_errors.append(results['convergence_history'][-1] if results['convergence_history'] else np.inf)
                
            robustness_results['measurement_error_analysis'][error_level] = {
                'avg_convergence_time': np.mean(convergence_times),
                'avg_final_error': np.mean(final_errors),
                'success_rate': np.sum(np.array(final_errors) < self.config.tolerance) / len(final_errors)
            }
            
        return robustness_results
        
    def _run_consensus_simulation(self, adjacency_matrix: np.ndarray) -> Dict[str, Any]:
        """Run a single consensus simulation."""
        initial_estimates = self._generate_initial_estimates()
        
        consensus_alg = ChebyshevAcceleratedConsensus(
            ConsensusParams(
                max_iterations=self.config.max_iterations,
                tolerance=self.config.tolerance
            )
        )
        
        results = consensus_alg.run_consensus(initial_estimates, adjacency_matrix)
        
        return {
            'convergence_time': results['iterations'],
            'final_error': results['convergence_history'][-1] if results['convergence_history'] else np.inf,
            'converged': results['converged']
        }
        
    def _generate_initial_estimates(self) -> np.ndarray:
        """Generate initial parameter estimates for all nodes."""
        # Each node has initial estimates for [delay, frequency_offset]
        # Add realistic variations around true values
        true_delay = 1e-6  # 1 microsecond
        true_freq = 1e3    # 1 kHz
        
        delay_variation = 1e-7  # 100 ns variation
        freq_variation = 100    # 100 Hz variation
        
        delays = true_delay + np.random.normal(0, delay_variation, self.config.n_nodes)
        frequencies = true_freq + np.random.normal(0, freq_variation, self.config.n_nodes)
        
        return np.column_stack([delays, frequencies])
        
    def _save_results(self, results: Dict[str, Any]):
        """Save simulation results to file."""
        results_file = os.path.join(self.config.results_dir, 'phase2_results.json')
        
        # Convert numpy arrays to lists for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, dict):
                return {key: convert_numpy(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            else:
                return obj
                
        json_results = convert_numpy(results)
        
        with open(results_file, 'w') as f:
            json.dump(json_results, f, indent=2)
            
        print(f"Results saved to {results_file}")
        
    def _generate_plots(self, results: Dict[str, Any]):
        """Generate visualization plots."""
        self._plot_topology_analysis(results['topology_analysis'])
        self._plot_consensus_comparison(results['consensus_comparison'])
        self._plot_scalability_analysis(results['scalability_analysis'])
        self._plot_robustness_analysis(results['robustness_analysis'])
        
    def _plot_topology_analysis(self, topo_results: Dict[str, Any]):
        """Plot topology analysis results."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
        
        connectivity_levels = topo_results['connectivity_levels']
        topologies = topo_results['topologies']
        
        # Convergence time comparison
        for topo in topologies:
            conv_times = [topo_results['convergence_metrics'][topo][f'connectivity_{c}']['avg_convergence_time'] 
                         for c in connectivity_levels]
            ax1.plot(connectivity_levels, conv_times, 'o-', label=topo)
            
        ax1.set_xlabel('Connectivity Probability')
        ax1.set_ylabel('Average Convergence Time (iterations)')
        ax1.set_title('Convergence Time vs Connectivity')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Success rate comparison
        for topo in topologies:
            success_rates = [topo_results['convergence_metrics'][topo][f'connectivity_{c}']['success_rate'] 
                           for c in connectivity_levels]
            ax2.plot(connectivity_levels, success_rates, 'o-', label=topo)
            
        ax2.set_xlabel('Connectivity Probability')
        ax2.set_ylabel('Success Rate')
        ax2.set_title('Success Rate vs Connectivity')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Average degree vs convergence time
        for topo in topologies:
            degrees = [topo_results['topology_properties'][topo][f'connectivity_{c}']['avg_degree'] 
                      for c in connectivity_levels]
            conv_times = [topo_results['convergence_metrics'][topo][f'connectivity_{c}']['avg_convergence_time'] 
                         for c in connectivity_levels]
            ax3.plot(degrees, conv_times, 'o-', label=topo)
            
        ax3.set_xlabel('Average Node Degree')
        ax3.set_ylabel('Average Convergence Time (iterations)')
        ax3.set_title('Convergence Time vs Node Degree')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Clustering coefficient vs convergence time
        for topo in topologies:
            clustering = [topo_results['topology_properties'][topo][f'connectivity_{c}']['avg_clustering'] 
                         for c in connectivity_levels]
            conv_times = [topo_results['convergence_metrics'][topo][f'connectivity_{c}']['avg_convergence_time'] 
                         for c in connectivity_levels]
            ax4.plot(clustering, conv_times, 'o-', label=topo)
            
        ax4.set_xlabel('Average Clustering Coefficient')
        ax4.set_ylabel('Average Convergence Time (iterations)')
        ax4.set_title('Convergence Time vs Clustering')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        if self.config.save_results:
            plt.savefig(os.path.join(self.config.results_dir, 'topology_analysis.png'), dpi=300)
        plt.show()
        
    def _plot_consensus_comparison(self, consensus_results: Dict[str, Any]):
        """Plot consensus algorithm comparison."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
        
        algorithms = consensus_results['algorithms']
        
        # Convergence time comparison
        conv_times = [consensus_results['performance_metrics'][alg]['avg_convergence_time'] 
                     for alg in algorithms]
        ax1.bar(algorithms, conv_times)
        ax1.set_ylabel('Average Convergence Time (iterations)')
        ax1.set_title('Convergence Time Comparison')
        ax1.grid(True, alpha=0.3)
        
        # Computation time comparison
        comp_times = [consensus_results['performance_metrics'][alg]['avg_computation_time'] 
                     for alg in algorithms]
        ax2.bar(algorithms, comp_times)
        ax2.set_ylabel('Average Computation Time (s)')
        ax2.set_title('Computation Time Comparison')
        ax2.grid(True, alpha=0.3)
        
        # Success rate comparison
        success_rates = [consensus_results['performance_metrics'][alg]['success_rate'] 
                        for alg in algorithms]
        ax3.bar(algorithms, success_rates)
        ax3.set_ylabel('Success Rate')
        ax3.set_title('Success Rate Comparison')
        ax3.grid(True, alpha=0.3)
        
        # Convergence history examples
        for alg in algorithms:
            histories = consensus_results['performance_metrics'][alg]['convergence_histories']
            if histories:
                # Plot first few convergence histories
                for i, history in enumerate(histories[:3]):
                    ax4.semilogy(history, alpha=0.7, label=f'{alg}_{i}' if i == 0 else "")
                    
        ax4.set_xlabel('Iteration')
        ax4.set_ylabel('Consensus Error')
        ax4.set_title('Convergence Histories')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        if self.config.save_results:
            plt.savefig(os.path.join(self.config.results_dir, 'consensus_comparison.png'), dpi=300)
        plt.show()
        
    def _plot_scalability_analysis(self, scalability_results: Dict[str, Any]):
        """Plot scalability analysis results."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
        
        network_sizes = scalability_results['network_sizes']
        
        # Convergence time vs network size
        conv_times = [scalability_results['convergence_metrics'][n]['avg_convergence_time'] 
                     for n in network_sizes]
        ax1.plot(network_sizes, conv_times, 'bo-')
        ax1.set_xlabel('Network Size (nodes)')
        ax1.set_ylabel('Average Convergence Time (iterations)')
        ax1.set_title('Convergence Time vs Network Size')
        ax1.grid(True, alpha=0.3)
        
        # Computation time vs network size
        comp_times = [scalability_results['computational_complexity'][n]['avg_computation_time'] 
                     for n in network_sizes]
        ax2.loglog(network_sizes, comp_times, 'ro-')
        ax2.set_xlabel('Network Size (nodes)')
        ax2.set_ylabel('Average Computation Time (s)')
        ax2.set_title('Computation Time vs Network Size (log-log)')
        ax2.grid(True, alpha=0.3)
        
        # Success rate vs network size
        success_rates = [scalability_results['convergence_metrics'][n]['success_rate'] 
                        for n in network_sizes]
        ax3.plot(network_sizes, success_rates, 'go-')
        ax3.set_xlabel('Network Size (nodes)')
        ax3.set_ylabel('Success Rate')
        ax3.set_title('Success Rate vs Network Size')
        ax3.grid(True, alpha=0.3)
        
        # Time per iteration vs network size
        time_per_iter = [scalability_results['computational_complexity'][n]['time_per_iteration'] 
                        for n in network_sizes]
        ax4.loglog(network_sizes, time_per_iter, 'mo-')
        ax4.set_xlabel('Network Size (nodes)')
        ax4.set_ylabel('Time per Iteration (s)')
        ax4.set_title('Time per Iteration vs Network Size (log-log)')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        if self.config.save_results:
            plt.savefig(os.path.join(self.config.results_dir, 'scalability_analysis.png'), dpi=300)
        plt.show()
        
    def _plot_robustness_analysis(self, robustness_results: Dict[str, Any]):
        """Plot robustness analysis results."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
        
        # Node failure analysis
        failure_rates = list(robustness_results['node_failure_analysis'].keys())
        
        conv_times_failure = [robustness_results['node_failure_analysis'][fr]['avg_convergence_time'] 
                             for fr in failure_rates]
        ax1.plot([fr*100 for fr in failure_rates], conv_times_failure, 'ro-')
        ax1.set_xlabel('Node Failure Rate (%)')
        ax1.set_ylabel('Average Convergence Time (iterations)')
        ax1.set_title('Convergence Time vs Node Failures')
        ax1.grid(True, alpha=0.3)
        
        success_rates_failure = [robustness_results['node_failure_analysis'][fr]['success_rate'] 
                               for fr in failure_rates]
        ax2.plot([fr*100 for fr in failure_rates], success_rates_failure, 'bo-')
        ax2.set_xlabel('Node Failure Rate (%)')
        ax2.set_ylabel('Success Rate')
        ax2.set_title('Success Rate vs Node Failures')
        ax2.grid(True, alpha=0.3)
        
        # Measurement error analysis
        error_levels = list(robustness_results['measurement_error_analysis'].keys())
        
        conv_times_error = [robustness_results['measurement_error_analysis'][el]['avg_convergence_time'] 
                           for el in error_levels]
        ax3.semilogx(error_levels, conv_times_error, 'go-')
        ax3.set_xlabel('Measurement Error Level')
        ax3.set_ylabel('Average Convergence Time (iterations)')
        ax3.set_title('Convergence Time vs Measurement Error')
        ax3.grid(True, alpha=0.3)
        
        success_rates_error = [robustness_results['measurement_error_analysis'][el]['success_rate'] 
                              for el in error_levels]
        ax4.semilogx(error_levels, success_rates_error, 'mo-')
        ax4.set_xlabel('Measurement Error Level')
        ax4.set_ylabel('Success Rate')
        ax4.set_title('Success Rate vs Measurement Error')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        if self.config.save_results:
            plt.savefig(os.path.join(self.config.results_dir, 'robustness_analysis.png'), dpi=300)
        plt.show()


def main():
    """Main function to run Phase 2 simulation."""
    # Configure simulation parameters
    config = Phase2Config(
        n_nodes=50,
        n_monte_carlo=50,  # Reduced for faster execution
        n_topologies=10,   # Reduced for faster execution
        save_results=True,
        plot_results=True
    )
    
    # Run simulation
    simulator = Phase2Simulator(config)
    results = simulator.run_full_simulation()
    
    print("Phase 2 simulation completed successfully!")
    return results


if __name__ == "__main__":
    main()
