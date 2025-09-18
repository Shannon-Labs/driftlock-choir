"""
Phase 3 Simulation: Mobility, scale, and oscillator sweep analysis.

This simulation explores advanced scenarios including node mobility,
large-scale networks, and various oscillator characteristics.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, List, Tuple
import json
import os
from dataclasses import dataclass

# Add src to path for imports
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from net.topo import RandomGeometricGraph, TopologyParams
from phy.osc import OscillatorParams, AllanDeviationGenerator
from hw.trx import TransceiverNode, TransceiverConfig
from alg.consensus import DistributedSynchronization, ConsensusParams


@dataclass
class Phase3Config:
    """Configuration for Phase 3 simulation."""
    # Mobility parameters
    mobility_speeds: List[float]  # m/s
    mobility_models: List[str]    # 'random_walk', 'linear', 'circular'
    
    # Scale parameters
    network_sizes: List[int]      # Number of nodes
    area_sizes: List[float]       # Network area sizes (m)
    
    # Oscillator parameters
    allan_dev_values: List[float]  # Allan deviation at 1s
    drift_rates: List[float]       # Linear drift rates (Hz/s)
    
    # Simulation parameters
    simulation_duration: float = 10.0  # seconds
    n_monte_carlo: int = 50
    
    # Fixed parameters
    carrier_freq: float = 2.4e9
    sample_rate: float = 1e6
    comm_range: float = 200.0
    
    # Output configuration
    save_results: bool = True
    plot_results: bool = True
    results_dir: str = "results/phase3"


class MobilityModel:
    """Node mobility models."""
    
    @staticmethod
    def random_walk(initial_positions: np.ndarray, speed: float, 
                   duration: float, dt: float = 0.1) -> List[np.ndarray]:
        """Random walk mobility model."""
        n_nodes = initial_positions.shape[0]
        n_steps = int(duration / dt)
        
        positions_history = [initial_positions.copy()]
        current_positions = initial_positions.copy()
        
        for step in range(n_steps):
            # Random direction for each node
            directions = np.random.uniform(0, 2*np.pi, n_nodes)
            
            # Update positions
            dx = speed * dt * np.cos(directions)
            dy = speed * dt * np.sin(directions)
            
            current_positions[:, 0] += dx
            current_positions[:, 1] += dy
            
            # Boundary reflection (simple model)
            current_positions[:, 0] = np.clip(current_positions[:, 0], 0, 1000)
            current_positions[:, 1] = np.clip(current_positions[:, 1], 0, 1000)
            
            positions_history.append(current_positions.copy())
            
        return positions_history
        
    @staticmethod
    def linear_mobility(initial_positions: np.ndarray, speed: float,
                       duration: float, dt: float = 0.1) -> List[np.ndarray]:
        """Linear mobility model."""
        n_nodes = initial_positions.shape[0]
        n_steps = int(duration / dt)
        
        # Random initial directions for each node
        directions = np.random.uniform(0, 2*np.pi, n_nodes)
        
        positions_history = [initial_positions.copy()]
        
        for step in range(1, n_steps + 1):
            # Linear movement
            dx = speed * dt * step * np.cos(directions)
            dy = speed * dt * step * np.sin(directions)
            
            new_positions = initial_positions.copy()
            new_positions[:, 0] += dx
            new_positions[:, 1] += dy
            
            # Boundary wrapping
            new_positions[:, 0] = new_positions[:, 0] % 1000
            new_positions[:, 1] = new_positions[:, 1] % 1000
            
            positions_history.append(new_positions)
            
        return positions_history


class Phase3Simulator:
    """Advanced scenario simulator for Phase 3."""
    
    def __init__(self, config: Phase3Config):
        self.config = config
        
        # Create results directory
        if self.config.save_results:
            os.makedirs(self.config.results_dir, exist_ok=True)
            
    def run_full_simulation(self) -> Dict[str, Any]:
        """Run complete Phase 3 simulation suite."""
        print("Starting Phase 3 Simulation: Mobility, scale, and oscillator analysis")
        
        results = {
            'mobility_analysis': self.analyze_mobility_effects(),
            'scalability_analysis': self.analyze_large_scale_networks(),
            'oscillator_analysis': self.analyze_oscillator_variations(),
            'combined_effects': self.analyze_combined_effects()
        }
        
        if self.config.save_results:
            self._save_results(results)
            
        if self.config.plot_results:
            self._generate_plots(results)
            
        return results
        
    def analyze_mobility_effects(self) -> Dict[str, Any]:
        """Analyze effects of node mobility on synchronization."""
        print("Analyzing mobility effects...")
        
        mobility_results = {
            'mobility_speeds': self.config.mobility_speeds,
            'mobility_models': self.config.mobility_models,
            'performance_metrics': {}
        }
        
        # Fixed network parameters
        n_nodes = 30
        area_size = 1000.0
        
        for model in self.config.mobility_models:
            mobility_results['performance_metrics'][model] = {}
            
            for speed in self.config.mobility_speeds:
                print(f"  Processing {model} mobility at {speed} m/s")
                
                sync_errors = []
                convergence_times = []
                
                for trial in range(self.config.n_monte_carlo):
                    # Generate initial topology
                    topo_params = TopologyParams(
                        n_nodes=n_nodes,
                        area_size=area_size,
                        comm_range=self.config.comm_range,
                        connectivity_prob=0.3
                    )
                    
                    topo_gen = RandomGeometricGraph(topo_params)
                    topo_results = topo_gen.generate_topology(seed=trial)
                    
                    initial_positions = topo_results['node_positions']
                    
                    # Generate mobility trace
                    if model == 'random_walk':
                        positions_history = MobilityModel.random_walk(
                            initial_positions, speed, self.config.simulation_duration
                        )
                    elif model == 'linear':
                        positions_history = MobilityModel.linear_mobility(
                            initial_positions, speed, self.config.simulation_duration
                        )
                    else:
                        # Static case
                        positions_history = [initial_positions] * 100
                        
                    # Simulate synchronization with mobility
                    sync_result = self._simulate_mobile_synchronization(
                        positions_history, n_nodes
                    )
                    
                    sync_errors.append(sync_result['final_sync_error'])
                    convergence_times.append(sync_result['convergence_time'])
                    
                mobility_results['performance_metrics'][model][speed] = {
                    'avg_sync_error': np.mean(sync_errors),
                    'std_sync_error': np.std(sync_errors),
                    'avg_convergence_time': np.mean(convergence_times),
                    'success_rate': np.sum(np.array(sync_errors) < 1e-6) / len(sync_errors)
                }
                
        return mobility_results
        
    def analyze_large_scale_networks(self) -> Dict[str, Any]:
        """Analyze scalability to large networks."""
        print("Analyzing large-scale networks...")
        
        scale_results = {
            'network_sizes': self.config.network_sizes,
            'area_sizes': self.config.area_sizes,
            'performance_metrics': {}
        }
        
        for n_nodes in self.config.network_sizes:
            scale_results['performance_metrics'][n_nodes] = {}
            
            for area_size in self.config.area_sizes:
                print(f"  Processing {n_nodes} nodes in {area_size}m area")
                
                # Adjust communication range based on area to maintain connectivity
                adjusted_comm_range = self.config.comm_range * np.sqrt(area_size / 1000.0)
                
                sync_errors = []
                convergence_times = []
                computation_times = []
                
                n_trials = min(20, self.config.n_monte_carlo)  # Reduce trials for large networks
                
                for trial in range(n_trials):
                    # Generate topology
                    topo_params = TopologyParams(
                        n_nodes=n_nodes,
                        area_size=area_size,
                        comm_range=adjusted_comm_range,
                        connectivity_prob=0.3
                    )
                    
                    topo_gen = RandomGeometricGraph(topo_params)
                    topo_results = topo_gen.generate_topology(seed=trial)
                    
                    # Check connectivity
                    if not topo_results['topology_metrics']['is_connected']:
                        continue
                        
                    # Simulate synchronization
                    sync_result = self._simulate_static_synchronization(
                        topo_results['adjacency_matrix'], n_nodes
                    )
                    
                    sync_errors.append(sync_result['final_sync_error'])
                    convergence_times.append(sync_result['convergence_time'])
                    computation_times.append(sync_result['computation_time'])
                    
                if sync_errors:  # Only store if we have valid results
                    scale_results['performance_metrics'][n_nodes][area_size] = {
                        'avg_sync_error': np.mean(sync_errors),
                        'avg_convergence_time': np.mean(convergence_times),
                        'avg_computation_time': np.mean(computation_times),
                        'success_rate': len(sync_errors) / n_trials
                    }
                    
        return scale_results
        
    def analyze_oscillator_variations(self) -> Dict[str, Any]:
        """Analyze effects of different oscillator characteristics."""
        print("Analyzing oscillator variations...")
        
        osc_results = {
            'allan_dev_values': self.config.allan_dev_values,
            'drift_rates': self.config.drift_rates,
            'performance_metrics': {}
        }
        
        # Fixed network parameters
        n_nodes = 25
        area_size = 1000.0
        
        # Allan deviation sweep
        osc_results['performance_metrics']['allan_dev_sweep'] = {}
        
        for allan_dev in self.config.allan_dev_values:
            print(f"  Processing Allan deviation: {allan_dev:.0e}")
            
            sync_errors = []
            
            for trial in range(self.config.n_monte_carlo):
                # Generate network topology
                topo_params = TopologyParams(
                    n_nodes=n_nodes,
                    area_size=area_size,
                    comm_range=self.config.comm_range,
                    connectivity_prob=0.3
                )
                
                topo_gen = RandomGeometricGraph(topo_params)
                topo_results = topo_gen.generate_topology(seed=trial)
                
                # Simulate with specific oscillator parameters
                sync_result = self._simulate_oscillator_synchronization(
                    topo_results['adjacency_matrix'], n_nodes, allan_dev, 1e-9
                )
                
                sync_errors.append(sync_result['final_sync_error'])
                
            osc_results['performance_metrics']['allan_dev_sweep'][allan_dev] = {
                'avg_sync_error': np.mean(sync_errors),
                'std_sync_error': np.std(sync_errors)
            }
            
        # Drift rate sweep
        osc_results['performance_metrics']['drift_rate_sweep'] = {}
        
        for drift_rate in self.config.drift_rates:
            print(f"  Processing drift rate: {drift_rate:.0e} Hz/s")
            
            sync_errors = []
            
            for trial in range(self.config.n_monte_carlo):
                topo_params = TopologyParams(
                    n_nodes=n_nodes,
                    area_size=area_size,
                    comm_range=self.config.comm_range,
                    connectivity_prob=0.3
                )
                
                topo_gen = RandomGeometricGraph(topo_params)
                topo_results = topo_gen.generate_topology(seed=trial)
                
                sync_result = self._simulate_oscillator_synchronization(
                    topo_results['adjacency_matrix'], n_nodes, 1e-9, drift_rate
                )
                
                sync_errors.append(sync_result['final_sync_error'])
                
            osc_results['performance_metrics']['drift_rate_sweep'][drift_rate] = {
                'avg_sync_error': np.mean(sync_errors),
                'std_sync_error': np.std(sync_errors)
            }
            
        return osc_results
        
    def analyze_combined_effects(self) -> Dict[str, Any]:
        """Analyze combined effects of mobility, scale, and oscillator variations."""
        print("Analyzing combined effects...")
        
        # Test specific combinations that represent realistic scenarios
        scenarios = [
            {
                'name': 'low_mobility_good_osc',
                'n_nodes': 30,
                'mobility_speed': 1.0,  # 1 m/s
                'allan_dev': 1e-10,
                'drift_rate': 1e-9
            },
            {
                'name': 'high_mobility_poor_osc',
                'n_nodes': 30,
                'mobility_speed': 10.0,  # 10 m/s
                'allan_dev': 1e-8,
                'drift_rate': 1e-7
            },
            {
                'name': 'large_static_good_osc',
                'n_nodes': 100,
                'mobility_speed': 0.0,
                'allan_dev': 1e-10,
                'drift_rate': 1e-10
            },
            {
                'name': 'medium_mobile_medium_osc',
                'n_nodes': 50,
                'mobility_speed': 5.0,
                'allan_dev': 1e-9,
                'drift_rate': 1e-8
            }
        ]
        
        combined_results = {
            'scenarios': scenarios,
            'performance_metrics': {}
        }
        
        for scenario in scenarios:
            print(f"  Processing scenario: {scenario['name']}")
            
            sync_errors = []
            convergence_times = []
            
            n_trials = min(30, self.config.n_monte_carlo)
            
            for trial in range(n_trials):
                # Generate topology
                topo_params = TopologyParams(
                    n_nodes=scenario['n_nodes'],
                    area_size=1000.0,
                    comm_range=self.config.comm_range,
                    connectivity_prob=0.3
                )
                
                topo_gen = RandomGeometricGraph(topo_params)
                topo_results = topo_gen.generate_topology(seed=trial)
                
                # Simulate combined scenario
                if scenario['mobility_speed'] > 0:
                    positions_history = MobilityModel.random_walk(
                        topo_results['node_positions'], 
                        scenario['mobility_speed'],
                        self.config.simulation_duration
                    )
                    sync_result = self._simulate_mobile_synchronization(
                        positions_history, scenario['n_nodes'],
                        scenario['allan_dev'], scenario['drift_rate']
                    )
                else:
                    sync_result = self._simulate_oscillator_synchronization(
                        topo_results['adjacency_matrix'], 
                        scenario['n_nodes'],
                        scenario['allan_dev'], 
                        scenario['drift_rate']
                    )
                    
                sync_errors.append(sync_result['final_sync_error'])
                convergence_times.append(sync_result['convergence_time'])
                
            combined_results['performance_metrics'][scenario['name']] = {
                'avg_sync_error': np.mean(sync_errors),
                'std_sync_error': np.std(sync_errors),
                'avg_convergence_time': np.mean(convergence_times),
                'success_rate': np.sum(np.array(sync_errors) < 1e-6) / len(sync_errors)
            }
            
        return combined_results
        
    def _simulate_mobile_synchronization(self, positions_history: List[np.ndarray],
                                       n_nodes: int, allan_dev: float = 1e-9,
                                       drift_rate: float = 1e-9) -> Dict[str, Any]:
        """Simulate synchronization with mobile nodes."""
        # Simplified mobile synchronization simulation
        # In practice, this would involve updating adjacency matrices over time
        
        # Use average connectivity over the mobility trace
        avg_adjacency = np.zeros((n_nodes, n_nodes))
        
        for positions in positions_history:
            # Compute distances
            distances = np.sqrt(np.sum((positions[:, None, :] - positions[None, :, :]) ** 2, axis=2))
            
            # Create adjacency based on communication range
            adjacency = (distances <= self.config.comm_range).astype(float)
            np.fill_diagonal(adjacency, 0)
            
            avg_adjacency += adjacency
            
        avg_adjacency /= len(positions_history)
        avg_adjacency = (avg_adjacency > 0.5).astype(float)  # Threshold for reliable links
        
        # Run consensus on average topology
        return self._simulate_oscillator_synchronization(avg_adjacency, n_nodes, allan_dev, drift_rate)
        
    def _simulate_static_synchronization(self, adjacency_matrix: np.ndarray,
                                       n_nodes: int) -> Dict[str, Any]:
        """Simulate static synchronization."""
        import time
        
        start_time = time.time()
        
        # Generate initial estimates
        initial_estimates = np.random.randn(n_nodes, 2) * 1e-6
        
        # Run consensus
        consensus_params = ConsensusParams(
            max_iterations=1000,
            tolerance=1e-8
        )
        
        sync_system = DistributedSynchronization(consensus_params, use_acceleration=True)
        
        # Convert to node estimates format
        node_estimates = {i: (initial_estimates[i, 0], initial_estimates[i, 1]) 
                         for i in range(n_nodes)}
        
        results = sync_system.synchronize_network(node_estimates, adjacency_matrix)
        
        computation_time = time.time() - start_time
        
        # Calculate final synchronization error
        final_estimates = np.array([[results['node_sync_params'][i]['delay'],
                                   results['node_sync_params'][i]['frequency']] 
                                  for i in range(n_nodes)])
        
        sync_error = np.max(np.std(final_estimates, axis=0))
        
        return {
            'final_sync_error': sync_error,
            'convergence_time': results['convergence_metrics']['iterations'],
            'computation_time': computation_time
        }
        
    def _simulate_oscillator_synchronization(self, adjacency_matrix: np.ndarray,
                                           n_nodes: int, allan_dev: float,
                                           drift_rate: float) -> Dict[str, Any]:
        """Simulate synchronization with specific oscillator parameters."""
        # Generate initial estimates with oscillator-dependent variations
        base_delay = 1e-6
        base_freq = 1e3
        
        # Oscillator-dependent variations
        delay_variation = allan_dev * 1e3  # Scale Allan deviation for delay variation
        freq_variation = drift_rate * 1e6   # Scale drift rate for frequency variation
        
        delays = base_delay + np.random.normal(0, delay_variation, n_nodes)
        frequencies = base_freq + np.random.normal(0, freq_variation, n_nodes)
        
        initial_estimates = np.column_stack([delays, frequencies])
        
        # Run consensus
        consensus_params = ConsensusParams(
            max_iterations=1000,
            tolerance=1e-8
        )
        
        sync_system = DistributedSynchronization(consensus_params, use_acceleration=True)
        
        node_estimates = {i: (initial_estimates[i, 0], initial_estimates[i, 1]) 
                         for i in range(n_nodes)}
        
        results = sync_system.synchronize_network(node_estimates, adjacency_matrix)
        
        # Calculate synchronization error
        final_estimates = np.array([[results['node_sync_params'][i]['delay'],
                                   results['node_sync_params'][i]['frequency']] 
                                  for i in range(n_nodes)])
        
        sync_error = np.max(np.std(final_estimates, axis=0))
        
        return {
            'final_sync_error': sync_error,
            'convergence_time': results['convergence_metrics']['iterations']
        }
        
    def _save_results(self, results: Dict[str, Any]):
        """Save simulation results to file."""
        results_file = os.path.join(self.config.results_dir, 'phase3_results.json')
        
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
        self._plot_mobility_analysis(results['mobility_analysis'])
        self._plot_scalability_analysis(results['scalability_analysis'])
        self._plot_oscillator_analysis(results['oscillator_analysis'])
        self._plot_combined_effects(results['combined_effects'])
        
    def _plot_mobility_analysis(self, mobility_results: Dict[str, Any]):
        """Plot mobility analysis results."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
        
        speeds = mobility_results['mobility_speeds']
        models = mobility_results['mobility_models']
        
        # Synchronization error vs mobility speed
        for model in models:
            sync_errors = [mobility_results['performance_metrics'][model][speed]['avg_sync_error'] 
                          for speed in speeds]
            ax1.semilogy(speeds, sync_errors, 'o-', label=model)
            
        ax1.set_xlabel('Mobility Speed (m/s)')
        ax1.set_ylabel('Average Sync Error')
        ax1.set_title('Sync Error vs Mobility Speed')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Convergence time vs mobility speed
        for model in models:
            conv_times = [mobility_results['performance_metrics'][model][speed]['avg_convergence_time'] 
                         for speed in speeds]
            ax2.plot(speeds, conv_times, 'o-', label=model)
            
        ax2.set_xlabel('Mobility Speed (m/s)')
        ax2.set_ylabel('Average Convergence Time (iterations)')
        ax2.set_title('Convergence Time vs Mobility Speed')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Success rate vs mobility speed
        for model in models:
            success_rates = [mobility_results['performance_metrics'][model][speed]['success_rate'] 
                           for speed in speeds]
            ax3.plot(speeds, success_rates, 'o-', label=model)
            
        ax3.set_xlabel('Mobility Speed (m/s)')
        ax3.set_ylabel('Success Rate')
        ax3.set_title('Success Rate vs Mobility Speed')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Error standard deviation vs mobility speed
        for model in models:
            error_stds = [mobility_results['performance_metrics'][model][speed]['std_sync_error'] 
                         for speed in speeds]
            ax4.semilogy(speeds, error_stds, 'o-', label=model)
            
        ax4.set_xlabel('Mobility Speed (m/s)')
        ax4.set_ylabel('Sync Error Standard Deviation')
        ax4.set_title('Error Variability vs Mobility Speed')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        if self.config.save_results:
            plt.savefig(os.path.join(self.config.results_dir, 'mobility_analysis.png'), dpi=300)
        plt.show()
        
    def _plot_scalability_analysis(self, scale_results: Dict[str, Any]):
        """Plot scalability analysis results."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
        
        network_sizes = scale_results['network_sizes']
        area_sizes = scale_results['area_sizes']
        
        # Sync error vs network size (for different area sizes)
        for area_size in area_sizes:
            sync_errors = []
            valid_sizes = []
            
            for n_nodes in network_sizes:
                if n_nodes in scale_results['performance_metrics'] and \
                   area_size in scale_results['performance_metrics'][n_nodes]:
                    sync_errors.append(scale_results['performance_metrics'][n_nodes][area_size]['avg_sync_error'])
                    valid_sizes.append(n_nodes)
                    
            if sync_errors:
                ax1.semilogy(valid_sizes, sync_errors, 'o-', label=f'Area: {area_size}m')
                
        ax1.set_xlabel('Network Size (nodes)')
        ax1.set_ylabel('Average Sync Error')
        ax1.set_title('Sync Error vs Network Size')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Computation time vs network size
        for area_size in area_sizes:
            comp_times = []
            valid_sizes = []
            
            for n_nodes in network_sizes:
                if n_nodes in scale_results['performance_metrics'] and \
                   area_size in scale_results['performance_metrics'][n_nodes]:
                    comp_times.append(scale_results['performance_metrics'][n_nodes][area_size]['avg_computation_time'])
                    valid_sizes.append(n_nodes)
                    
            if comp_times:
                ax2.loglog(valid_sizes, comp_times, 'o-', label=f'Area: {area_size}m')
                
        ax2.set_xlabel('Network Size (nodes)')
        ax2.set_ylabel('Average Computation Time (s)')
        ax2.set_title('Computation Time vs Network Size')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Create heatmap for remaining plots if data is available
        if network_sizes and area_sizes:
            # Success rate heatmap
            success_grid = np.zeros((len(area_sizes), len(network_sizes)))
            
            for i, area_size in enumerate(area_sizes):
                for j, n_nodes in enumerate(network_sizes):
                    if n_nodes in scale_results['performance_metrics'] and \
                       area_size in scale_results['performance_metrics'][n_nodes]:
                        success_grid[i, j] = scale_results['performance_metrics'][n_nodes][area_size]['success_rate']
                    else:
                        success_grid[i, j] = 0
                        
            im = ax3.imshow(success_grid, aspect='auto', cmap='viridis')
            ax3.set_xlabel('Network Size Index')
            ax3.set_ylabel('Area Size Index')
            ax3.set_title('Success Rate Heatmap')
            plt.colorbar(im, ax=ax3)
            
        plt.tight_layout()
        if self.config.save_results:
            plt.savefig(os.path.join(self.config.results_dir, 'scalability_analysis.png'), dpi=300)
        plt.show()
        
    def _plot_oscillator_analysis(self, osc_results: Dict[str, Any]):
        """Plot oscillator analysis results."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
        
        # Allan deviation sweep
        allan_devs = osc_results['allan_dev_values']
        allan_errors = [osc_results['performance_metrics']['allan_dev_sweep'][ad]['avg_sync_error'] 
                       for ad in allan_devs]
        
        ax1.loglog(allan_devs, allan_errors, 'bo-')
        ax1.set_xlabel('Allan Deviation at 1s')
        ax1.set_ylabel('Average Sync Error')
        ax1.set_title('Sync Error vs Allan Deviation')
        ax1.grid(True, alpha=0.3)
        
        # Drift rate sweep
        drift_rates = osc_results['drift_rates']
        drift_errors = [osc_results['performance_metrics']['drift_rate_sweep'][dr]['avg_sync_error'] 
                       for dr in drift_rates]
        
        ax2.loglog(drift_rates, drift_errors, 'ro-')
        ax2.set_xlabel('Drift Rate (Hz/s)')
        ax2.set_ylabel('Average Sync Error')
        ax2.set_title('Sync Error vs Drift Rate')
        ax2.grid(True, alpha=0.3)
        
        # Error standard deviations
        allan_stds = [osc_results['performance_metrics']['allan_dev_sweep'][ad]['std_sync_error'] 
                     for ad in allan_devs]
        
        ax3.loglog(allan_devs, allan_stds, 'go-')
        ax3.set_xlabel('Allan Deviation at 1s')
        ax3.set_ylabel('Sync Error Standard Deviation')
        ax3.set_title('Error Variability vs Allan Deviation')
        ax3.grid(True, alpha=0.3)
        
        drift_stds = [osc_results['performance_metrics']['drift_rate_sweep'][dr]['std_sync_error'] 
                     for dr in drift_rates]
        
        ax4.loglog(drift_rates, drift_stds, 'mo-')
        ax4.set_xlabel('Drift Rate (Hz/s)')
        ax4.set_ylabel('Sync Error Standard Deviation')
        ax4.set_title('Error Variability vs Drift Rate')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        if self.config.save_results:
            plt.savefig(os.path.join(self.config.results_dir, 'oscillator_analysis.png'), dpi=300)
        plt.show()
        
    def _plot_combined_effects(self, combined_results: Dict[str, Any]):
        """Plot combined effects analysis."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
        
        scenarios = [s['name'] for s in combined_results['scenarios']]
        
        # Sync error comparison
        sync_errors = [combined_results['performance_metrics'][s]['avg_sync_error'] for s in scenarios]
        ax1.bar(scenarios, sync_errors)
        ax1.set_ylabel('Average Sync Error')
        ax1.set_title('Sync Error by Scenario')
        ax1.tick_params(axis='x', rotation=45)
        
        # Convergence time comparison
        conv_times = [combined_results['performance_metrics'][s]['avg_convergence_time'] for s in scenarios]
        ax2.bar(scenarios, conv_times)
        ax2.set_ylabel('Average Convergence Time (iterations)')
        ax2.set_title('Convergence Time by Scenario')
        ax2.tick_params(axis='x', rotation=45)
        
        # Success rate comparison
        success_rates = [combined_results['performance_metrics'][s]['success_rate'] for s in scenarios]
        ax3.bar(scenarios, success_rates)
        ax3.set_ylabel('Success Rate')
        ax3.set_title('Success Rate by Scenario')
        ax3.tick_params(axis='x', rotation=45)
        
        # Error standard deviation comparison
        error_stds = [combined_results['performance_metrics'][s]['std_sync_error'] for s in scenarios]
        ax4.bar(scenarios, error_stds)
        ax4.set_ylabel('Sync Error Standard Deviation')
        ax4.set_title('Error Variability by Scenario')
        ax4.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        if self.config.save_results:
            plt.savefig(os.path.join(self.config.results_dir, 'combined_effects.png'), dpi=300)
        plt.show()


def main():
    """Main function to run Phase 3 simulation."""
    # Configure simulation parameters
    config = Phase3Config(
        mobility_speeds=[0.0, 1.0, 2.0, 5.0, 10.0, 20.0],  # m/s
        mobility_models=['random_walk', 'linear'],
        network_sizes=[20, 30, 50, 75, 100],
        area_sizes=[500.0, 1000.0, 2000.0],  # m
        allan_dev_values=[1e-11, 1e-10, 1e-9, 1e-8, 1e-7],
        drift_rates=[1e-10, 1e-9, 1e-8, 1e-7, 1e-6],  # Hz/s
        n_monte_carlo=30,  # Reduced for faster execution
        save_results=True,
        plot_results=True
    )
    
    # Run simulation
    simulator = Phase3Simulator(config)
    results = simulator.run_full_simulation()
    
    print("Phase 3 simulation completed successfully!")
    return results


if __name__ == "__main__":
    main()
