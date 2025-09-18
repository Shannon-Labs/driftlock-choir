#!/usr/bin/env python3
"""
Fire-and-forget Monte Carlo runner for DriftLock simulations.

This script provides a convenient interface to run large-scale Monte Carlo
simulations with different configurations and parameter sweeps.
"""

import argparse
import yaml
import json
import os
import sys
import time
import multiprocessing as mp
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from pathlib import Path

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'sim'))

# Import simulation modules
from phase1 import Phase1Simulator, Phase1Config
from phase2 import Phase2Simulator, Phase2Config
from phase3 import Phase3Simulator, Phase3Config


@dataclass
class MCRunConfig:
    """Configuration for Monte Carlo run."""
    simulation_type: str        # 'phase1', 'phase2', 'phase3', 'all'
    config_file: str           # Path to YAML configuration file
    n_workers: int = 1         # Number of parallel workers
    output_dir: str = "results"
    run_id: Optional[str] = None
    verbose: bool = True
    save_intermediate: bool = False


class MonteCarloRunner:
    """Monte Carlo simulation runner with parallel execution."""
    
    def __init__(self, config: MCRunConfig):
        self.config = config
        
        # Load simulation configuration
        self.sim_config = self._load_config(config.config_file)
        
        # Setup output directory
        self.output_dir = Path(config.output_dir)
        if config.run_id:
            self.output_dir = self.output_dir / config.run_id
        else:
            # Generate run ID based on timestamp
            self.output_dir = self.output_dir / f"run_{int(time.time())}"
            
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save configuration for reproducibility
        self._save_run_config()
        
    def _load_config(self, config_file: str) -> Dict[str, Any]:
        """Load YAML configuration file."""
        try:
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
            return config
        except FileNotFoundError:
            print(f"Error: Configuration file '{config_file}' not found.")
            sys.exit(1)
        except yaml.YAMLError as e:
            print(f"Error parsing YAML configuration: {e}")
            sys.exit(1)
            
    def _save_run_config(self):
        """Save run configuration for reproducibility."""
        run_info = {
            'timestamp': time.time(),
            'simulation_type': self.config.simulation_type,
            'config_file': self.config.config_file,
            'n_workers': self.config.n_workers,
            'simulation_config': self.sim_config
        }
        
        with open(self.output_dir / 'run_config.json', 'w') as f:
            json.dump(run_info, f, indent=2, default=str)
            
    def run_simulations(self) -> Dict[str, Any]:
        """Run the specified simulations."""
        print(f"Starting Monte Carlo simulations: {self.config.simulation_type}")
        print(f"Output directory: {self.output_dir}")
        print(f"Using {self.config.n_workers} worker(s)")
        
        results = {}
        
        if self.config.simulation_type in ['phase1', 'all']:
            print("\n=== Running Phase 1 Simulation ===")
            results['phase1'] = self._run_phase1()
            
        if self.config.simulation_type in ['phase2', 'all']:
            print("\n=== Running Phase 2 Simulation ===")
            results['phase2'] = self._run_phase2()
            
        if self.config.simulation_type in ['phase3', 'all']:
            print("\n=== Running Phase 3 Simulation ===")
            results['phase3'] = self._run_phase3()
            
        # Save combined results
        self._save_final_results(results)
        
        print(f"\nSimulations completed! Results saved to: {self.output_dir}")
        return results
        
    def _run_phase1(self) -> Dict[str, Any]:
        """Run Phase 1 simulation."""
        # Create Phase 1 configuration from loaded config
        phase1_config = Phase1Config(
            snr_range_db=self._generate_range(self.sim_config['ranges']['snr_db']),
            bandwidth_range=self._generate_range(self.sim_config['ranges']['bandwidth']),
            duration_range=self._generate_range(self.sim_config['ranges']['duration']),
            carrier_freq=self.sim_config['physical']['carrier_freq'],
            sample_rate=self.sim_config['physical']['sample_rate'],
            n_monte_carlo=self.sim_config['simulation']['n_monte_carlo'],
            save_results=True,
            plot_results=self.sim_config['simulation']['plot_results'],
            results_dir=str(self.output_dir / 'phase1')
        )
        
        simulator = Phase1Simulator(phase1_config)
        
        if self.config.n_workers > 1:
            # Parallel execution for different parameter combinations
            return self._run_phase1_parallel(simulator)
        else:
            return simulator.run_full_simulation()
            
    def _run_phase2(self) -> Dict[str, Any]:
        """Run Phase 2 simulation."""
        phase2_config = Phase2Config(
            n_nodes=self.sim_config['network']['n_nodes'],
            area_size=self.sim_config['network']['area_size'],
            comm_range=self.sim_config['network']['comm_range'],
            max_iterations=self.sim_config['algorithms']['max_iterations'],
            tolerance=self.sim_config['algorithms']['tolerance'],
            n_monte_carlo=self.sim_config['simulation']['n_monte_carlo'],
            carrier_freq=self.sim_config['physical']['carrier_freq'],
            sample_rate=self.sim_config['physical']['sample_rate'],
            snr_db=self.sim_config['physical']['snr_db'],
            save_results=True,
            plot_results=self.sim_config['simulation']['plot_results'],
            results_dir=str(self.output_dir / 'phase2')
        )
        
        simulator = Phase2Simulator(phase2_config)
        return simulator.run_full_simulation()
        
    def _run_phase3(self) -> Dict[str, Any]:
        """Run Phase 3 simulation."""
        mobility_config = self.sim_config.get('mobility', {})
        
        phase3_config = Phase3Config(
            mobility_speeds=[0.0, 1.0, 5.0, 10.0] if not mobility_config.get('enabled') 
                           else [mobility_config.get('speed', 1.0)],
            mobility_models=['random_walk', 'linear'],
            network_sizes=self._generate_range(self.sim_config['ranges']['network_size']),
            area_sizes=[self.sim_config['network']['area_size']],
            allan_dev_values=self._generate_range(self.sim_config['ranges']['allan_dev']),
            drift_rates=self._generate_range(self.sim_config['ranges']['drift_rate']),
            simulation_duration=self.sim_config['simulation']['duration'],
            n_monte_carlo=self.sim_config['simulation']['n_monte_carlo'],
            carrier_freq=self.sim_config['physical']['carrier_freq'],
            sample_rate=self.sim_config['physical']['sample_rate'],
            comm_range=self.sim_config['network']['comm_range'],
            save_results=True,
            plot_results=self.sim_config['simulation']['plot_results'],
            results_dir=str(self.output_dir / 'phase3')
        )
        
        simulator = Phase3Simulator(phase3_config)
        return simulator.run_full_simulation()
        
    def _run_phase1_parallel(self, simulator) -> Dict[str, Any]:
        """Run Phase 1 simulation with parallel processing."""
        # Split Monte Carlo trials across workers
        n_trials_per_worker = simulator.config.n_monte_carlo // self.config.n_workers
        
        if self.config.verbose:
            print(f"Splitting {simulator.config.n_monte_carlo} trials across {self.config.n_workers} workers")
            print(f"Each worker will run {n_trials_per_worker} trials")
            
        # Create worker configurations
        worker_configs = []
        for i in range(self.config.n_workers):
            worker_config = simulator.config
            worker_config.n_monte_carlo = n_trials_per_worker
            worker_config.results_dir = str(self.output_dir / 'phase1' / f'worker_{i}')
            worker_configs.append(worker_config)
            
        # Run workers in parallel
        with mp.Pool(self.config.n_workers) as pool:
            worker_results = pool.map(self._run_phase1_worker, worker_configs)
            
        # Aggregate results
        return self._aggregate_phase1_results(worker_results)
        
    def _run_phase1_worker(self, config: Phase1Config) -> Dict[str, Any]:
        """Run Phase 1 simulation for a single worker."""
        simulator = Phase1Simulator(config)
        return simulator.run_full_simulation()
        
    def _aggregate_phase1_results(self, worker_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate results from multiple Phase 1 workers."""
        # Simple aggregation - average the results
        # In practice, this would be more sophisticated
        
        if not worker_results:
            return {}
            
        # Take structure from first worker
        aggregated = worker_results[0].copy()
        
        # Average numerical results
        for key in ['snr_sweep', 'bandwidth_sweep', 'duration_sweep']:
            if key in aggregated:
                for metric in ['estimator_mse_delay', 'estimator_mse_frequency', 
                              'estimator_bias_delay', 'estimator_bias_frequency']:
                    if metric in aggregated[key]:
                        values = [wr[key][metric] for wr in worker_results if key in wr and metric in wr[key]]
                        if values:
                            aggregated[key][metric] = [sum(v[i] for v in values) / len(values) 
                                                     for i in range(len(values[0]))]
                                                     
        return aggregated
        
    def _generate_range(self, range_spec: List) -> List[float]:
        """Generate parameter range from specification."""
        if len(range_spec) == 3:
            min_val, max_val, step_or_type = range_spec
            
            if step_or_type == "log":
                # Logarithmic spacing
                return list(np.logspace(np.log10(min_val), np.log10(max_val), 10))
            else:
                # Linear spacing
                return list(np.arange(min_val, max_val + step_or_type, step_or_type))
        else:
            # Direct list
            return range_spec
            
    def _save_final_results(self, results: Dict[str, Any]):
        """Save final aggregated results."""
        results_file = self.output_dir / 'final_results.json'
        
        def convert_numpy(obj):
            if hasattr(obj, 'tolist'):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_numpy(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            else:
                return obj
                
        json_results = convert_numpy(results)
        
        with open(results_file, 'w') as f:
            json.dump(json_results, f, indent=2, default=str)
            
        print(f"Final results saved to: {results_file}")
        
    def generate_report(self) -> str:
        """Generate a summary report of the simulation results."""
        report_lines = [
            "=== DriftLock Monte Carlo Simulation Report ===",
            f"Run ID: {self.output_dir.name}",
            f"Timestamp: {time.ctime()}",
            f"Configuration: {self.config.config_file}",
            f"Simulation Type: {self.config.simulation_type}",
            f"Workers: {self.config.n_workers}",
            "",
            "Simulation Summary:",
        ]
        
        # Load final results if available
        results_file = self.output_dir / 'final_results.json'
        if results_file.exists():
            with open(results_file, 'r') as f:
                results = json.load(f)
                
            for phase, phase_results in results.items():
                report_lines.append(f"  {phase.upper()}:")
                # Add phase-specific summary
                if phase == 'phase1':
                    report_lines.append(f"    - SNR range: {len(phase_results.get('snr_sweep', {}).get('snr_values_db', []))} points")
                    report_lines.append(f"    - Bandwidth range: {len(phase_results.get('bandwidth_sweep', {}).get('bandwidth_values', []))} points")
                elif phase == 'phase2':
                    report_lines.append(f"    - Network topologies analyzed")
                    report_lines.append(f"    - Consensus algorithms compared")
                elif phase == 'phase3':
                    report_lines.append(f"    - Mobility effects analyzed")
                    report_lines.append(f"    - Oscillator variations studied")
                    
        report_lines.extend([
            "",
            f"Output directory: {self.output_dir}",
            "=== End Report ==="
        ])
        
        report_text = "\n".join(report_lines)
        
        # Save report
        report_file = self.output_dir / 'simulation_report.txt'
        with open(report_file, 'w') as f:
            f.write(report_text)
            
        return report_text


def main():
    """Main function for command-line interface."""
    parser = argparse.ArgumentParser(description='DriftLock Monte Carlo Simulation Runner')
    
    parser.add_argument('simulation_type', choices=['phase1', 'phase2', 'phase3', 'all'],
                       help='Type of simulation to run')
    
    parser.add_argument('-c', '--config', default='sim/configs/default.yaml',
                       help='Path to YAML configuration file')
    
    parser.add_argument('-w', '--workers', type=int, default=1,
                       help='Number of parallel workers')
    
    parser.add_argument('-o', '--output', default='results',
                       help='Output directory')
    
    parser.add_argument('-r', '--run-id', 
                       help='Run ID for organizing results')
    
    parser.add_argument('-v', '--verbose', action='store_true',
                       help='Verbose output')
    
    parser.add_argument('--save-intermediate', action='store_true',
                       help='Save intermediate results')
    
    args = parser.parse_args()
    
    # Create configuration
    config = MCRunConfig(
        simulation_type=args.simulation_type,
        config_file=args.config,
        n_workers=args.workers,
        output_dir=args.output,
        run_id=args.run_id,
        verbose=args.verbose,
        save_intermediate=args.save_intermediate
    )
    
    # Create and run Monte Carlo runner
    runner = MonteCarloRunner(config)
    
    try:
        start_time = time.time()
        results = runner.run_simulations()
        end_time = time.time()
        
        print(f"\nTotal simulation time: {end_time - start_time:.2f} seconds")
        
        # Generate and display report
        report = runner.generate_report()
        print("\n" + report)
        
    except KeyboardInterrupt:
        print("\nSimulation interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\nError during simulation: {e}")
        sys.exit(1)


if __name__ == "__main__":
    # Import numpy here to avoid import errors in argument parsing
    import numpy as np
    main()
