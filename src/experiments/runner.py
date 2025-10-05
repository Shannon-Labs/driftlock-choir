"""
Experiment runner framework for Driftlock Choir.
"""

import numpy as np
import time
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime, timezone

from ..core.types import (
    ExperimentConfig, ExperimentResult, PerformanceMetrics,
    Timestamp, Seconds, Picoseconds, PPB, MeasurementQuality
)
from ..core.constants import PhysicalConstants


@dataclass
class ExperimentContext:
    """Context for experiment execution."""
    config: ExperimentConfig
    output_dir: str = "results"
    random_seed: Optional[int] = None
    verbose: bool = True
    
    def __post_init__(self):
        if self.random_seed is not None:
            np.random.seed(self.random_seed)


class ExperimentRunner:
    """
    Orchestrates systematic experiments for Driftlock Choir.
    
    This class provides a framework for running experiments, collecting results,
    and managing parameter sweeps and Monte Carlo campaigns.
    """
    
    def __init__(self, context: ExperimentContext):
        """
        Initialize experiment runner.
        
        Args:
            context: Experiment context
        """
        self.context = context
        self.results: List[ExperimentResult] = []
        self.start_time = None
        self.end_time = None
    
    def run_single_experiment(self, 
                             experiment_func,
                             parameters: Dict[str, Any]) -> ExperimentResult:
        """
        Run a single experiment.
        
        Args:
            experiment_func: Function to execute the experiment
            parameters: Experiment parameters
            
        Returns:
            Experiment result
        """
        if self.context.verbose:
            print(f"Running experiment: {self.context.config.experiment_id}")
            print(f"Parameters: {parameters}")
        
        self.start_time = time.time()
        
        try:
            # Execute experiment
            result = experiment_func(self.context, parameters)
            
            # Record execution time
            self.end_time = time.time()
            execution_time = self.end_time - self.start_time
            
            if self.context.verbose:
                print(f"Experiment completed in {execution_time:.2f} seconds")
                print(f"Success: {result.success}")
                if not result.success:
                    print(f"Error: {result.error_message}")
            
            self.results.append(result)
            return result
            
        except Exception as e:
            # Create error result
            error_result = ExperimentResult(
                config=self.context.config,
                metrics=PerformanceMetrics(
                    rmse_timing=Picoseconds(float('inf')),
                    rmse_frequency=PPB(float('inf')),
                    convergence_time=Seconds(float('inf')),
                    iterations_to_convergence=-1,
                    final_spectral_gap=0.0,
                    communication_overhead=0,
                    computation_time=Seconds(time.time() - self.start_time)
                ),
                telemetry=[],
                final_state=None,
                success=False,
                error_message=str(e),
                completion_time=Timestamp.from_ps(
                    PhysicalConstants.seconds_to_ps(time.time())
                )
            )
            
            self.results.append(error_result)
            return error_result
    
    def run_parameter_sweep(self,
                           experiment_func,
                           parameter_ranges: Dict[str, List[Any]]) -> List[ExperimentResult]:
        """
        Run systematic parameter sweep.
        
        Args:
            experiment_func: Function to execute the experiment
            parameter_ranges: Dictionary of parameter ranges to sweep
            
        Returns:
            List of experiment results
        """
        if self.context.verbose:
            print(f"Starting parameter sweep for {self.context.config.experiment_id}")
            print(f"Parameter ranges: {parameter_ranges}")
        
        # Generate all parameter combinations
        parameter_combinations = self._generate_parameter_combinations(parameter_ranges)
        
        if self.context.verbose:
            print(f"Total combinations: {len(parameter_combinations)}")
        
        results = []
        for i, params in enumerate(parameter_combinations):
            if self.context.verbose:
                print(f"\n--- Combination {i+1}/{len(parameter_combinations)} ---")
            
            # Update experiment config with parameters
            updated_config = ExperimentConfig(
                experiment_id=f"{self.context.config.experiment_id}_sweep_{i+1}",
                description=f"{self.context.config.description} - Sweep {i+1}",
                parameters={**self.context.config.parameters, **params},
                seed=self.context.config.seed,
                start_time=self.context.config.start_time,
                expected_duration=self.context.config.expected_duration
            )
            
            # Create updated context
            updated_context = ExperimentContext(
                config=updated_config,
                output_dir=self.context.output_dir,
                random_seed=self.context.random_seed,
                verbose=self.context.verbose
            )
            
            # Run experiment
            runner = ExperimentRunner(updated_context)
            result = runner.run_single_experiment(experiment_func, params)
            results.append(result)
        
        return results
    
    def run_monte_carlo(self,
                       experiment_func,
                       n_runs: int,
                       parameter_distributions: Dict[str, Any]) -> List[ExperimentResult]:
        """
        Run Monte Carlo campaign.
        
        Args:
            experiment_func: Function to execute the experiment
            n_runs: Number of Monte Carlo runs
            parameter_distributions: Dictionary of parameter distributions
            
        Returns:
            List of experiment results
        """
        if self.context.verbose:
            print(f"Starting Monte Carlo campaign for {self.context.config.experiment_id}")
            print(f"Number of runs: {n_runs}")
            print(f"Parameter distributions: {parameter_distributions}")
        
        results = []
        for i in range(n_runs):
            if self.context.verbose:
                print(f"\n--- Monte Carlo Run {i+1}/{n_runs} ---")
            
            # Sample parameters from distributions
            params = self._sample_parameters(parameter_distributions)
            
            # Update experiment config with run number
            updated_config = ExperimentConfig(
                experiment_id=f"{self.context.config.experiment_id}_mc_{i+1}",
                description=f"{self.context.config.description} - MC Run {i+1}",
                parameters={**self.context.config.parameters, **params},
                seed=self.context.config.seed + i if self.context.config.seed else None,
                start_time=self.context.config.start_time,
                expected_duration=self.context.config.expected_duration
            )
            
            # Create updated context
            updated_context = ExperimentContext(
                config=updated_config,
                output_dir=self.context.output_dir,
                random_seed=self.context.random_seed + i if self.context.random_seed else None,
                verbose=self.context.verbose
            )
            
            # Run experiment
            runner = ExperimentRunner(updated_context)
            result = runner.run_single_experiment(experiment_func, params)
            results.append(result)
        
        return results
    
    def _generate_parameter_combinations(self, 
                                        parameter_ranges: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
        """
        Generate all combinations of parameters for sweep.
        
        Args:
            parameter_ranges: Dictionary of parameter ranges
            
        Returns:
            List of parameter dictionaries
        """
        import itertools
        
        # Get parameter names and value lists
        param_names = list(parameter_ranges.keys())
        param_values = list(parameter_ranges.values())
        
        # Generate all combinations
        combinations = list(itertools.product(*param_values))
        
        # Convert to list of dictionaries
        return [dict(zip(param_names, combo)) for combo in combinations]
    
    def _sample_parameters(self, 
                          parameter_distributions: Dict[str, Any]) -> Dict[str, Any]:
        """
        Sample parameters from distributions.
        
        Args:
            parameter_distributions: Dictionary of parameter distributions
            
        Returns:
            Dictionary of sampled parameters
        """
        params = {}
        
        for param_name, distribution in parameter_distributions.items():
            if isinstance(distribution, dict):
                dist_type = distribution.get('type', 'uniform')
                
                if dist_type == 'uniform':
                    low = distribution.get('low', 0.0)
                    high = distribution.get('high', 1.0)
                    params[param_name] = np.random.uniform(low, high)
                
                elif dist_type == 'normal':
                    mean = distribution.get('mean', 0.0)
                    std = distribution.get('std', 1.0)
                    params[param_name] = np.random.normal(mean, std)
                
                elif dist_type == 'exponential':
                    scale = distribution.get('scale', 1.0)
                    params[param_name] = np.random.exponential(scale)
                
                else:
                    # Default to uniform
                    params[param_name] = np.random.uniform(0.0, 1.0)
            
            else:
                # Assume it's a fixed value
                params[param_name] = distribution
        
        return params
    
    def get_summary_statistics(self, 
                             results: Optional[List[ExperimentResult]] = None) -> Dict[str, Any]:
        """
        Get summary statistics for experiment results.
        
        Args:
            results: List of results (uses self.results if None)
            
        Returns:
            Dictionary of summary statistics
        """
        if results is None:
            results = self.results
        
        if not results:
            return {}
        
        # Filter successful results
        successful_results = [r for r in results if r.success]
        
        if not successful_results:
            return {
                'total_runs': len(results),
                'successful_runs': 0,
                'success_rate': 0.0
            }
        
        # Extract metrics
        timing_errors = [r.metrics.rmse_timing for r in successful_results if r.metrics.rmse_timing != float('inf')]
        frequency_errors = [r.metrics.rmse_frequency for r in successful_results if r.metrics.rmse_frequency != float('inf')]
        convergence_times = [r.metrics.convergence_time for r in successful_results if r.metrics.convergence_time != float('inf')]
        
        summary = {
            'total_runs': len(results),
            'successful_runs': len(successful_results),
            'success_rate': len(successful_results) / len(results),
        }
        
        if timing_errors:
            summary['timing_rmse'] = {
                'mean_ps': np.mean(timing_errors),
                'std_ps': np.std(timing_errors),
                'min_ps': np.min(timing_errors),
                'max_ps': np.max(timing_errors)
            }
        
        if frequency_errors:
            summary['frequency_rmse'] = {
                'mean_ppb': np.mean(frequency_errors),
                'std_ppb': np.std(frequency_errors),
                'min_ppb': np.min(frequency_errors),
                'max_ppb': np.max(frequency_errors)
            }
        
        if convergence_times:
            summary['convergence_time'] = {
                'mean_seconds': np.mean(convergence_times),
                'std_seconds': np.std(convergence_times),
                'min_seconds': np.min(convergence_times),
                'max_seconds': np.max(convergence_times)
            }
        
        return summary
    
    def save_results(self, filename: Optional[str] = None):
        """
        Save experiment results to file.
        
        Args:
            filename: Output filename (auto-generated if None)
        """
        import json
        import os
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{self.context.config.experiment_id}_{timestamp}.json"
        
        filepath = os.path.join(self.context.output_dir, filename)
        os.makedirs(self.context.output_dir, exist_ok=True)
        
        # Convert results to serializable format
        serializable_results = []
        for result in self.results:
            result_dict = {
                'config': {
                    'experiment_id': result.config.experiment_id,
                    'description': result.config.description,
                    'parameters': result.config.parameters,
                    'success': result.success,
                    'error_message': result.error_message
                },
                'metrics': {
                    'rmse_timing_ps': result.metrics.rmse_timing,
                    'rmse_frequency_ppb': result.metrics.rmse_frequency,
                    'convergence_time_seconds': result.metrics.convergence_time,
                    'iterations_to_convergence': result.metrics.iterations_to_convergence,
                    'final_spectral_gap': result.metrics.final_spectral_gap,
                    'communication_overhead_bytes': result.metrics.communication_overhead,
                    'computation_time_seconds': result.metrics.computation_time
                }
            }
            serializable_results.append(result_dict)
        
        # Add summary statistics
        summary = self.get_summary_statistics()
        
        output_data = {
            'experiment_config': {
                'experiment_id': self.context.config.experiment_id,
                'description': self.context.config.description,
                'parameters': self.context.config.parameters,
                'start_time': self.context.config.start_time.to_datetime().isoformat(),
                'expected_duration_seconds': self.context.config.expected_duration
            },
            'summary_statistics': summary,
            'results': serializable_results
        }
        
        with open(filepath, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        if self.context.verbose:
            print(f"Results saved to: {filepath}")