"""
Phase 1 Simulation: Two-node physical-limit sweep.

This simulation explores the fundamental physical limits of synchronization
between two nodes under various SNR, bandwidth, and duration conditions.
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

from phy.osc import OscillatorParams, AllanDeviationGenerator
from phy.noise import NoiseParams, NoiseGenerator
from hw.trx import TransceiverNode, TransceiverConfig
from alg.ci import ClosedFormEstimator, EstimatorParams
from metrics.crlb import JointCRLBCalculator, CRLBParams
from metrics.biasvar import BiasVarianceAnalyzer, BiasVarianceParams


@dataclass
class Phase1Config:
    """Configuration for Phase 1 simulation."""
    # SNR sweep parameters
    snr_range_db: List[float]
    
    # Bandwidth sweep parameters  
    bandwidth_range: List[float]
    
    # Duration sweep parameters
    duration_range: List[float]
    
    # Fixed simulation parameters
    carrier_freq: float = 2.4e9
    sample_rate: float = 1e6
    n_monte_carlo: int = 500
    
    # Output configuration
    save_results: bool = True
    plot_results: bool = True
    results_dir: str = "results/phase1"


class Phase1Simulator:
    """Two-node physical limit analysis simulator."""
    
    def __init__(self, config: Phase1Config):
        self.config = config
        
        # Create results directory
        if self.config.save_results:
            os.makedirs(self.config.results_dir, exist_ok=True)
            
    def run_full_simulation(self) -> Dict[str, Any]:
        """Run complete Phase 1 simulation suite."""
        print("Starting Phase 1 Simulation: Two-node physical limits")
        
        results = {
            'snr_sweep': self.run_snr_sweep(),
            'bandwidth_sweep': self.run_bandwidth_sweep(),
            'duration_sweep': self.run_duration_sweep(),
            'joint_analysis': self.run_joint_parameter_analysis()
        }
        
        if self.config.save_results:
            self._save_results(results)
            
        if self.config.plot_results:
            self._generate_plots(results)
            
        return results
        
    def run_snr_sweep(self) -> Dict[str, Any]:
        """Run SNR sweep analysis."""
        print("Running SNR sweep...")
        
        snr_results = {
            'snr_values_db': self.config.snr_range_db,
            'crlb_delay': [],
            'crlb_frequency': [],
            'estimator_mse_delay': [],
            'estimator_mse_frequency': [],
            'estimator_bias_delay': [],
            'estimator_bias_frequency': []
        }
        
        # Fixed parameters for SNR sweep
        bandwidth = 1e6  # 1 MHz
        duration = 0.001  # 1 ms
        
        for snr_db in self.config.snr_range_db:
            print(f"  Processing SNR = {snr_db} dB")
            
            # Compute theoretical CRLB
            crlb_params = CRLBParams(
                snr_db=snr_db,
                bandwidth=bandwidth,
                duration=duration,
                carrier_freq=self.config.carrier_freq,
                sample_rate=self.config.sample_rate
            )
            
            crlb_calc = JointCRLBCalculator(crlb_params)
            crlb_results = crlb_calc.compute_joint_crlb()
            
            snr_results['crlb_delay'].append(crlb_results['delay_crlb_std'])
            snr_results['crlb_frequency'].append(crlb_results['frequency_crlb_std'])
            
            # Run Monte Carlo simulation
            mc_results = self._run_monte_carlo_estimation(snr_db, bandwidth, duration)
            
            snr_results['estimator_mse_delay'].append(mc_results['delay_mse'])
            snr_results['estimator_mse_frequency'].append(mc_results['frequency_mse'])
            snr_results['estimator_bias_delay'].append(mc_results['delay_bias'])
            snr_results['estimator_bias_frequency'].append(mc_results['frequency_bias'])
            
        return snr_results
        
    def run_bandwidth_sweep(self) -> Dict[str, Any]:
        """Run bandwidth sweep analysis."""
        print("Running bandwidth sweep...")
        
        bw_results = {
            'bandwidth_values': self.config.bandwidth_range,
            'crlb_delay': [],
            'crlb_frequency': [],
            'estimator_mse_delay': [],
            'estimator_mse_frequency': []
        }
        
        # Fixed parameters for bandwidth sweep
        snr_db = 20.0
        duration = 0.001
        
        for bandwidth in self.config.bandwidth_range:
            print(f"  Processing Bandwidth = {bandwidth/1e6:.1f} MHz")
            
            # Compute CRLB
            crlb_params = CRLBParams(
                snr_db=snr_db,
                bandwidth=bandwidth,
                duration=duration,
                carrier_freq=self.config.carrier_freq,
                sample_rate=self.config.sample_rate
            )
            
            crlb_calc = JointCRLBCalculator(crlb_params)
            crlb_results = crlb_calc.compute_joint_crlb()
            
            bw_results['crlb_delay'].append(crlb_results['delay_crlb_std'])
            bw_results['crlb_frequency'].append(crlb_results['frequency_crlb_std'])
            
            # Monte Carlo simulation
            mc_results = self._run_monte_carlo_estimation(snr_db, bandwidth, duration)
            
            bw_results['estimator_mse_delay'].append(mc_results['delay_mse'])
            bw_results['estimator_mse_frequency'].append(mc_results['frequency_mse'])
            
        return bw_results
        
    def run_duration_sweep(self) -> Dict[str, Any]:
        """Run observation duration sweep analysis."""
        print("Running duration sweep...")
        
        duration_results = {
            'duration_values': self.config.duration_range,
            'crlb_delay': [],
            'crlb_frequency': [],
            'estimator_mse_delay': [],
            'estimator_mse_frequency': []
        }
        
        # Fixed parameters for duration sweep
        snr_db = 15.0
        bandwidth = 1e6
        
        for duration in self.config.duration_range:
            print(f"  Processing Duration = {duration*1000:.1f} ms")
            
            # Compute CRLB
            crlb_params = CRLBParams(
                snr_db=snr_db,
                bandwidth=bandwidth,
                duration=duration,
                carrier_freq=self.config.carrier_freq,
                sample_rate=self.config.sample_rate
            )
            
            crlb_calc = JointCRLBCalculator(crlb_params)
            crlb_results = crlb_calc.compute_joint_crlb()
            
            duration_results['crlb_delay'].append(crlb_results['delay_crlb_std'])
            duration_results['crlb_frequency'].append(crlb_results['frequency_crlb_std'])
            
            # Monte Carlo simulation
            mc_results = self._run_monte_carlo_estimation(snr_db, bandwidth, duration)
            
            duration_results['estimator_mse_delay'].append(mc_results['delay_mse'])
            duration_results['estimator_mse_frequency'].append(mc_results['frequency_mse'])
            
        return duration_results
        
    def run_joint_parameter_analysis(self) -> Dict[str, Any]:
        """Run joint parameter space analysis."""
        print("Running joint parameter analysis...")
        
        # Create parameter grid
        snr_grid = np.linspace(0, 30, 15)  # 0 to 30 dB
        bw_grid = np.logspace(5, 7, 15)    # 100 kHz to 10 MHz
        
        joint_results = {
            'snr_grid': snr_grid,
            'bandwidth_grid': bw_grid,
            'delay_crlb_grid': np.zeros((len(snr_grid), len(bw_grid))),
            'frequency_crlb_grid': np.zeros((len(snr_grid), len(bw_grid))),
            'joint_crlb_determinant': np.zeros((len(snr_grid), len(bw_grid)))
        }
        
        duration = 0.001  # Fixed 1 ms
        
        for i, snr_db in enumerate(snr_grid):
            for j, bandwidth in enumerate(bw_grid):
                crlb_params = CRLBParams(
                    snr_db=snr_db,
                    bandwidth=bandwidth,
                    duration=duration,
                    carrier_freq=self.config.carrier_freq,
                    sample_rate=self.config.sample_rate
                )
                
                crlb_calc = JointCRLBCalculator(crlb_params)
                crlb_results = crlb_calc.compute_joint_crlb()
                
                joint_results['delay_crlb_grid'][i, j] = crlb_results['delay_crlb_std']
                joint_results['frequency_crlb_grid'][i, j] = crlb_results['frequency_crlb_std']
                joint_results['joint_crlb_determinant'][i, j] = crlb_results['determinant_fim']
                
        return joint_results
        
    def _run_monte_carlo_estimation(self, snr_db: float, bandwidth: float, 
                                  duration: float) -> Dict[str, float]:
        """Run Monte Carlo estimation simulation."""
        # True parameter values
        true_delay = 1e-6  # 1 microsecond
        true_freq_offset = 1e3  # 1 kHz
        
        delay_estimates = []
        freq_estimates = []
        
        # Setup estimator
        estimator_params = EstimatorParams(
            sample_rate=self.config.sample_rate,
            carrier_freq=self.config.carrier_freq,
            bandwidth=bandwidth,
            estimation_method='ml'
        )
        estimator = ClosedFormEstimator(estimator_params)
        
        # Setup noise generator
        noise_params = NoiseParams(
            snr_db=snr_db,
            phase_noise_psd=-80,  # dBc/Hz
            jitter_rms=1e-12      # 1 ps
        )
        noise_gen = NoiseGenerator(noise_params, self.config.sample_rate)
        
        for trial in range(self.config.n_monte_carlo):
            # Generate reference signal
            n_samples = int(duration * self.config.sample_rate)
            t = np.arange(n_samples) / self.config.sample_rate
            
            # Simple sinusoidal reference signal
            ref_signal = np.exp(1j * 2 * np.pi * self.config.carrier_freq * t)
            
            # Generate received signal with delay and frequency offset
            t_delayed = t - true_delay
            rx_signal = np.exp(1j * 2 * np.pi * (self.config.carrier_freq + true_freq_offset) * t_delayed)
            
            # Add noise
            rx_signal = noise_gen.add_awgn(rx_signal)
            
            # Estimate parameters
            try:
                delay_est, freq_est = estimator.estimate_delay_and_frequency(ref_signal, rx_signal)
                delay_estimates.append(delay_est)
                freq_estimates.append(freq_est)
            except:
                # Handle estimation failures
                delay_estimates.append(true_delay)
                freq_estimates.append(true_freq_offset)
                
        # Compute statistics
        delay_estimates = np.array(delay_estimates)
        freq_estimates = np.array(freq_estimates)
        
        delay_mse = np.mean((delay_estimates - true_delay) ** 2)
        freq_mse = np.mean((freq_estimates - true_freq_offset) ** 2)
        delay_bias = np.mean(delay_estimates - true_delay)
        freq_bias = np.mean(freq_estimates - true_freq_offset)
        
        return {
            'delay_mse': delay_mse,
            'frequency_mse': freq_mse,
            'delay_bias': delay_bias,
            'frequency_bias': freq_bias
        }
        
    def _save_results(self, results: Dict[str, Any]):
        """Save simulation results to file."""
        results_file = os.path.join(self.config.results_dir, 'phase1_results.json')
        
        # Convert numpy arrays to lists for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
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
        # SNR sweep plots
        self._plot_snr_sweep(results['snr_sweep'])
        
        # Bandwidth sweep plots
        self._plot_bandwidth_sweep(results['bandwidth_sweep'])
        
        # Duration sweep plots
        self._plot_duration_sweep(results['duration_sweep'])
        
        # Joint analysis plots
        self._plot_joint_analysis(results['joint_analysis'])
        
    def _plot_snr_sweep(self, snr_results: Dict[str, Any]):
        """Plot SNR sweep results."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        snr_db = snr_results['snr_values_db']
        
        # Delay performance
        ax1.semilogy(snr_db, snr_results['crlb_delay'], 'b-', label='CRLB', linewidth=2)
        ax1.semilogy(snr_db, np.sqrt(snr_results['estimator_mse_delay']), 'ro-', 
                    label='Estimator RMSE', markersize=4)
        ax1.set_xlabel('SNR (dB)')
        ax1.set_ylabel('Delay Error (s)')
        ax1.set_title('Delay Estimation vs SNR')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Frequency performance
        ax2.semilogy(snr_db, snr_results['crlb_frequency'], 'b-', label='CRLB', linewidth=2)
        ax2.semilogy(snr_db, np.sqrt(snr_results['estimator_mse_frequency']), 'ro-', 
                    label='Estimator RMSE', markersize=4)
        ax2.set_xlabel('SNR (dB)')
        ax2.set_ylabel('Frequency Error (Hz)')
        ax2.set_title('Frequency Estimation vs SNR')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        if self.config.save_results:
            plt.savefig(os.path.join(self.config.results_dir, 'snr_sweep.png'), dpi=300)
        plt.show()
        
    def _plot_bandwidth_sweep(self, bw_results: Dict[str, Any]):
        """Plot bandwidth sweep results."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        bw_mhz = np.array(bw_results['bandwidth_values']) / 1e6
        
        # Delay performance
        ax1.loglog(bw_mhz, bw_results['crlb_delay'], 'b-', label='CRLB', linewidth=2)
        ax1.loglog(bw_mhz, np.sqrt(bw_results['estimator_mse_delay']), 'ro-', 
                  label='Estimator RMSE', markersize=4)
        ax1.set_xlabel('Bandwidth (MHz)')
        ax1.set_ylabel('Delay Error (s)')
        ax1.set_title('Delay Estimation vs Bandwidth')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Frequency performance  
        ax2.loglog(bw_mhz, bw_results['crlb_frequency'], 'b-', label='CRLB', linewidth=2)
        ax2.loglog(bw_mhz, np.sqrt(bw_results['estimator_mse_frequency']), 'ro-', 
                  label='Estimator RMSE', markersize=4)
        ax2.set_xlabel('Bandwidth (MHz)')
        ax2.set_ylabel('Frequency Error (Hz)')
        ax2.set_title('Frequency Estimation vs Bandwidth')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        if self.config.save_results:
            plt.savefig(os.path.join(self.config.results_dir, 'bandwidth_sweep.png'), dpi=300)
        plt.show()
        
    def _plot_duration_sweep(self, duration_results: Dict[str, Any]):
        """Plot duration sweep results."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        duration_ms = np.array(duration_results['duration_values']) * 1000
        
        # Delay performance
        ax1.loglog(duration_ms, duration_results['crlb_delay'], 'b-', label='CRLB', linewidth=2)
        ax1.loglog(duration_ms, np.sqrt(duration_results['estimator_mse_delay']), 'ro-', 
                  label='Estimator RMSE', markersize=4)
        ax1.set_xlabel('Duration (ms)')
        ax1.set_ylabel('Delay Error (s)')
        ax1.set_title('Delay Estimation vs Duration')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Frequency performance
        ax2.loglog(duration_ms, duration_results['crlb_frequency'], 'b-', label='CRLB', linewidth=2)
        ax2.loglog(duration_ms, np.sqrt(duration_results['estimator_mse_frequency']), 'ro-', 
                  label='Estimator RMSE', markersize=4)
        ax2.set_xlabel('Duration (ms)')
        ax2.set_ylabel('Frequency Error (Hz)')
        ax2.set_title('Frequency Estimation vs Duration')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        if self.config.save_results:
            plt.savefig(os.path.join(self.config.results_dir, 'duration_sweep.png'), dpi=300)
        plt.show()
        
    def _plot_joint_analysis(self, joint_results: Dict[str, Any]):
        """Plot joint parameter analysis."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
        
        snr_grid = joint_results['snr_grid']
        bw_grid = joint_results['bandwidth_grid'] / 1e6  # Convert to MHz
        
        # Delay CRLB heatmap
        im1 = ax1.imshow(joint_results['delay_crlb_grid'], 
                        extent=[bw_grid[0], bw_grid[-1], snr_grid[0], snr_grid[-1]],
                        aspect='auto', origin='lower', cmap='viridis')
        ax1.set_xlabel('Bandwidth (MHz)')
        ax1.set_ylabel('SNR (dB)')
        ax1.set_title('Delay CRLB (s)')
        plt.colorbar(im1, ax=ax1)
        
        # Frequency CRLB heatmap
        im2 = ax2.imshow(joint_results['frequency_crlb_grid'],
                        extent=[bw_grid[0], bw_grid[-1], snr_grid[0], snr_grid[-1]],
                        aspect='auto', origin='lower', cmap='viridis')
        ax2.set_xlabel('Bandwidth (MHz)')
        ax2.set_ylabel('SNR (dB)')
        ax2.set_title('Frequency CRLB (Hz)')
        plt.colorbar(im2, ax=ax2)
        
        # Joint CRLB determinant
        im3 = ax3.imshow(np.log10(joint_results['joint_crlb_determinant']),
                        extent=[bw_grid[0], bw_grid[-1], snr_grid[0], snr_grid[-1]],
                        aspect='auto', origin='lower', cmap='plasma')
        ax3.set_xlabel('Bandwidth (MHz)')
        ax3.set_ylabel('SNR (dB)')
        ax3.set_title('log₁₀(FIM Determinant)')
        plt.colorbar(im3, ax=ax3)
        
        # Combined performance metric
        combined_metric = joint_results['delay_crlb_grid'] * joint_results['frequency_crlb_grid']
        im4 = ax4.imshow(np.log10(combined_metric),
                        extent=[bw_grid[0], bw_grid[-1], snr_grid[0], snr_grid[-1]],
                        aspect='auto', origin='lower', cmap='plasma')
        ax4.set_xlabel('Bandwidth (MHz)')
        ax4.set_ylabel('SNR (dB)')
        ax4.set_title('log₁₀(Combined CRLB)')
        plt.colorbar(im4, ax=ax4)
        
        plt.tight_layout()
        if self.config.save_results:
            plt.savefig(os.path.join(self.config.results_dir, 'joint_analysis.png'), dpi=300)
        plt.show()


def main():
    """Main function to run Phase 1 simulation."""
    # Configure simulation parameters
    config = Phase1Config(
        snr_range_db=list(range(-10, 31, 2)),  # -10 to 30 dB, step 2 dB
        bandwidth_range=[1e5, 2e5, 5e5, 1e6, 2e6, 5e6, 1e7],  # 100 kHz to 10 MHz
        duration_range=[1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2],  # 0.1 to 10 ms
        n_monte_carlo=100,  # Reduced for faster execution
        save_results=True,
        plot_results=True
    )
    
    # Run simulation
    simulator = Phase1Simulator(config)
    results = simulator.run_full_simulation()
    
    print("Phase 1 simulation completed successfully!")
    return results


if __name__ == "__main__":
    main()
