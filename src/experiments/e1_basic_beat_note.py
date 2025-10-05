"""
Experiment E1: Basic Beat-Note Formation and Analysis

This experiment validates the fundamental physics model of chronometric interferometry
by generating clean beat-note waveforms from two oscillators with known offsets and
implementing basic τ/Δf estimation using phase slope analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
import time
from typing import Dict, Any

from ..core.types import (
    ExperimentConfig, ExperimentResult, PerformanceMetrics,
    Timestamp, Seconds, Picoseconds, Hertz, MeasurementQuality,
    OscillatorModel, BeatNoteData, EstimationResult, PPB
)
from ..core.constants import PhysicalConstants
from ..signal_processing.oscillator import Oscillator
from ..signal_processing.beat_note import BeatNoteProcessor
from ..signal_processing.channel import ChannelSimulator
from ..algorithms.estimator import EstimatorFactory
from .runner import ExperimentContext


class ExperimentE1:
    """
    Experiment E1: Basic Beat-Note Formation and Analysis.
    
    Hypothesis: Two-way chronometric interferometry can extract τ and Δf from 
    simple beat patterns with known ground truth.
    
    Objectives:
    - Generate clean beat-note waveforms from two oscillators with known offsets
    - Implement basic τ/Δf estimation using phase slope analysis
    - Validate against analytical solutions
    - Characterize variance floor due to numerical precision
    """
    
    def __init__(self):
        """Initialize experiment E1."""
        self.name = "E1_Basic_Beat_Note"
        self.description = "Basic beat-note formation and analysis"
        
    def create_default_config(self) -> ExperimentConfig:
        """Create default experiment configuration."""
        return ExperimentConfig(
            experiment_id=self.name,
            description=self.description,
            parameters={
                # Oscillator parameters
                'tx_frequency_hz': 2.4e9,  # 2.4 GHz
                'rx_frequency_hz': 2.4e9 + 100.0,  # 2.4 GHz + 100 Hz offset
                'sampling_rate_hz': 10e6,  # 10 MS/s
                'duration_seconds': 0.1,  # 100 ms
                
                # True values for validation
                'true_tau_ps': 1000.0,  # 1 ns time-of-flight
                'true_delta_f_hz': 50.0,  # 50 Hz frequency offset
                
                # Signal parameters
                'snr_db': 40.0,  # High SNR for clean measurement
                'add_noise': True,
                'add_phase_noise': False,  # Start with ideal oscillators
                
                # Estimation parameters
                'estimation_method': 'phase_slope',
                
                # Visualization parameters
                'plot_results': True,
                'save_plots': True,
            },
            seed=42,
            start_time=Timestamp.from_ps(0.0),
            expected_duration=Seconds(60.0)  # 1 minute expected
        )
    
    def run_experiment(self, context: ExperimentContext, parameters: Dict[str, Any]) -> ExperimentResult:
        """
        Run the experiment.
        
        Args:
            context: Experiment context
            parameters: Experiment parameters
            
        Returns:
            Experiment result
        """
        # Extract parameters
        tx_frequency = Hertz(parameters.get('tx_frequency_hz', 2.4e9))
        rx_frequency = Hertz(parameters.get('rx_frequency_hz', 2.4e9 + 100.0))
        sampling_rate = Hertz(parameters.get('sampling_rate_hz', 10e6))
        duration = Seconds(parameters.get('duration_seconds', 0.1))
        true_tau = Picoseconds(parameters.get('true_tau_ps', 1000.0))
        true_delta_f = Hertz(parameters.get('true_delta_f_hz', 50.0))
        snr_db = parameters.get('snr_db', 40.0)
        add_noise = parameters.get('add_noise', True)
        add_phase_noise = parameters.get('add_phase_noise', False)
        estimation_method = parameters.get('estimation_method', 'phase_slope')
        plot_results = parameters.get('plot_results', True)
        save_plots = parameters.get('save_plots', True)
        
        # Create oscillators
        if add_phase_noise:
            tx_oscillator_model = Oscillator.create_tcxo_model(tx_frequency)
            rx_oscillator_model = Oscillator.create_tcxo_model(rx_frequency)
        else:
            tx_oscillator_model = Oscillator.create_ideal_oscillator(tx_frequency)
            rx_oscillator_model = Oscillator.create_ideal_oscillator(rx_frequency)
        
        tx_oscillator = Oscillator(tx_oscillator_model)
        rx_oscillator = Oscillator(rx_oscillator_model, initial_phase=0.0)
        
        # Generate signals
        tx_time, tx_signal = tx_oscillator.generate_signal(
            duration=duration,
            sampling_rate=sampling_rate,
            frequency_offset=Hertz(0.0),
            phase_noise_enabled=add_phase_noise
        )
        
        rx_time, rx_signal = rx_oscillator.generate_signal(
            duration=duration,
            sampling_rate=sampling_rate,
            frequency_offset=true_delta_f,
            phase_noise_enabled=add_phase_noise
        )
        
        # Apply time delay to simulate time-of-flight
        delay_samples = int(true_tau * sampling_rate * 1e-12)
        if delay_samples > 0:
            rx_signal = np.concatenate([np.zeros(delay_samples, dtype=complex), rx_signal[:-delay_samples]])
        
        # Create channel simulator (AWGN for now)
        channel_sim = ChannelSimulator(sampling_rate)
        channel_model = channel_sim.create_awgn_channel(delay=Picoseconds(0.0))
        
        # Apply channel
        rx_signal = channel_sim.apply_channel(rx_signal, channel_model, rx_frequency)
        
        # Add thermal noise if requested
        if add_noise:
            rx_signal = channel_sim.add_thermal_noise(rx_signal, snr_db=snr_db)
        
        # Create beat note processor
        beat_processor = BeatNoteProcessor(sampling_rate)
        
        # Generate beat note
        timestamp = Timestamp.from_ps(0.0)
        beat_note = beat_processor.generate_beat_note(
            tx_signal=tx_signal,
            rx_signal=rx_signal,
            tx_frequency=tx_frequency,
            rx_frequency=rx_frequency,
            duration=duration,
            timestamp=timestamp,
            add_noise=False,  # Noise already added
            snr_db=snr_db
        )
        
        # Estimate τ and Δf
        estimator = EstimatorFactory.create_estimator(estimation_method)
        estimation_result = estimator.estimate(beat_note)
        
        # Calculate errors
        tau_error = abs(estimation_result.tau - true_tau)
        delta_f_error = abs(estimation_result.delta_f - true_delta_f)
        
        # Create performance metrics
        metrics = PerformanceMetrics(
            rmse_timing=Picoseconds(tau_error),
            rmse_frequency=delta_f_error / tx_frequency * 1e9,  # Convert to ppb
            convergence_time=Seconds(0.0),  # Not applicable for single measurement
            iterations_to_convergence=1,
            final_spectral_gap=1.0,  # Not applicable for single measurement
            communication_overhead=0,  # Not applicable for single measurement
            computation_time=Seconds(0.0)  # Would need to measure this
        )
        
        # Generate plots if requested
        if plot_results:
            self._plot_results(context, beat_note, estimation_result, true_tau, true_delta_f, save_plots)
        
        # Determine success
        success = (
            tau_error < 10.0 and  # Within 10 ps
            delta_f_error < 1.0 and  # Within 1 Hz
            estimation_result.quality != MeasurementQuality.INVALID
        )
        
        error_message = None if success else f"Large errors: τ={tau_error:.1f}ps, Δf={delta_f_error:.1f}Hz"
        
        return ExperimentResult(
            config=context.config,
            metrics=metrics,
            telemetry=[],
            final_state=None,
            success=success,
            error_message=error_message,
            completion_time=Timestamp.from_ps(PhysicalConstants.seconds_to_ps(time.time()))
        )
    
    def _plot_results(self, 
                     context: ExperimentContext,
                     beat_note: BeatNoteData,
                     estimation_result: EstimationResult,
                     true_tau: Picoseconds,
                     true_delta_f: Hertz,
                     save_plots: bool):
        """
        Generate plots for experiment results.
        
        Args:
            context: Experiment context
            beat_note: Beat note data
            estimation_result: Estimation result
            true_tau: True time-of-flight
            true_delta_f: True frequency offset
            save_plots: Whether to save plots
        """
        from ..signal_processing.beat_note import BeatNoteProcessor
        processor = BeatNoteProcessor(beat_note.sampling_rate)
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f"Experiment E1: Basic Beat-Note Analysis\n"
                    f"True: τ={true_tau:.1f}ps, Δf={true_delta_f:.1f}Hz | "
                    f"Estimated: τ={estimation_result.tau:.1f}±{estimation_result.tau_uncertainty:.1f}ps, "
                    f"Δf={estimation_result.delta_f:.1f}±{estimation_result.delta_f_uncertainty:.1f}Hz",
                    fontsize=12)
        
        # Plot 1: Beat note waveform (real part)
        time_vector = beat_note.get_time_vector() * 1000  # Convert to ms
        axes[0, 0].plot(time_vector, beat_note.waveform.real)
        axes[0, 0].set_title("Beat Note Waveform (Real Part)")
        axes[0, 0].set_xlabel("Time (ms)")
        axes[0, 0].set_ylabel("Amplitude")
        axes[0, 0].grid(True)
        
        # Plot 2: Beat note spectrum
        fft_result = np.fft.fft(beat_note.waveform)
        freqs = np.fft.fftfreq(len(beat_note.waveform), 1.0 / beat_note.sampling_rate)
        positive_freq_idx = freqs > 0
        positive_freqs = freqs[positive_freq_idx] / 1000  # Convert to kHz
        positive_magnitude = np.abs(fft_result[positive_freq_idx])
        
        axes[0, 1].plot(positive_freqs, 20 * np.log10(positive_magnitude))
        axes[0, 1].set_title("Beat Note Spectrum")
        axes[0, 1].set_xlabel("Frequency (kHz)")
        axes[0, 1].set_ylabel("Magnitude (dB)")
        axes[0, 1].grid(True)
        
        # Plot 3: Instantaneous phase
        _, instantaneous_phase = processor.extract_instantaneous_phase(beat_note)
        axes[1, 0].plot(time_vector, instantaneous_phase)
        axes[1, 0].set_title("Instantaneous Phase")
        axes[1, 0].set_xlabel("Time (ms)")
        axes[1, 0].set_ylabel("Phase (rad)")
        axes[1, 0].grid(True)
        
        # Plot 4: Instantaneous frequency
        _, instantaneous_freq = processor.extract_instantaneous_frequency(beat_note)
        axes[1, 1].plot(time_vector, instantaneous_freq)
        axes[1, 1].axhline(y=beat_note.get_beat_frequency(), color='r', linestyle='--', 
                          label=f"Expected: {beat_note.get_beat_frequency():.1f} Hz")
        axes[1, 1].set_title("Instantaneous Frequency")
        axes[1, 1].set_xlabel("Time (ms)")
        axes[1, 1].set_ylabel("Frequency (Hz)")
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        if save_plots:
            import os
            os.makedirs(context.output_dir, exist_ok=True)
            filename = os.path.join(context.output_dir, f"{context.config.experiment_id}_plot.png")
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Plot saved to: {filename}")
        
        plt.show()
    
    def run_parameter_sweep(self, 
                          context: ExperimentContext,
                          snr_range: list = None,
                          frequency_offset_range: list = None) -> Dict[str, Any]:
        """
        Run parameter sweep for SNR and frequency offset.
        
        Args:
            context: Experiment context
            snr_range: List of SNR values to test
            frequency_offset_range: List of frequency offsets to test
            
        Returns:
            Sweep results
        """
        if snr_range is None:
            snr_range = [20, 30, 40, 50, 60]  # dB
        
        if frequency_offset_range is None:
            frequency_offset_range = [10, 50, 100, 500, 1000]  # Hz
        
        from .runner import ExperimentRunner
        
        # Create parameter ranges
        parameter_ranges = {
            'snr_db': snr_range,
            'true_delta_f_hz': frequency_offset_range
        }
        
        # Run sweep
        runner = ExperimentRunner(context)
        results = runner.run_parameter_sweep(self.run_experiment, parameter_ranges)
        
        # Analyze results
        sweep_results = self._analyze_sweep_results(results, snr_range, frequency_offset_range)
        
        return sweep_results
    
    def _analyze_sweep_results(self, 
                              results: list,
                              snr_range: list,
                              frequency_offset_range: list) -> Dict[str, Any]:
        """
        Analyze parameter sweep results.
        
        Args:
            results: List of experiment results
            snr_range: SNR values tested
            frequency_offset_range: Frequency offset values tested
            
        Returns:
            Analysis results
        """
        # Create matrices for results
        tau_errors = np.zeros((len(snr_range), len(frequency_offset_range)))
        delta_f_errors = np.zeros((len(snr_range), len(frequency_offset_range)))
        success_rates = np.zeros((len(snr_range), len(frequency_offset_range)))
        
        for i, result in enumerate(results):
            if result.success:
                snr_idx = i // len(frequency_offset_range)
                freq_idx = i % len(frequency_offset_range)
                
                tau_errors[snr_idx, freq_idx] = result.metrics.rmse_timing
                delta_f_errors[snr_idx, freq_idx] = result.metrics.rmse_frequency
                success_rates[snr_idx, freq_idx] = 1.0
        
        return {
            'tau_errors': tau_errors,
            'delta_f_errors': delta_f_errors,
            'success_rates': success_rates,
            'snr_range': snr_range,
            'frequency_offset_range': frequency_offset_range
        }


def main():
    """Main function to run Experiment E1."""
    print("Running Experiment E1: Basic Beat-Note Formation and Analysis")
    
    # Create experiment
    experiment = ExperimentE1()
    
    # Create configuration
    config = experiment.create_default_config()
    
    # Create context
    context = ExperimentContext(
        config=config,
        output_dir="results/e1_basic_beat_note",
        random_seed=42,
        verbose=True
    )
    
    # Run single experiment
    print("\n=== Running Single Experiment ===")
    result = experiment.run_experiment(context, config.parameters)
    
    print(f"Success: {result.success}")
    print(f"Timing RMSE: {result.metrics.rmse_timing:.1f} ps")
    print(f"Frequency RMSE: {result.metrics.rmse_frequency:.1f} ppb")
    
    if not result.success:
        print(f"Error: {result.error_message}")
    
    # Run parameter sweep
    print("\n=== Running Parameter Sweep ===")
    sweep_results = experiment.run_parameter_sweep(context)
    
    print(f"Sweep completed with {len(sweep_results['snr_range'])} SNR values and "
          f"{len(sweep_results['frequency_offset_range'])} frequency offsets")
    
    # Save results
    from .runner import ExperimentRunner
    runner = ExperimentRunner(context)
    runner.results = [result]  # Add single result for saving
    runner.save_results()
    
    print("\nExperiment E1 completed!")


if __name__ == "__main__":
    main()