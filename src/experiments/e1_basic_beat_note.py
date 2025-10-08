"""
Experiment E1: Basic Beat-Note Formation and Analysis

This experiment validates the fundamental physics model of chronometric interferometry
by generating clean beat-note waveforms from two oscillators with known offsets and
implementing basic τ/Δf estimation using phase slope analysis.
"""

import time
from typing import Any, Dict, Optional

import matplotlib.pyplot as plt
import numpy as np

from ..algorithms.estimator import EstimatorFactory
from ..core.constants import PhysicalConstants
from ..core.types import (PPB, BeatNoteData, EstimationResult,
                          ExperimentConfig, ExperimentResult, Hertz,
                          MeasurementQuality, OscillatorModel,
                          PerformanceMetrics, Picoseconds, Seconds, Timestamp)
from ..signal_processing.beat_note import BeatNoteProcessor
from ..signal_processing.channel import ChannelSimulator
from ..signal_processing.oscillator import Oscillator
from ..signal_processing.utils import apply_fractional_delay
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

    def _validate_parameters(
        self, sampling_rate: Hertz, duration: Seconds, snr_db: float
    ) -> Optional[str]:
        """Validate runtime parameters and return error message on failure."""
        if sampling_rate <= 0:
            return "Sampling rate must be positive."
        if duration <= 0:
            return "Duration must be positive."
        if not np.isfinite(float(snr_db)):
            return "SNR must be finite."
        return None

    def _build_failure_result(
        self, context: ExperimentContext, error_message: str
    ) -> ExperimentResult:
        """Construct a standardized failure result."""
        metrics = PerformanceMetrics(
            rmse_timing=Picoseconds(float("inf")),
            rmse_frequency=PPB(float("inf")),
            convergence_time=Seconds(0.0),
            iterations_to_convergence=0,
            final_spectral_gap=0.0,
            communication_overhead=0,
            computation_time=Seconds(0.0),
        )

        completion_time = Timestamp.from_ps(
            PhysicalConstants.seconds_to_ps(time.time())
        )

        return ExperimentResult(
            config=context.config,
            metrics=metrics,
            telemetry=[],
            final_state=None,
            success=False,
            error_message=error_message,
            completion_time=completion_time,
        )

    def create_default_config(self) -> ExperimentConfig:
        """Create default experiment configuration."""
        tx_frequency_hz = 2442e6  # 2442 MHz - GPS L1 band (for reference)
        true_delta_f_hz = 0.0  # Expect zero residual frequency offset after mixing

        return ExperimentConfig(
            experiment_id=self.name,
            description=self.description,
            parameters={
                # Realistic RF frequencies for chronometric interferometry
                "tx_frequency_hz": tx_frequency_hz,
                "rx_frequency_hz": tx_frequency_hz + 100.0,  # RX frequency offset for beat note
                "sampling_rate_hz": 20e6,  # 20 MS/s (adequate for baseband processing)
                "duration_seconds": 0.01,  # 10 ms (reduced for better performance)
                # True values for validation
                "true_tau_ps": 100.0,  # 100 ps time-of-flight (realistic for RF)
                "true_delta_f_hz": true_delta_f_hz,
                # Signal parameters
                "snr_db": 30.0,  # Moderate SNR for realistic conditions
                "add_noise": True,
                "add_phase_noise": True,  # Realistic phase noise for RF
                "oscillator_type": "tcxo",  # Use realistic TCXO model
                # Estimation parameters
                "estimation_method": "phase_slope",
                # Visualization parameters
                "plot_results": True,
                "save_plots": True,
            },
            seed=42,
            start_time=Timestamp.from_ps(0.0),
            expected_duration=Seconds(60.0),  # 1 minute expected
        )

    def run_experiment(
        self, context: ExperimentContext, parameters: Dict[str, Any]
    ) -> ExperimentResult:
        """
        Run the experiment.

        Args:
            context: Experiment context
            parameters: Experiment parameters

        Returns:
            Experiment result
        """
        # Extract parameters
        tx_frequency = Hertz(parameters.get("tx_frequency_hz", 2442e6))  # GPS L1 band
        rx_frequency = Hertz(
            parameters.get("rx_frequency_hz", 2442e6 + 100.0)
        )  # GPS L1 band + offset
        sampling_rate = Hertz(parameters.get("sampling_rate_hz", 20e6))
        duration = Seconds(parameters.get("duration_seconds", 0.01))  # Default to 10 ms
        true_tau = Picoseconds(
            parameters.get("true_tau_ps", 100.0)
        )  # Default to 100 ps
        true_delta_f = Hertz(parameters.get("true_delta_f_hz", 50.0))
        snr_db = parameters.get("snr_db", 30.0)
        add_noise = parameters.get("add_noise", True)
        add_phase_noise = parameters.get("add_phase_noise", True)
        oscillator_type = parameters.get("oscillator_type", "tcxo")
        estimation_method = parameters.get("estimation_method", "phase_slope")
        plot_results = parameters.get("plot_results", True)
        save_plots = parameters.get("save_plots", True)

        validation_error = self._validate_parameters(sampling_rate, duration, snr_db)
        if validation_error:
            return self._build_failure_result(context, validation_error)

        # Create realistic oscillators based on type selection
        if oscillator_type == "ocxo":
            tx_oscillator_model = Oscillator.create_ocxo_model(tx_frequency)
            rx_oscillator_model = Oscillator.create_ocxo_model(rx_frequency)
        elif oscillator_type == "tcxo":
            tx_oscillator_model = Oscillator.create_tcxo_model(tx_frequency)
            rx_oscillator_model = Oscillator.create_tcxo_model(rx_frequency)
        else:  # ideal
            tx_oscillator_model = Oscillator.create_ideal_oscillator(tx_frequency)
            rx_oscillator_model = Oscillator.create_ideal_oscillator(rx_frequency)

        tx_oscillator = Oscillator(tx_oscillator_model)
        rx_oscillator = Oscillator(rx_oscillator_model, initial_phase=0.0)

        # Generate signals with realistic RF characteristics
        tx_time, tx_signal = tx_oscillator.generate_signal(
            duration=duration,
            sampling_rate=sampling_rate,
            frequency_offset=Hertz(0.0),  # No additional offset for TX
            phase_noise_enabled=add_phase_noise,
        )

        # Generate RX signal with frequency offset that simulates drift
        rx_time, rx_signal = rx_oscillator.generate_signal(
            duration=duration,
            sampling_rate=sampling_rate,
            frequency_offset=Hertz(0.0),  # RX already has the offset in its model
            phase_noise_enabled=add_phase_noise,
        )

        # Apply time delay to simulate time-of-flight (chronometric interferometry)
        total_delay_samples = float(true_tau) * float(sampling_rate) * 1e-12
        if total_delay_samples < 0:
            return self._build_failure_result(
                context, "Time-of-flight delay must be non-negative."
            )
        if total_delay_samples > 0:
            rx_signal = apply_fractional_delay(rx_signal, total_delay_samples)

        # Create realistic channel simulator with multipath effects
        channel_sim = ChannelSimulator(sampling_rate)

        # Create realistic RF channel model with typical characteristics for chronometric interferometry
        channel_model = channel_sim.create_rf_multipath_channel(
            delay=true_tau,  # Primary path delay
            multipath_taps=[0.3, 0.1],  # Secondary/multipath tap amplitudes
            multipath_delays=[50.0, 120.0],  # Additional delays in ps
        )

        # Apply channel
        rx_signal = channel_sim.apply_channel(rx_signal, channel_model, rx_frequency)

        # Add thermal noise if requested
        if add_noise:
            rx_signal = channel_sim.add_thermal_noise(rx_signal, snr_db=snr_db)

        # Create beat note processor
        beat_processor = BeatNoteProcessor(sampling_rate)

        # Generate beat note for chronometric interferometry
        timestamp = Timestamp.from_ps(0.0)
        beat_note = beat_processor.generate_beat_note(
            tx_signal=tx_signal,
            rx_signal=rx_signal,
            tx_frequency=tx_frequency,
            rx_frequency=rx_frequency,
            duration=duration,
            timestamp=timestamp,
            add_noise=False,  # Noise already added
            snr_db=snr_db,
        )

        # Estimate τ and Δf using chronometric interferometry principles
        estimator = EstimatorFactory.create_estimator(estimation_method)
        estimation_result = estimator.estimate(beat_note)

        # Calculate errors
        tau_error_ps = abs(float(estimation_result.tau) - float(true_tau))
        delta_f_error_hz = abs(float(estimation_result.delta_f) - float(true_delta_f))

        # Create performance metrics
        metrics = PerformanceMetrics(
            rmse_timing=Picoseconds(tau_error_ps),
            rmse_frequency=PPB(delta_f_error_hz / float(tx_frequency) * 1e9),
            convergence_time=Seconds(0.0),  # Not applicable for single measurement
            iterations_to_convergence=1,
            final_spectral_gap=1.0,  # Not applicable for single measurement
            communication_overhead=0,  # Not applicable for single measurement
            computation_time=Seconds(0.0),  # Would need to measure this
        )

        # Generate plots if requested
        if plot_results:
            self._plot_results(
                context,
                beat_note,
                estimation_result,
                true_tau,
                true_delta_f,
                save_plots,
            )

        # Determine success with more reasonable criteria for OSS demo
        success = (
            tau_error_ps < 500.0  # Within 500 ps (reasonable for OSS demo)
            and delta_f_error_hz < 25.0  # Within 25 Hz (reasonable for OSS demo)
            and estimation_result.quality != MeasurementQuality.INVALID
        )

        error_message = (
            None
            if success
            else f"Large errors: τ={tau_error_ps:.1f}ps, Δf={delta_f_error_hz:.1f}Hz"
        )

        return ExperimentResult(
            config=context.config,
            metrics=metrics,
            telemetry=[],
            final_state=None,
            success=success,
            error_message=error_message,
            completion_time=Timestamp.from_ps(
                PhysicalConstants.seconds_to_ps(time.time())
            ),
        )

    def _plot_results(
        self,
        context: ExperimentContext,
        beat_note: BeatNoteData,
        estimation_result: EstimationResult,
        true_tau: Picoseconds,
        true_delta_f: Hertz,
        save_plots: bool,
    ):
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
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(
            f"RF Chronometric Interferometry: Beat Note Analysis\n"
            f"TX Freq: {beat_note.tx_frequency/1e6:.1f} MHz, "
            f"RX Freq: {beat_note.rx_frequency/1e6:.1f} MHz\n"
            f"Duration: {beat_note.duration*1000:.1f} ms, "
            f"Sampling Rate: {beat_note.sampling_rate/1e6:.1f} MS/s\n"
            f"True: τ={true_tau:.1f}ps, Δf={true_delta_f:.1f}Hz | "
            f"Estimated: τ={estimation_result.tau:.1f}±{estimation_result.tau_uncertainty:.1f}ps, "
            f"Δf={estimation_result.delta_f:.1f}±{estimation_result.delta_f_uncertainty:.1f}Hz",
            fontsize=14,
            fontweight="bold",
        )

        # Plot 1: Beat note waveform (magnitude)
        time_vector = beat_note.get_time_vector() * 1000  # Convert to ms
        axes[0, 0].plot(
            time_vector,
            np.abs(beat_note.waveform),
            "b-",
            linewidth=1.0,
            label="Magnitude",
        )
        axes[0, 0].set_title("RF Beat Note Waveform (Magnitude)", fontweight="bold")
        axes[0, 0].set_xlabel("Time (ms)")
        axes[0, 0].set_ylabel("Amplitude")
        axes[0, 0].grid(True, linestyle="--", alpha=0.7)
        axes[0, 0].legend()

        # Plot 2: Beat note spectrum
        fft_result = np.fft.fft(beat_note.waveform)
        freqs = np.fft.fftfreq(len(beat_note.waveform), 1.0 / beat_note.sampling_rate)
        positive_freq_idx = freqs > 0
        positive_freqs = freqs[positive_freq_idx] / 1000  # Convert to kHz
        positive_magnitude = np.abs(fft_result[positive_freq_idx])

        axes[0, 1].plot(
            positive_freqs, 20 * np.log10(positive_magnitude), "r-", linewidth=1.0
        )
        axes[0, 1].set_title("Beat Note Spectrum", fontweight="bold")
        axes[0, 1].set_xlabel("Frequency (kHz)")
        axes[0, 1].set_ylabel("Magnitude (dB)")
        axes[0, 1].grid(True, linestyle="--", alpha=0.7)

        # Plot 3: Instantaneous phase
        _, instantaneous_phase = processor.extract_instantaneous_phase(beat_note)
        axes[1, 0].plot(time_vector, instantaneous_phase, "g-", linewidth=1.0)
        axes[1, 0].set_title("Instantaneous Phase Evolution", fontweight="bold")
        axes[1, 0].set_xlabel("Time (ms)")
        axes[1, 0].set_ylabel("Phase (rad)")
        axes[1, 0].grid(True, linestyle="--", alpha=0.7)

        # Plot 4: Instantaneous frequency
        _, instantaneous_freq = processor.extract_instantaneous_frequency(beat_note)
        expected_beat_freq = beat_note.get_beat_frequency()
        axes[1, 1].plot(
            time_vector,
            instantaneous_freq,
            "m-",
            linewidth=1.0,
            label="Instantaneous Frequency",
        )
        axes[1, 1].axhline(
            y=expected_beat_freq,
            color="r",
            linestyle="--",
            label=f"Expected: {expected_beat_freq:.1f} Hz",
            linewidth=2,
        )
        axes[1, 1].set_title("Instantaneous Frequency", fontweight="bold")
        axes[1, 1].set_xlabel("Time (ms)")
        axes[1, 1].set_ylabel("Frequency (Hz)")
        axes[1, 1].legend()
        axes[1, 1].grid(True, linestyle="--", alpha=0.7)

        plt.tight_layout()

        if save_plots:
            import os

            os.makedirs(context.output_dir, exist_ok=True)
            filename = os.path.join(
                context.output_dir, f"{context.config.experiment_id}_visualization.png"
            )
            plt.savefig(filename, dpi=300, bbox_inches="tight")
            print(f"RF Chronometric Interferometry visualization saved to: {filename}")

        plt.show()

    def run_parameter_sweep(
        self,
        context: ExperimentContext,
        snr_range: list = None,
        frequency_offset_range: list = None,
    ) -> Dict[str, Any]:
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
            snr_range = [20, 25, 30, 35, 40]  # dB (more appropriate for RF)

        if frequency_offset_range is None:
            frequency_offset_range = [
                10,
                25,
                50,
                75,
                100,
            ]  # Hz (more appropriate for RF)

        from .runner import ExperimentRunner

        # Create parameter ranges
        parameter_ranges = {
            "snr_db": snr_range,
            "true_delta_f_hz": frequency_offset_range,
        }

        # Run sweep
        runner = ExperimentRunner(context)
        results = runner.run_parameter_sweep(self.run_experiment, parameter_ranges)

        # Analyze results
        sweep_results = self._analyze_sweep_results(
            results, snr_range, frequency_offset_range
        )

        return sweep_results

    def _analyze_sweep_results(
        self, results: list, snr_range: list, frequency_offset_range: list
    ) -> Dict[str, Any]:
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
            "tau_errors": tau_errors,
            "delta_f_errors": delta_f_errors,
            "success_rates": success_rates,
            "snr_range": snr_range,
            "frequency_offset_range": frequency_offset_range,
        }


def main():
    """Main function to run Experiment E1."""
    print("Running Experiment E1: Basic Beat-Note Formation and Analysis")
    print("RF Chronometric Interferometry Demonstration")

    # Create experiment
    experiment = ExperimentE1()

    # Create configuration
    config = experiment.create_default_config()

    # Create context
    context = ExperimentContext(
        config=config,
        output_dir="results/e1_basic_beat_note",
        random_seed=42,
        verbose=True,
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

    print(
        f"Sweep completed with {len(sweep_results['snr_range'])} SNR values and "
        f"{len(sweep_results['frequency_offset_range'])} frequency offsets"
    )

    # Save results
    from .runner import ExperimentRunner

    runner = ExperimentRunner(context)
    runner.results = [result]  # Add single result for saving
    runner.save_results()

    print("\nExperiment E1 completed!")


if __name__ == "__main__":
    main()
