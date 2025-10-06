"""
Basic Beat Note Generation Demo

This example demonstrates the fundamental chronometric interferometry
principles by generating clean beat notes from two oscillators and
extracting timing and frequency information.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np

from src.algorithms.estimator import EstimatorFactory
from src.core.types import Hertz, Picoseconds, Seconds, Timestamp
from src.signal_processing.beat_note import BeatNoteProcessor
from src.signal_processing.channel import ChannelSimulator
from src.signal_processing.oscillator import Oscillator


def main():
    """Demonstrate basic beat note generation and analysis."""
    print("Basic Beat Note Generation Demo")
    print("=" * 40)

    # Configuration - using realistic RF frequencies for wireless deployment
    tx_frequency = Hertz(
        2.4e9
    )  # 2.4 GHz ISM band (realistic for RF chronometric interferometry)
    rx_frequency = Hertz(2.4e9 + 100.0)  # 2.4 GHz + 100 Hz offset
    sampling_rate = Hertz(20e6)  # 20 MS/s (adequate for baseband processing)
    duration = Seconds(0.01)  # 10 ms
    true_tau = Picoseconds(100.0)  # 100 ps time-of-flight
    true_delta_f = Hertz(50.0)  # 50 Hz frequency offset
    snr_db = 30.0  # 30 dB SNR

    print(f"TX Frequency: {tx_frequency/1e9:.1f} GHz")
    print(f"RX Frequency: {rx_frequency/1e9:.1f} GHz")
    print(f"Sampling Rate: {sampling_rate/1e6:.1f} MS/s")
    print(f"Duration: {duration*1000:.1f} ms")
    print(f"True τ: {true_tau:.1f} ps")
    print(f"True Δf: {true_delta_f:.1f} Hz")

    # Create ideal oscillators (no phase noise for clarity)
    print("\nCreating oscillators...")
    tx_oscillator_model = Oscillator.create_ideal_oscillator(tx_frequency)
    rx_oscillator_model = Oscillator.create_ideal_oscillator(rx_frequency)

    tx_oscillator = Oscillator(tx_oscillator_model)
    rx_oscillator = Oscillator(rx_oscillator_model, initial_phase=0.0)

    # Generate signals
    print("Generating signals...")
    tx_time, tx_signal = tx_oscillator.generate_signal(
        duration=duration,
        sampling_rate=sampling_rate,
        frequency_offset=Hertz(0.0),
        phase_noise_enabled=False,
    )

    rx_time, rx_signal = rx_oscillator.generate_signal(
        duration=duration,
        sampling_rate=sampling_rate,
        frequency_offset=Hertz(0.0),  # RX already has offset in its nominal frequency
        phase_noise_enabled=False,
    )

    print(f"Generated {len(tx_signal)} samples per signal")

    # Apply time delay to simulate time-of-flight
    print("Applying time-of-flight delay...")
    delay_samples = int(true_tau * sampling_rate * 1e-12)
    if delay_samples > 0:
        rx_signal = np.concatenate(
            [np.zeros(delay_samples, dtype=complex), rx_signal[:-delay_samples]]
        )

    # Create channel simulator and add noise
    print("Adding channel effects and noise...")
    channel_sim = ChannelSimulator(sampling_rate)
    rx_signal = channel_sim.add_thermal_noise(rx_signal, snr_db=snr_db)

    # Create beat note processor
    print("Processing beat note...")
    beat_processor = BeatNoteProcessor(sampling_rate)

    # Generate beat note
    timestamp = Timestamp.from_ps(0.0)
    beat_note = beat_processor.generate_beat_note(
        tx_signal=tx_signal,
        rx_signal=rx_signal,
        tx_frequency=tx_frequency,
        rx_frequency=rx_frequency,  # Use the RX frequency (already includes offset)
        duration=duration,
        timestamp=timestamp,
        add_noise=False,  # Noise already added
        snr_db=snr_db,
    )

    print(f"Beat note generated with {len(beat_note.waveform)} samples")
    print(f"Expected beat frequency: {beat_note.get_beat_frequency():.1f} Hz")

    # Extract beat frequency
    beat_freq, freq_uncertainty = beat_processor.extract_beat_frequency(beat_note)
    print(f"Measured beat frequency: {beat_freq:.1f} ± {freq_uncertainty:.1f} Hz")

    # Estimate τ and Δf using phase slope method
    print("\nEstimating τ and Δf...")
    estimator = EstimatorFactory.create_estimator("phase_slope")
    estimation_result = estimator.estimate(beat_note)

    print(
        f"Estimated τ: {estimation_result.tau:.1f} ± {estimation_result.tau_uncertainty:.1f} ps"
    )
    print(
        f"Estimated Δf: {estimation_result.delta_f:.1f} ± {estimation_result.delta_f_uncertainty:.1f} Hz"
    )

    # Calculate errors
    tau_error = abs(estimation_result.tau - true_tau)
    delta_f_error = abs(estimation_result.delta_f - true_delta_f)

    print(f"\nEstimation Errors:")
    print(f"τ error: {tau_error:.1f} ps")
    print(f"Δf error: {delta_f_error:.1f} Hz")

    # Success criteria
    success = (tau_error < 100.0) and (delta_f_error < 10.0)
    print(f"\nEstimation Success: {'✓' if success else '✗'}")

    # Generate plots
    print("\nGenerating plots...")
    plot_beat_note_analysis(
        beat_note, beat_processor, estimation_result, true_tau, true_delta_f
    )

    print("\nDemo completed!")


def plot_beat_note_analysis(
    beat_note, processor, estimation_result, true_tau, true_delta_f
):
    """Generate comprehensive plots of beat note analysis."""
    try:
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(
            f"Beat Note Analysis Demo\n"
            f"True: τ={true_tau:.1f}ps, Δf={true_delta_f:.1f}Hz | "
            f"Estimated: τ={estimation_result.tau:.1f}±{estimation_result.tau_uncertainty:.1f}ps, "
            f"Δf={estimation_result.delta_f:.1f}±{estimation_result.delta_f_uncertainty:.1f}Hz",
            fontsize=12,
        )

        # Time vector in milliseconds
        time_vector = beat_note.get_time_vector() * 1000

        # Plot 1: Beat note waveform (real part)
        axes[0, 0].plot(time_vector, beat_note.waveform.real, "b-", linewidth=1)
        axes[0, 0].set_title("Beat Note Waveform (Real Part)")
        axes[0, 0].set_xlabel("Time (ms)")
        axes[0, 0].set_ylabel("Amplitude")
        axes[0, 0].grid(True, alpha=0.3)

        # Plot 2: Beat note spectrum
        fft_result = np.fft.fft(beat_note.waveform)
        freqs = np.fft.fftfreq(len(beat_note.waveform), 1.0 / beat_note.sampling_rate)
        positive_freq_idx = freqs > 0
        positive_freqs = freqs[positive_freq_idx]
        positive_magnitude = np.abs(fft_result[positive_freq_idx])

        # Only plot up to 1 kHz for clarity
        plot_idx = positive_freqs <= 1000
        plot_freqs = positive_freqs[plot_idx]
        plot_magnitude = positive_magnitude[plot_idx]

        axes[0, 1].plot(
            plot_freqs, 20 * np.log10(plot_magnitude + 1e-10), "r-", linewidth=1
        )
        axes[0, 1].axvline(
            x=beat_note.get_beat_frequency(),
            color="g",
            linestyle="--",
            label=f"Expected: {beat_note.get_beat_frequency():.1f} Hz",
        )
        axes[0, 1].set_title("Beat Note Spectrum")
        axes[0, 1].set_xlabel("Frequency (Hz)")
        axes[0, 1].set_ylabel("Magnitude (dB)")
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # Plot 3: Instantaneous phase
        _, instantaneous_phase = processor.extract_instantaneous_phase(beat_note)
        axes[1, 0].plot(time_vector, instantaneous_phase, "g-", linewidth=1)
        axes[1, 0].set_title("Instantaneous Phase")
        axes[1, 0].set_xlabel("Time (ms)")
        axes[1, 0].set_ylabel("Phase (rad)")
        axes[1, 0].grid(True, alpha=0.3)

        # Plot 4: Instantaneous frequency
        _, instantaneous_freq = processor.extract_instantaneous_frequency(beat_note)
        axes[1, 1].plot(time_vector, instantaneous_freq, "m-", linewidth=1)
        axes[1, 1].axhline(
            y=beat_note.get_beat_frequency(),
            color="r",
            linestyle="--",
            label=f"Expected: {beat_note.get_beat_frequency():.1f} Hz",
        )
        axes[1, 1].set_title("Instantaneous Frequency")
        axes[1, 1].set_xlabel("Time (ms)")
        axes[1, 1].set_ylabel("Frequency (Hz)")
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()

        # Save plot
        plt.savefig("beat_note_demo.png", dpi=300, bbox_inches="tight")
        print("Plot saved as 'beat_note_demo.png'")

        # Show plot if in interactive environment
        plt.show()

    except ImportError:
        print("Matplotlib not available. Install with: pip install matplotlib")
    except Exception as e:
        print(f"Error generating plots: {e}")


if __name__ == "__main__":
    main()
