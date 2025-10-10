#!/usr/bin/env python3
"""
Hardware Experiment Controller
Part of Driftlock Choir - RF Chronometric Interferometry Hardware Implementation

This script controls the hardware chronometric interferometry demonstration using:
- Two Adafruit Feather M4 Express boards for signal generation
- RTL-SDR for signal capture and analysis
- USB serial communication for timing coordination

The experiment demonstrates chronometric interferometry by:
1. Generating two signals with known frequency offset (100 Hz)
2. Capturing both signals simultaneously with RTL-SDR
3. Performing beat note analysis to extract timing and frequency information
4. Comparing results with theoretical predictions
"""

import argparse
import queue
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Add project root to path to allow importing from src
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.types import Hertz, Seconds, Timestamp
from src.signal_processing.beat_note import BeatNoteProcessor

import matplotlib.pyplot as plt
import numpy as np
import serial

# RTL-SDR imports
try:
    from rtlsdr import RtlSdr

    RTL_SDR_AVAILABLE = True
except ImportError:
    print("Warning: pyrtlsdr not installed. Install with: pip install pyrtlsdr")
    RTL_SDR_AVAILABLE = False

import concurrent.futures

# Signal processing imports
from scipy import signal as scipy_signal
from scipy.fft import fft, fftfreq


class FeatherController:
    """Controls a single Adafruit Feather via USB serial."""

    def __init__(self, port: str, name: str, baudrate: int = 115200):
        """
        Initialize Feather controller.

        Args:
            port: Serial port (e.g., '/dev/ttyACM0' or 'COM3')
            name: Human-readable name for this Feather
            baudrate: Serial communication baudrate
        """
        self.port = port
        self.name = name
        self.baudrate = baudrate
        self.serial_conn = None
        self.is_connected = False
        self.signal_active = False

    def connect(self) -> bool:
        """
        Connect to the Feather.

        Returns:
            True if connection successful
        """
        try:
            print(f"Connecting to {self.name} on {self.port}...")
            self.serial_conn = serial.Serial(self.port, self.baudrate, timeout=2)
            time.sleep(2)  # Allow Arduino to reset

            # Test connection
            response = self.send_command("STATUS")
            if response:
                self.is_connected = True
                print(f"✓ Connected to {self.name}")
                return True
            else:
                print(f"✗ Failed to get response from {self.name}")
                return False

        except Exception as e:
            print(f"✗ Failed to connect to {self.name}: {e}")
            return False

    def disconnect(self):
        """Disconnect from the Feather."""
        if self.serial_conn:
            self.send_command("STOP")  # Stop any active signal generation
            self.serial_conn.close()
            self.is_connected = False
            print(f"Disconnected from {self.name}")

    def send_command(self, command: str) -> Optional[str]:
        """
        Send command to Feather and get response.

        Args:
            command: Command string to send

        Returns:
            Response string or None if failed
        """
        if not self.is_connected or not self.serial_conn:
            return None

        try:
            # Send command
            self.serial_conn.write(f"{command}\n".encode())

            # Read response (with timeout)
            response_lines = []
            start_time = time.time()
            while time.time() - start_time < 3:  # 3 second timeout
                if self.serial_conn.in_waiting > 0:
                    line = self.serial_conn.readline().decode().strip()
                    if line:
                        response_lines.append(line)
                        # Stop reading after status or completion message
                        if any(
                            keyword in line.lower()
                            for keyword in [
                                "complete",
                                "started",
                                "stopped",
                                "active",
                                "mhz",
                            ]
                        ):
                            break
                time.sleep(0.01)

            return "\n".join(response_lines) if response_lines else None

        except Exception as e:
            print(f"Error communicating with {self.name}: {e}")
            return None

    def start_signal(self) -> bool:
        """Start signal generation."""
        response = self.send_command("START")
        if response and "started" in response.lower():
            self.signal_active = True
            return True
        return False

    def stop_signal(self) -> bool:
        """Stop signal generation."""
        response = self.send_command("STOP")
        if response and "stopped" in response.lower():
            self.signal_active = False
            return True
        return False

    def get_status(self) -> Dict:
        """Get detailed status from Feather."""
        response = self.send_command("STATUS")
        status = {
            "name": self.name,
            "connected": self.is_connected,
            "signal_active": False,
            "frequency": "Unknown",
            "runtime": "Unknown",
            "samples": "Unknown",
        }

        if response:
            lines = response.split("\n")
            for line in lines:
                if "active" in line.lower():
                    status["signal_active"] = "active" in line.lower()
                elif "frequency:" in line.lower():
                    status["frequency"] = line.split(":")[1].strip()
                elif "runtime:" in line.lower():
                    status["runtime"] = line.split(":")[1].strip()
                elif "samples:" in line.lower():
                    status["samples"] = line.split(":")[1].strip()

        return status


class RTLSDRController:
    """Controls RTL-SDR for signal capture and analysis."""

    def __init__(self):
        """Initialize RTL-SDR controller."""
        self.sdr = None
        self.is_connected = False
        self.center_freq = 433.05e6  # 433.05 MHz to capture both signals
        self.sample_rate = 2.4e6  # 2.4 MS/s
        self.gain = "auto"

    def connect(self) -> bool:
        """
        Connect to RTL-SDR.

        Returns:
            True if connection successful
        """
        if not RTL_SDR_AVAILABLE:
            print("✗ RTL-SDR library not available")
            return False

        try:
            print("Connecting to RTL-SDR...")
            self.sdr = RtlSdr()

            # Configure RTL-SDR
            self.sdr.sample_rate = self.sample_rate
            self.sdr.center_freq = self.center_freq
            self.sdr.gain = self.gain

            self.is_connected = True
            print(f"✓ RTL-SDR connected")
            print(f"  Center Frequency: {self.center_freq/1e6:.1f} MHz")
            print(f"  Sample Rate: {self.sample_rate/1e6:.1f} MS/s")
            print(f"  Gain: {self.gain}")

            return True

        except Exception as e:
            print(f"✗ Failed to connect to RTL-SDR: {e}")
            return False

    def disconnect(self):
        """Disconnect from RTL-SDR."""
        if self.sdr:
            self.sdr.close()
            self.is_connected = False
            print("Disconnected from RTL-SDR")

    def capture_samples(
        self, duration: float, num_samples: Optional[int] = None
    ) -> np.ndarray:
        """
        Capture samples from RTL-SDR.

        Args:
            duration: Capture duration in seconds
            num_samples: Number of samples (if None, calculated from duration)

        Returns:
            Complex samples array
        """
        if not self.is_connected or not self.sdr:
            raise RuntimeError("RTL-SDR not connected")

        if num_samples is None:
            num_samples = int(self.sample_rate * duration)

        print(f"Capturing {num_samples:,} samples ({duration:.1f}s)...")

        # Capture samples
        samples = self.sdr.read_samples(num_samples)

        print(f"✓ Captured {len(samples):,} samples")
        return samples

    def get_frequency_spectrum(
        self, samples: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate frequency spectrum of samples.

        Args:
            samples: Complex samples

        Returns:
            Tuple of (frequencies, magnitude_db)
        """
        # Calculate FFT
        fft_result = fft(samples)
        freqs = fftfreq(len(samples), 1.0 / self.sample_rate)

        # Shift to center frequency
        freqs = freqs + self.center_freq
        fft_shifted = np.fft.fftshift(fft_result)

        # Convert to dB
        magnitude_db = 20 * np.log10(np.abs(fft_shifted) + 1e-10)

        # Shift frequencies too
        freqs_shifted = np.fft.fftshift(freqs)

        return freqs_shifted, magnitude_db


class BeatNoteAnalyzer:
    """Analyzes beat notes for chronometric interferometry."""

    def __init__(self, sample_rate: float):
        """
        Initialize beat note analyzer.

        Args:
            sample_rate: Sample rate of the captured data
        """
        self.sample_rate = sample_rate

    def extract_beat_signal(
        self, samples: np.ndarray, freq1: float, freq2: float, bandwidth: float = 10000
    ) -> np.ndarray:
        """
        Extract beat signal from captured samples.

        Args:
            samples: Complex samples from RTL-SDR
            freq1: First signal frequency (Hz)
            freq2: Second signal frequency (Hz)
            bandwidth: Filter bandwidth around each signal (Hz)

        Returns:
            Beat signal
        """
        print(
            f"Extracting beat signal from {freq1/1e6:.3f} MHz and {freq2/1e6:.3f} MHz signals..."
        )

        # Create time vector
        t = np.arange(len(samples)) / self.sample_rate

        # Extract each signal using frequency shifting and filtering
        signal1 = self._extract_single_signal(samples, freq1, bandwidth)
        signal2 = self._extract_single_signal(samples, freq2, bandwidth)

        # Generate beat note
        beat_signal = signal1 * np.conj(signal2)

        print(f"✓ Beat signal extracted (length: {len(beat_signal):,} samples)")
        return beat_signal

    def _extract_single_signal(
        self, samples: np.ndarray, center_freq: float, bandwidth: float
    ) -> np.ndarray:
        """Extract a single signal using frequency shifting and filtering."""
        # Create time vector
        t = np.arange(len(samples)) / self.sample_rate

        # Shift signal to baseband
        shift_freq = center_freq - self.sample_rate / 4  # Shift away from DC
        shifted_signal = samples * np.exp(-1j * 2 * np.pi * shift_freq * t)

        # Design low-pass filter
        nyquist = self.sample_rate / 2
        cutoff = bandwidth / 2
        b, a = scipy_signal.butter(4, cutoff / nyquist, btype="low")

        # Apply filter
        filtered_signal = scipy_signal.filtfilt(b, a, shifted_signal)

        return filtered_signal

    def analyze_beat_frequency(self, beat_signal: np.ndarray) -> Dict:
        """
        Analyze beat frequency from beat signal.

        Args:
            beat_signal: Beat signal to analyze

        Returns:
            Analysis results dictionary
        """
        print("Analyzing beat frequency...")

        # Calculate FFT
        fft_result = fft(beat_signal)
        freqs = fftfreq(len(beat_signal), 1.0 / self.sample_rate)

        # Find peak in positive frequencies
        positive_idx = freqs > 0
        positive_freqs = freqs[positive_idx]
        positive_magnitude = np.abs(fft_result[positive_idx])

        # Find peak frequency
        peak_idx = np.argmax(positive_magnitude)
        beat_freq = positive_freqs[peak_idx]
        peak_magnitude = positive_magnitude[peak_idx]

        # Calculate SNR (simplified, as the ratio of the peak signal to the median noise floor)
        noise_floor = np.median(positive_magnitude)
        snr_db = 20 * np.log10(peak_magnitude / (noise_floor + 1e-10))

        # Estimate frequency uncertainty using a simplified model based on SNR.
        # This is an approximation related to the Cramer-Rao Lower Bound (CRLB) for
        # estimating a single tone's frequency in white Gaussian noise, where
        # uncertainty is inversely proportional to the signal-to-noise ratio (SNR).
        # A higher SNR allows for a more precise frequency estimate.
        freq_resolution = self.sample_rate / len(beat_signal)
        snr_linear = 10 ** (snr_db / 10)
        freq_uncertainty = freq_resolution / np.sqrt(snr_linear)

        results = {
            "beat_frequency_hz": beat_freq,
            "frequency_uncertainty_hz": freq_uncertainty,
            "snr_db": snr_db,
            "peak_magnitude": peak_magnitude,
            "noise_floor": noise_floor,
            "frequency_resolution_hz": freq_resolution,
        }

        print(f"✓ Beat frequency: {beat_freq:.1f} ± {freq_uncertainty:.1f} Hz")
        print(f"✓ SNR: {snr_db:.1f} dB")

        return results

    def estimate_timing_offset(self, beat_signal: np.ndarray, beat_freq: float) -> Dict:
        """
        Estimate timing offset from beat signal phase.

        Args:
            beat_signal: Beat signal
            beat_freq: Beat frequency (Hz)

        Returns:
            Timing analysis results
        """
        print("Estimating timing offset...")

        # Extract instantaneous phase
        analytic_signal = scipy_signal.hilbert(beat_signal.real)
        instantaneous_phase = np.unwrap(np.angle(analytic_signal))

        # Create time vector
        t = np.arange(len(beat_signal)) / self.sample_rate

        # Estimate timing offset (τ) from the initial phase (φ₀) of the beat note.
        # The relationship is given by the core chronometry equation: τ = φ₀ / (2π * f_beat).
        # This tells us how much of a head start one signal has over the other,
        # expressed as a time delay.
        initial_phase = instantaneous_phase[0]
        timing_offset_seconds = initial_phase / (2 * np.pi * beat_freq)
        timing_offset_ps = timing_offset_seconds * 1e12

        # Estimate timing uncertainty from the standard deviation of the phase difference (phase noise).
        # A noisier phase measurement leads to a larger uncertainty in the timing estimate.
        # The uncertainty in tau is proportional to the phase uncertainty.
        phase_std = np.std(np.diff(instantaneous_phase))
        timing_uncertainty_ps = abs(phase_std / (2 * np.pi * beat_freq)) * 1e12

        results = {
            "timing_offset_ps": timing_offset_ps,
            "timing_uncertainty_ps": timing_uncertainty_ps,
            "initial_phase_rad": initial_phase,
            "phase_std_rad": phase_std,
        }

        print(
            f"✓ Timing offset: {timing_offset_ps:.1f} ± {timing_uncertainty_ps:.1f} ps"
        )

        return results


class ChronometricInterferometryHardwareExperiment:
    """Main controller for chronometric interferometry demonstration."""

    def __init__(self):
        """Initialize chronometric interferometry demonstration."""
        self.reference_feather = None
        self.offset_feather = None
        self.rtlsdr = None
        self.analyzer = None

        # Experiment parameters
        self.reference_freq = 433.0e6  # 433.0 MHz
        self.offset_freq = 433.1e6  # 433.1 MHz (100 Hz * 433)
        self.expected_beat_freq = 100.0  # 100 Hz
        self.capture_duration = 10.0  # 10 seconds

    def setup_hardware(self, ref_port: str, offset_port: str, dry_run: bool = False) -> bool:
        """
        Setup all hardware connections.

        Args:
            ref_port: Serial port for reference Feather
            offset_port: Serial port for offset Feather
            dry_run: If True, skip actual hardware connections

        Returns:
            True if all connections successful
        """
        if dry_run:
            print("=== Skipping hardware setup (--dry-run mode) ===")
            # Use a default sample rate for the analyzer in dry run mode
            self.analyzer = BeatNoteAnalyzer(2.4e6)
            return True

        print("=== Setting up hardware ===")

        # Connect to Feathers
        self.reference_feather = FeatherController(ref_port, "Reference")
        self.offset_feather = FeatherController(offset_port, "Offset")

        ref_ok = self.reference_feather.connect()
        offset_ok = self.offset_feather.connect()

        if not (ref_ok and offset_ok):
            print("✗ Failed to connect to one or more Feathers")
            return False

        # Connect to RTL-SDR
        self.rtlsdr = RTLSDRController()
        rtl_ok = self.rtlsdr.connect()

        if not rtl_ok:
            print("✗ Failed to connect to RTL-SDR")
            return False

        # Initialize analyzer
        self.analyzer = BeatNoteAnalyzer(self.rtlsdr.sample_rate)

        print("✓ All hardware connected successfully!")
        return True

    def _generate_simulated_samples(self, duration: float) -> np.ndarray:
        """Generates simulated IQ samples for a dry run."""
        # Use the same sample rate as the real hardware
        sample_rate = 2.4e6
        num_samples = int(duration * sample_rate)
        t = np.arange(num_samples) / sample_rate

        # Create two clean signals (complex exponentials)
        ref_signal = np.exp(1j * 2 * np.pi * self.reference_freq * t)

        # Introduce a small, known delay to the offset signal to be detected
        tau_ps = 13.5
        tau_sec = tau_ps * 1e-12
        offset_signal = np.exp(1j * 2 * np.pi * self.offset_freq * (t - tau_sec))

        # Combine them to simulate what the SDR would receive
        combined_signal = ref_signal + offset_signal

        # Add a small amount of noise for realism
        signal_power = np.mean(np.abs(combined_signal) ** 2)
        snr_db = 40.0  # High SNR for a clean dry run
        noise_power = signal_power / (10 ** (snr_db / 10))
        noise = np.random.normal(
            0, np.sqrt(noise_power / 2), num_samples
        ) + 1j * np.random.normal(0, np.sqrt(noise_power / 2), num_samples)

        simulated_samples = combined_signal + noise

        print(
            f"✓ Generated {len(simulated_samples):,} simulated samples with tau={tau_ps} ps."
        )
        return simulated_samples

    def run_experiment(self, dry_run: bool = False) -> Dict:
        """
        Run the complete chronometric interferometry demonstration.

        Args:
            dry_run: If True, use simulated data instead of hardware.

        Returns:
            Experiment results dictionary
        """
        print("\n=== Running Hardware Chronometric Interferometry Demo ===")

        results = {
            "timestamp": datetime.now().isoformat(),
            "success": False,
            "error_message": None,
            "dry_run": dry_run,
        }

        try:
            # Start signal generation (hardware only)
            if not dry_run:
                print("\nInitial hardware status:")
                ref_status = self.reference_feather.get_status()
                offset_status = self.offset_feather.get_status()
                print(
                    f"Reference: {ref_status['frequency']} - {'Active' if ref_status['signal_active'] else 'Idle'}"
                )
                print(
                    f"Offset: {offset_status['frequency']} - {'Active' if offset_status['signal_active'] else 'Idle'}"
                )

                print("\nStarting signal generation...")
                ref_start = self.reference_feather.start_signal()
                offset_start = self.offset_feather.start_signal()

                if not (ref_start and offset_start):
                    results["error_message"] = "Failed to start signal generation"
                    return results

                print("Waiting for signals to stabilize...")
                time.sleep(2)

            # Capture samples (from hardware or simulation)
            if dry_run:
                print(f"\nGenerating simulated samples for {self.capture_duration}s...")
                samples = self._generate_simulated_samples(self.capture_duration)
            else:
                print(f"\nCapturing samples for {self.capture_duration}s...")
                samples = self.rtlsdr.capture_samples(self.capture_duration)

            # Stop signal generation (hardware only)
            if not dry_run:
                print("\nStopping signal generation...")
                self.reference_feather.stop_signal()
                self.offset_feather.stop_signal()

            # Analyze samples
            print("\nAnalyzing captured samples...")
            analysis_results = self._analyze_samples(samples)
            results.update(analysis_results)

            # Generate plots
            print("\nGenerating analysis plots...")
            self._generate_plots(samples, results)

            results["success"] = True
            print("\n✓ Experiment completed successfully!")

        except Exception as e:
            results["error_message"] = str(e)
            print(f"\n✗ Experiment failed: {e}")

        finally:
            # Ensure signals are stopped (hardware only)
            if not dry_run:
                if self.reference_feather:
                    self.reference_feather.stop_signal()
                if self.offset_feather:
                    self.offset_feather.stop_signal()

        return results

    def _analyze_samples(self, samples: np.ndarray) -> Dict:
        """Analyze captured samples for beat note."""
        # Extract beat signal
        beat_signal = self.analyzer.extract_beat_signal(
            samples, self.reference_freq, self.offset_freq
        )

        # Analyze beat frequency
        freq_results = self.analyzer.analyze_beat_frequency(beat_signal)

        # Estimate timing offset
        timing_results = self.analyzer.estimate_timing_offset(
            beat_signal, freq_results["beat_frequency_hz"]
        )

        # Calculate errors compared to expected values
        freq_error = abs(freq_results["beat_frequency_hz"] - self.expected_beat_freq)

        return {
            "beat_signal": beat_signal,
            "frequency_analysis": freq_results,
            "timing_analysis": timing_results,
            "frequency_error_hz": freq_error,
            "expected_beat_freq_hz": self.expected_beat_freq,
        }

    def _generate_plots(self, samples: np.ndarray, analysis: Dict):
        """Generate analysis plots."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(
            "Hardware Chronometric Interferometry Demo - Chronometric Interferometry Analysis",
            fontsize=14,
            fontweight="bold",
        )

        # Plot 1: RF Spectrum
        freqs, magnitude = self.rtlsdr.get_frequency_spectrum(samples)
        axes[0, 0].plot(freqs / 1e6, magnitude)
        axes[0, 0].axvline(
            self.reference_freq / 1e6,
            color="r",
            linestyle="--",
            label=f"Ref: {self.reference_freq/1e6:.1f} MHz",
        )
        axes[0, 0].axvline(
            self.offset_freq / 1e6,
            color="g",
            linestyle="--",
            label=f"Offset: {self.offset_freq/1e6:.1f} MHz",
        )
        axes[0, 0].set_title("RF Spectrum")
        axes[0, 0].set_xlabel("Frequency (MHz)")
        axes[0, 0].set_ylabel("Magnitude (dB)")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Plot 2: Beat Signal Time Domain
        beat_signal = analysis["beat_signal"]
        t_beat = np.arange(len(beat_signal)) / self.rtlsdr.sample_rate
        axes[0, 1].plot(
            t_beat[:10000], beat_signal.real[:10000]
        )  # Show first 10k samples
        axes[0, 1].set_title("Beat Signal (Time Domain)")
        axes[0, 1].set_xlabel("Time (s)")
        axes[0, 1].set_ylabel("Amplitude")
        axes[0, 1].grid(True, alpha=0.3)

        # Plot 3: Beat Signal Spectrum
        beat_fft = fft(beat_signal)
        beat_freqs = fftfreq(len(beat_signal), 1.0 / self.rtlsdr.sample_rate)
        pos_idx = beat_freqs > 0
        axes[1, 0].plot(
            beat_freqs[pos_idx], 20 * np.log10(np.abs(beat_fft[pos_idx]) + 1e-10)
        )
        axes[1, 0].axvline(
            analysis["frequency_analysis"]["beat_frequency_hz"],
            color="r",
            linestyle="--",
            label=f"Measured: {analysis['frequency_analysis']['beat_frequency_hz']:.1f} Hz",
        )
        axes[1, 0].axvline(
            self.expected_beat_freq,
            color="g",
            linestyle="--",
            label=f"Expected: {self.expected_beat_freq:.1f} Hz",
        )
        axes[1, 0].set_title("Beat Frequency Spectrum")
        axes[1, 0].set_xlabel("Frequency (Hz)")
        axes[1, 0].set_ylabel("Magnitude (dB)")
        axes[1, 0].set_xlim(0, 500)  # Focus on beat frequency range
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # Plot 4: Results Summary
        axes[1, 1].axis("off")
        summary_text = f"""
Experiment Results Summary

Beat Frequency:
  Measured: {analysis['frequency_analysis']['beat_frequency_hz']:.1f} ± {analysis['frequency_analysis']['frequency_uncertainty_hz']:.1f} Hz
  Expected: {self.expected_beat_freq:.1f} Hz
  Error: {analysis['frequency_error_hz']:.1f} Hz

Timing Analysis:
  Timing Offset: {analysis['timing_analysis']['timing_offset_ps']:.1f} ± {analysis['timing_analysis']['timing_uncertainty_ps']:.1f} ps
  Initial Phase: {analysis['timing_analysis']['initial_phase_rad']:.3f} rad

Signal Quality:
  SNR: {analysis['frequency_analysis']['snr_db']:.1f} dB
  Frequency Resolution: {analysis['frequency_analysis']['frequency_resolution_hz']:.3f} Hz

Hardware Configuration:
  Reference: {self.reference_freq/1e6:.1f} MHz
  Offset: {self.offset_freq/1e6:.1f} MHz
  Sample Rate: {self.rtlsdr.sample_rate/1e6:.1f} MS/s
  Duration: {self.capture_duration:.1f} s
        """
        axes[1, 1].text(
            0.05,
            0.95,
            summary_text,
            transform=axes[1, 1].transAxes,
            fontsize=10,
            verticalalignment="top",
            fontfamily="monospace",
        )

        plt.tight_layout()

        # Save plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"e1_hardware_experiment_{timestamp}.png"
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        print(f"✓ Analysis plot saved as: {filename}")

        plt.show()

    def cleanup(self, dry_run: bool = False):
        """Clean up hardware connections."""
        if dry_run:
            print("\nSkipping hardware cleanup (--dry-run mode).")
            print("✓ Cleanup complete")
            return

        print("\nCleaning up hardware connections...")

        if self.reference_feather:
            self.reference_feather.disconnect()

        if self.offset_feather:
            self.offset_feather.disconnect()

        if self.rtlsdr:
            self.rtlsdr.disconnect()

        print("✓ Cleanup complete")


def main():
    """Main function for chronometric interferometry demonstration."""
    parser = argparse.ArgumentParser(
        description="Hardware Chronometric Interferometry Demo - Chronometric Interferometry with Feathers and RTL-SDR"
    )
    parser.add_argument(
        "--ref-port",
        help="Serial port for reference Feather (e.g., /dev/ttyACM0 or COM3)",
    )
    parser.add_argument(
        "--offset-port",
        help="Serial port for offset Feather (e.g., /dev/ttyACM1 or COM4)",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=10.0,
        help="Capture duration in seconds (default: 10.0)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run without hardware, using simulated data for testing.",
    )

    args = parser.parse_args()

    # If not a dry run, serial ports are required
    if not args.dry_run:
        if not args.ref_port or not args.offset_port:
            parser.error(
                "--ref-port and --offset-port are required unless --dry-run is specified."
            )

    print("=== Hardware Chronometric Interferometry Demo ===")
    if args.dry_run:
        print("Running in --dry-run mode (no hardware required).")
    else:
        print(f"Reference Feather: {args.ref_port}")
        print(f"Offset Feather: {args.offset_port}")
    print(f"Capture Duration: {args.duration}s")

    # Create experiment instance
    experiment = ChronometricInterferometryHardwareExperiment()
    experiment.capture_duration = args.duration

    try:
        # Setup hardware
        if not experiment.setup_hardware(
            args.ref_port, args.offset_port, dry_run=args.dry_run
        ):
            print("✗ Hardware setup failed")
            return 1

        # Run experiment
        results = experiment.run_experiment(dry_run=args.dry_run)

        # Print final results
        print("\n=== Final Results ===")
        if results["success"]:
            freq_analysis = results["frequency_analysis"]
            timing_analysis = results["timing_analysis"]

            print(
                f"✓ Beat Frequency: {freq_analysis['beat_frequency_hz']:.1f} ± {freq_analysis['frequency_uncertainty_hz']:.1f} Hz"
            )
            print(f"✓ Frequency Error: {results['frequency_error_hz']:.1f} Hz")
            print(
                f"✓ Timing Offset: {timing_analysis['timing_offset_ps']:.1f} ± {timing_analysis['timing_uncertainty_ps']:.1f} ps"
            )
            print(f"✓ SNR: {freq_analysis['snr_db']:.1f} dB")

            # Success criteria
            freq_success = results["frequency_error_hz"] < 10.0  # Within 10 Hz
            snr_success = freq_analysis["snr_db"] > 20.0  # At least 20 dB SNR

            if freq_success and snr_success:
                print(
                    "\n🎉 EXPERIMENT SUCCESS! Chronometric interferometry demonstrated!"
                )
            else:
                print(f"\n⚠️  Marginal results - check setup and signal levels")
        else:
            print(
                f"✗ Experiment failed: {results.get('error_message', 'Unknown error')}"
            )

        return 0 if results["success"] else 1

    except KeyboardInterrupt:
        print("\n\nExperiment interrupted by user")
        return 1

    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        return 1

    finally:
        experiment.cleanup(dry_run=args.dry_run)


if __name__ == "__main__":
    sys.exit(main())
