"""
Beat note processing for chronometric interferometry.
"""

import numpy as np
from typing import Tuple, Optional, List
from scipy import signal as scipy_signal
from scipy.fft import fft, fftfreq, fftshift

from ..core.types import (
    BeatNoteData, Hertz, Seconds, Picoseconds, Timestamp,
    MeasurementQuality, EstimationResult, PhaseMeasurement
)
from ..core.constants import PhysicalConstants


class BeatNoteProcessor:
    """
    Processes beat-note signals for chronometric interferometry.
    
    This class implements the core signal processing for extracting
    timing and frequency information from two-way beat-note exchanges.
    """
    
    def __init__(self, sampling_rate: Hertz):
        """
        Initialize beat note processor.
        
        Args:
            sampling_rate: Sampling rate for signal processing
        """
        self.sampling_rate = sampling_rate
        
    def generate_beat_note(self,
                          tx_signal: np.ndarray,
                          rx_signal: np.ndarray,
                          tx_frequency: Hertz,
                          rx_frequency: Hertz,
                          duration: Seconds,
                          timestamp: Timestamp,
                          add_noise: bool = True,
                          snr_db: float = 30.0) -> BeatNoteData:
        """
        Generate beat-note signal from two oscillators.
        
        Args:
            tx_signal: Transmit oscillator signal
            rx_signal: Receive oscillator signal  
            tx_frequency: Transmit frequency
            rx_frequency: Receive frequency
            duration: Signal duration
            timestamp: Timestamp for measurement
            add_noise: Whether to add noise
            snr_db: Signal-to-noise ratio in dB
            
        Returns:
            Beat note data
        """
        if len(tx_signal) != len(rx_signal):
            raise ValueError("TX and RX signals must have same length")
        
        # Generate beat note by mixing signals
        beat_signal = tx_signal * np.conj(rx_signal)
        
        # Add noise if requested
        if add_noise:
            beat_signal = self._add_noise(beat_signal, snr_db)
        
        # Determine quality based on SNR
        quality = self._determine_quality_from_snr(snr_db)
        
        return BeatNoteData(
            tx_frequency=tx_frequency,
            rx_frequency=rx_frequency,
            sampling_rate=self.sampling_rate,
            duration=duration,
            waveform=beat_signal,
            timestamp=timestamp,
            snr=snr_db,
            quality=quality
        )
    
    def extract_beat_frequency(self, beat_note: BeatNoteData) -> Tuple[Hertz, Hertz]:
        """
        Extract beat frequency from beat note using FFT.
        
        Args:
            beat_note: Beat note data
            
        Returns:
            Tuple of (beat_frequency, frequency_uncertainty)
        """
        # Compute FFT
        n_samples = len(beat_note.waveform)
        fft_result = fft(beat_note.waveform)
        freqs = fftfreq(n_samples, 1.0 / beat_note.sampling_rate)
        
        # Find peak frequency (positive frequencies only)
        positive_freq_idx = freqs > 0
        positive_freqs = freqs[positive_freq_idx]
        positive_magnitude = np.abs(fft_result[positive_freq_idx])
        
        # Find peak
        peak_idx = np.argmax(positive_magnitude)
        beat_freq = positive_freqs[peak_idx]
        
        # Estimate uncertainty (simplified)
        # Use Cramér-Rao bound for frequency estimation
        snr_linear = 10 ** (beat_note.snr / 10)
        freq_uncertainty = beat_note.sampling_rate / (2 * np.pi * np.sqrt(snr_linear) * n_samples)
        
        return Hertz(beat_freq), Hertz(freq_uncertainty)
    
    def extract_instantaneous_phase(self, beat_note: BeatNoteData) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract instantaneous phase and frequency from beat note.
        
        Args:
            beat_note: Beat note data
            
        Returns:
            Tuple of (time_vector, instantaneous_phase)
        """
        # Get analytic signal using Hilbert transform
        analytic_signal = scipy_signal.hilbert(beat_note.waveform.real)
        
        # Extract instantaneous phase
        instantaneous_phase = np.unwrap(np.angle(analytic_signal))
        
        # Get time vector
        time_vector = beat_note.get_time_vector()
        
        return time_vector, instantaneous_phase
    
    def extract_instantaneous_frequency(self, beat_note: BeatNoteData) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract instantaneous frequency from beat note.
        
        Args:
            beat_note: Beat note data
            
        Returns:
            Tuple of (time_vector, instantaneous_frequency)
        """
        time_vector, instantaneous_phase = self.extract_instantaneous_phase(beat_note)
        
        # Differentiate phase to get frequency
        dt = time_vector[1] - time_vector[0] if len(time_vector) > 1 else 1.0
        instantaneous_freq = np.gradient(instantaneous_phase, dt) / (2 * np.pi)
        
        return time_vector, instantaneous_freq
    
    def estimate_tau_delta_f(self, 
                           beat_note: BeatNoteData,
                           method: str = "phase_slope") -> EstimationResult:
        """
        Estimate time-of-flight (τ) and frequency offset (Δf) from beat note.
        
        Args:
            beat_note: Beat note data
            method: Estimation method ("phase_slope", "fft_peak", "ml")
            
        Returns:
            Estimation result
        """
        if method == "phase_slope":
            return self._estimate_phase_slope(beat_note)
        elif method == "fft_peak":
            return self._estimate_fft_peak(beat_note)
        elif method == "ml":
            return self._estimate_maximum_likelihood(beat_note)
        else:
            raise ValueError(f"Unknown estimation method: {method}")
    
    def _estimate_phase_slope(self, beat_note: BeatNoteData) -> EstimationResult:
        """
        Estimate τ and Δf using phase slope method.
        
        This method uses the linear relationship between phase and frequency
        to estimate time-of-flight from the phase slope.
        """
        # Get instantaneous frequency
        time_vector, instantaneous_freq = self.extract_instantaneous_frequency(beat_note)
        
        # Expected beat frequency
        expected_beat_freq = beat_note.get_beat_frequency()
        
        # Estimate frequency offset (Δf) as mean deviation from expected beat frequency
        delta_f = np.mean(instantaneous_freq - expected_beat_freq)
        delta_f_uncertainty = np.std(instantaneous_freq - expected_beat_freq) / np.sqrt(len(instantaneous_freq))
        
        # For phase slope method, we need multi-frequency data
        # With single frequency, we can only estimate τ from the mean phase
        # This is a simplified version - full implementation needs multiple frequencies
        
        # Get instantaneous phase
        _, instantaneous_phase = self.extract_instantaneous_phase(beat_note)
        
        # Use mean phase offset to estimate τ
        # τ = φ / (2π * f_beat)
        mean_phase = np.mean(instantaneous_phase)
        beat_freq = beat_note.get_beat_frequency()
        
        if beat_freq > 0:
            tau_seconds = mean_phase / (2 * np.pi * beat_freq)
            tau_ps = PhysicalConstants.seconds_to_ps(tau_seconds)
            
            # Uncertainty estimate (simplified)
            phase_uncertainty = np.std(instantaneous_phase) / np.sqrt(len(instantaneous_phase))
            tau_uncertainty_ps = abs(phase_uncertainty / (2 * np.pi * beat_freq)) * PhysicalConstants.PS_PER_SEC
        else:
            tau_ps = 0.0
            tau_uncertainty_ps = float('inf')
        
        # Covariance matrix (simplified)
        covariance = np.array([
            [tau_uncertainty_ps**2, 0],
            [0, delta_f_uncertainty**2]
        ])
        
        # Likelihood (simplified)
        likelihood = min(1.0, 10 ** (beat_note.snr / 20) / 100.0)
        
        return EstimationResult(
            tau=Picoseconds(tau_ps),
            tau_uncertainty=Picoseconds(tau_uncertainty_ps),
            delta_f=Hertz(delta_f),
            delta_f_uncertainty=Hertz(delta_f_uncertainty),
            covariance=covariance,
            likelihood=likelihood,
            quality=beat_note.quality,
            method="phase_slope",
            timestamp=beat_note.timestamp
        )
    
    def _estimate_fft_peak(self, beat_note: BeatNoteData) -> EstimationResult:
        """
        Estimate τ and Δf using FFT peak method.
        
        This method uses the peak frequency from FFT to estimate Δf
        and the phase at that frequency to estimate τ.
        """
        # Extract beat frequency
        beat_freq, freq_uncertainty = self.extract_beat_frequency(beat_note)
        
        # Expected beat frequency
        expected_beat_freq = beat_note.get_beat_frequency()
        
        # Frequency offset is difference from expected
        delta_f = beat_freq - expected_beat_freq
        
        # Get phase at beat frequency
        n_samples = len(beat_note.waveform)
        fft_result = fft(beat_note.waveform)
        freqs = fftfreq(n_samples, 1.0 / beat_note.sampling_rate)
        
        # Find peak index
        peak_idx = np.argmin(np.abs(freqs - beat_freq))
        phase_at_peak = np.angle(fft_result[peak_idx])
        
        # Estimate τ from phase
        if beat_freq > 0:
            tau_seconds = phase_at_peak / (2 * np.pi * beat_freq)
            tau_ps = PhysicalConstants.seconds_to_ps(tau_seconds)
            
            # Uncertainty estimate
            tau_uncertainty_ps = abs(1.0 / (2 * np.pi * beat_freq)) * PhysicalConstants.PS_PER_SEC
        else:
            tau_ps = 0.0
            tau_uncertainty_ps = float('inf')
        
        # Covariance matrix
        covariance = np.array([
            [tau_uncertainty_ps**2, 0],
            [0, freq_uncertainty**2]
        ])
        
        # Likelihood based on FFT peak magnitude
        peak_magnitude = np.abs(fft_result[peak_idx])
        total_magnitude = np.sum(np.abs(fft_result))
        likelihood = min(1.0, peak_magnitude / total_magnitude)
        
        return EstimationResult(
            tau=Picoseconds(tau_ps),
            tau_uncertainty=Picoseconds(tau_uncertainty_ps),
            delta_f=Hertz(delta_f),
            delta_f_uncertainty=Hertz(freq_uncertainty),
            covariance=covariance,
            likelihood=likelihood,
            quality=beat_note.quality,
            method="fft_peak",
            timestamp=beat_note.timestamp
        )
    
    def _estimate_maximum_likelihood(self, beat_note: BeatNoteData) -> EstimationResult:
        """
        Estimate τ and Δf using maximum likelihood estimation.
        
        This is a simplified ML estimator - full implementation would
        involve more sophisticated optimization.
        """
        # For now, fall back to FFT peak method
        # In a full implementation, this would use numerical optimization
        # to find the τ and Δf that maximize the likelihood function
        return self._estimate_fft_peak(beat_note)
    
    def _add_noise(self, signal: np.ndarray, snr_db: float) -> np.ndarray:
        """
        Add complex Gaussian noise to achieve desired SNR.
        
        Args:
            signal: Input signal
            snr_db: Desired SNR in dB
            
        Returns:
            Signal with added noise
        """
        signal_power = np.mean(np.abs(signal) ** 2)
        snr_linear = 10 ** (snr_db / 10)
        noise_power = signal_power / snr_linear
        
        # Generate complex Gaussian noise
        noise = np.sqrt(noise_power / 2) * (
            np.random.normal(0, 1, len(signal)) + 
            1j * np.random.normal(0, 1, len(signal))
        )
        
        return signal + noise
    
    def _determine_quality_from_snr(self, snr_db: float) -> MeasurementQuality:
        """
        Determine measurement quality from SNR.
        
        Args:
            snr_db: Signal-to-noise ratio in dB
            
        Returns:
            Measurement quality
        """
        if snr_db >= 40:
            return MeasurementQuality.EXCELLENT
        elif snr_db >= 30:
            return MeasurementQuality.GOOD
        elif snr_db >= 20:
            return MeasurementQuality.FAIR
        elif snr_db >= 10:
            return MeasurementQuality.POOR
        else:
            return MeasurementQuality.INVALID
    
    def get_snr_estimate(self, beat_note: BeatNoteData) -> float:
        """
        Estimate SNR from beat note signal.
        
        Args:
            beat_note: Beat note data
            
        Returns:
            Estimated SNR in dB
        """
        # Use signal power vs noise power in frequency domain
        fft_result = fft(beat_note.waveform)
        freqs = fftfreq(len(beat_note.waveform), 1.0 / beat_note.sampling_rate)
        
        # Find signal band (around beat frequency)
        beat_freq = beat_note.get_beat_frequency()
        signal_bandwidth = 1000.0  # Hz
        signal_mask = np.abs(freqs - beat_freq) < signal_bandwidth
        
        # Signal power
        signal_power = np.mean(np.abs(fft_result[signal_mask]) ** 2)
        
        # Noise power (away from signal)
        noise_mask = ~signal_mask
        noise_power = np.mean(np.abs(fft_result[noise_mask]) ** 2)
        
        if noise_power > 0:
            snr_linear = signal_power / noise_power
            snr_db = 10 * np.log10(snr_linear)
        else:
            snr_db = 60.0  # Very high SNR
        
        return snr_db