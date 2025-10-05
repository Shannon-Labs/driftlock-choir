"""
Channel simulation for wireless propagation modeling.
"""

import numpy as np
from typing import List, Tuple, Optional
from ..core.types import (
    ChannelModel, Hertz, Seconds, Picoseconds, Meters,
    MeasurementQuality, BeatNoteData
)
from ..core.constants import PhysicalConstants


class ChannelSimulator:
    """
    Simulates wireless propagation channels for beat-note experiments.
    
    This class implements various channel models including AWGN, multipath,
    and Doppler effects to simulate realistic wireless propagation conditions.
    """
    
    def __init__(self, sampling_rate: Hertz):
        """
        Initialize channel simulator.
        
        Args:
            sampling_rate: Sampling rate for signal processing
        """
        self.sampling_rate = sampling_rate
    
    def apply_channel(self, 
                     signal: np.ndarray,
                     channel_model: ChannelModel,
                     tx_frequency: Hertz) -> np.ndarray:
        """
        Apply channel effects to a signal.
        
        Args:
            signal: Input signal
            channel_model: Channel model to apply
            tx_frequency: Transmit frequency
            
        Returns:
            Signal with channel effects applied
        """
        # Get impulse response
        impulse_response = channel_model.get_impulse_response(self.sampling_rate)
        
        # Apply multipath using convolution
        if len(impulse_response) > 1:
            output_signal = np.convolve(signal, impulse_response, mode='same')
        else:
            output_signal = signal.copy()
        
        # Apply Doppler shift
        if channel_model.doppler_shift != 0:
            output_signal = self._apply_doppler_shift(output_signal, channel_model.doppler_shift)
        
        # Apply path loss (simple free space model)
        if len(channel_model.path_delays) > 0:
            # Use first path distance for path loss calculation
            first_path_distance = PhysicalConstants.ps_to_meters(channel_model.path_delays[0])
            path_loss_db = self._calculate_free_space_path_loss(first_path_distance, tx_frequency)
            output_signal = self._apply_path_loss(output_signal, path_loss_db)
        
        return output_signal
    
    def create_awgn_channel(self, 
                           delay: Picoseconds = 0.0,
                           noise_figure_db: float = 5.0) -> ChannelModel:
        """
        Create an AWGN (Additive White Gaussian Noise) channel model.
        
        Args:
            delay: Propagation delay in picoseconds
            noise_figure_db: Noise figure in dB
            
        Returns:
            AWGN channel model
        """
        return ChannelModel(
            delay_spread=Picoseconds(0.0),
            path_delays=[delay],
            path_gains=[1.0],
            doppler_shift=Hertz(0.0),
            temperature=25.0,
            humidity=50.0
        )
    
    def create_multipath_channel(self,
                                delays: List[Picoseconds],
                                powers: List[float],
                                doppler_shift: Hertz = 0.0) -> ChannelModel:
        """
        Create a multipath channel model.
        
        Args:
            delays: List of path delays in picoseconds
            powers: List of relative path powers (linear scale)
            doppler_shift: Doppler shift in Hz
            
        Returns:
            Multipath channel model
        """
        if len(delays) != len(powers):
            raise ValueError("Delays and powers must have same length")
        
        # Convert powers to linear gains
        gains = [np.sqrt(p) for p in powers]
        
        # Calculate delay spread
        if len(delays) > 1:
            delay_mean = np.mean(delays)
            delay_spread = Picoseconds(np.sqrt(np.mean([(d - delay_mean)**2 for d in delays])))
        else:
            delay_spread = Picoseconds(0.0)
        
        return ChannelModel(
            delay_spread=delay_spread,
            path_delays=delays,
            path_gains=gains,
            doppler_shift=doppler_shift,
            temperature=25.0,
            humidity=50.0
        )
    
    def create_indoor_channel(self, 
                            distance_m: float,
                            tx_frequency: Hertz) -> ChannelModel:
        """
        Create a typical indoor multipath channel model.
        
        Args:
            distance_m: Distance in meters
            tx_frequency: Transmit frequency
            
        Returns:
            Indoor channel model
        """
        # Convert distance to delay
        delay_ps = PhysicalConstants.meters_to_ps(distance_m)
        
        # Typical indoor multipath profile
        # Direct path + several reflections with decreasing power
        delays = [
            delay_ps,  # Direct path
            delay_ps + 50,  # First reflection
            delay_ps + 120,  # Second reflection
            delay_ps + 200,  # Third reflection
            delay_ps + 350,  # Fourth reflection
        ]
        
        # Power decreases with path length (simplified model)
        powers = [
            1.0,      # Direct path
            0.3,      # First reflection
            0.1,      # Second reflection
            0.05,     # Third reflection
            0.02,     # Fourth reflection
        ]
        
        return self.create_multipath_channel(delays, powers)
    
    def create_outdoor_channel(self,
                             distance_m: float,
                             tx_frequency: Hertz) -> ChannelModel:
        """
        Create a typical outdoor multipath channel model.
        
        Args:
            distance_m: Distance in meters
            tx_frequency: Transmit frequency
            
        Returns:
            Outdoor channel model
        """
        # Convert distance to delay
        delay_ps = PhysicalConstants.meters_to_ps(distance_m)
        
        # Typical outdoor multipath profile (fewer reflections, longer delays)
        delays = [
            delay_ps,  # Direct path
            delay_ps + 200,  # Ground reflection
            delay_ps + 500,  # Building reflection
        ]
        
        # Outdoor typically has stronger direct path
        powers = [
            1.0,      # Direct path
            0.2,      # Ground reflection
            0.05,     # Building reflection
        ]
        
        return self.create_multipath_channel(delays, powers)
    
    def _apply_doppler_shift(self, signal: np.ndarray, doppler_shift: Hertz) -> np.ndarray:
        """
        Apply Doppler shift to signal.
        
        Args:
            signal: Input signal
            doppler_shift: Doppler shift in Hz
            
        Returns:
            Signal with Doppler shift applied
        """
        n_samples = len(signal)
        t = np.arange(n_samples) / self.sampling_rate
        
        # Create Doppler phase shift
        doppler_phase = 2 * np.pi * doppler_shift * t
        
        # Apply phase shift
        return signal * np.exp(1j * doppler_phase)
    
    def _calculate_free_space_path_loss(self, distance: Meters, frequency: Hertz) -> float:
        """
        Calculate free space path loss.
        
        Args:
            distance: Distance in meters
            frequency: Frequency in Hz
            
        Returns:
            Path loss in dB
        """
        if distance <= 0:
            return 0.0
        
        # FSPL = 20*log10(d) + 20*log10(f) - 147.55
        path_loss_db = 20 * np.log10(distance) + 20 * np.log10(frequency) - 147.55
        return max(0.0, path_loss_db)
    
    def _apply_path_loss(self, signal: np.ndarray, path_loss_db: float) -> np.ndarray:
        """
        Apply path loss to signal.
        
        Args:
            signal: Input signal
            path_loss_db: Path loss in dB
            
        Returns:
            Signal with path loss applied
        """
        gain_linear = 10 ** (-path_loss_db / 20)
        return signal * gain_linear
    
    def add_thermal_noise(self,
                         signal: np.ndarray,
                         snr_db: float = None,
                         noise_figure_db: float = 5.0,
                         temperature_kelvin: float = 290.0) -> np.ndarray:
        """
        Add thermal noise to signal.
        
        Args:
            signal: Input signal
            snr_db: Target SNR in dB (if provided, overrides noise_figure calculation)
            noise_figure_db: Noise figure in dB
            temperature_kelvin: Temperature in Kelvin
            
        Returns:
            Signal with thermal noise added
        """
        if snr_db is not None:
            # Use SNR-based calculation
            signal_power = np.mean(np.abs(signal) ** 2)
            noise_power = signal_power / (10 ** (snr_db / 10))
        else:
            # Use traditional thermal noise calculation
            bandwidth_hz = self.sampling_rate / 2  # Nyquist bandwidth
            noise_power_dbm = PhysicalConstants.THERMAL_NOISE_DBM_PER_HZ + 10 * np.log10(bandwidth_hz) + noise_figure_db
            noise_power_linear = 10 ** ((noise_power_dbm - 30) / 10)  # Convert to watts
            
            # Convert to signal units (assuming normalized signal)
            signal_power = np.mean(np.abs(signal) ** 2)
            if signal_power > 0:
                noise_power = noise_power_linear / signal_power
            else:
                noise_power = noise_power_linear
        
        # Generate complex Gaussian noise
        noise = np.sqrt(noise_power / 2) * (
            np.random.normal(0, 1, len(signal)) + 
            1j * np.random.normal(0, 1, len(signal))
        )
        
        return signal + noise

    
    def estimate_channel_from_beat_note(self, beat_note: BeatNoteData) -> ChannelModel:
        """
        Estimate channel parameters from beat note data.
        
        This is a simplified channel estimation - full implementation
        would use more sophisticated techniques.
        
        Args:
            beat_note: Beat note data
            
        Returns:
            Estimated channel model
        """
        # Get impulse response from beat note
        analytic_signal = beat_note.get_analytic_signal()
        
        # Use inverse FFT to get impulse response
        impulse_response = np.fft.ifft(analytic_signal)
        
        # Find significant paths
        threshold = 0.1 * np.max(np.abs(impulse_response))
        significant_indices = np.where(np.abs(impulse_response) > threshold)[0]
        
        if len(significant_indices) > 0:
            # Convert indices to delays
            delays = [PhysicalConstants.seconds_to_ps(idx / self.sampling_rate) 
                     for idx in significant_indices]
            gains = [impulse_response[idx] for idx in significant_indices]
            powers = [np.abs(g)**2 for g in gains]
            
            # Calculate delay spread
            if len(delays) > 1:
                delay_mean = np.mean(delays)
                delay_spread = Picoseconds(np.sqrt(np.mean([(d - delay_mean)**2 for d in delays])))
            else:
                delay_spread = Picoseconds(0.0)
            
            return ChannelModel(
                delay_spread=delay_spread,
                path_delays=delays,
                path_gains=[np.sqrt(p) for p in powers],
                doppler_shift=Hertz(0.0),  # Would need more analysis to estimate
                temperature=25.0,
                humidity=50.0
            )
        else:
            # No significant paths found, return AWGN channel
            return self.create_awgn_channel()
    
    def get_channel_impulse_response(self, channel_model: ChannelModel) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get channel impulse response.
        
        Args:
            channel_model: Channel model
            
        Returns:
            Tuple of (time_vector, impulse_response)
        """
        impulse_response = channel_model.get_impulse_response(self.sampling_rate)
        time_vector = np.arange(len(impulse_response)) / self.sampling_rate
        
        return time_vector, impulse_response
    
    def get_channel_frequency_response(self, channel_model: ChannelModel) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get channel frequency response.
        
        Args:
            channel_model: Channel model
            
        Returns:
            Tuple of (frequency_vector, frequency_response)
        """
        impulse_response = channel_model.get_impulse_response(self.sampling_rate)
        frequency_response = np.fft.fft(impulse_response)
        frequency_vector = np.fft.fftfreq(len(impulse_response), 1.0 / self.sampling_rate)
        
        # Return only positive frequencies
        positive_freq_idx = frequency_vector >= 0
        return frequency_vector[positive_freq_idx], frequency_response[positive_freq_idx]