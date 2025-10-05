#!/usr/bin/env python3
"""
Debug script to understand beat frequency calculation issues.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))

import numpy as np
from src.signal_processing.oscillator import Oscillator
from src.signal_processing.beat_note import BeatNoteProcessor
from src.core.types import Hertz, Seconds, Timestamp

def debug_beat_frequency():
    """Debug beat frequency calculation."""
    print("=== Beat Frequency Debug ===")
    
    # Parameters - using realistic frequencies
    tx_freq = Hertz(100e3)  # 100 kHz
    rx_freq_nominal = Hertz(100e3 + 100.0)  # 100 kHz + 100 Hz offset
    rx_freq_additional = Hertz(50.0)  # Additional 50 Hz offset
    sampling_rate = Hertz(1e6)  # 1 MS/s
    duration = Seconds(0.001)  # 1 ms
    
    print(f"TX frequency: {tx_freq:.1f} Hz")
    print(f"RX frequency (nominal): {rx_freq_nominal:.1f} Hz")
    print(f"RX frequency (additional): {rx_freq_additional:.1f} Hz")
    print(f"Expected total offset: {100.0 + 50.0} Hz")
    
    # Create oscillators
    tx_oscillator = Oscillator(Oscillator.create_ideal_oscillator(tx_freq))
    rx_oscillator = Oscillator(Oscillator.create_ideal_oscillator(rx_freq_nominal))
    
    # Generate signals
    tx_time, tx_signal = tx_oscillator.generate_signal(
        duration=duration,
        sampling_rate=sampling_rate,
        frequency_offset=Hertz(0.0),
        phase_noise_enabled=False
    )
    
    rx_time, rx_signal = rx_oscillator.generate_signal(
        duration=duration,
        sampling_rate=sampling_rate,
        frequency_offset=rx_freq_additional,
        phase_noise_enabled=False
    )
    
    print(f"Generated {len(tx_signal)} samples per signal")
    
    # Create beat note
    processor = BeatNoteProcessor(sampling_rate)
    timestamp = Timestamp.from_ps(0.0)
    
    beat_note = processor.generate_beat_note(
        tx_signal=tx_signal,
        rx_signal=rx_signal,
        tx_frequency=tx_freq,
        rx_frequency=rx_freq_nominal,
        duration=duration,
        timestamp=timestamp,
        add_noise=False,
        snr_db=40.0
    )
    
    # Get beat frequency from BeatNoteData
    calculated_beat_freq = beat_note.get_beat_frequency()
    print(f"Beat frequency from BeatNoteData: {calculated_beat_freq:.1f} Hz")
    
    # Extract beat frequency using processor
    measured_beat_freq, freq_uncertainty = processor.extract_beat_frequency(beat_note)
    print(f"Measured beat frequency: {measured_beat_freq:.1f} ± {freq_uncertainty:.1f} Hz")
    
    # Manual FFT analysis
    fft_result = np.fft.fft(beat_note.waveform)
    freqs = np.fft.fftfreq(len(beat_note.waveform), 1.0 / sampling_rate)
    
    # Find peak in positive frequencies
    positive_freq_idx = freqs > 0
    positive_freqs = freqs[positive_freq_idx]
    positive_magnitude = np.abs(fft_result[positive_freq_idx])
    
    peak_idx = np.argmax(positive_magnitude)
    peak_freq = positive_freqs[peak_idx]
    
    print(f"FFT peak frequency: {peak_freq:.1f} Hz")
    
    # Look at frequency spectrum around beat frequency
    print("\nFrequency spectrum analysis:")
    freq_range = 2000  # Look at ±1 kHz around expected beat
    freq_mask = np.abs(positive_freqs) < freq_range
    plot_freqs = positive_freqs[freq_mask]
    plot_magnitudes = positive_magnitude[freq_mask]
    
    # Find top 5 peaks
    sorted_indices = np.argsort(plot_magnitudes)[-5:]
    print("Top 5 frequency peaks:")
    for i, idx in enumerate(reversed(sorted_indices)):
        freq = plot_freqs[idx]
        mag = plot_magnitudes[idx]
        print(f"  {i+1}. {freq:.1f} Hz (magnitude: {mag:.2e})")
    
    # Check the actual beat signal properties
    print(f"\nBeat signal properties:")
    print(f"Signal length: {len(beat_note.waveform)} samples")
    print(f"Signal duration: {duration:.6f} seconds")
    print(f"Sampling rate: {sampling_rate:.1f} Hz")
    
    # Simple beat frequency calculation
    # Beat frequency should be |f_tx - f_rx_effective|
    f_tx = tx_freq
    f_rx_effective = rx_freq_nominal + rx_freq_additional
    expected_beat = abs(f_tx - f_rx_effective)
    print(f"Expected beat frequency: {expected_beat:.1f} Hz")
    
    # Check if frequencies are too close to the sampling rate
    print(f"\nFrequency analysis:")
    print(f"TX freq / sampling rate: {tx_freq / sampling_rate:.2e}")
    print(f"RX freq / sampling rate: {(rx_freq_nominal + rx_freq_additional) / sampling_rate:.2e}")
    
    # Check signal properties
    print(f"\nSignal analysis:")
    print(f"TX signal mean magnitude: {np.mean(np.abs(tx_signal)):.6f}")
    print(f"RX signal mean magnitude: {np.mean(np.abs(rx_signal)):.6f}")
    print(f"Beat signal mean magnitude: {np.mean(np.abs(beat_note.waveform)):.6f}")
    
    # Look at the phase difference between signals
    phase_diff = np.angle(rx_signal) - np.angle(tx_signal)
    unwrapped_phase_diff = np.unwrap(phase_diff)
    
    # Estimate frequency from phase derivative
    dt = 1.0 / sampling_rate
    phase_derivative = np.gradient(unwrapped_phase_diff, dt)
    instantaneous_freq_diff = phase_derivative / (2 * np.pi)
    
    print(f"\nPhase analysis:")
    print(f"Mean instantaneous frequency difference: {np.mean(instantaneous_freq_diff):.1f} Hz")
    print(f"Std instantaneous frequency difference: {np.std(instantaneous_freq_diff):.1f} Hz")
    
    # The issue might be aliasing due to undersampling
    print(f"\nAliasing analysis:")
    print(f"Nyquist frequency: {sampling_rate / 2:.1f} Hz")
    print(f"Carrier frequencies are {tx_freq / (sampling_rate / 2):.0f}x higher than Nyquist")
    print(f"This causes aliasing - the baseband signal may not represent the true beat frequency")
    
    return measured_beat_freq, expected_beat

if __name__ == "__main__":
    debug_beat_frequency()