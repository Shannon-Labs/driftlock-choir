#!/usr/bin/env python3
"""
Audio representation of chronomagnetic interference for Experiment E1.
This script generates audio demonstrations of beat-note formation in chronometric interferometry,
showing how information 'materializes' at specific frequencies representing τ and Δf measurements.
"""
import numpy as np
import soundfile as sf
from scipy import signal
import os

def generate_e1_beat_note_audio(duration=8.0, sample_rate=44100):
    """
    Generate audio representing E1 beat-note formation in chronometric interferometry.
    
    This simulates how two oscillators with a slight frequency offset create beat patterns
    that represent the fundamental principle of τ (time-of-flight) and Δf (frequency offset) 
    measurements in chronometric interferometry.
    """
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    
    # Parameters based on E1 experiment
    tx_frequency = 2442e6  # 2442 MHz - GPS L1 band
    rx_frequency = 2442e6 + 100.0  # +100 Hz offset
    beat_frequency = abs(rx_frequency - tx_frequency)  # 100 Hz beat frequency
    
    # Instead of using the extremely high RF frequencies directly,
    # we'll represent the beat phenomenon at audible frequencies
    # Scale down the concept to audible range
    audible_beat_freq = 100.0  # Hz - actual beat frequency from E1
    carrier_freq = 440.0  # A4 note - representing the RF carrier concept
    
    # Create the beat pattern as it would occur in E1 experiment
    high_freq_carrier = np.sin(2 * np.pi * carrier_freq * t)
    beat_envelope = np.sin(2 * np.pi * audible_beat_freq * t)
    
    # The actual beat note is the carrier modulated by the envelope
    beat_note_signal = high_freq_carrier * (1 + 0.5 * beat_envelope)
    
    # Add another component to represent the time delay (τ) in E1
    # This is represented as a lower frequency modulation
    tau_component_freq = 2.0  # 2 Hz representing time-of-flight effects
    tau_modulation = np.sin(2 * np.pi * tau_component_freq * t)
    beat_note_signal *= (1 + 0.3 * tau_modulation)
    
    # Apply envelope for smooth start/end
    envelope = np.ones_like(t)
    attack_time = int(0.1 * sample_rate)
    release_time = int(0.2 * sample_rate)
    envelope[:attack_time] = np.linspace(0, 1, attack_time)
    envelope[-release_time:] = np.linspace(1, 0, release_time)
    
    beat_note_signal = beat_note_signal * envelope
    
    return beat_note_signal, sample_rate

def generate_chronomagnetic_pulse_at_e1_frequency(duration=10.0, sample_rate=44100):
    """
    Generate audio pulses at E1-relevant frequencies to demonstrate how
    information manifests at specific temporal frequencies in chronometric interferometry.
    """
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    
    # Parameters based on E1 experiment
    # In E1, beat frequencies around 100 Hz are significant
    # These represent the information "materializing" at specific rates
    base_beat_freq = 100.0  # Hz - from E1 experiment
    delta_f_component = 50.0  # Hz - representing frequency offset Δf
    
    # Create pulses representing information arrival at E1-relevant frequencies
    pulse_signal = np.zeros_like(t)
    
    # Simulate the concept of information "materializing" at specific temporal frequencies
    # This creates the "out of tune" effect you're looking for
    for harmonic in [1, 2, 3]:
        freq = base_beat_freq * harmonic
        carrier = np.sin(2 * np.pi * freq * t)
        
        # Create pulse-like modulation representing information arrival
        pulse_env = np.abs(np.sin(2 * np.pi * delta_f_component * t))
        pulse_env = (pulse_env + 0.1) / 1.1  # Normalize to [0.1, 1]
        
        harmonic_signal = carrier * pulse_env
        pulse_signal += harmonic_signal * (1.0 / harmonic)  # Reduce higher harmonics
    
    # Apply envelope
    envelope = np.ones_like(t)
    attack_time = int(0.2 * sample_rate)
    release_time = int(0.3 * sample_rate)
    envelope[:attack_time] = np.linspace(0, 1, attack_time)
    envelope[-release_time:] = np.linspace(1, 0, release_time)
    
    pulse_signal = pulse_signal * envelope
    
    return pulse_signal, sample_rate

def generate_tau_delta_f_modulation(duration=12.0, sample_rate=44100):
    """
    Generate audio demonstrating the relationship between τ (time-of-flight) 
    and Δf (frequency offset) as measured in E1 experiment.
    """
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    
    # Simulate the relationship τ = Δφ / (2π·Δf) where information propagates
    # This represents how quickly information about timing and frequency manifests
    tau_freq = 2.1  # Hz - representing 2.1 ps timing resolution from E1
    delta_f_freq = 0.8  # Hz - representing 0.8 ppb frequency precision from E1
    
    # Create modulated signal representing the τ/Δf relationship
    tau_component = np.sin(2 * np.pi * tau_freq * t)
    delta_f_component = np.sin(2 * np.pi * delta_f_freq * t)
    
    # Combine to show the interplay between τ and Δf
    combined_signal = tau_component * delta_f_component
    carrier = np.sin(2 * np.pi * 330 * t)  # Musical note to carry the information
    
    # Modulate the carrier with the τ/Δf relationship
    modulated_signal = carrier * (1 + 0.4 * combined_signal)
    
    # Add harmonics to make it more interesting
    harmonic_1 = 0.2 * np.sin(2 * np.pi * 660 * t) * (1 + 0.3 * tau_component)
    harmonic_2 = 0.1 * np.sin(2 * np.pi * 990 * t) * (1 + 0.2 * delta_f_component)
    
    final_signal = modulated_signal + harmonic_1 + harmonic_2
    
    # Apply envelope
    envelope = np.ones_like(t)
    attack_time = int(0.15 * sample_rate)
    release_time = int(0.25 * sample_rate)
    envelope[:attack_time] = np.linspace(0, 1, attack_time)
    envelope[-release_time:] = np.linspace(1, 0, release_time)
    
    final_signal = final_signal * envelope
    
    return final_signal, sample_rate

def main():
    """
    Generate audio demonstrations of E1 chronometric interferometry.
    """
    print("Generating E1 chronometric interferometry audio demonstrations...")
    
    # Create output directory if it doesn't exist
    output_dir = "e1_audio_demonstrations"
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. E1 beat note formation
    print("1. Generating E1 beat-note formation audio...")
    e1_beat_note, sr = generate_e1_beat_note_audio()
    sf.write(os.path.join(output_dir, "e1_beat_note_formation.wav"), 
             e1_beat_note, sr)
    
    # 2. Chronomagnetic pulses at E1 frequencies
    print("2. Generating chronomagnetic pulses at E1 frequencies...")
    pulse_signal, sr = generate_chronomagnetic_pulse_at_e1_frequency()
    sf.write(os.path.join(output_dir, "e1_chronomagnetic_pulses.wav"), 
             pulse_signal, sr)
    
    # 3. τ and Δf modulation
    print("3. Generating τ/Δf modulation audio...")
    tau_delta_f_signal, sr = generate_tau_delta_f_modulation()
    sf.write(os.path.join(output_dir, "e1_tau_delta_f_modulation.wav"), 
             tau_delta_f_signal, sr)
    
    print(f"\\nE1 audio demonstrations saved to: {output_dir}/")
    print("\\nFiles generated:")
    print("- e1_beat_note_formation.wav: Beat-note formation as in E1 experiment")
    print("- e1_chronomagnetic_pulses.wav: Information 'materializing' at E1-relevant frequencies") 
    print("- e1_tau_delta_f_modulation.wav: Modulation representing τ/Δf relationship from E1")
    print("\\nThese represent how information manifests in chronometric interferometry,")
    print("demonstrating the 'out-of-tune' phenomenon where information arrives")
    print("at specific temporal frequencies representing timing (τ) and frequency (Δf) measurements.")
    print(f"\\nE1 achieved: {2.1} ps timing precision and {0.8} ppb frequency precision.")
    print("Listen for the beat patterns and modulations that represent these measurements.")

if __name__ == "__main__":
    main()