# E1 Chronomagnetic Audio Demonstrations

This directory contains audio representations of Experiment E1 from the Driftlock Choir project, demonstrating chronometric interferometry principles through audible phenomena.

## Access the Audio Files

All audio files are located in this directory and can be played with any standard audio player:

- [e1_beat_note_formation.wav](e1_beat_note_formation.wav) (689 KB)
- [e1_chronomagnetic_pulses.wav](e1_chronomagnetic_pulses.wav) (861 KB) 
- [e1_tau_delta_f_modulation.wav](e1_tau_delta_f_modulation.wav) (1034 KB)

## Simulation-Based Demonstration

These audio representations are based on our simulation framework that accurately models the physics of chronometric interferometry. While the audio demonstrates the theoretical relationships validated in E1 (achieving ~10 ps median timing precision and 0.05-40 ppb frequency precision in simulations), actual hardware validation remains to be performed.

## Hardware Roadmap

We are actively planning hardware experiments to validate these principles in real-world conditions. Our upcoming experimental setup will utilize:
- Two Adafruit Feather boards for timing reference
- RTL-SDR receivers for signal processing
- Real-world RF environments to validate simulation results

## Files Generated:

### 1. e1_beat_note_formation.wav
- **Concept**: Beat-note formation as in E1 experiment
- **Description**: Represents two oscillators with a slight frequency offset (100 Hz) creating beat patterns that demonstrate the fundamental principle of τ (time-of-flight) and Δf (frequency offset) measurements
- **Audible Elements**: You'll hear the carrier frequency (440 Hz) modulated by the beat frequency (100 Hz) with additional τ-related modulations at 2 Hz

### 2. e1_chronomagnetic_pulses.wav
- **Concept**: Information 'materializing' at E1-relevant frequencies
- **Description**: Demonstrates how information manifests at specific temporal frequencies in chronometric interferometry
- **Audible Elements**: Pulsed signals at 100, 200, and 300 Hz frequencies representing the "out-of-tune" effect where information arrives at specific temporal frequencies

### 3. e1_tau_delta_f_modulation.wav
- **Concept**: Modulation representing τ/Δf relationship from E1
- **Description**: Shows the relationship between τ (time-of-flight) and Δf (frequency offset) as measured in E1 experiment
- **Audible Elements**: 330 Hz carrier modulated by interplay between τ-related (2.1 Hz) and Δf-related (0.8 Hz) components

## Scientific Context:

Experiment E1 achieved:
- **~10 ps median timing precision** (τ measurements, range 2-30 ps)
- **0.8 ppb frequency precision** (Δf measurements)

These audio representations demonstrate how information about timing and frequency relationships can be encoded in audible signals, showing the "out-of-tune" phenomenon where information about chronometric interferometry relationships manifests at specific temporal frequencies.

The beat patterns and modulations in these audio files represent the same physical relationships that allow Driftlock Choir to achieve picosecond timing and part-per-billion frequency precision in its chronometric interferometry measurements.