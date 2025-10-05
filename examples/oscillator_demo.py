"""
Oscillator and Signal Generation Demo

This example demonstrates the oscillator models and signal generation
capabilities that form the foundation of chronometric interferometry.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib.pyplot as plt

from src.signal_processing.oscillator import Oscillator
from src.signal_processing.channel import ChannelSimulator
from src.core.types import Hertz, Seconds


def demo_ideal_oscillator():
    """Demonstrate ideal oscillator signal generation."""
    print("=== Ideal Oscillator Demo ===")
    
    # Create ideal oscillator at 2.4 GHz
    frequency = Hertz(2.4e9)
    oscillator_model = Oscillator.create_ideal_oscillator(frequency)
    oscillator = Oscillator(oscillator_model)
    
    print(f"Created ideal oscillator at {frequency/1e9:.1f} GHz")
    
    # Generate signal
    duration = Seconds(1e-6)  # 1 microsecond
    sampling_rate = Hertz(10e6)  # 10 MS/s
    
    time, signal = oscillator.generate_signal(
        duration=duration,
        sampling_rate=sampling_rate,
        frequency_offset=Hertz(0.0),
        phase_noise_enabled=False
    )
    
    print(f"Generated {len(signal)} samples over {duration*1e6:.1f} μs")
    print(f"Signal magnitude: {np.mean(np.abs(signal)):.3f}")
    print(f"Signal phase stability: {np.std(np.angle(signal)):.6f} rad")
    
    return time, signal, "Ideal"


def demo_tcxo_oscillator():
    """Demonstrate TCXO oscillator with phase noise."""
    print("\n=== TCXO Oscillator Demo ===")
    
    # Create TCXO oscillator at 2.4 GHz
    frequency = Hertz(2.4e9)
    oscillator_model = Oscillator.create_tcxo_model(frequency)
    oscillator = Oscillator(oscillator_model)
    
    print(f"Created TCXO oscillator at {frequency/1e9:.1f} GHz")
    print(f"Phase noise enabled: {oscillator_model.phase_noise_enabled}")
    
    # Generate signal with phase noise
    duration = Seconds(1e-6)  # 1 microsecond
    sampling_rate = Hertz(10e6)  # 10 MS/s
    
    time, signal = oscillator.generate_signal(
        duration=duration,
        sampling_rate=sampling_rate,
        frequency_offset=Hertz(0.0),
        phase_noise_enabled=True
    )
    
    print(f"Generated {len(signal)} samples over {duration*1e6:.1f} μs")
    print(f"Signal magnitude: {np.mean(np.abs(signal)):.3f}")
    print(f"Signal phase noise: {np.std(np.angle(signal)):.6f} rad")
    
    return time, signal, "TCXO"


def demo_frequency_offset():
    """Demonstrate frequency offset effects."""
    print("\n=== Frequency Offset Demo ===")
    
    frequency = Hertz(2.4e9)
    frequency_offset = Hertz(1000.0)  # 1 kHz offset
    
    oscillator_model = Oscillator.create_ideal_oscillator(frequency)
    oscillator = Oscillator(oscillator_model)
    
    print(f"Base frequency: {frequency/1e9:.1f} GHz")
    print(f"Frequency offset: {frequency_offset:.1f} Hz")
    
    # Generate signal with frequency offset
    duration = Seconds(1e-3)  # 1 millisecond (longer to see offset effect)
    sampling_rate = Hertz(1e6)  # 1 MS/s
    
    time, signal = oscillator.generate_signal(
        duration=duration,
        sampling_rate=sampling_rate,
        frequency_offset=frequency_offset,
        phase_noise_enabled=False
    )
    
    print(f"Generated {len(signal)} samples over {duration*1e3:.1f} ms")
    
    # Analyze instantaneous frequency
    instantaneous_phase = np.unwrap(np.angle(signal))
    instantaneous_freq = np.diff(instantaneous_phase) * sampling_rate / (2 * np.pi)
    
    print(f"Mean instantaneous frequency offset: {np.mean(instantaneous_freq):.1f} Hz")
    print(f"Expected frequency offset: {frequency_offset:.1f} Hz")
    
    return time, signal, "Frequency Offset"


def demo_channel_effects():
    """Demonstrate channel simulation effects."""
    print("\n=== Channel Effects Demo ===")
    
    # Create clean signal
    frequency = Hertz(2.4e9)
    oscillator_model = Oscillator.create_ideal_oscillator(frequency)
    oscillator = Oscillator(oscillator_model)
    
    duration = Seconds(1e-4)  # 100 μs
    sampling_rate = Hertz(1e6)  # 1 MS/s
    
    time, clean_signal = oscillator.generate_signal(
        duration=duration,
        sampling_rate=sampling_rate,
        frequency_offset=Hertz(0.0),
        phase_noise_enabled=False
    )
    
    print(f"Clean signal power: {10*np.log10(np.mean(np.abs(clean_signal)**2)):.1f} dB")
    
    # Apply channel effects
    channel_sim = ChannelSimulator(sampling_rate)
    
    # Add thermal noise at different SNR levels
    snr_levels = [10, 20, 30, 40]
    noisy_signals = {}
    
    for snr_db in snr_levels:
        noisy_signal = channel_sim.add_thermal_noise(clean_signal.copy(), snr_db=snr_db)
        noisy_signals[snr_db] = noisy_signal
        
        noise_power = np.mean(np.abs(noisy_signal - clean_signal)**2)
        print(f"SNR {snr_db} dB: Noise power = {10*np.log10(noise_power):.1f} dB")
    
    return time, clean_signal, noisy_signals


def plot_oscillator_comparison(demos):
    """Plot comparison of different oscillator types."""
    try:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Oscillator Signal Generation Demo', fontsize=14)
        
        # Combine all signals for plotting
        ideal_time, ideal_signal, _ = demos[0]
        tcxo_time, tcxo_signal, _ = demos[1]
        offset_time, offset_signal, _ = demos[2]
        
        # Plot 1: Ideal vs TCXO signals (real part, zoomed in)
        plot_samples = min(100, len(ideal_signal))
        time_ms = ideal_time[:plot_samples] * 1e6  # Convert to microseconds
        
        axes[0, 0].plot(time_ms, ideal_signal[:plot_samples].real, 'b-', label='Ideal', linewidth=1)
        axes[0, 0].plot(time_ms, tcxo_signal[:plot_samples].real, 'r-', label='TCXO', linewidth=1)
        axes[0, 0].set_title('Signal Comparison (Real Part)')
        axes[0, 0].set_xlabel('Time (μs)')
        axes[0, 0].set_ylabel('Amplitude')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Magnitude comparison
        axes[0, 1].plot(time_ms, np.abs(ideal_signal[:plot_samples]), 'b-', label='Ideal', linewidth=1)
        axes[0, 1].plot(time_ms, np.abs(tcxo_signal[:plot_samples]), 'r-', label='TCXO', linewidth=1)
        axes[0, 1].set_title('Signal Magnitude')
        axes[0, 1].set_xlabel('Time (μs)')
        axes[0, 1].set_ylabel('Magnitude')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Phase comparison
        axes[1, 0].plot(time_ms, np.angle(ideal_signal[:plot_samples]), 'b-', label='Ideal', linewidth=1)
        axes[1, 0].plot(time_ms, np.angle(tcxo_signal[:plot_samples]), 'r-', label='TCXO', linewidth=1)
        axes[1, 0].set_title('Signal Phase')
        axes[1, 0].set_xlabel('Time (μs)')
        axes[1, 0].set_ylabel('Phase (rad)')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Frequency offset effect (longer timescale)
        offset_samples = min(1000, len(offset_signal))
        offset_time_ms = offset_time[:offset_samples] * 1e3  # Convert to ms
        
        # Calculate instantaneous frequency
        instantaneous_phase = np.unwrap(np.angle(offset_signal[:offset_samples]))
        instantaneous_freq = np.diff(instantaneous_phase) * 1e6 / (2 * np.pi)  # Hz
        
        axes[1, 1].plot(offset_time_ms[1:], instantaneous_freq, 'g-', linewidth=1)
        axes[1, 1].set_title('Instantaneous Frequency (with 1kHz offset)')
        axes[1, 1].set_xlabel('Time (ms)')
        axes[1, 1].set_ylabel('Frequency Offset (Hz)')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('oscillator_demo.png', dpi=300, bbox_inches='tight')
        print("\nPlot saved as 'oscillator_demo.png'")
        plt.show()
        
    except ImportError:
        print("\nMatplotlib not available. Install with: pip install matplotlib")
    except Exception as e:
        print(f"Error generating plots: {e}")


def plot_channel_effects(time, clean_signal, noisy_signals):
    """Plot channel effects comparison."""
    try:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Channel Effects Demo', fontsize=14)
        
        # Plot subset of samples for clarity
        plot_samples = min(500, len(clean_signal))
        time_us = time[:plot_samples] * 1e6  # Convert to microseconds
        
        # Plot 1: Clean vs noisy signals (real part)
        axes[0, 0].plot(time_us, clean_signal[:plot_samples].real, 'b-', label='Clean', linewidth=2)
        axes[0, 0].plot(time_us, noisy_signals[20][:plot_samples].real, 'r-', alpha=0.7, label='SNR 20dB')
        axes[0, 0].set_title('Clean vs Noisy Signal (Real Part)')
        axes[0, 0].set_xlabel('Time (μs)')
        axes[0, 0].set_ylabel('Amplitude')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Magnitude comparison
        axes[0, 1].plot(time_us, np.abs(clean_signal[:plot_samples]), 'b-', label='Clean', linewidth=2)
        axes[0, 1].plot(time_us, np.abs(noisy_signals[20][:plot_samples]), 'r-', alpha=0.7, label='SNR 20dB')
        axes[0, 1].set_title('Signal Magnitude')
        axes[0, 1].set_xlabel('Time (μs)')
        axes[0, 1].set_ylabel('Magnitude')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: SNR comparison (constellation plot)
        axes[1, 0].scatter(clean_signal[:plot_samples].real, clean_signal[:plot_samples].imag, 
                          c='b', alpha=0.5, s=1, label='Clean')
        axes[1, 0].scatter(noisy_signals[20][:plot_samples].real, noisy_signals[20][:plot_samples].imag, 
                          c='r', alpha=0.3, s=1, label='SNR 20dB')
        axes[1, 0].set_title('Constellation Plot')
        axes[1, 0].set_xlabel('In-Phase')
        axes[1, 0].set_ylabel('Quadrature')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].axis('equal')
        
        # Plot 4: SNR vs signal quality
        snr_values = list(noisy_signals.keys())
        signal_qualities = []
        
        for snr in snr_values:
            # Calculate signal quality as correlation with clean signal
            correlation = np.abs(np.corrcoef(clean_signal.real, noisy_signals[snr].real)[0, 1])
            signal_qualities.append(correlation)
        
        axes[1, 1].plot(snr_values, signal_qualities, 'go-', linewidth=2, markersize=8)
        axes[1, 1].set_title('Signal Quality vs SNR')
        axes[1, 1].set_xlabel('SNR (dB)')
        axes[1, 1].set_ylabel('Signal Correlation')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].set_ylim([0, 1])
        
        plt.tight_layout()
        plt.savefig('channel_effects_demo.png', dpi=300, bbox_inches='tight')
        print("Plot saved as 'channel_effects_demo.png'")
        plt.show()
        
    except ImportError:
        print("\nMatplotlib not available. Install with: pip install matplotlib")
    except Exception as e:
        print(f"Error generating plots: {e}")


def main():
    """Main demonstration function."""
    print("Oscillator and Signal Generation Demo")
    print("=" * 50)
    
    # Run oscillator demos
    ideal_demo = demo_ideal_oscillator()
    tcxo_demo = demo_tcxo_oscillator()
    offset_demo = demo_frequency_offset()
    
    # Run channel effects demo
    time, clean_signal, noisy_signals = demo_channel_effects()
    
    # Generate plots
    print("\n" + "=" * 50)
    print("Generating plots...")
    
    oscillator_demos = [ideal_demo, tcxo_demo, offset_demo]
    plot_oscillator_comparison(oscillator_demos)
    plot_channel_effects(time, clean_signal, noisy_signals)
    
    print("\nDemo completed!")
    print("\nKey Takeaways:")
    print("- Ideal oscillators provide clean, stable signals")
    print("- TCXO oscillators include realistic phase noise")
    print("- Frequency offsets create measurable signal changes")
    print("- Channel effects degrade signal quality predictably")
    print("- Higher SNR preserves signal integrity better")


if __name__ == "__main__":
    main()