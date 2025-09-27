import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import os

from src.hw.trx import TrxParams, Transceiver
from src.phy.osc import OscillatorParams
from src.utils.plotting import save_figure
from src.utils.io import ensure_directory

# --- Configuration ---
SAMPLE_RATE = 100e3
CHANNEL_BW = 25e3
CARRIER_FREQ = 150e6
DURATION = 0.01  # 10 ms duration
N_SAMPLES = int(SAMPLE_RATE * DURATION)
# Two peaks within +/- 12.5 kHz baseband
FORMANT_FREQS = [3000.0, 8000.0]
OUTPUT_DIR = "results/formant_feasibility"

# --- Signal Generation ---

def generate_formant_signal(
    freqs: list[float],
    duration: float,
    sample_rate: float,
    amplitude: float = 1.0,
) -> np.ndarray:
    """Creates a complex baseband signal with multiple tones (formants)."""
    t = np.arange(int(sample_rate * duration)) / sample_rate
    signal_out = np.zeros_like(t, dtype=np.complex128)
    
    # Use equal amplitude for simplicity
    amp_per_tone = amplitude / len(freqs)
    
    for freq in freqs:
        signal_out += amp_per_tone * np.exp(1j * 2.0 * np.pi * freq * t)
        
    return signal_out

# --- Analysis Helpers ---

def calculate_snr(spectrum: np.ndarray, freqs: np.ndarray, signal_freqs: list[float], noise_band_start: float = 1000.0) -> float:
    """
    Calculate SNR by comparing power at signal peaks vs. average noise floor.
    Assumes spectrum is magnitude squared (Power Spectral Density).
    This is a heuristic metric for feasibility assessment.
    """
    
    # 1. Identify signal power (sum of power at peak frequencies)
    signal_power = 0.0
    for f_sig in signal_freqs:
        # Find the index closest to the signal frequency
        idx = np.argmin(np.abs(freqs - f_sig))
        signal_power += spectrum[idx]
        
    # 2. Identify noise power
    # Define noise band (excluding DC and signal peaks)
    noise_mask = (np.abs(freqs) > noise_band_start)
    
    # Exclude signal peaks from noise calculation (simple exclusion zone)
    for f_sig in signal_freqs:
        exclusion_zone = 500.0 # 500 Hz exclusion around peaks
        noise_mask &= (np.abs(freqs - f_sig) > exclusion_zone)
        
    noise_spectrum = spectrum[noise_mask]
    
    if noise_spectrum.size == 0:
        return float('inf')
        
    # Average noise power per bin
    avg_noise_power = np.mean(noise_spectrum)
    
    # Total noise power normalized to the number of signal bins
    total_noise_power = avg_noise_power * len(signal_freqs)
    
    if total_noise_power == 0.0:
        return float('inf')
        
    snr_linear = signal_power / total_noise_power
    return 10.0 * np.log10(snr_linear)


def plot_spectrum(
    signal_in: np.ndarray,
    signal_out: np.ndarray,
    sample_rate: float,
    channel_bw: float,
    output_path: str,
) -> None:
    """Plots the magnitude spectrum of the input and output signals."""
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    fig.suptitle(f"Formant Feasibility Validation (BW={channel_bw/1e3:.1f} kHz)")

    def plot_fft(ax: plt.Axes, data: np.ndarray, title: str):
        N = data.size
        # Apply windowing
        window = np.hanning(N)
        data_windowed = data * window
        
        # FFT
        spectrum = np.fft.fftshift(np.fft.fft(data_windowed))
        magnitudes = np.abs(spectrum)
        power_spectrum = magnitudes**2
        
        # Frequencies
        freqs = np.fft.fftshift(np.fft.fftfreq(N, d=1.0 / sample_rate))
        
        # Calculate SNR
        snr_db = calculate_snr(power_spectrum, freqs, FORMANT_FREQS)
        
        # Plot
        ax.plot(freqs / 1e3, 10 * np.log10(power_spectrum), label=f'SNR: {snr_db:.2f} dB')
        ax.axvline(channel_bw / 2e3, color='r', linestyle='--', label='BW/2 Limit')
        ax.axvline(-channel_bw / 2e3, color='r', linestyle='--')
        
        for f in FORMANT_FREQS:
            ax.axvline(f / 1e3, color='g', linestyle=':', alpha=0.7, label=f'{f/1e3:.1f} kHz Formant' if f == FORMANT_FREQS[0] else None)
            
        ax.set_title(title)
        ax.set_ylabel('Power/Frequency (dB)')
        ax.grid(True, which='both', linestyle='--', alpha=0.5)
        ax.legend()
        ax.set_xlim([-sample_rate / 2e3, sample_rate / 2e3])
        
        return power_spectrum, freqs, snr_db

    # Plot Input Spectrum
    plot_fft(ax1, signal_in, "Input Baseband Signal Spectrum (Ideal)")
    
    # Plot Output Spectrum
    power_spectrum_out, freqs_out, snr_db_out = plot_fft(ax2, signal_out, "Received Baseband Signal Spectrum (Processed)")
    
    ax2.set_xlabel('Frequency (kHz)')
    
    save_figure(fig, output_path)
    print(f"Saved spectrum plot to {output_path}")
    print(f"Received Signal SNR: {snr_db_out:.2f} dB")


# --- Main Simulation ---

def main():
    print("Starting Formant Feasibility Simulation...")
    
    # 1. Setup Configuration
    # Use realistic, but minimal, oscillator noise for component validation
    osc_params = OscillatorParams(
        allan_dev_1s=1e-11,
        drift_rate=0.0,
        white_noise_level=1e-6,
    )
    
    trx_params = TrxParams(
        node_id=1,
        carrier_freq=CARRIER_FREQ,
        sample_rate=SAMPLE_RATE,
        tx_power=0.0,
        osc_params=osc_params,
        adc_bits=12,
        iq_imbalance_db=0.5,
        channel_bandwidth_hz=CHANNEL_BW,
        p1db_dbm=-10.0,
    )
    
    # Ensure output directory exists
    ensure_directory(OUTPUT_DIR)
    
    # 2. Initialize Transceiver
    trx = Transceiver(trx_params)
    
    # 3. Formant Signal Generation
    tx_baseband_signal = generate_formant_signal(
        freqs=FORMANT_FREQS,
        duration=DURATION,
        sample_rate=SAMPLE_RATE,
        amplitude=1.0,
    )
    
    # 4. Simulation Run (TX -> Ideal Channel -> RX)
    
    # TX: Upconversion
    rf_signal_tx = trx.transmit(tx_baseband_signal, timestamp=0.0)
    
    # Ideal Channel: No loss, no noise added here
    rf_signal_rx = rf_signal_tx
    
    # RX: Downconversion and processing (Non-linearity, Filtering, ADC, IQ correction)
    # Use DURATION as timestamp for RX to simulate time passage
    rx_baseband_signal = trx.receive(rf_signal_rx, timestamp=DURATION)
    
    # 5. Analysis and Plotting
    plot_spectrum(
        tx_baseband_signal,
        rx_baseband_signal,
        SAMPLE_RATE,
        CHANNEL_BW,
        os.path.join(OUTPUT_DIR, "formant_spectrum_validation.png"),
    )
    
    print("Simulation finished.")


if __name__ == "__main__":
    main()