"""
Record the beat frequency using RTL-SDR
The ~1 kHz beat contains timing information in its phase.

Tip: keep gain modest to avoid ADC clipping; ensure both TX are active.
"""
import time as _time
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

try:
    from rtlsdr import RtlSdr
except Exception as exc:
    raise SystemExit("pyrtlsdr not installed or RTL-SDR drivers missing. See experiment/requirements.txt and hardware_setup.md") from exc


def main():
    sdr = RtlSdr()
    sdr.sample_rate = 2.048e6  # Hz
    sdr.center_freq = 915e6    # Hz (center between the two tones)
    sdr.freq_correction = 0    # ppm (adjust if needed)
    sdr.gain = 20              # dB (tune to avoid clipping)

    print("RTL-SDR configured:")
    print(f"  Center frequency: {sdr.center_freq/1e6:.6f} MHz")
    print(f"  Sample rate:     {sdr.sample_rate/1e6:.3f} MHz")
    print(f"  Gain:            {sdr.gain} dB")
    print("Recording the ~1 kHz beat frequency...")

    duration = 1.0  # seconds
    n = int(sdr.sample_rate * duration)
    samples = sdr.read_samples(n)
    sdr.close()

    # Save raw IQ samples
    import os
    os.makedirs('results', exist_ok=True)
    timestamp = int(_time.time())
    out_np = f'results/beat_capture_{timestamp}.npy'
    np.save(out_np, samples)
    print(f"Saved raw IQ to: {out_np}")

    # Quick visualization
    fig, axs = plt.subplots(3, 1, figsize=(12, 8))

    # Time domain envelope (|IQ|). With two nearby carriers, envelope beats at Δf.
    env = np.abs(samples)
    axs[0].plot(env[:20000])
    axs[0].set_title('Beat Signal Envelope (~1 kHz)')
    axs[0].set_xlabel('Samples')
    axs[0].set_ylabel('Amplitude')

    # Spectrum around baseband
    freqs, psd = signal.welch(samples, sdr.sample_rate, nperseg=8192)
    axs[1].semilogy((freqs - sdr.sample_rate/2)/1e3, np.fft.fftshift(psd))
    axs[1].set_title('Spectrum (two tones ~1 kHz apart w.r.t. center)')
    axs[1].set_xlabel('Frequency from DC (kHz)')
    axs[1].set_xlim([-10, 10])

    # Band-pass around ~1 kHz on the envelope
    sos = signal.butter(4, [500, 1500], btype='band', fs=sdr.sample_rate, output='sos')
    beat_filtered = signal.sosfilt(sos, env)
    axs[2].plot(beat_filtered[:20000])
    axs[2].set_title('Filtered Envelope (~1 kHz beat)')
    axs[2].set_xlabel('Samples')
    axs[2].set_ylabel('Amplitude')

    plt.tight_layout()
    out_png = f'results/beat_visualization_{timestamp}.png'
    fig.savefig(out_png)
    print(f"Saved visualization: {out_png}")
    plt.show()


if __name__ == "__main__":
    main()
