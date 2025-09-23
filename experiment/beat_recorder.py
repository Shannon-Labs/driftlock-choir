"""Record the beat frequency from two offset transmitters using an RTL-SDR."""
from __future__ import annotations

import argparse
import json
import time as _time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

try:
    from rtlsdr import RtlSdr
except Exception as exc:  # pragma: no cover - hardware import guard
    raise SystemExit(
        "pyrtlsdr not installed or RTL-SDR drivers missing. "
        "See experiment/requirements.txt and hardware_setup.md."
    ) from exc


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Capture the ~1 kHz beat from the Driftlock hardware demo and "
            "store IQ samples plus a quick-look visualization."
        )
    )
    parser.add_argument(
        "--device-index",
        type=int,
        default=0,
        help="RTL-SDR device index (default: 0)",
    )
    parser.add_argument(
        "--center-freq",
        type=float,
        default=915.000,
        help="Center frequency in MHz (default: 915.000)",
    )
    parser.add_argument(
        "--sample-rate",
        type=float,
        default=2.048,
        help="Sample rate in Msps (default: 2.048)",
    )
    parser.add_argument(
        "--gain",
        type=float,
        default=20.0,
        help="Tuner gain in dB (default: 20 dB)",
    )
    parser.add_argument(
        "--freq-correction",
        type=float,
        default=0.0,
        help="Frequency correction in ppm (default: 0)",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=1.0,
        help="Capture duration in seconds (default: 1.0)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results"),
        help="Directory for capture artifacts (default: results/)",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip showing the matplotlib window (still saves PNG)",
    )
    parser.add_argument(
        "--band-low",
        type=float,
        default=500.0,
        help="Envelope band-pass low cut in Hz (default: 500)",
    )
    parser.add_argument(
        "--band-high",
        type=float,
        default=1500.0,
        help="Envelope band-pass high cut in Hz (default: 1500)",
    )
    return parser.parse_args()


def capture_beat(
    *,
    device_index: int = 0,
    center_freq_mhz: float = 915.000,
    sample_rate_msps: float = 2.048,
    gain_db: float = 20.0,
    freq_correction_ppm: float = 0.0,
    duration_s: float = 1.0,
    output_dir: Path = Path("results"),
    band_low_hz: float = 500.0,
    band_high_hz: float = 1500.0,
    show_plot: bool = True,
    quiet: bool = False,
) -> dict[str, object]:
    """Capture IQ samples and quick-look plots for the Driftlock beat demo."""

    sample_rate_hz = sample_rate_msps * 1e6
    center_freq_hz = center_freq_mhz * 1e6

    sdr = RtlSdr(device_index)
    sdr.sample_rate = sample_rate_hz
    sdr.center_freq = center_freq_hz
    sdr.freq_correction = freq_correction_ppm
    sdr.gain = gain_db

    if not quiet:
        print("RTL-SDR configured:")
        print(f"  Device index:     {device_index}")
        print(f"  Center frequency: {sdr.center_freq/1e6:.6f} MHz")
        print(f"  Sample rate:      {sdr.sample_rate/1e6:.3f} Msps")
        print(f"  Gain:             {sdr.gain:.1f} dB")
        print(f"  Freq correction:  {sdr.freq_correction:.1f} ppm")
        print(
            f"Recording {duration_s:.3f} s of IQ (~{int(sample_rate_hz * duration_s):,} samples)..."
        )

    n_samples = int(sample_rate_hz * duration_s)
    start_ts = _time.time()
    samples = sdr.read_samples(n_samples)
    sdr.close()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = int(start_ts)
    capture_path = output_dir / f"beat_capture_{stamp}.npy"
    np.save(capture_path, samples)

    metadata = {
        "timestamp_unix": stamp,
        "duration_s": float(duration_s),
        "center_frequency_hz": float(center_freq_hz),
        "sample_rate_hz": float(sample_rate_hz),
        "gain_db": float(gain_db),
        "freq_correction_ppm": float(freq_correction_ppm),
        "bandpass_low_hz": float(band_low_hz),
        "bandpass_high_hz": float(band_high_hz),
        "device_index": int(device_index),
        "samples": int(n_samples),
    }
    metadata_path = output_dir / f"beat_capture_{stamp}.json"
    metadata_path.write_text(json.dumps(metadata, indent=2))

    if not quiet:
        print(f"Saved raw IQ to:    {capture_path}")
        print(f"Saved metadata to: {metadata_path}")

    fig, axs = plt.subplots(3, 1, figsize=(12, 8))

    env = np.abs(samples)
    axs[0].plot(env[:20000])
    axs[0].set_title("Beat Signal Envelope (~Δf)")
    axs[0].set_xlabel("Samples")
    axs[0].set_ylabel("Amplitude")

    freqs, psd = signal.welch(samples, sample_rate_hz, nperseg=8192)
    axs[1].semilogy((freqs - sample_rate_hz / 2) / 1e3, np.fft.fftshift(psd))
    axs[1].set_title("Spectrum around baseband")
    axs[1].set_xlabel("Frequency from DC (kHz)")
    axs[1].set_xlim([-10, 10])

    sos = signal.butter(
        4,
        [band_low_hz, band_high_hz],
        btype="band",
        fs=sample_rate_hz,
        output="sos",
    )
    beat_filtered = signal.sosfilt(sos, env)
    axs[2].plot(beat_filtered[:20000])
    axs[2].set_title("Filtered Envelope (beat)")
    axs[2].set_xlabel("Samples")
    axs[2].set_ylabel("Amplitude")

    plt.tight_layout()
    vis_path = output_dir / f"beat_visualization_{stamp}.png"
    fig.savefig(vis_path)
    if not quiet:
        print(f"Saved visualization: {vis_path}")

    if show_plot:
        plt.show()
    plt.close(fig)

    return {
        "capture_path": capture_path,
        "metadata_path": metadata_path,
        "visualization_path": vis_path,
        "metadata": metadata,
        "samples": samples,
    }


def main() -> None:
    args = _parse_args()
    capture_beat(
        device_index=args.device_index,
        center_freq_mhz=args.center_freq,
        sample_rate_msps=args.sample_rate,
        gain_db=args.gain,
        freq_correction_ppm=args.freq_correction,
        duration_s=args.duration,
        output_dir=args.output_dir,
        band_low_hz=args.band_low,
        band_high_hz=args.band_high,
        show_plot=not args.no_plot,
    )


if __name__ == "__main__":
    main()
