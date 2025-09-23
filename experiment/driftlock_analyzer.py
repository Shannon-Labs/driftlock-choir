"""Driftlock Analyzer: Extract timing information from the beat phase."""
from __future__ import annotations

import argparse
import json
import time as _time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from scipy.signal import hilbert


def extract_timing_from_beat(
    filename: Path,
    *,
    fs: float,
    f_carrier: float,
    band: tuple[float, float],
):
    samples = np.load(filename)
    env = np.abs(samples)
    sos = signal.butter(6, [band[0], band[1]], btype="band", fs=fs, output="sos")
    beat = signal.sosfilt(sos, env)

    analytic = hilbert(beat)
    phase = np.unwrap(np.angle(analytic))
    t = np.arange(len(phase)) / fs

    a, b = np.polyfit(t, phase, 1)
    f_beat = a / (2 * np.pi)
    tau_abs = b / (2 * np.pi * f_carrier)

    fit = a * t + b
    resid = phase - fit
    rms_resid = np.sqrt(np.mean(resid ** 2))

    return {
        "f_beat_hz": float(f_beat),
        "phase_intercept": float(b),
        "tau_abs_s": float(tau_abs),
        "tau_abs_ps": float(tau_abs * 1e12),
        "rms_resid_rad": float(rms_resid),
        "t": t,
        "phase": phase,
        "fit": fit,
        "beat": beat,
    }

EXPERIMENT_DIR = Path(__file__).resolve().parent
DEFAULT_RESULTS_DIR = EXPERIMENT_DIR / "results"


def _sorted_captures(directory: Path) -> list[Path]:
    if not directory.exists():
        return []
    return sorted(directory.glob("beat_capture_*.npy"))


def _find_latest_capture() -> Path | None:
    search_dirs = {
        Path.cwd() / "results",
        DEFAULT_RESULTS_DIR,
    }
    candidates: list[Path] = []
    for directory in search_dirs:
        candidates.extend(_sorted_captures(directory))
    return candidates[-1] if candidates else None


def _find_previous_capture(current: Path) -> Path | None:
    current = current.resolve()
    directories = [current.parent]
    default_dir = DEFAULT_RESULTS_DIR.resolve()
    cwd_results = (Path.cwd() / "results").resolve()
    for directory in (default_dir, cwd_results):
        if directory not in directories:
            directories.append(directory)

    for directory in directories:
        captures = _sorted_captures(directory)
        if not captures:
            continue
        captures = [c.resolve() for c in captures]
        if current in captures and captures.index(current) >= 1:
            return captures[captures.index(current) - 1]
        if captures[-1] != current:
            return captures[-1]
    return None


def _load_metadata(capture: Path) -> dict[str, float] | None:
    meta_path = capture.with_suffix(".json")
    if not meta_path.exists():
        return None
    try:
        return json.loads(meta_path.read_text())
    except json.JSONDecodeError:
        print(f"Warning: metadata JSON {meta_path} is malformed; ignoring.")
        return None


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze a captured Driftlock beat to estimate beat frequency and "
            "relative timing shifts."
        )
    )
    parser.add_argument(
        "--capture",
        type=Path,
        help="Path to beat_capture_*.npy (defaults to most recent)",
    )
    parser.add_argument(
        "--reference",
        type=Path,
        help="Optional second capture for Δτ calculation",
    )
    parser.add_argument(
        "--sample-rate",
        type=float,
        help="Sample rate in Msps (overrides metadata/default)",
    )
    parser.add_argument(
        "--band-low",
        type=float,
        help="Envelope band-pass low cut in Hz (overrides metadata/default)",
    )
    parser.add_argument(
        "--band-high",
        type=float,
        help="Envelope band-pass high cut in Hz (overrides metadata/default)",
    )
    parser.add_argument(
        "--carrier",
        type=float,
        default=915.000,
        help="Carrier frequency in MHz (default: 915.000)",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip showing the matplotlib window (still saves PNG)",
    )
    return parser.parse_args()


def analyze_capture(
    capture: Path,
    *,
    reference: Path | None = None,
    sample_rate_hz: float | None = None,
    band_low_hz: float | None = None,
    band_high_hz: float | None = None,
    carrier_hz: float = 915e6,
    show_plot: bool = True,
    auto_reference: bool = True,
    quiet: bool = False,
) -> dict[str, object]:
    metadata = _load_metadata(capture)
    effective_sample_rate = sample_rate_hz
    if effective_sample_rate is None and metadata:
        sr_meta = metadata.get("sample_rate_hz")
        if sr_meta:
            effective_sample_rate = float(sr_meta)
    if effective_sample_rate is None:
        effective_sample_rate = 2.048e6

    low = band_low_hz
    high = band_high_hz
    if low is None and metadata:
        low = metadata.get("bandpass_low_hz")
    if high is None and metadata:
        high = metadata.get("bandpass_high_hz")
    if low is None:
        low = 900.0
    if high is None:
        high = 1100.0
    band = (float(low), float(high))

    if not quiet:
        print(f"Analyzing capture: {capture}")
        if metadata:
            print(f"Loaded metadata: {capture.with_suffix('.json')}")
        print(f"Using sample rate: {effective_sample_rate/1e6:.3f} Msps")
        print(f"Band-pass window: {band[0]:.1f}–{band[1]:.1f} Hz")

    res = extract_timing_from_beat(
        capture,
        fs=effective_sample_rate,
        f_carrier=carrier_hz,
        band=band,
    )

    ref_path = reference
    if ref_path is None and auto_reference:
        ref_path = _find_previous_capture(capture)

    ref_result = None
    ref_metadata = None
    delta_tau_ps = None
    if ref_path and ref_path.exists():
        ref_metadata = _load_metadata(ref_path)
        ref_sample_rate = effective_sample_rate
        if sample_rate_hz is None and ref_metadata and ref_metadata.get("sample_rate_hz"):
            ref_sample_rate = float(ref_metadata["sample_rate_hz"])
        ref_result = extract_timing_from_beat(
            ref_path,
            fs=ref_sample_rate,
            f_carrier=carrier_hz,
            band=band,
        )
        delta_phase = res["phase_intercept"] - ref_result["phase_intercept"]
        delta_tau_s = delta_phase / (2 * np.pi * carrier_hz)
        delta_tau_ps = delta_tau_s * 1e12

    if not quiet:
        print("\n" + "=" * 50)
        print("DRIFTLOCK TIMING EXTRACTION")
        print("=" * 50)
        print(f"Measured beat frequency: {res['f_beat_hz']:.2f} Hz (nominal ~1000 Hz)")
        print(f"Beat phase linear fit residual (RMS): {res['rms_resid_rad']:.3f} rad")
        print(
            "Naive absolute timing (includes unknown phases): "
            f"{res['tau_abs_ps']:.1f} ps"
        )
        print(
            "Note: Absolute τ includes unknown initial phases; use deltas between captures "
            "for distance changes."
        )
        if ref_path and ref_result is not None and delta_tau_ps is not None:
            print(
                f"ΔTiming vs {ref_path.name}: {delta_tau_ps:.1f} ps"
                " (move node to change this)"
            )
            if ref_metadata:
                print(f"Reference metadata: {ref_path.with_suffix('.json')}")
        elif ref_path is None and auto_reference:
            print("No previous capture found for Δτ comparison.")

    t = res["t"]
    phase = res["phase"]
    fit = res["fit"]
    beat = res["beat"]

    plt.figure(figsize=(12, 10))
    n = min(len(t), 20000)
    plt.subplot(3, 1, 1)
    plt.plot(t[:n], phase[:n], label="Phase")
    plt.plot(t[:n], fit[:n], "r--", label=f"Fit (Δf ≈ {res['f_beat_hz']:.1f} Hz)")
    plt.xlabel("Time (s)")
    plt.ylabel("Phase (rad)")
    plt.title("Beat Phase Evolution")
    plt.grid(True)
    plt.legend()

    plt.subplot(3, 1, 2)
    resid = phase - fit
    plt.plot(t[:n], resid[:n])
    plt.xlabel("Time (s)")
    plt.ylabel("Residual (rad)")
    plt.title(f"Phase Residual (RMS: {res['rms_resid_rad']:.3f} rad)")
    plt.grid(True)

    plt.subplot(3, 1, 3)
    plt.plot(t[:n], beat[:n])
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title("Envelope near ~Δf (beat)")
    plt.grid(True)

    plt.tight_layout()
    out_dir = capture.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = int(_time.time())
    out = out_dir / f"driftlock_analysis_{stamp}.png"
    plt.savefig(out)
    if not quiet:
        print(f"Saved plot: {out}")
    if show_plot:
        plt.show()
    plt.close()

    return {
        "capture_path": capture,
        "result": res,
        "metadata": metadata,
        "reference_path": ref_path,
        "reference_result": ref_result,
        "reference_metadata": ref_metadata,
        "delta_tau_ps": float(delta_tau_ps) if delta_tau_ps is not None else None,
        "plot_path": out,
        "band": band,
        "sample_rate_hz": float(effective_sample_rate),
    }


def main() -> None:
    args = _parse_args()

    capture = args.capture or _find_latest_capture()
    if capture is None:
        print("No captures found. Run beat_recorder.py first!")
        return

    sample_rate_override = args.sample_rate * 1e6 if args.sample_rate is not None else None
    band_low_override = args.band_low
    band_high_override = args.band_high
    carrier_hz = args.carrier * 1e6

    analyze_capture(
        capture,
        reference=args.reference,
        sample_rate_hz=sample_rate_override,
        band_low_hz=band_low_override,
        band_high_hz=band_high_override,
        carrier_hz=carrier_hz,
        show_plot=not args.no_plot,
        auto_reference=args.reference is None,
        quiet=False,
    )


if __name__ == "__main__":
    main()
