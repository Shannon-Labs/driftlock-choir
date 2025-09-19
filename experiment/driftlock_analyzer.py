"""
Driftlock Analyzer: Extract timing information from beat phase

Core idea: beat phase φ(t) evolves ~ 2π Δf (t - τ) + constants. 
By fitting the phase trajectory we can estimate Δf and a phase intercept.
Absolute τ requires knowing initial phases (or doing two-way). For this demo,
we show that moving the nodes produces a consistent phase/timing shift (relative).
"""
import glob
import time as _time
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.signal import hilbert


def extract_timing_from_beat(filename, fs=2.048e6, f_carrier=915e6, band=(900, 1100)):
    samples = np.load(filename)
    # Beat appears on the envelope (magnitude) of the complex baseband sum
    env = np.abs(samples)
    # Band-pass the envelope near the nominal beat (default ~1 kHz)
    sos = signal.butter(6, [band[0], band[1]], btype='band', fs=fs, output='sos')
    beat = signal.sosfilt(sos, env)

    # Instantaneous phase of the beat via Hilbert transform
    analytic = hilbert(beat)
    phase = np.unwrap(np.angle(analytic))
    t = np.arange(len(phase)) / fs

    # Fit φ(t) ≈ a t + b (a = 2π Δf)
    a, b = np.polyfit(t, phase, 1)
    f_beat = a / (2 * np.pi)
    # A naive absolute τ estimate (includes unknown initial phase terms)
    tau_abs = b / (2 * np.pi * f_carrier)
    tau_ns = tau_abs * 1e9
    tau_ps = tau_abs * 1e12

    # Residuals for SNR/quality indication
    fit = a * t + b
    resid = phase - fit
    rms_resid = np.sqrt(np.mean(resid ** 2))

    return {
        'f_beat_hz': float(f_beat),
        'phase_intercept': float(b),
        'tau_abs_s': float(tau_abs),
        'tau_abs_ns': float(tau_ns),
        'tau_abs_ps': float(tau_ps),
        'rms_resid_rad': float(rms_resid),
        't': t,
        'phase': phase,
        'fit': fit,
        'beat': beat,
    }


def main():
    captures = sorted(glob.glob('results/beat_capture_*.npy'))
    if not captures:
        print("No captures found. Run beat_recorder.py first!")
        return

    latest = captures[-1]
    print(f"Analyzing: {latest}")
    res = extract_timing_from_beat(latest)

    print("\n" + "=" * 50)
    print("DRIFTLOCK TIMING EXTRACTION")
    print("=" * 50)
    print(f"Measured beat frequency: {res['f_beat_hz']:.2f} Hz (nominal ~1000 Hz)")
    print(f"Beat phase linear fit residual (RMS): {res['rms_resid_rad']:.3f} rad")
    print(f"Naive absolute timing (includes unknown phases): {res['tau_abs_ps']:.1f} ps")
    print("Note: Absolute τ includes unknown initial phases; use delta between captures for distance changes.")

    # If there is a previous capture, compute delta timing
    if len(captures) >= 2:
        prev = captures[-2]
        res_prev = extract_timing_from_beat(prev)
        delta_phase = res['phase_intercept'] - res_prev['phase_intercept']
        # Δτ ≈ Δφ / (2π f_carrier)
        f_carrier = 915e6
        delta_tau_s = delta_phase / (2 * np.pi * f_carrier)
        delta_tau_ps = delta_tau_s * 1e12
        print(f"ΔTiming vs previous capture: {delta_tau_ps:.1f} ps (move node to change this)")

    # Visualization
    t = res['t']
    phase = res['phase']
    fit = res['fit']
    beat = res['beat']

    plt.figure(figsize=(12, 10))
    # Phase evolution + fit
    plt.subplot(3, 1, 1)
    n = min(len(t), 20000)
    plt.plot(t[:n], phase[:n], label='Phase')
    plt.plot(t[:n], fit[:n], 'r--', label=f"Fit (Δf ≈ {res['f_beat_hz']:.1f} Hz)")
    plt.xlabel('Time (s)')
    plt.ylabel('Phase (rad)')
    plt.title('Beat Phase Evolution')
    plt.grid(True)
    plt.legend()

    # Residuals
    plt.subplot(3, 1, 2)
    resid = phase - fit
    plt.plot(t[:n], resid[:n])
    plt.xlabel('Time (s)')
    plt.ylabel('Residual (rad)')
    plt.title(f'Phase Residual (RMS: {res["rms_resid_rad"]:.3f} rad)')
    plt.grid(True)

    # Beat (envelope, band‑passed)
    plt.subplot(3, 1, 3)
    plt.plot(t[:n], beat[:n])
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title('Envelope near ~1 kHz (beat)')
    plt.grid(True)

    plt.tight_layout()
    import os
    os.makedirs('results', exist_ok=True)
    stamp = int(_time.time())
    out = f'results/driftlock_analysis_{stamp}.png'
    plt.savefig(out)
    print(f"Saved plot: {out}")
    plt.show()


if __name__ == "__main__":
    main()
