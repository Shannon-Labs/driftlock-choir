"""
Driftlock Phase Delta Analyzer

Compute the timing and distance change between two captures by comparing
the beat phase intercepts. Δτ ≈ Δφ / (2π f_carrier),  Δd = c * Δτ.

Usage:
  python experiment/driftlock_phase_delta.py [older.npy newer.npy]
If filepaths are omitted, picks the two most recent captures in experiment/results/.
"""
import sys
import glob
import numpy as np
from scipy import signal
from scipy.signal import hilbert

C = 299_792_458.0  # speed of light (m/s)


def phase_intercept(filename, fs=2.048e6, band=(900, 1100)):
    x = np.load(filename)
    env = np.abs(x)
    sos = signal.butter(6, [band[0], band[1]], btype='band', fs=fs, output='sos')
    y = signal.sosfilt(sos, env)
    analytic = hilbert(y)
    phase = np.unwrap(np.angle(analytic))
    t = np.arange(len(phase)) / fs
    a, b = np.polyfit(t, phase, 1)
    f_beat = a / (2 * np.pi)
    return b, f_beat


def main():
    args = sys.argv[1:]
    if len(args) == 2:
        f1, f2 = args
    else:
        # Use two most recent captures in experiment/results/
        captures = sorted(glob.glob('experiment/results/beat_capture_*.npy'))
        if len(captures) < 2:
            captures = sorted(glob.glob('results/beat_capture_*.npy'))
        if len(captures) < 2:
            print("Need at least two captures. Run beat_recorder.py twice.")
            sys.exit(1)
        f1, f2 = captures[-2], captures[-1]

    print(f"Comparing: {f1}  →  {f2}")
    b1, f1b = phase_intercept(f1)
    b2, f2b = phase_intercept(f2)
    dphi = b2 - b1
    f_carrier = 915e6
    d_tau_s = dphi / (2 * np.pi * f_carrier)
    d_tau_ps = d_tau_s * 1e12
    d_m = C * d_tau_s
    d_cm = d_m * 100

    print("\n=== DRIFTLOCK PHASE DELTA ===")
    print(f"Beat estimates (older,newer): {f1b:.2f} Hz, {f2b:.2f} Hz")
    print(f"Phase intercepts (b1,b2): {b1:.3f} rad, {b2:.3f} rad")
    print(f"Δφ: {dphi:.3f} rad")
    print(f"Δτ: {d_tau_ps:.1f} ps")
    print(f"Δd: {d_cm:.2f} cm")


if __name__ == "__main__":
    main()

