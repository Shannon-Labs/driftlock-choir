from __future__ import annotations

import numpy as np
from .phase_schedules import newman_phases, amplitude_taper


def qpsk_symbols(n: int, rng: np.random.Generator) -> np.ndarray:
    bits = rng.integers(0, 4, size=n)
    const = np.array([1+1j, -1+1j, -1-1j, 1-1j]) / np.sqrt(2)
    return const[bits]


def generate_comb(
    fs: float,
    duration: float,
    df: float,
    m: int,
    omit_fundamental: bool = True,
    amp_mode: str = "equal",
    phase_mode: str = "newman",
    payload_qpsk_fraction: float = 0.0,
    payload_symbol_rate: float = 1000.0,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate complex baseband multi-tone comb around 0 Hz.

    Returns (x, fk, pilot_mask).
    """
    rng = np.random.default_rng() if rng is None else rng
    n = int(round(fs * duration))
    t = np.arange(n) / fs

    # Define carrier indices
    # Missing fundamental: omit k=0; use symmetric set around 0
    half = m // 2
    if omit_fundamental:
        ks = np.concatenate([-(np.arange(1, half+1)), (np.arange(1, m - half + 1))])
        ks = ks[:m]  # ensure length m
    else:
        ks = np.arange(-half, -half + m)

    fk = ks * df
    amps = amplitude_taper(m, amp_mode)

    if phase_mode == "newman":
        ph = newman_phases(m)
    elif phase_mode == "none":
        ph = np.zeros(m)
    else:
        raise ValueError("phase_mode must be 'newman' or 'none'")

    # Payload selection
    payload_count = int(max(0, round(payload_qpsk_fraction * m)))
    pilot_mask = np.ones(m, dtype=bool)
    if payload_count > 0:
        # Choose the highest-frequency bins for payload to avoid DC
        order = np.argsort(np.abs(fk))
        payload_idx = order[::-1][:payload_count]
        pilot_mask[payload_idx] = False
        # Build QPSK stream per payload tone
        sym_dur = int(max(1, round(fs / payload_symbol_rate)))
        nsyms = int(np.ceil(n / sym_dur))
        syms = qpsk_symbols(nsyms, rng)
    else:
        payload_idx = np.array([], dtype=int)
        sym_dur = 1
        syms = np.array([1+0j])

    x = np.zeros(n, dtype=complex)
    for i, (f, a, p) in enumerate(zip(fk, amps, ph)):
        if i in payload_idx:
            # piecewise constant QPSK symbol modulation
            s = np.repeat(syms, sym_dur)[:n]
        else:
            s = 1.0
        x += a * s * np.exp(1j * (2 * np.pi * f * t + p))

    # Normalize power
    x /= np.sqrt(np.mean(np.abs(x)**2) + 1e-12)
    return x, fk, pilot_mask

