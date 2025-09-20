from __future__ import annotations

from typing import Any, Dict

import numpy as np
from .phase_schedules import newman_phases, amplitude_taper


def qpsk_symbols(n: int, rng: np.random.Generator) -> np.ndarray:
    bits = rng.integers(0, 4, size=n)
    const = np.array([1 + 1j, -1 + 1j, -1 - 1j, 1 - 1j]) / np.sqrt(2)
    return const[bits]


CombBase = tuple[np.ndarray, np.ndarray, np.ndarray]
CombWithPayload = tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any] | None]


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
    return_payload: bool = False,
) -> CombBase | CombWithPayload:
    """Generate complex baseband multi-tone comb around 0 Hz.

    Returns ``(x, fk, pilot_mask)`` by default. When ``return_payload`` is true, the
    tuple additionally includes a metadata dictionary describing the payload
    carriers (or ``None`` when no payload is scheduled).
    """
    rng = np.random.default_rng() if rng is None else rng
    n = int(round(fs * duration))
    t = np.arange(n) / fs

    # Define carrier indices. In missing-fundamental mode, omit k=0 and use a
    # symmetric set around 0 Hz.
    half = m // 2
    if omit_fundamental:
        ks = np.concatenate([-(np.arange(1, half + 1)), np.arange(1, m - half + 1)])
        ks = ks[:m]
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

    payload_count = int(max(0, round(payload_qpsk_fraction * m)))
    pilot_mask = np.ones(m, dtype=bool)
    payload_mask = np.zeros(m, dtype=bool)

    sym_dur = int(max(1, round(fs / payload_symbol_rate)))
    payload_indices = np.array([], dtype=int)
    payload_symbols = np.array([], dtype=complex)

    if payload_count > 0:
        # Choose the highest-frequency bins for payload to avoid DC.
        order = np.argsort(np.abs(fk))
        payload_idx = order[::-1][:payload_count]
        payload_mask[payload_idx] = True
        pilot_mask[payload_idx] = False

        # Build QPSK stream that is shared across payload tones.
        n_syms = int(np.ceil(n / sym_dur))
        payload_symbols = qpsk_symbols(n_syms, rng)
    else:
        payload_symbols = np.array([], dtype=complex)

    x = np.zeros(n, dtype=complex)
    for i, (f, a, p) in enumerate(zip(fk, amps, ph)):
        if payload_mask[i]:
            s = np.repeat(payload_symbols, sym_dur)[:n]
        else:
            s = 1.0
        x += a * s * np.exp(1j * (2 * np.pi * f * t + p))

    # Normalize power
    x /= np.sqrt(np.mean(np.abs(x) ** 2) + 1e-12)

    if return_payload:
        meta: Dict[str, Any] = {
            "phases": ph.copy(),
            "amps": amps.copy(),
            "payload_indices": np.where(payload_mask)[0],
            "payload_symbols": payload_symbols,
            "samples_per_symbol": sym_dur,
            "symbol_rate": payload_symbol_rate,
        }
        return x, fk, pilot_mask, meta
    return x, fk, pilot_mask

