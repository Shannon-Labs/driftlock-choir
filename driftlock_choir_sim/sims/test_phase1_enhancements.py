from __future__ import annotations

import numpy as np

from driftlock_choir_sim.dsp.tx_comb import generate_comb
from driftlock_choir_sim.dsp.rx_coherent import (
    estimate_tone_phasors,
    estimate_noise_power,
    per_tone_snr,
    traditional_wls_delay,
    unwrap_phase,
    wls_delay,
)
from driftlock_choir_sim.dsp.time_delay import impose_fractional_delay_fft
from driftlock_choir_sim.dsp.channel_models import awgn


def _synthesise_measurements(
    truth_tau: float = 50e-9,
    snr_db: float = 25.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Generate a compact tone comb with a deterministic timing offset."""
    fs = 1e6
    duration = 0.003
    df = 10_000.0
    m = 9
    rng = np.random.default_rng(2024)

    x, fk, *_ = generate_comb(
        fs,
        duration,
        df,
        m=m,
        omit_fundamental=True,
        rng=rng,
        return_payload=False,
    )
    x = impose_fractional_delay_fft(x, fs, truth_tau)
    x = awgn(x, snr_db, rng)

    phasors = estimate_tone_phasors(x, fs, fk)
    phases = unwrap_phase(np.angle(phasors), fk)
    noise_power = estimate_noise_power(x, fs, fk, phasors)
    snr = per_tone_snr(phasors, noise_power, len(x))
    return fk, phases, snr, truth_tau


def test_enhanced_wls_refines_delay_estimate() -> None:
    """The enhanced estimator should outperform the traditional weighting."""
    fk, phases, snr, truth_tau = _synthesise_measurements()
    traditional_tau, _ = traditional_wls_delay(fk, phases, snr)
    enhanced_tau, _ = wls_delay(fk, phases, snr)

    assert abs(enhanced_tau - truth_tau) < abs(traditional_tau - truth_tau)


def test_generate_comb_metadata_is_well_formed() -> None:
    """Comb metadata should describe the emitted tones and payload schedule."""
    fs = 1e6
    duration = 0.002
    df = 8_000.0
    m = 9
    rng = np.random.default_rng(7)

    x, fk, pilot_mask, meta = generate_comb(
        fs,
        duration,
        df,
        m=m,
        omit_fundamental=True,
        rng=rng,
        payload_qpsk_fraction=0.4,
        return_payload=True,
    )

    assert x.shape == (int(round(fs * duration)),)
    assert pilot_mask.dtype == bool
    assert meta["amps"].shape == fk.shape
    assert np.isfinite(meta["reconstruction_snr_db"])
    assert meta["samples_per_symbol"] >= 1
    assert np.isclose(np.mean(np.abs(x) ** 2), 1.0, rtol=0.05)
