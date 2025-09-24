from __future__ import annotations

import numpy as np
import pytest

from driftlock_choir_sim.dsp.tx_comb import generate_comb
from driftlock_choir_sim.dsp.rx_coherent import (
    estimate_tone_phasors,
    estimate_noise_power,
    per_tone_snr,
)
from driftlock_choir_sim.dsp.time_delay import impose_fractional_delay_fft
from driftlock_choir_sim.dsp.impairments import apply_cfo
from driftlock_choir_sim.dsp.channel_models import awgn
from driftlock_choir_sim.dsp.closed_form_estimator import closed_form_tau_df_estimator


def _make_snapshot() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create a compact observation with non-zero τ and Δf."""
    fs = 500_000.0
    duration = 0.003
    df = 10_000.0
    m = 7
    truth_tau = 40e-9
    truth_df = 500.0
    rng = np.random.default_rng(321)

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
    x = apply_cfo(x, fs, truth_df / fs)
    x = awgn(x, 35.0, rng)

    phasors = estimate_tone_phasors(x, fs, fk)
    noise_power = estimate_noise_power(x, fs, fk, phasors)
    snr = per_tone_snr(phasors, noise_power, len(x))
    return fk, phasors, snr


@pytest.mark.parametrize("method", ["geometric", "algebraic", "hybrid"])
def test_closed_form_estimator_produces_finite_stats(method: str) -> None:
    """All estimator variants should yield finite estimates and metadata."""
    fk, phasors, snr = _make_snapshot()
    tau_est, df_est, stats = closed_form_tau_df_estimator(
        phasors,
        fk,
        snr,
        method=method,
        return_stats=True,
    )

    assert np.isfinite(tau_est)
    assert np.isfinite(df_est)
    assert stats["crlb_tau_ps"] > 0.0
    assert stats["crlb_df_hz"] >= 0.0

    if method == "hybrid":
        assert stats["method_used"] == "hybrid"
        assert 0.0 <= stats["weight_geometric"] <= 1.0
        assert 0.0 <= stats["weight_algebraic"] <= 1.0
