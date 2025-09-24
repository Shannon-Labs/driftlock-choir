from __future__ import annotations

import numpy as np
import pytest

from driftlock_choir_sim.dsp.tx_comb import generate_comb
from driftlock_choir_sim.dsp.rx_coherent import (
    estimate_tone_phasors,
    estimate_noise_power,
    per_tone_snr,
)
from driftlock_choir_sim.dsp.ultra_precision_estimator import (
    PrecisionMode,
    ultra_precision_timing_estimator,
)
from driftlock_choir_sim.dsp.time_delay import impose_fractional_delay_fft
from driftlock_choir_sim.dsp.channel_models import awgn


def _snapshot(snr_db: float = 35.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    fs = 500_000.0
    duration = 0.003
    df = 10_000.0
    m = 7
    truth_tau = 20e-9
    rng = np.random.default_rng(987)

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
    noise_power = estimate_noise_power(x, fs, fk, phasors)
    snr = per_tone_snr(phasors, noise_power, len(x))
    return fk, phasors, snr


@pytest.mark.parametrize(
    "mode",
    [PrecisionMode.HIGH_PRECISION, PrecisionMode.ROBUST],
)
def test_ultra_precision_estimator_reports_diagnostics(mode: PrecisionMode) -> None:
    fk, phasors, snr = _snapshot()
    tau_est, df_est, stats = ultra_precision_timing_estimator(
        phasors,
        fk,
        snr,
        mode=mode,
        return_stats=True,
    )

    assert np.isfinite(tau_est)
    assert np.isfinite(df_est)
    assert stats["precision_mode"] == mode.value
    assert stats["hypothesis_weights"].ndim == 1
    assert np.isclose(np.sum(stats["hypothesis_weights"]), 1.0)
    assert np.isfinite(stats["optimal_bandwidth_hz"])
