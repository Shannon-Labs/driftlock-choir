from __future__ import annotations

import numpy as np

from driftlock_choir_sim.dsp.tx_comb import generate_comb
from driftlock_choir_sim.dsp.rx_coherent import (
    estimate_tone_phasors,
    estimate_noise_power,
    per_tone_snr,
)
from driftlock_choir_sim.dsp.time_delay import impose_fractional_delay_fft
from driftlock_choir_sim.dsp.channel_models import awgn
from driftlock_choir_sim.dsp.super_resolution_estimator import (
    SuperResolutionConfig,
    SuperResolutionEstimator,
    SuperResolutionMethod,
)


def _snapshot() -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    fs = 500_000.0
    duration = 0.003
    df = 10_000.0
    m = 7
    truth_tau = 30e-9
    rng = np.random.default_rng(444)

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
    x = awgn(x, 30.0, rng)

    phasors = estimate_tone_phasors(x, fs, fk)
    noise_power = estimate_noise_power(x, fs, fk, phasors)
    snr = per_tone_snr(phasors, noise_power, len(x))
    return fk, phasors, snr, fs


def test_compressed_sensing_super_resolution_returns_metadata() -> None:
    fk, phasors, snr, fs = _snapshot()
    config = SuperResolutionConfig(
        method=SuperResolutionMethod.COMPRESSED_SENSING,
        n_sources=2,
        n_snapshots=5,
        threshold_db=-20.0,
    )
    estimator = SuperResolutionEstimator(fs=fs, config=config)
    tau_est, df_est, stats = estimator.estimate_timing(
        phasors,
        fk,
        snr,
        return_stats=True,
    )

    assert np.isfinite(tau_est)
    assert np.isfinite(df_est)
    assert stats["method"] == SuperResolutionMethod.COMPRESSED_SENSING.value
    assert stats["n_sources_detected"] == config.n_sources
    assert 0.0 <= stats["confidence"] <= 1.0
    assert np.isfinite(stats["snr_db"])
