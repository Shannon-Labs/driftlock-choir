from __future__ import annotations

import numpy as np
import pytest

from src.metrics.crlb import CRLBParams, JointCRLBCalculator
from driftlock_choir_sim.dsp.tx_comb import generate_comb
from driftlock_choir_sim.dsp.rx_coherent import (
    estimate_tone_phasors,
    estimate_noise_power,
    per_tone_snr,
)
from driftlock_choir_sim.dsp.time_delay import impose_fractional_delay_fft
from driftlock_choir_sim.dsp.channel_models import awgn


def _make_simple_observation() -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float, float]:
    """Synthesize a small comb observation with a known delay."""
    fs = 200_000.0
    duration = 0.002  # two milliseconds keeps the FFT sizes tiny
    df = 5_000.0
    m = 5
    truth_tau = 4e-8
    rng = np.random.default_rng(1234)

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
    return fk, phasors, snr, truth_tau, fs, duration


def test_joint_crlb_returns_finite_values() -> None:
    """The closed-form CRLB computation should yield finite, positive bounds."""
    fk, *_, fs, duration = _make_simple_observation()
    params = CRLBParams(
        snr_db=25.0,
        bandwidth=float(np.ptp(fk)) or 1.0,
        duration=duration,
        carrier_freq=2.4e9,
        sample_rate=fs,
    )
    calculator = JointCRLBCalculator(params)
    results = calculator.compute_joint_crlb()

    assert results["delay_crlb_std"] > 0.0
    assert results["frequency_crlb_std"] > 0.0
    assert np.isfinite(results["determinant_fim"])


@pytest.mark.filterwarnings("ignore:invalid value encountered")
def test_verify_crlb_consistency_handles_infinite_bounds() -> None:
    """Even when the CRLB is infinite the helper should return sensible fields."""
    fk, *_, fs, duration = _make_simple_observation()
    params = CRLBParams(
        snr_db=30.0,
        bandwidth=float(np.ptp(fk)) or 1.0,
        duration=duration,
        carrier_freq=2.45e9,
        sample_rate=fs,
    )
    calculator = JointCRLBCalculator(params)
    baseline = calculator.compute_joint_crlb()

    delay_std = baseline["delay_crlb_std"]
    freq_std = baseline["frequency_crlb_std"]
    ls_covariance = np.diag([max(delay_std**2, 1.0), max(freq_std**2, 1.0)])

    consistency = calculator.verify_crlb_consistency(
        mc_rmse_delay=max(delay_std, 1.0),
        mc_rmse_freq=max(freq_std, 1.0),
        ls_covariance=ls_covariance,
    )

    required_keys = {
        "mc_efficiency_delay",
        "mc_efficiency_freq",
        "crlb_vs_ls_delay_ratio",
        "crlb_vs_ls_freq_ratio",
        "crlb_consistent",
        "mc_reasonable",
        "overall_consistent",
    }
    assert required_keys.issubset(consistency)
    assert consistency["theoretical_crlb"]["delay_crlb_std"] == delay_std
    assert consistency["theoretical_crlb"]["frequency_crlb_std"] == freq_std
    assert isinstance(consistency["crlb_consistent"], (bool, np.bool_))
    assert isinstance(consistency["mc_reasonable"], (bool, np.bool_))
    assert isinstance(consistency["overall_consistent"], (bool, np.bool_))
