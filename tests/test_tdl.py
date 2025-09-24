import numpy as np
import types

from alg.chronometric_handshake import (
    ChronometricHandshakeConfig,
    ChronometricHandshakeSimulator,
    ChronometricNode,
    ChronometricNodeConfig,
)
from chan.tdl import TappedDelayLine


def _make_nodes() -> tuple[ChronometricNode, ChronometricNode]:
    node_a = ChronometricNode(
        ChronometricNodeConfig(
            node_id=0,
            carrier_freq_hz=2.4e9,
            phase_offset_rad=0.0,
            clock_bias_s=0.0,
            freq_error_ppm=0.0,
        )
    )
    node_b = ChronometricNode(
        ChronometricNodeConfig(
            node_id=1,
            carrier_freq_hz=2.4e9 + 100e3,
            phase_offset_rad=0.0,
            clock_bias_s=0.0,
            freq_error_ppm=0.0,
        )
    )
    return node_a, node_b


def test_narrowband_response_phase_alignment() -> None:
    tau = 120e-9
    tdl = TappedDelayLine(
        delays_s=np.array([0.0, tau]),
        gains_c=np.array([1.0 + 0j, 0.5 + 0j]),
    )
    response = tdl.narrowband_response(1.0e9)
    expected_phase = -2.0 * np.pi * 1.0e9 * tau
    phase_error = np.unwrap([0.0, np.angle(response) - expected_phase])[1]
    assert abs(phase_error) < 1e-6


def test_coarse_delay_estimate_under_tdl_high_snr() -> None:
    delays = np.array([0.0, 35e-9])
    gains = np.array([1.0 + 0j, 0.4 + 0j])
    channel = TappedDelayLine(delays_s=delays, gains_c=gains)
    cfg = ChronometricHandshakeConfig(
        coarse_enabled=True,
        coarse_bandwidth_hz=20e6,
        coarse_duration_s=5e-6,
    )
    simulator = ChronometricHandshakeSimulator(cfg)
    simulator._sample_channel = types.MethodType(lambda self, rng: channel, simulator)
    node_a, node_b = _make_nodes()

    result, _ = simulator.run_two_way(
        node_a=node_a,
        node_b=node_b,
        distance_m=150.0,
        snr_db=55.0,
        rng=np.random.default_rng(77),
    )

    sample_rate = max(cfg.coarse_bandwidth_hz, cfg.min_baseband_rate_hz)
    sample_period = 1.0 / sample_rate
    forward_error = abs(result.forward.coarse_tau_est_s - result.forward.tau_true_s)

    assert forward_error <= sample_period
