import numpy as np

from alg.chronometric_handshake import (
    ChronometricHandshakeConfig,
    ChronometricHandshakeSimulator,
    ChronometricNode,
    ChronometricNodeConfig,
)


def _make_nodes() -> tuple[ChronometricNode, ChronometricNode]:
    node_a = ChronometricNode(
        ChronometricNodeConfig(
            node_id=0,
            carrier_freq_hz=2.4e9,
            phase_offset_rad=0.1,
            clock_bias_s=2.0e-9,
            freq_error_ppm=0.2,
        )
    )
    node_b = ChronometricNode(
        ChronometricNodeConfig(
            node_id=1,
            carrier_freq_hz=2.4e9 + 100e3,
            phase_offset_rad=2.1,
            clock_bias_s=-3.0e-9,
            freq_error_ppm=-0.1,
        )
    )
    return node_a, node_b


def _run_bias(calib_mode: str) -> float:
    cfg = ChronometricHandshakeConfig(
        retune_offsets_hz=(1e6,),
        d_tx_ns={0: 20.0, 1: 24.0},
        d_rx_ns={0: 12.0, 1: 18.0},
        calibration_mode=calib_mode,
        loopback_cal_noise_ps=5.0,
        delta_t_schedule_us=(0.0, 1.5),
    )
    simulator = ChronometricHandshakeSimulator(cfg)
    node_a, node_b = _make_nodes()
    result, _ = simulator.run_two_way(
        node_a=node_a,
        node_b=node_b,
        distance_m=120.0,
        snr_db=40.0,
        rng=np.random.default_rng(99),
    )
    return abs(result.reciprocity_bias_s) * 1e12


def test_reciprocity_calibration_modes() -> None:
    bias_off = _run_bias('off')
    bias_perfect = _run_bias('perfect')
    bias_loopback = _run_bias('loopback')

    assert bias_perfect < 1e-6
    assert bias_loopback < 0.1 * bias_off
