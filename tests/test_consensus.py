import math
import os
import sys
from dataclasses import replace

import networkx as nx
import numpy as np
import pytest

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from alg.chronometric_handshake import (
    ChronometricHandshakeConfig,
    ChronometricHandshakeSimulator,
    ChronometricNode,
    ChronometricNodeConfig,
    simulate_handshake_pair,
)
from alg.consensus import ConsensusOptions, DecentralizedChronometricConsensus
from sim.phase2 import Phase2Config, Phase2Simulation


def _build_nodes(rng: np.random.Generator) -> dict[int, ChronometricNode]:
    base_freq = 2.4e9
    delta_f = 100e3
    nodes: dict[int, ChronometricNode] = {}
    for node_id in range(4):
        ppm = rng.normal(0.0, 0.3)
        intentional = delta_f / 2.0 if node_id % 2 else -delta_f / 2.0
        carrier = (base_freq + intentional) * (1.0 + ppm * 1e-6)
        nodes[node_id] = ChronometricNode(
            ChronometricNodeConfig(
                node_id=node_id,
                carrier_freq_hz=carrier,
                phase_offset_rad=rng.uniform(0.0, 2.0 * np.pi),
                clock_bias_s=rng.normal(0.0, 20e-12),
                freq_error_ppm=ppm,
            )
        )
    return nodes


def _oracle_state(nodes: dict[int, ChronometricNode]) -> np.ndarray:
    clock_offsets = np.array([node.clock_bias_s for node in nodes.values()], dtype=float)
    freq_offsets = np.array([node.carrier_freq_hz for node in nodes.values()], dtype=float)
    clock_offsets -= np.mean(clock_offsets)
    freq_offsets -= np.mean(freq_offsets)
    return np.column_stack((clock_offsets, freq_offsets))


def _dense_phase2_config(local_kf: bool, rng_seed: int, results_dir: str) -> Phase2Config:
    area_size = 350.0
    density = 0.22
    comm_range = math.sqrt(density) * area_size / math.sqrt(math.pi)
    common_kwargs = dict(
        n_nodes=64,
        area_size_m=area_size,
        comm_range_m=comm_range,
        snr_db=20.0,
        freq_offset_span_hz=80e3,
        clock_bias_std_ps=25.0,
        clock_ppm_std=2.0,
        handshake_delta_f_hz=100e3,
        retune_offsets_hz=(1e6,),
        coarse_enabled=True,
        coarse_bandwidth_hz=20e6,
        coarse_duration_s=5e-6,
        coarse_variance_floor_ps=50.0,
        max_iterations=2000,
        timestep_s=0.5e-3,
        convergence_threshold_ps=90.0,
        asynchronous=False,
        rng_seed=rng_seed,
        save_results=False,
        plot_results=False,
        spectral_margin=0.8,
        epsilon_override=0.02,
        target_rmse_ps=90.0,
        target_streak=3,
        results_dir=results_dir,
    )
    if local_kf:
        return Phase2Config(
            weighting='metropolis_var',
            local_kf_enabled=True,
            local_kf_sigma_T_ps=5.0,
            local_kf_sigma_f_hz=2.0,
            local_kf_init_var_T_ps=5.0e3,
            local_kf_init_var_f_hz=150.0,
            local_kf_max_abs_ps=30.0,
            local_kf_max_abs_freq_hz=2000.0,
            local_kf_clock_gain=0.32,
            local_kf_freq_gain=0.03,
            local_kf_iterations=1,
            **common_kwargs,
        )
    return Phase2Config(
        weighting='metropolis',
        local_kf_enabled=False,
        **common_kwargs,
    )


def test_variance_weighted_consensus_converges() -> None:
    rng = np.random.default_rng(2025)
    simulator = ChronometricHandshakeSimulator(
        ChronometricHandshakeConfig(retune_offsets_hz=(1e6,))
    )
    nodes = _build_nodes(rng)

    graph = nx.Graph()
    graph.add_edges_from([(0, 1), (1, 2), (2, 3), (0, 3), (0, 2)])
    positions = {
        0: np.array([0.0, 0.0]),
        1: np.array([60.0, 0.0]),
        2: np.array([60.0, 60.0]),
        3: np.array([0.0, 60.0]),
    }

    for u, v in graph.edges():
        distance = float(np.linalg.norm(positions[u] - positions[v]))
        result, _ = simulate_handshake_pair(
            node_a=nodes[u],
            node_b=nodes[v],
            distance_m=distance,
            snr_db=55.0,
            rng=rng,
            simulator=simulator,
            retune_offsets_hz=(1e6,),
        )

        var_tau_u = result.forward.running_variance_tau or float(result.forward.effective_tau_variance_s2)
        var_tau_v = result.reverse.running_variance_tau or float(result.reverse.effective_tau_variance_s2)
        var_df_u = result.forward.running_variance_delta_f or float(result.forward.covariance[1, 1])
        var_df_v = result.reverse.running_variance_delta_f or float(result.reverse.covariance[1, 1])

        sigma_tau_sq = max(float(var_tau_u + var_tau_v), (50e-12) ** 2)
        sigma_df_sq = max(float(var_df_u + var_df_v), 1.0)

        weight_diag = np.array(
            [
                1.0 / max(sigma_tau_sq * 1e24, 1.0),
                1.0 / max(sigma_df_sq, 1.0),
            ],
            dtype=float,
        )
        graph.edges[u, v]['measurement'] = np.array(
            [
                result.forward.tau_est_s - result.reverse.tau_est_s,
                result.forward.delta_f_est_hz - result.reverse.delta_f_est_hz,
            ],
            dtype=float,
        )
        graph.edges[u, v]['weight_matrix'] = np.diag(weight_diag)
        graph.edges[u, v]['orientation'] = (u, v)

    initial_state = np.zeros((len(nodes), 2), dtype=float)
    true_state = _oracle_state(nodes)

    options = ConsensusOptions(
        max_iterations=200,
        tolerance_ps=80.0,
        asynchronous=False,
        spectral_margin=0.7,
    )
    solver = DecentralizedChronometricConsensus(graph, options)
    result = solver.run(initial_state, true_state)

    assert result.converged
    assert result.timing_rms_ps[-1] < 80.0
    assert np.isfinite(result.frequency_rms_hz[-1])
    assert result.lambda_2 >= 0.0
    assert math.isclose(result.spectral_gap, result.lambda_2)


def test_phase2_local_kf_metrics(tmp_path) -> None:
    base_cfg = Phase2Config(
        n_nodes=10,
        area_size_m=200.0,
        comm_range_m=140.0,
        snr_db=22.0,
        retune_offsets_hz=(1e6,),
        save_results=False,
        plot_results=False,
        rng_seed=322,
        max_iterations=120,
        target_rmse_ps=150.0,
        target_streak=2,
        results_dir=str(tmp_path / 'kf_on'),
    )
    sim_on = Phase2Simulation(base_cfg)
    telemetry_on = sim_on.run()
    kf_on = telemetry_on['local_kf']

    assert kf_on['enabled']
    assert kf_on['mode'] == 'on'
    assert kf_on['clock_improvement_ps'] is not None
    assert kf_on['clock_improvement_ps'] > -100.0
    assert kf_on['freq_improvement_hz'] is not None
    assert kf_on['freq_improvement_hz'] >= 0.0
    if kf_on['clock_ratio'] is not None:
        assert math.isfinite(kf_on['clock_ratio'])

    off_cfg = replace(base_cfg, local_kf_enabled=False, results_dir=str(tmp_path / 'kf_off'))
    sim_off = Phase2Simulation(off_cfg)
    telemetry_off = sim_off.run()
    kf_off = telemetry_off['local_kf']

    assert not kf_off['enabled']
    assert kf_off['mode'] == 'off'
    assert pytest.approx(0.0, abs=1e-6) == kf_off['clock_improvement_ps']
    assert pytest.approx(0.0, abs=1e-6) == kf_off['freq_improvement_hz']
    if kf_off['clock_ratio'] is not None:
        assert pytest.approx(1.0, abs=1e-6) == kf_off['clock_ratio']
    assert kf_on['clock_improvement_ps'] != kf_off['clock_improvement_ps']


def test_dense_kf_vs_baseline(tmp_path) -> None:
    seed = 5001
    cfg_on = _dense_phase2_config(True, seed, str(tmp_path / 'dense_kf_on'))
    sim_on = Phase2Simulation(cfg_on)
    rmse_on = sim_on.run()['consensus']['timing_rms_ps'][-1]

    cfg_off = _dense_phase2_config(False, seed, str(tmp_path / 'dense_kf_off'))
    sim_off = Phase2Simulation(cfg_off)
    rmse_off = sim_off.run()['consensus']['timing_rms_ps'][-1]

    improvement_ps = rmse_off - rmse_on
    assert rmse_on < rmse_off
    assert improvement_ps >= 1.0
