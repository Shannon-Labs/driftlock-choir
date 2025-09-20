"""
Phase 2 Simulation: variance-weighted decentralized consensus across a
random geometric network using Chronometric Interferometry measurements.
"""

from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

# Add src/ to import path for local execution
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from alg.chronometric_handshake import (
    ChronometricHandshakeConfig,
    ChronometricHandshakeSimulator,
    ChronometricNode,
    ChronometricNodeConfig,
    simulate_handshake_pair,
)
from alg.consensus import (
    ConsensusOptions,
    ConsensusResult,
    DecentralizedChronometricConsensus,
)
from alg.spectral_predictor import predict_iterations_to_rmse
from alg.weights import build_weight_matrix
from utils.io import append_csv_row, dump_run_config, echo_config, ensure_directory, save_json
from utils.plotting import save_figure


@dataclass
class Phase2Config:
    """Configuration for the Phase 2 network consensus study."""

    n_nodes: int = 50
    area_size_m: float = 500.0
    comm_range_m: float = 180.0
    snr_db: float = 20.0
    base_carrier_hz: float = 2.4e9
    freq_offset_span_hz: float = 80e3
    clock_bias_std_ps: float = 25.0
    clock_ppm_std: float = 2.0
    handshake_delta_f_hz: float = 100e3

    retune_offsets_hz: Tuple[float, ...] = (1e6,)
    coarse_enabled: bool = True
    coarse_bandwidth_hz: float = 20e6
    coarse_duration_s: float = 5e-6
    coarse_variance_floor_ps: float = 50.0
    multipath_two_ray_alpha: Optional[float] = None
    multipath_two_ray_delay_s: Optional[float] = None

    max_iterations: int = 1000
    timestep_s: float = 1e-3
    convergence_threshold_ps: float = 100.0
    asynchronous: bool = False

    rng_seed: Optional[int] = 42
    save_results: bool = True
    plot_results: bool = True
    results_dir: str = "results/phase2"

    spectral_margin: float = 0.8
    epsilon_override: Optional[float] = None
    weighting: str = 'inverse_variance'
    target_rmse_ps: float = 100.0
    target_streak: int = 3
    local_kf_enabled: bool = True
    local_kf_sigma_T_ps: float = 10.0
    local_kf_sigma_f_hz: float = 5.0
    local_kf_init_var_T_ps: float = 1e5
    local_kf_init_var_f_hz: float = 1e3
    local_kf_max_abs_ps: float = 500.0
    local_kf_max_abs_freq_hz: float = 1e5
    local_kf_clock_gain: float = 0.18
    local_kf_freq_gain: float = 0.05
    local_kf_iterations: int = 1

    def handshake_config(self) -> ChronometricHandshakeConfig:
        return ChronometricHandshakeConfig(
            beat_duration_s=20e-6,
            baseband_rate_factor=20.0,
            min_baseband_rate_hz=200_000.0,
            min_adc_rate_hz=20_000.0,
            filter_relative_bw=1.4,
            phase_noise_psd=-80.0,
            jitter_rms_s=1e-12,
            retune_offsets_hz=self.retune_offsets_hz,
            coarse_enabled=self.coarse_enabled,
            coarse_bandwidth_hz=self.coarse_bandwidth_hz,
            coarse_duration_s=self.coarse_duration_s,
            coarse_variance_floor_ps=self.coarse_variance_floor_ps,
            multipath_two_ray_alpha=self.multipath_two_ray_alpha,
            multipath_two_ray_delay_s=self.multipath_two_ray_delay_s,
        )


class Phase2Simulation:
    """Phase 2 driver orchestrating topology, measurements, and consensus."""

    def __init__(self, config: Phase2Config):
        self.config = config
        if self.config.save_results:
            ensure_directory(self.config.results_dir)
        self.handshake = ChronometricHandshakeSimulator(self.config.handshake_config())
        self.rng = np.random.default_rng(self.config.rng_seed)

    def run(self) -> Dict[str, Any]:
        positions = self._sample_positions()
        graph = self._build_graph(positions)
        nodes = self._build_nodes()
        self._populate_measurements(graph, nodes, positions)

        true_state = self._oracle_state(nodes)

        initial_state, kf_metrics = self._prepare_initial_state(graph, true_state)

        options = ConsensusOptions(
            max_iterations=self.config.max_iterations,
            epsilon=self.config.epsilon_override,
            tolerance_ps=self.config.convergence_threshold_ps,
            asynchronous=self.config.asynchronous,
            rng_seed=None if self.config.rng_seed is None else self.config.rng_seed + 1,
            enforce_zero_mean=True,
            spectral_margin=self.config.spectral_margin,
        )
        solver = DecentralizedChronometricConsensus(graph, options)
        result = solver.run(initial_state, true_state)
        predicted_iterations, measured_iterations = self._predict_iterations(result)

        telemetry = self._compile_results(
            graph=graph,
            positions=positions,
            nodes=nodes,
            result=result,
            predicted_iterations=predicted_iterations,
            measured_iterations=measured_iterations,
            kf_metrics=kf_metrics,
        )

        if self.config.save_results:
            self._append_run_csv(
                telemetry=telemetry,
                predicted_iterations=predicted_iterations,
                measured_iterations=measured_iterations,
                kf_metrics=kf_metrics,
            )
            self._save_results(telemetry)
        if self.config.plot_results:
            self._generate_plots(graph, positions, result)

        return telemetry

    # Network construction --------------------------------------------

    def _sample_positions(self) -> Dict[int, np.ndarray]:
        positions: Dict[int, np.ndarray] = {}
        for node in range(self.config.n_nodes):
            positions[node] = self.rng.uniform(
                0.0,
                self.config.area_size_m,
                size=2,
            )
        return positions

    def _build_graph(self, positions: Dict[int, np.ndarray]) -> nx.Graph:
        attempts = 0
        while True:
            graph = nx.random_geometric_graph(
                self.config.n_nodes,
                radius=self.config.comm_range_m,
                pos=positions,
                seed=None,
            )
            if nx.is_connected(graph):
                break
            attempts += 1
            if attempts > 10:
                raise RuntimeError("Failed to draw a connected topology within 10 attempts")
            # Resample positions and try again
            positions.update(self._sample_positions())
        return graph

    def _build_nodes(self) -> Dict[int, ChronometricNode]:
        nodes: Dict[int, ChronometricNode] = {}
        for node_id in range(self.config.n_nodes):
            freq_offset = self.rng.uniform(
                -self.config.freq_offset_span_hz / 2.0,
                self.config.freq_offset_span_hz / 2.0,
            )
            ppm_error = self.rng.normal(0.0, self.config.clock_ppm_std)
            intentional_offset = (
                self.config.handshake_delta_f_hz / 2.0
                if node_id % 2
                else -self.config.handshake_delta_f_hz / 2.0
            )
            carrier = (
                self.config.base_carrier_hz
                + intentional_offset
                + freq_offset
            ) * (1.0 + ppm_error * 1e-6)
            clock_bias = self.rng.normal(0.0, self.config.clock_bias_std_ps * 1e-12)
            phase = self.rng.uniform(0.0, 2.0 * np.pi)

            nodes[node_id] = ChronometricNode(
                ChronometricNodeConfig(
                    node_id=node_id,
                    carrier_freq_hz=carrier,
                    phase_offset_rad=phase,
                    clock_bias_s=clock_bias,
                    freq_error_ppm=ppm_error,
                )
            )
        return nodes

    def _populate_measurements(
        self,
        graph: nx.Graph,
        nodes: Dict[int, ChronometricNode],
        positions: Dict[int, np.ndarray],
    ) -> None:
        for u, v in graph.edges():
            distance = float(np.linalg.norm(positions[u] - positions[v]))
            result, _ = simulate_handshake_pair(
                node_a=nodes[u],
                node_b=nodes[v],
                distance_m=distance,
                snr_db=self.config.snr_db,
                rng=self.rng,
                simulator=self.handshake,
                retune_offsets_hz=self.config.retune_offsets_hz,
            )

            var_tau_u = result.forward.running_variance_tau or float(result.forward.effective_tau_variance_s2)
            var_tau_v = result.reverse.running_variance_tau or float(result.reverse.effective_tau_variance_s2)
            var_df_u = result.forward.running_variance_delta_f or float(result.forward.covariance[1, 1])
            var_df_v = result.reverse.running_variance_delta_f or float(result.reverse.covariance[1, 1])

            sigma_tau_sq = float(var_tau_u + var_tau_v)
            sigma_df_sq = float(var_df_u + var_df_v)

            tau_floor_s2 = (self.config.coarse_variance_floor_ps * 1e-12) ** 2
            tau_variance_floor = tau_floor_s2  # seconds^2 floor
            # Include coarse floor contribution for forward and reverse in predicted edge variance
            sigma_tau_sq = sigma_tau_sq + 2.0 * tau_floor_s2
            freq_variance_floor = 1.0  # Hz^2 floor to avoid singular weights
            sigma_tau_sq = max(sigma_tau_sq, tau_variance_floor)
            sigma_df_sq = max(sigma_df_sq, freq_variance_floor)

            measurement = np.array(
                [
                    result.forward.tau_est_s - result.reverse.tau_est_s,
                    result.forward.delta_f_est_hz - result.reverse.delta_f_est_hz,
                ],
                dtype=float,
            )

            sigma_tau_sq_ps = sigma_tau_sq * 1e24
            weight_matrix = build_weight_matrix(
                self.config.weighting,
                graph,
                u,
                v,
                sigma_tau_sq_ps,
                sigma_df_sq,
            )

            graph.edges[u, v]['distance_m'] = distance
            graph.edges[u, v]['measurement'] = measurement
            graph.edges[u, v]['measurement_true'] = np.array([
                result.forward.tau_true_s - result.reverse.tau_true_s,
                result.forward.delta_f_true_hz - result.reverse.delta_f_true_hz,
            ], dtype=float)
            graph.edges[u, v]['weight_matrix'] = weight_matrix
            graph.edges[u, v]['sigma_tau_sq'] = sigma_tau_sq
            graph.edges[u, v]['sigma_df_sq'] = sigma_df_sq
            graph.edges[u, v]['tau_variance_floor_s2'] = tau_floor_s2
            graph.edges[u, v]['orientation'] = (u, v)
            graph.edges[u, v]['residual_phase_rms'] = (
                result.forward.residual_phase_rms + result.reverse.residual_phase_rms
            ) * 0.5
            graph.edges[u, v]['coarse_tof_est_s'] = result.coarse_tof_est_s
            graph.edges[u, v]['coarse_error_s'] = (
                result.coarse_tof_est_s - result.tof_true_s if result.coarse_tof_est_s is not None else None
            )
            graph.edges[u, v]['carrier_count'] = len(result.forward.carrier_frequencies_hz)
            graph.edges[u, v]['weighting'] = self.config.weighting
            # Store LS covariance diagnostics per direction for telemetry
            graph.edges[u, v]['forward_cov'] = result.forward.covariance
            graph.edges[u, v]['reverse_cov'] = result.reverse.covariance
            # Alias-resolution diagnostics (if available)
            graph.edges[u, v]['alias_resolved_forward'] = bool(result.forward.alias_resolved) if result.forward.alias_resolved is not None else None
            graph.edges[u, v]['alias_resolved_reverse'] = bool(result.reverse.alias_resolved) if result.reverse.alias_resolved is not None else None

    def _oracle_state(self, nodes: Dict[int, ChronometricNode]) -> np.ndarray:
        clock_offsets = np.array([node.clock_bias_s for node in nodes.values()], dtype=float)
        freq_offsets = np.array(
            [node.carrier_freq_hz for node in nodes.values()],
            dtype=float,
        )

        clock_offsets -= np.mean(clock_offsets)
        freq_offsets -= np.mean(freq_offsets)

        return np.column_stack((clock_offsets, freq_offsets))

    # Results ----------------------------------------------------------

    def _prepare_initial_state(
        self,
        graph: nx.Graph,
        true_state: np.ndarray,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        baseline_state = np.zeros((self.config.n_nodes, 2), dtype=float)
        initial_clock_rms_ps, initial_freq_rms_hz = self._state_rms(baseline_state, true_state)
        metrics: Dict[str, Any] = {
            'enabled': False,
            'mode': 'off',
            'initial_clock_rms_ps': initial_clock_rms_ps,
            'initial_freq_rms_hz': initial_freq_rms_hz,
            'filtered_clock_rms_ps': initial_clock_rms_ps,
            'filtered_freq_rms_hz': initial_freq_rms_hz,
        }
        result_state = baseline_state
        if self.config.local_kf_enabled:
            filtered_state = self._run_local_kf(graph)
            filtered_clock_rms_ps, filtered_freq_rms_hz = self._state_rms(filtered_state, true_state)
            metrics['enabled'] = True
            metrics['mode'] = 'on'
            metrics['filtered_clock_rms_ps'] = filtered_clock_rms_ps
            metrics['filtered_freq_rms_hz'] = filtered_freq_rms_hz
            result_state = filtered_state
        metrics['clock_improvement_ps'] = float(
            metrics['initial_clock_rms_ps'] - metrics['filtered_clock_rms_ps']
        )
        metrics['freq_improvement_hz'] = float(
            metrics['initial_freq_rms_hz'] - metrics['filtered_freq_rms_hz']
        )
        metrics['clock_ratio'] = _safe_ratio(
            metrics['filtered_clock_rms_ps'],
            metrics['initial_clock_rms_ps'],
        )
        metrics['freq_ratio'] = _safe_ratio(
            metrics['filtered_freq_rms_hz'],
            metrics['initial_freq_rms_hz'],
        )
        return result_state, metrics

    def _run_local_kf(self, graph: nx.Graph) -> np.ndarray:
        clock_gain = max(float(self.config.local_kf_clock_gain), 0.0)
        freq_gain = max(float(self.config.local_kf_freq_gain), 0.0)
        iterations = max(int(self.config.local_kf_iterations), 1)

        state = np.zeros((self.config.n_nodes, 2), dtype=float)
        if clock_gain <= 0.0 and freq_gain <= 0.0:
            return state

        tau_floor = max((self.config.local_kf_sigma_T_ps * 1e-12) ** 2, 1e-24)
        freq_floor = max(self.config.local_kf_sigma_f_hz ** 2, 1.0)
        clamp_clock = self.config.local_kf_max_abs_ps * 1e-12 if self.config.local_kf_max_abs_ps > 0 else None
        clamp_freq = self.config.local_kf_max_abs_freq_hz if self.config.local_kf_max_abs_freq_hz > 0 else None

        for _ in range(iterations):
            updated = state.copy()
            for node in graph.nodes():
                sum_clock = 0.0
                weight_clock = 0.0
                sum_freq = 0.0
                weight_freq = 0.0

                for neighbor, data in graph[node].items():
                    measurement = np.array(data['measurement'], dtype=float)
                    orientation = data.get('orientation', (node, neighbor))
                    if orientation[0] != node:
                        measurement = -measurement

                    sigma_tau_sq = float(data.get('sigma_tau_sq', tau_floor))
                    sigma_df_sq = float(data.get('sigma_df_sq', freq_floor))
                    w_clock = 1.0 / max(sigma_tau_sq, tau_floor)
                    w_freq = 1.0 / max(sigma_df_sq, freq_floor)

                    sum_clock += w_clock * measurement[0]
                    weight_clock += w_clock
                    sum_freq += w_freq * measurement[1]
                    weight_freq += w_freq

                if weight_clock > 0.0 and clock_gain > 0.0:
                    avg_clock = sum_clock / weight_clock
                    if clamp_clock is not None:
                        avg_clock = float(np.clip(avg_clock, -clamp_clock, clamp_clock))
                    updated[node, 0] = state[node, 0] - clock_gain * avg_clock
                if weight_freq > 0.0 and freq_gain > 0.0:
                    avg_freq = sum_freq / weight_freq
                    if clamp_freq is not None:
                        avg_freq = float(np.clip(avg_freq, -clamp_freq, clamp_freq))
                    updated[node, 1] = state[node, 1] - freq_gain * avg_freq

            updated[:, 0] -= np.mean(updated[:, 0])
            updated[:, 1] -= np.mean(updated[:, 1])
            state = updated

        if clamp_clock is not None:
            state[:, 0] = np.clip(state[:, 0], -clamp_clock, clamp_clock)
        if clamp_freq is not None:
            state[:, 1] = np.clip(state[:, 1], -clamp_freq, clamp_freq)
        return state

    def _state_rms(self, state: np.ndarray, true_state: np.ndarray) -> Tuple[float, float]:
        error = state - true_state
        clock_rms_ps = float(np.sqrt(np.mean(error[:, 0] ** 2)) * 1e12)
        freq_rms_hz = float(np.sqrt(np.mean(error[:, 1] ** 2)))
        return clock_rms_ps, freq_rms_hz

    def _predict_iterations(self, result: ConsensusResult) -> Tuple[Optional[float], Optional[int]]:
        if result.timing_rms_ps.size == 0:
            return None, None
        initial_rmse_ps = float(result.timing_rms_ps[0])
        predicted = predict_iterations_to_rmse(
            target_rmse_ps=self.config.target_rmse_ps,
            initial_rmse_ps=initial_rmse_ps,
            lambda2=result.lambda_2,
            epsilon=result.epsilon,
        )
        predicted_val = float(predicted) if math.isfinite(predicted) else None
        measured_val = self._measure_iterations(result.timing_rms_ps)
        return predicted_val, measured_val

    def _measure_iterations(self, timing_series: np.ndarray) -> Optional[int]:
        target = self.config.target_rmse_ps
        streak = max(self.config.target_streak, 1)
        for idx in range(len(timing_series)):
            window = timing_series[idx : idx + streak]
            if window.size < streak:
                break
            if np.all(window <= target):
                return int(idx)
        return None

    def _compile_results(
        self,
        graph: nx.Graph,
        positions: Dict[int, np.ndarray],
        nodes: Dict[int, ChronometricNode],
        result: ConsensusResult,
        predicted_iterations: Optional[float],
        measured_iterations: Optional[int],
        kf_metrics: Dict[str, Any],
    ) -> Dict[str, Any]:
        final_state = result.state_history[-1]
        iteration_axis = np.arange(result.timing_rms_ps.size)
        time_ms = iteration_axis * self.config.timestep_s * 1e3

        convergence_iteration = result.convergence_iteration
        convergence_time_ms = (
            None
            if convergence_iteration is None
            else convergence_iteration * self.config.timestep_s * 1e3
        )
        success_under_5ms = (
            convergence_time_ms is not None and convergence_time_ms <= 5.0
        )

        timing_residuals_ps = [
            float(residual[0] * 1e12) for residual in result.edge_residuals.values()
        ]
        freq_residuals_hz = [
            float(residual[1]) for residual in result.edge_residuals.values()
        ]

        degrees = np.array([deg for _, deg in graph.degree()], dtype=float)
        distances = [graph.edges[u, v]['distance_m'] for u, v in graph.edges()]

        edge_carriers = [data.get('carrier_count', 1) for _, _, data in graph.edges(data=True)]
        coarse_errors = [
            data['coarse_error_s']
            for _, _, data in graph.edges(data=True)
            if data.get('coarse_error_s') is not None
        ]

        telemetry: Dict[str, Any] = {
            'config': asdict(self.config),
            'network': {
                'n_nodes': self.config.n_nodes,
                'n_edges': graph.number_of_edges(),
                'avg_degree': float(np.mean(degrees)) if degrees.size else 0.0,
                'avg_distance_m': float(np.mean(distances)) if distances else 0.0,
                'positions': {str(node): positions[node].tolist() for node in graph.nodes()},
            },
            'consensus': {
                'epsilon': result.epsilon,
                'iterations': int(result.state_history.shape[0] - 1),
                'converged': result.converged,
                'convergence_iteration': convergence_iteration,
                'convergence_time_ms': convergence_time_ms,
                'success_under_5ms': success_under_5ms,
                'timing_rms_ps': result.timing_rms_ps.tolist(),
                'frequency_rms_hz': result.frequency_rms_hz.tolist(),
                'time_axis_ms': time_ms.tolist(),
                'lambda_max': result.lambda_max,
                'lambda_2': result.lambda_2,
                'spectral_gap': result.spectral_gap,
                'predicted_iterations': _clean_numeric(predicted_iterations),
                'measured_iterations': measured_iterations if measured_iterations is not None else None,
                'target_rmse_ps': self.config.target_rmse_ps,
                'target_streak': self.config.target_streak,
                'weighting_strategy': self.config.weighting,
            },
            'residuals': {
                'timing_ps': timing_residuals_ps,
                'frequency_hz': freq_residuals_hz,
            },
            'final_state': {
                'clock_offsets_ps': (final_state[:, 0] * 1e12).astype(float).tolist(),
                'frequency_offsets_hz': final_state[:, 1].astype(float).tolist(),
            },
            'edge_diagnostics': {
                'avg_carrier_count': _clean_numeric(float(np.mean(edge_carriers)) if edge_carriers else None),
                'coarse_tof_rmse_ps': _clean_numeric(
                    float(np.sqrt(np.mean(np.square(coarse_errors))) * 1e12)
                    if coarse_errors
                    else None
                ),
                # Estimator sanity telemetry (per-edge)
                'measurement_rmse_tau_ps': _edge_rmse(graph, index=0, scale=1e12),
                'measurement_rmse_df_hz': _edge_rmse(graph, index=1, scale=1.0),
                'avg_ls_tau_std_ps': _edge_avg_ls_std(graph, index=0, scale=1e12),
                'avg_ls_df_std_hz': _edge_avg_ls_std(graph, index=1, scale=1.0),
                'predicted_tau_std_ps': _edge_predicted_std(graph, index=0, scale=1e12),
                'predicted_df_std_hz': _edge_predicted_std(graph, index=1, scale=1.0),
                'rmse_over_predicted_tau': _safe_ratio(
                    _edge_rmse(graph, index=0, scale=1e12) or 0.0,
                    _edge_predicted_std(graph, index=0, scale=1e12) or float('nan')
                ),
                'rmse_over_predicted_df': _safe_ratio(
                    _edge_rmse(graph, index=1, scale=1.0) or 0.0,
                    _edge_predicted_std(graph, index=1, scale=1.0) or float('nan')
                ),
                'alias_resolved_rate': _alias_resolved_rate(graph),
            },
            'local_kf': kf_metrics,
        }
        return telemetry


    def _save_results(self, telemetry: Dict[str, Any]) -> None:
        out_path = os.path.join(self.config.results_dir, 'phase2_results.json')
        save_json(telemetry, out_path)
        print(f"Results saved to {out_path}")

    def _append_run_csv(
        self,
        telemetry: Dict[str, Any],
        predicted_iterations: Optional[float],
        measured_iterations: Optional[int],
        kf_metrics: Dict[str, Any],
    ) -> None:
        csv_path = os.path.join(self.config.results_dir, 'phase2_runs.csv')
        network = telemetry['network']
        consensus = telemetry['consensus']
        n_nodes = int(network['n_nodes'])
        n_edges = int(network['n_edges'])
        density = 0.0
        if n_nodes > 1:
            density = (2.0 * n_edges) / (n_nodes * (n_nodes - 1))
        fieldnames = [
            'timestamp',
            'n_nodes',
            'n_edges',
            'density',
            'weighting',
            'epsilon',
            'lambda2',
            'predicted_iterations',
            'measured_iterations',
            'convergence_iteration',
            'converged',
            'success_under_5ms',
            'target_rmse_ps',
            'target_streak',
            'local_kf_enabled',
            'local_kf_mode',
            'kf_initial_clock_rms_ps',
            'kf_filtered_clock_rms_ps',
            'kf_initial_freq_rms_hz',
            'kf_filtered_freq_rms_hz',
            'kf_clock_improvement_ps',
            'kf_freq_improvement_hz',
            'kf_clock_ratio',
            'kf_freq_ratio',
        ]
        epsilon_val = consensus['epsilon']
        epsilon_val = float(epsilon_val) if epsilon_val is not None else None
        lambda2_val = consensus['lambda_2']
        lambda2_val = float(lambda2_val) if lambda2_val is not None else None
        convergence_iteration = consensus['convergence_iteration']
        row = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'n_nodes': n_nodes,
            'n_edges': n_edges,
            'density': density,
            'weighting': self.config.weighting,
            'epsilon': epsilon_val,
            'lambda2': lambda2_val,
            'predicted_iterations': predicted_iterations,
            'measured_iterations': measured_iterations,
            'convergence_iteration': convergence_iteration if convergence_iteration is not None else None,
            'converged': consensus['converged'],
            'success_under_5ms': consensus['success_under_5ms'],
            'target_rmse_ps': self.config.target_rmse_ps,
            'target_streak': self.config.target_streak,
            'local_kf_enabled': kf_metrics['enabled'],
            'local_kf_mode': kf_metrics.get('mode'),
            'kf_initial_clock_rms_ps': kf_metrics['initial_clock_rms_ps'],
            'kf_filtered_clock_rms_ps': kf_metrics['filtered_clock_rms_ps'],
            'kf_initial_freq_rms_hz': kf_metrics['initial_freq_rms_hz'],
            'kf_filtered_freq_rms_hz': kf_metrics['filtered_freq_rms_hz'],
            'kf_clock_improvement_ps': kf_metrics['clock_improvement_ps'],
            'kf_freq_improvement_hz': kf_metrics['freq_improvement_hz'],
            'kf_clock_ratio': kf_metrics['clock_ratio'],
            'kf_freq_ratio': kf_metrics['freq_ratio'],
        }
        append_csv_row(csv_path, fieldnames, row)

    def _generate_plots(
        self,
        graph: nx.Graph,
        positions: Dict[int, np.ndarray],
        result: ConsensusResult,
    ) -> None:
        self._plot_topology(graph, positions, result)
        self._plot_convergence(result)
        self._plot_residuals(result)
        self._plot_measurement_diagnostics(graph)

    def _plot_topology(
        self,
        graph: nx.Graph,
        positions: Dict[int, np.ndarray],
        result: ConsensusResult,
    ) -> None:
        fig, ax = plt.subplots(figsize=(6, 6))
        pos_dict = {node: positions[node] for node in graph.nodes()}
        final_state = result.state_history[-1]
        node_colors = final_state[:, 0] * 1e12  # Convert to ps for visualization

        degrees = [deg for _, deg in graph.degree()]
        weight_values = [graph.edges[u, v]['weight_matrix'][0, 0] for u, v in graph.edges()]
        width_scale = 1.0
        if weight_values:
            max_weight = max(weight_values)
            if max_weight > 0:
                width_scale = 2.5 / max_weight

        nx.draw_networkx_nodes(
            graph,
            pos_dict,
            node_color=node_colors,
            cmap='coolwarm',
            node_size=80,
            ax=ax,
        )
        nx.draw_networkx_edges(
            graph,
            pos_dict,
            width=[0.5 + width_scale * w for w in weight_values],
            alpha=0.8,
            ax=ax,
        )
        ax.set_title('Topology with Clock Offsets (ps)')
        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')
        sm = plt.cm.ScalarMappable(cmap='coolwarm')
        sm.set_array(node_colors)
        plt.colorbar(sm, ax=ax, label='ΔT estimate (ps)')
        path = os.path.join(self.config.results_dir, 'phase2_topology.png')
        save_figure(fig, path)
        print(f"Saved topology plot to {path}")

    def _plot_convergence(self, result: ConsensusResult) -> None:
        time_ms = np.arange(result.timing_rms_ps.size) * self.config.timestep_s * 1e3

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].semilogy(time_ms, result.timing_rms_ps, marker='o')
        axes[0].axhline(
            self.config.convergence_threshold_ps,
            color='tab:orange',
            linestyle='--',
            label='Target (100 ps)',
        )
        axes[0].set_title('Timing RMS convergence')
        axes[0].set_xlabel('Time (ms)')
        axes[0].set_ylabel('RMSE (ps)')
        axes[0].legend()

        axes[1].semilogy(time_ms, np.maximum(result.frequency_rms_hz, 1e-6), marker='o')
        axes[1].set_title('Frequency RMS convergence')
        axes[1].set_xlabel('Time (ms)')
        axes[1].set_ylabel('RMSE (Hz)')

        path = os.path.join(self.config.results_dir, 'phase2_convergence.png')
        save_figure(fig, path)
        print(f"Saved convergence plot to {path}")

    def _plot_residuals(self, result: ConsensusResult) -> None:
        timing_residuals_ps = np.array(
            [residual[0] * 1e12 for residual in result.edge_residuals.values()],
            dtype=float,
        )
        freq_residuals_hz = np.array(
            [residual[1] for residual in result.edge_residuals.values()],
            dtype=float,
        )

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].hist(timing_residuals_ps, bins=20, color='tab:blue', alpha=0.8)
        axes[0].set_title('Edge timing residuals')
        axes[0].set_xlabel('Residual (ps)')
        axes[0].set_ylabel('Count')

        axes[1].hist(freq_residuals_hz, bins=20, color='tab:green', alpha=0.8)
        axes[1].set_title('Edge frequency residuals')
        axes[1].set_xlabel('Residual (Hz)')
        axes[1].set_ylabel('Count')

        path = os.path.join(self.config.results_dir, 'phase2_residuals.png')
        save_figure(fig, path)
        print(f"Saved residual plot to {path}")

    def _plot_measurement_diagnostics(self, graph: nx.Graph) -> None:
        # Gather metrics
        rmse_tau_ps = _edge_rmse(graph, index=0, scale=1e12) or 0.0
        rmse_df_hz = _edge_rmse(graph, index=1, scale=1.0) or 0.0
        ls_tau_std_ps = _edge_avg_ls_std(graph, index=0, scale=1e12) or 0.0
        ls_df_std_hz = _edge_avg_ls_std(graph, index=1, scale=1.0) or 0.0
        pred_tau_std_ps = _edge_predicted_std(graph, index=0, scale=1e12) or 0.0
        pred_df_std_hz = _edge_predicted_std(graph, index=1, scale=1.0) or 0.0
        alias_rate = _alias_resolved_rate(graph)

        labels = ['τ (ps)', 'Δf (Hz)']
        rmse_vals = [rmse_tau_ps, rmse_df_hz]
        pred_vals = [pred_tau_std_ps, pred_df_std_hz]
        ls_vals = [ls_tau_std_ps, ls_df_std_hz]

        x = np.arange(len(labels))
        width = 0.28
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.bar(x - width, rmse_vals, width, label='RMSE')
        ax.bar(x,         pred_vals, width, label='Predicted σ (fwd+rev)')
        ax.bar(x + width, ls_vals,   width, label='Avg LS σ (per-dir)')
        ax.set_xticks(x, labels)
        ax.set_title('Estimator Diagnostics')
        ax.legend()
        if alias_rate is not None:
            ax.text(0.02, 0.90, f'Alias resolved: {alias_rate:.2f}', transform=ax.transAxes)
        # Global Δf CRLB (approximate) annotation
        df_crlb_std = _global_df_crlb_std(graph)
        if df_crlb_std is not None:
            ax.text(0.02, 0.82, f'Global Δf CRLB≈{df_crlb_std:.1f} Hz', transform=ax.transAxes)

        path = os.path.join(self.config.results_dir, 'phase2_measurement_diag.png')
        save_figure(fig, path)
        print(f"Saved measurement diagnostics plot to {path}")


def _clean_numeric(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return float(value)


def _edge_rmse(graph: nx.Graph, index: int, scale: float) -> Optional[float]:
    """Compute RMSE of measurement errors across edges for component index (0: tau, 1: df)."""
    errors: List[float] = []
    for _, _, data in graph.edges(data=True):
        m = data.get('measurement')
        t = data.get('measurement_true')
        if m is None or t is None:
            continue
        try:
            errors.append(float(m[index] - t[index]))
        except Exception:
            continue
    if not errors:
        return None
    return float(np.sqrt(np.mean(np.square(errors))) * scale)


def _edge_avg_ls_std(graph: nx.Graph, index: int, scale: float) -> Optional[float]:
    """Average LS-derived standard deviation across edges for component index.

    index: 0 for tau, 1 for df; uses forward and reverse covariances averaged per edge.
    """
    vars_: List[float] = []
    for _, _, data in graph.edges(data=True):
        f_cov = data.get('forward_cov')
        r_cov = data.get('reverse_cov')
        if f_cov is None or r_cov is None:
            continue
        try:
            v = 0.5 * (float(f_cov[index, index]) + float(r_cov[index, index]))
            vars_.append(v)
        except Exception:
            continue
    if not vars_:
        return None
    return float(np.sqrt(np.mean(vars_)) * scale)


def _alias_resolved_rate(graph: nx.Graph) -> Optional[float]:
    flags: List[float] = []
    for _, _, data in graph.edges(data=True):
        fr = data.get('alias_resolved_forward')
        rr = data.get('alias_resolved_reverse')
        if isinstance(fr, bool) and isinstance(rr, bool):
            flags.append(1.0 if (fr and rr) else 0.0)
    if not flags:
        return None
    return float(np.mean(flags))


def _edge_predicted_std(graph: nx.Graph, index: int, scale: float) -> Optional[float]:
    """Predicted standard deviation for edge measurement (difference of forward and reverse).

    For τ and Δf, the edge measurement is forward - reverse, so Var = Var_fwd + Var_rev.
    index: 0 for τ (seconds^2), 1 for Δf (Hz^2).
    """
    pred_vars: List[float] = []
    for _, _, data in graph.edges(data=True):
        f_cov = data.get('forward_cov')
        r_cov = data.get('reverse_cov')
        if f_cov is None or r_cov is None:
            continue
        try:
            v = float(f_cov[index, index]) + float(r_cov[index, index])
            # Include coarse floor for τ (index 0) if available
            if index == 0 and 'tau_variance_floor_s2' in data:
                v += 2.0 * float(data['tau_variance_floor_s2'])
            pred_vars.append(v)
        except Exception:
            continue
    if not pred_vars:
        return None
    return float(np.sqrt(np.mean(pred_vars)) * scale)


def _global_df_crlb_std(graph: nx.Graph) -> Optional[float]:
    """Approximate a global Δf CRLB standard deviation using beat duration and ADC heuristic.

    Uses per-edge average |Δf_true| to select an effective ADC rate (≈max(4·|Δf|, min_adc_rate)),
    beat duration from the Phase 2 config defaults (20 µs), and residual phase RMS as σ_phase.
    This is an approximation for quick telemetry only.
    """
    # Gather residual phase RMS and true Δf if present
    df_abs: List[float] = []
    sigma_phase: List[float] = []
    for _, _, data in graph.edges(data=True):
        m_true = data.get('measurement_true')
        if m_true is None:
            continue
        df_abs.append(abs(float(m_true[1])))
        if 'residual_phase_rms' in data and isinstance(data['residual_phase_rms'], float):
            sigma_phase.append(float(data['residual_phase_rms']))
    if not df_abs or not sigma_phase:
        return None
    df_mean = float(np.mean(df_abs))
    sigma_phase_mean = float(np.mean(sigma_phase))

    beat_duration_s = 20e-6
    min_adc_rate_hz = 20_000.0
    adc_rate = max(4.0 * max(df_mean, 1.0), min_adc_rate_hz)
    n = max(int(beat_duration_s * adc_rate), 8)
    # Sum t^2 for t_k = k / adc_rate, k=0..n-1
    k = np.arange(n)
    sum_t2 = float(np.sum((k / adc_rate) ** 2))
    # Var(Δf) ≈ Var(slope)/(2π)^2, Var(slope) ≈ σ_phase^2 / sum(t^2)
    var_df = (sigma_phase_mean ** 2) / max(sum_t2, 1e-18) / ((2.0 * np.pi) ** 2)
    return float(np.sqrt(max(var_df, 0.0)))


def _safe_ratio(numerator: float, denominator: float) -> Optional[float]:
    if not math.isfinite(denominator) or abs(denominator) < 1e-12:
        return None
    if not math.isfinite(numerator):
        return None
    return float(numerator / denominator)


def build_phase2_cli(parser: argparse.ArgumentParser) -> None:
    parser.add_argument('--nodes', type=int, default=Phase2Config.n_nodes, help='Number of nodes in the random geometric graph.')
    parser.add_argument('--area-m', type=float, default=Phase2Config.area_size_m, help='Side length (meters) of the square deployment region.')
    parser.add_argument('--density', type=float, default=None, help='Target edge density (0-1] used to derive the communication range.')
    parser.add_argument('--comm-range-m', type=float, default=None, help='Communication radius in meters; overrides --density when provided.')
    parser.add_argument('--snr-db', type=float, default=Phase2Config.snr_db, help='Link SNR in dB for handshake measurements.')
    parser.add_argument('--weighting', type=str, choices=['inverse_variance', 'metropolis', 'metropolis_var', 'bx_surrogate'], default=Phase2Config.weighting, help='Consensus weighting strategy.')
    parser.add_argument('--target-rmse-ps', type=float, default=Phase2Config.target_rmse_ps, help='RMSE target (ps) for auto-stopping heuristics.')
    parser.add_argument('--target-streak', type=int, default=Phase2Config.target_streak, help='Number of consecutive iterations required under the RMSE target.')
    parser.add_argument('--max-iterations', type=int, default=Phase2Config.max_iterations, help='Maximum consensus iterations.')
    parser.add_argument('--timestep-ms', type=float, default=Phase2Config.timestep_s * 1e3, help='Consensus timestep in milliseconds.')
    parser.add_argument('--epsilon', type=float, default=None, help='Optional step size override for the consensus solver.')
    parser.add_argument('--spectral-margin', type=float, default=Phase2Config.spectral_margin, help='Spectral margin applied when epsilon is auto-selected.')
    parser.add_argument('--rng-seed', type=int, default=Phase2Config.rng_seed, help='Seed for reproducible randomness.')
    parser.add_argument('--retune-offsets-hz', type=str, default='1e6', help='Comma-separated retune offsets (Hz) used during handshakes.')
    parser.add_argument('--coarse-bw-hz', type=float, default=Phase2Config.coarse_bandwidth_hz, help='Coarse preamble bandwidth (Hz).')
    parser.add_argument('--coarse-duration-us', type=float, default=Phase2Config.coarse_duration_s * 1e6, help='Coarse preamble duration (microseconds).')
    parser.add_argument('--coarse-off', dest='coarse_enabled', action='store_false', help='Disable coarse delay estimation.')
    parser.set_defaults(coarse_enabled=Phase2Config.coarse_enabled)
    parser.add_argument('--local-kf', type=str, choices=['auto', 'on', 'off', 'baseline'], default='auto', help='Enable/disable the local Kalman pre-filter (auto uses config default, baseline matches legacy off).')
    parser.add_argument('--local-kf-sigma-T-ps', type=float, default=Phase2Config.local_kf_sigma_T_ps, help='Process noise σ_T for the local KF (ps).')
    parser.add_argument('--local-kf-sigma-f-hz', type=float, default=Phase2Config.local_kf_sigma_f_hz, help='Process noise σ_f for the local KF (Hz).')
    parser.add_argument('--local-kf-init-var-T-ps', type=float, default=Phase2Config.local_kf_init_var_T_ps, help='Initial variance of the KF clock state (ps^2).')
    parser.add_argument('--local-kf-init-var-f-hz', type=float, default=Phase2Config.local_kf_init_var_f_hz, help='Initial variance of the KF frequency state (Hz^2).')
    parser.add_argument('--local-kf-max-abs-ps', type=float, default=Phase2Config.local_kf_max_abs_ps, help='Clamp absolute KF clock offset magnitude (ps).')
    parser.add_argument('--local-kf-max-abs-f-hz', type=float, default=Phase2Config.local_kf_max_abs_freq_hz, help='Clamp absolute KF frequency offset magnitude (Hz).')
    parser.add_argument('--local-kf-clock-gain', type=float, default=Phase2Config.local_kf_clock_gain, help='Shrink factor applied to variance-weighted neighbour timing residuals.')
    parser.add_argument('--local-kf-freq-gain', type=float, default=Phase2Config.local_kf_freq_gain, help='Shrink factor applied to variance-weighted neighbour frequency residuals.')
    parser.add_argument('--local-kf-iters', type=int, default=Phase2Config.local_kf_iterations, help='Number of local smoothing passes before consensus.')
    parser.add_argument('--save-results', dest='save_results', action='store_true', help='Persist JSON/CSV outputs (default on).')
    parser.add_argument('--no-save-results', dest='save_results', action='store_false', help='Skip writing JSON/CSV outputs.')
    parser.add_argument('--make-plots', dest='plot_results', action='store_true', help='Generate PNG plots of topology/convergence (default on).')
    parser.add_argument('--no-plots', dest='plot_results', action='store_false', help='Disable plot generation.')
    parser.set_defaults(save_results=True, plot_results=True)
    parser.add_argument('--results-dir', type=str, default='results/phase2', help='Directory for JSON/CSV/plot artifacts.')
    parser.add_argument('--dry-run', action='store_true', help='Echo resolved configuration and exit without executing.')
    parser.add_argument('--echo-config', action='store_true', help='Print the resolved Phase 2 configuration JSON before running.')


def main() -> None:
    parser = argparse.ArgumentParser(description='Phase 2 decentralized consensus harness')
    build_phase2_cli(parser)
    args = parser.parse_args()

    retune_offsets = _parse_float_sequence(args.retune_offsets_hz)
    if not retune_offsets:
        retune_offsets = (1e6,)

    comm_range = args.comm_range_m
    if args.density is not None:
        if comm_range is not None:
            raise ValueError('Specify at most one of --density or --comm-range-m')
        comm_range = _radius_from_density(args.density, args.area_m)
    if comm_range is None:
        comm_range = Phase2Config.comm_range_m

    local_kf_enabled = _resolve_local_kf_flag(args.local_kf, Phase2Config.local_kf_enabled)

    cfg = Phase2Config(
        n_nodes=args.nodes,
        area_size_m=args.area_m,
        comm_range_m=comm_range,
        snr_db=args.snr_db,
        results_dir=args.results_dir,
        save_results=args.save_results,
        plot_results=args.plot_results,
        spectral_margin=args.spectral_margin,
        epsilon_override=args.epsilon,
        weighting=args.weighting,
        target_rmse_ps=args.target_rmse_ps,
        target_streak=args.target_streak,
        max_iterations=args.max_iterations,
        timestep_s=args.timestep_ms * 1e-3,
        rng_seed=args.rng_seed,
        retune_offsets_hz=retune_offsets,
        coarse_enabled=args.coarse_enabled,
        coarse_bandwidth_hz=args.coarse_bw_hz,
        coarse_duration_s=args.coarse_duration_us * 1e-6,
        local_kf_enabled=local_kf_enabled,
        local_kf_sigma_T_ps=args.local_kf_sigma_T_ps,
        local_kf_sigma_f_hz=args.local_kf_sigma_f_hz,
        local_kf_init_var_T_ps=args.local_kf_init_var_T_ps,
        local_kf_init_var_f_hz=args.local_kf_init_var_f_hz,
        local_kf_max_abs_ps=args.local_kf_max_abs_ps,
        local_kf_max_abs_freq_hz=args.local_kf_max_abs_f_hz,
        local_kf_clock_gain=args.local_kf_clock_gain,
        local_kf_freq_gain=args.local_kf_freq_gain,
        local_kf_iterations=args.local_kf_iters,
    )

    if args.echo_config or args.dry_run:
        print(echo_config(cfg, label='Phase2Config'))
    if args.dry_run:
        return

    simulation = Phase2Simulation(cfg)
    if cfg.save_results:
        dump_run_config(cfg.results_dir, cfg, filename='phase2_config.json')

    telemetry = simulation.run()
    if not cfg.save_results:
        # Provide a minimal inline summary when nothing is persisted.
        final_rmse = telemetry['consensus']['timing_rms_ps'][-1]
        print(f"Final timing RMSE: {final_rmse:.2f} ps")


def _parse_float_sequence(text: str | None) -> Tuple[float, ...]:
    if not text:
        return ()
    values: List[float] = []
    for chunk in text.split(','):
        chunk = chunk.strip()
        if not chunk:
            continue
        try:
            values.append(float(chunk))
        except ValueError as exc:  # pragma: no cover - user input error path
            raise ValueError(f'Could not parse float from "{chunk}"') from exc
    return tuple(values)


def _radius_from_density(density: float, area_side_m: float) -> float:
    if density <= 0.0 or density > 1.0:
        raise ValueError('--density must lie in (0, 1]')
    radius = math.sqrt(density) * area_side_m / math.sqrt(math.pi)
    return float(min(radius, area_side_m))


def _resolve_local_kf_flag(flag: str, default: bool) -> bool:
    normalized = flag.lower()
    if normalized == 'auto':
        return default
    if normalized == 'on':
        return True
    if normalized in {'off', 'baseline'}:
        return False
    raise ValueError(f'Unknown --local-kf option: {flag}')


if __name__ == '__main__':
    main()
