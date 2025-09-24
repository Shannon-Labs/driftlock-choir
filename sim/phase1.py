"""
Phase 1: Chronometric Interferometry two-node handshake simulation.

This module now supports two primary workflows:

1. The legacy SNR sweep used during early validation (still available via
   `Phase1Simulator.run_full_simulation`).
2. A configurable alias-failure mapping sweep that scans retune offsets,
   coarse preamble bandwidth, and SNR to quantify synthetic-wavelength
   robustness. Command-line execution focuses on the alias map workflow.
"""

from __future__ import annotations

import argparse
import math
import os
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import yaml

from pathlib import Path

# Add src/ to import path for local execution
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from alg.chronometric_handshake import (
    ChronometricHandshakeConfig,
    ChronometricHandshakeSimulator,
    ChronometricNode,
    ChronometricNodeConfig,
    HandshakeTrace,
    TwoWayHandshakeResult,
)
from mac.scheduler import MacSlots
from chan.tdl import TappedDelayLine, tdl_exponential, TDL_PROFILES, tdl_from_profile
from utils.io import dump_run_config, echo_config, ensure_directory, save_json, write_csv
from utils.plotting import heatmap as plot_heatmap, save_figure


@dataclass
class Phase1Config:
    """Simulation configuration for legacy Phase 1 sweep."""

    snr_values_db: List[float] = field(default_factory=lambda: [0, 5, 10, 15, 20, 25, 30])
    n_monte_carlo: int = 200
    distance_m: float = 100.0
    carrier_freq_hz: float = 2.4e9
    delta_f_hz: float = 100e3
    beat_duration_s: float = 20e-6
    baseband_rate_factor: float = 20.0
    min_baseband_rate_hz: float = 200_000.0
    min_adc_rate_hz: float = 20_000.0
    filter_relative_bw: float = 1.4
    phase_noise_psd: float = -80.0
    jitter_rms_s: float = 1e-12
    clock_bias_std_ps: float = 25.0
    clock_ppm_std: float = 2.0
    retune_offsets_hz: Tuple[float, ...] = (1e6,)
    coarse_enabled: bool = True
    coarse_bandwidth_hz: float = 20e6
    coarse_duration_s: float = 5e-6
    coarse_variance_floor_ps: float = 50.0
    multipath_two_ray_alpha: Optional[float] = None
    multipath_two_ray_delay_s: Optional[float] = None
    rng_seed: Optional[int] = 1234
    capture_trace_snr_db: Optional[float] = None
    save_results: bool = True
    plot_results: bool = True
    results_dir: str = "results/phase1"
    channel_model: Optional[TappedDelayLine] = None
    tdl_profile: Optional[str] = None
    delta_t_us: Tuple[float, ...] = (0.0,)
    calib_mode: str = 'off'
    loopback_cal_noise_ps: float = 10.0
    d_tx_ns: Dict[int, float] = field(default_factory=dict)
    d_rx_ns: Dict[int, float] = field(default_factory=dict)
    mac_slots: Optional[MacSlots] = field(
        default_factory=lambda: MacSlots(preamble_len=1024, narrowband_len=512, guard_us=10.0)
    )
    # Atomic mode parameters (overridden by YAML if atomic_mode)
    atomic_mode: bool = False
    atomic_config_path: str = 'sim/configs/atomic_reconciliation.yaml'
    atomic_phase_noise_psd: float = -140.0
    atomic_jitter_rms_s: float = 1e-13
    atomic_clock_bias_std_ps: float = 1.0
    atomic_clock_ppm_std: float = 0.1
    atomic_snr_db: float = 40.0
    atomic_beat_duration_s: float = 1.0
    atomic_retune_offsets_hz: Tuple[float, ...] = (1e6, 2e6, 5e6)
    atomic_coarse_bw_hz: float = 100e6
    atomic_delta_f_hz: float = 1e3

    def handshake_config(self) -> ChronometricHandshakeConfig:
        return ChronometricHandshakeConfig(
            beat_duration_s=self.beat_duration_s,
            baseband_rate_factor=self.baseband_rate_factor,
            min_baseband_rate_hz=self.min_baseband_rate_hz,
            min_adc_rate_hz=self.min_adc_rate_hz,
            filter_relative_bw=self.filter_relative_bw,
            phase_noise_psd=self.phase_noise_psd,
            jitter_rms_s=self.jitter_rms_s,
            retune_offsets_hz=self.retune_offsets_hz,
            coarse_enabled=self.coarse_enabled,
            coarse_bandwidth_hz=self.coarse_bandwidth_hz,
            coarse_duration_s=self.coarse_duration_s,
            coarse_variance_floor_ps=self.coarse_variance_floor_ps,
            multipath_two_ray_alpha=self.multipath_two_ray_alpha,
            multipath_two_ray_delay_s=self.multipath_two_ray_delay_s,
            channel_model=self.channel_model,
            delta_t_schedule_us=self.delta_t_us,
            calibration_mode=self.calib_mode,
            loopback_cal_noise_ps=self.loopback_cal_noise_ps,
            d_tx_ns=self.d_tx_ns if self.d_tx_ns else None,
            d_rx_ns=self.d_rx_ns if self.d_rx_ns else None,
            mac=self.mac_slots,
        )


class Phase1Simulator:
    """Driver for Phase 1 Chronometric Interferometry validation."""

    def __init__(self, config: Phase1Config):
        self.config = config
        if self.config.atomic_mode:
            self._apply_atomic_overrides()
        if self.config.save_results:
            ensure_directory(self.config.results_dir)
        self.handshake = ChronometricHandshakeSimulator(self.config.handshake_config())
        self._base_handshake_cfg = self.config.handshake_config()

    def _apply_atomic_overrides(self) -> None:
        """Apply atomic baseline configuration overrides."""
        if not self.config.results_dir or self.config.results_dir == 'results/phase1':
            self.config.results_dir = 'results/phase1_atomic'

        common_overrides: Dict[str, Any] = {}
        phase1_overrides: Dict[str, Any] = {}
        config_path = Path(self.config.atomic_config_path)
        if config_path.exists():
            try:
                with open(config_path, 'r', encoding='utf-8') as handle:
                    data = yaml.safe_load(handle) or {}
                common_overrides = data.get('common', {}) or {}
                phase1_overrides = data.get('phase1', {}) or {}
            except (OSError, yaml.YAMLError):
                common_overrides = {}
                phase1_overrides = {}

        def _coerce_numeric(value: Any) -> Any:
            if isinstance(value, str):
                try:
                    return float(value)
                except ValueError:
                    return value
            if isinstance(value, (list, tuple)):
                coerced = [_coerce_numeric(v) for v in value]
                return type(value)(coerced)
            return value

        def _lookup(key: str, fallback: Any) -> Any:
            return _coerce_numeric(common_overrides.get(key, fallback))

        self.config.phase_noise_psd = _lookup('phase_noise_psd', self.config.atomic_phase_noise_psd)
        self.config.jitter_rms_s = _lookup('jitter_rms_s', self.config.atomic_jitter_rms_s)
        self.config.retune_offsets_hz = tuple(_lookup('retune_offsets_hz', list(self.config.atomic_retune_offsets_hz)))
        self.config.coarse_bandwidth_hz = _lookup('coarse_bw_hz', self.config.atomic_coarse_bw_hz)
        self.config.snr_values_db = [_lookup('snr_db', self.config.atomic_snr_db)]
        self.config.clock_bias_std_ps = _lookup('clock_bias_std_ps', self.config.atomic_clock_bias_std_ps)
        self.config.clock_ppm_std = _lookup('clock_ppm_std', self.config.atomic_clock_ppm_std)
        self.config.beat_duration_s = _lookup('beat_duration_s', self.config.atomic_beat_duration_s)
        self.config.delta_f_hz = _lookup('delta_f_hz', self.config.atomic_delta_f_hz)

        if phase1_overrides:
            for attr in (
                'baseband_rate_factor',
                'min_baseband_rate_hz',
                'min_adc_rate_hz',
                'filter_relative_bw',
                'coarse_duration_s',
                'coarse_variance_floor_ps',
            ):
                if attr in phase1_overrides and phase1_overrides[attr] is not None:
                    setattr(self.config, attr, _coerce_numeric(phase1_overrides[attr]))
        
        # Update handshake config post-override
        self.handshake = ChronometricHandshakeSimulator(self.config.handshake_config())
        self._base_handshake_cfg = self.config.handshake_config()

    # ------------------------------------------------------------------
    # Legacy sweep utilities

    def run_full_simulation(self) -> Dict[str, Any]:
        rng = np.random.default_rng(self.config.rng_seed)

        snr_results, exemplar_trace = self._run_snr_sweep(self.handshake, rng)

        results: Dict[str, Any] = {
            'snr_sweep': snr_results,
            'config': asdict(self.config),
        }
        if exemplar_trace:
            results['example_trace'] = self._serialize_trace(exemplar_trace)

        if self.config.save_results:
            self._save_results(results)
        if self.config.plot_results:
            self._generate_plots(snr_results, exemplar_trace)

        # Atomic mode: add CRLB comparison
        if self.config.atomic_mode:
            from metrics.crlb import MultiFrequencyCRLBCalculator, CRLBParams
            # Use config params for CRLB
            crlb_params = CRLBParams(
                snr_db=self.config.atomic_snr_db,
                bandwidth=self.config.atomic_coarse_bw_hz,
                duration=self.config.atomic_beat_duration_s,
                carrier_freq=self.config.carrier_freq_hz,
                sample_rate=self.config.min_baseband_rate_hz,
                carrier_frequencies=self.config.atomic_retune_offsets_hz,
                sigma_phase_rad=np.sqrt(10 ** (self.config.atomic_phase_noise_psd / 10) * self.config.min_baseband_rate_hz / 2),  # Approx
            )
            multi_crlb = MultiFrequencyCRLBCalculator(crlb_params)
            crlb_results = multi_crlb.compute_crlb()
            results['crlb'] = crlb_results
            # Plot simulated vs CRLB
            self._plot_atomic_validation(snr_results, crlb_results)

        return results

    def _run_snr_sweep(
        self,
        handshake: ChronometricHandshakeSimulator,
        rng: np.random.Generator,
    ) -> Tuple[Dict[str, Any], Optional[HandshakeTrace]]:
        snr_values = self.config.snr_values_db
        tau_rmse: List[float] = []
        clock_rmse: List[float] = []
        delta_f_rmse: List[float] = []
        tau_bias: List[float] = []
        delta_f_bias: List[float] = []
        phase_rms: List[float] = []
        coarse_rmse: List[Optional[float]] = []
        alias_success: List[Optional[float]] = []
        carrier_counts: List[Optional[float]] = []

        exemplar_trace: Optional[HandshakeTrace] = None
        exemplar_set = False

        for snr_db in snr_values:
            tof_errors: List[float] = []
            clock_errors: List[float] = []
            delta_f_errors: List[float] = []
            directional_phase_rms: List[float] = []

            coarse_errors: List[float] = []
            alias_flags: List[float] = []
            carrier_counts_local: List[float] = []

            for _ in range(self.config.n_monte_carlo):
                node_a, node_b = self._sample_nodes(rng)
                capture = False
                if not exemplar_set:
                    target_snr = self.config.capture_trace_snr_db or snr_values[0]
                    capture = np.isclose(snr_db, target_snr)

                result, traces = handshake.run_two_way(
                    node_a=node_a,
                    node_b=node_b,
                    distance_m=self.config.distance_m,
                    snr_db=snr_db,
                    rng=rng,
                    capture_trace=capture,
                    retune_offsets_hz=self.config.retune_offsets_hz,
                )

                tof_errors.append(result.tof_est_s - result.tof_true_s)
                clock_errors.append(result.clock_offset_est_s - result.clock_offset_true_s)
                delta_f_errors.append(result.delta_f_est_hz - result.delta_f_true_hz)
                directional_phase_rms.append(
                    0.5 * (result.forward.residual_phase_rms + result.reverse.residual_phase_rms)
                )
                if result.coarse_tof_est_s is not None:
                    coarse_errors.append(result.coarse_tof_est_s - result.tof_true_s)
                alias_forward = bool(result.forward.alias_resolved)
                alias_reverse = bool(result.reverse.alias_resolved)
                alias_flags.append(float(alias_forward and alias_reverse))
                carrier_counts_local.append(
                    0.5
                    * (
                        len(result.forward.carrier_frequencies_hz)
                        + len(result.reverse.carrier_frequencies_hz)
                    )
                )

                if capture and traces and not exemplar_set:
                    exemplar_trace = traces['forward']
                    exemplar_set = True

            tau_errors = np.array(tof_errors)
            clock_errors = np.array(clock_errors)
            delta_f_errors_arr = np.array(delta_f_errors)
            directional_phase_rms_arr = np.array(directional_phase_rms)

            tau_rmse.append(float(np.sqrt(np.mean(tau_errors**2))))
            clock_rmse.append(float(np.sqrt(np.mean(clock_errors**2))))
            delta_f_rmse.append(float(np.sqrt(np.mean(delta_f_errors_arr**2))))
            tau_bias.append(float(np.mean(tau_errors)))
            delta_f_bias.append(float(np.mean(delta_f_errors_arr)))
            phase_rms.append(float(np.mean(directional_phase_rms_arr)))
            if coarse_errors:
                coarse_value = float(np.sqrt(np.mean(np.square(coarse_errors))))
                coarse_rmse.append(coarse_value * 1e12)
            else:
                coarse_rmse.append(None)
            if alias_flags:
                alias_success.append(float(np.mean(alias_flags)))
            else:
                alias_success.append(None)
            if carrier_counts_local:
                carrier_counts.append(float(np.mean(carrier_counts_local)))
            else:
                carrier_counts.append(None)

        snr_results = {
            'snr_db': snr_values,
            'tof_rmse_ps': (np.array(tau_rmse) * 1e12).tolist(),
            'tof_bias_ps': (np.array(tau_bias) * 1e12).tolist(),
            'clock_rmse_ps': (np.array(clock_rmse) * 1e12).tolist(),
            'delta_f_rmse_hz': np.array(delta_f_rmse).tolist(),
            'delta_f_bias_hz': np.array(delta_f_bias).tolist(),
            'phase_fit_rms_rad': np.array(phase_rms).tolist(),
            'coarse_tof_rmse_ps': _clean_numeric_list(coarse_rmse),
            'alias_success_rate': _clean_numeric_list(alias_success),
            'avg_carrier_count': _clean_numeric_list(carrier_counts),
        }

        return snr_results, exemplar_trace

    def _sample_nodes(self, rng: np.random.Generator) -> Tuple[ChronometricNode, ChronometricNode]:
        base_freq = self.config.carrier_freq_hz
        delta_f = self.config.delta_f_hz

        ppm_a = rng.normal(0.0, self.config.clock_ppm_std)
        ppm_b = rng.normal(0.0, self.config.clock_ppm_std)

        freq_a = base_freq * (1.0 + ppm_a * 1e-6)
        freq_b = (base_freq + delta_f) * (1.0 + ppm_b * 1e-6)

        node_a = ChronometricNode(
            ChronometricNodeConfig(
                node_id=0,
                carrier_freq_hz=freq_a,
                phase_offset_rad=rng.uniform(0.0, 2.0 * np.pi),
                clock_bias_s=rng.normal(0.0, self.config.clock_bias_std_ps * 1e-12),
                freq_error_ppm=ppm_a,
            )
        )
        node_b = ChronometricNode(
            ChronometricNodeConfig(
                node_id=1,
                carrier_freq_hz=freq_b,
                phase_offset_rad=rng.uniform(0.0, 2.0 * np.pi),
                clock_bias_s=rng.normal(0.0, self.config.clock_bias_std_ps * 1e-12),
                freq_error_ppm=ppm_b,
            )
        )

        return node_a, node_b

    def _save_results(self, results: Dict[str, Any]) -> None:
        out_path = os.path.join(self.config.results_dir, 'phase1_results.json')
        save_json(results, out_path)
        print(f"Results saved to {out_path}")

    def _generate_plots(
        self,
        snr_results: Dict[str, Any],
        trace: Optional[HandshakeTrace],
    ) -> None:
        if trace:
            self._plot_waveforms(trace)
        self._plot_error_curves(snr_results)
    
    def _plot_atomic_validation(self, snr_results: Dict[str, Any], crlb_results: Dict[str, Any]) -> None:
        """Plot simulated RMSE vs CRLB for atomic validation."""
        fig, ax = plt.subplots(figsize=(8, 6))
        snr = np.array(snr_results['snr_db'])
        simulated_rmse_ps = np.array(snr_results['tof_rmse_ps'])
        crlb_std_ps = crlb_results['delay_crlb_std'] * 1e12  # Convert to ps
        ax.loglog(simulated_rmse_ps, 'bo-', label='Simulated RMSE')
        ax.loglog(crlb_std_ps, 'r--', label='CRLB std')
        ax.set_xlabel('Simulated RMSE (ps)')
        ax.set_ylabel('CRLB std (ps)')
        ax.set_title('Atomic Validation: Simulated vs CRLB')
        ax.legend()
        ax.grid(True)
        plot_path = os.path.join(self.config.results_dir, 'atomic_crlb_validation.png')
        save_figure(fig, plot_path)
        print(f"Atomic validation plot saved to {plot_path}")

    def _plot_waveforms(self, trace: HandshakeTrace) -> None:
        fig, axes = plt.subplots(3, 1, figsize=(10, 10))

        axes[0].plot(trace.time_us, trace.beat_raw.real, label='Real')
        axes[0].plot(trace.time_us, trace.beat_raw.imag, label='Imag', alpha=0.7)
        axes[0].set_title('Beat Signal (Noisy)')
        axes[0].set_xlabel('Time (µs)')
        axes[0].set_ylabel('Amplitude')
        axes[0].legend()

        axes[1].plot(trace.time_us, trace.beat_filtered.real, label='Real')
        axes[1].plot(trace.time_us, trace.beat_filtered.imag, label='Imag', alpha=0.7)
        axes[1].set_title('Band-pass Filtered Beat')
        axes[1].set_xlabel('Time (µs)')
        axes[1].set_ylabel('Amplitude')
        axes[1].legend()

        axes[2].plot(trace.adc_time_us, np.unwrap(np.angle(trace.adc_samples)), label='Phase')
        axes[2].plot(trace.adc_time_us, trace.phase_fit, label='Linear Fit', linestyle='--')
        axes[2].set_title('Phase Extraction at ADC Rate')
        axes[2].set_xlabel('Time (µs)')
        axes[2].set_ylabel('Phase (rad)')
        axes[2].legend()

        plot_path = os.path.join(self.config.results_dir, 'phase1_waveforms.png')
        save_figure(fig, plot_path)
        print(f"Saved waveform plot to {plot_path}")

    def _plot_error_curves(self, snr_results: Dict[str, Any]) -> None:
        snr = np.array(snr_results['snr_db'])
        tof_rmse = np.array(snr_results['tof_rmse_ps'])
        clock_rmse = np.array(snr_results['clock_rmse_ps'])
        delta_f_rmse = np.array(snr_results['delta_f_rmse_hz'])

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        axes[0].semilogy(snr, tof_rmse, marker='o')
        axes[0].set_title('ToF RMSE vs SNR')
        axes[0].set_xlabel('SNR (dB)')
        axes[0].set_ylabel('RMSE (ps)')

        axes[1].semilogy(snr, clock_rmse, marker='o', color='tab:orange')
        axes[1].set_title('Clock Offset RMSE vs SNR')
        axes[1].set_xlabel('SNR (dB)')
        axes[1].set_ylabel('RMSE (ps)')

        axes[2].semilogy(snr, np.maximum(delta_f_rmse, 1e-3), marker='o', color='tab:green')
        axes[2].set_title('Δf RMSE vs SNR')
        axes[2].set_xlabel('SNR (dB)')
        axes[2].set_ylabel('RMSE (Hz)')

        plot_path = os.path.join(self.config.results_dir, 'phase1_errors.png')
        save_figure(fig, plot_path)
        print(f"Saved error plot to {plot_path}")

    def _serialize_trace(self, trace: HandshakeTrace) -> Dict[str, Any]:
        return {
            'time_us': trace.time_us.tolist(),
            'beat_raw_real': trace.beat_raw.real.tolist(),
            'beat_raw_imag': trace.beat_raw.imag.tolist(),
            'beat_filtered_real': trace.beat_filtered.real.tolist(),
            'beat_filtered_imag': trace.beat_filtered.imag.tolist(),
            'adc_time_us': trace.adc_time_us.tolist(),
            'adc_samples_real': trace.adc_samples.real.tolist(),
            'adc_samples_imag': trace.adc_samples.imag.tolist(),
            'unwrapped_phase': trace.unwrapped_phase.tolist(),
            'phase_fit': trace.phase_fit.tolist(),
        }

    # ------------------------------------------------------------------
    # Alias failure mapping

    def run_alias_failure_map(
        self,
        retune_offsets_hz: Sequence[float],
        coarse_bw_hz: Sequence[float],
        snr_db: Sequence[float],
        num_trials: int,
        rng_seed: int,
        make_plots: bool = False,
    ) -> Dict[str, Any]:
        if self.config.save_results:
            alias_dir = os.path.join(self.config.results_dir, 'alias_map')
            os.makedirs(alias_dir, exist_ok=True)
        else:
            alias_dir = None

        rng = np.random.default_rng(rng_seed)

        retune_offsets = list(retune_offsets_hz)
        coarse_bandwidths = list(coarse_bw_hz)
        snr_values = list(snr_db)

        shape = (len(retune_offsets), len(coarse_bandwidths), len(snr_values))
        alias_fail_rate = np.full(shape, np.nan, dtype=float)
        tau_rmse_ps = np.full(shape, np.nan, dtype=float)
        tau_bias_ps = np.full(shape, np.nan, dtype=float)
        deltaf_rmse_hz = np.full(shape, np.nan, dtype=float)
        coarse_rmse_ns = np.full(shape, np.nan, dtype=float)
        carriers_avg = np.full(shape, np.nan, dtype=float)
        phase_bias_rad = np.full(shape, np.nan, dtype=float)
        reciprocity_bias_ps = np.full(shape, np.nan, dtype=float)

        for i, retune in enumerate(retune_offsets):
            for j, coarse_bw in enumerate(coarse_bandwidths):
                handshake_cfg = replace(
                    self._base_handshake_cfg,
                    retune_offsets_hz=(retune,),
                    coarse_bandwidth_hz=coarse_bw,
                )
                handshake = ChronometricHandshakeSimulator(handshake_cfg)

                for k, snr in enumerate(snr_values):
                    alias_flags: List[bool] = []
                    tau_errors_ps: List[float] = []
                    deltaf_errors_hz: List[float] = []
                    coarse_errors_ns: List[float] = []
                    carrier_counts: List[float] = []
                    phase_bias_vals: List[float] = []
                    reciprocity_vals_ps: List[float] = []

                    for _ in range(num_trials):
                        node_a, node_b = self._sample_nodes(rng)
                        result, _ = handshake.run_two_way(
                            node_a=node_a,
                            node_b=node_b,
                            distance_m=self.config.distance_m,
                            snr_db=snr,
                            rng=rng,
                            retune_offsets_hz=handshake_cfg.retune_offsets_hz,
                        )

                        alias_forward = bool(result.forward.alias_resolved)
                        alias_reverse = bool(result.reverse.alias_resolved)
                        alias_flags.append(alias_forward and alias_reverse)

                        tau_errors_ps.append((result.tof_est_s - result.tof_true_s) * 1e12)
                        deltaf_errors_hz.append(result.delta_f_est_hz - result.delta_f_true_hz)
                        if result.coarse_tof_est_s is not None:
                            coarse_errors_ns.append((result.coarse_tof_est_s - result.tof_true_s) * 1e9)
                        carrier_counts.append(
                            0.5
                            * (
                                len(result.forward.carrier_frequencies_hz)
                                + len(result.reverse.carrier_frequencies_hz)
                            )
                        )
                        phase_samples = [result.forward.phase_bias_rad, result.reverse.phase_bias_rad]
                        valid_phase = [val for val in phase_samples if val is not None]
                        if valid_phase:
                            phase_bias_vals.append(float(np.mean(valid_phase)))
                        reciprocity_vals_ps.append(result.reciprocity_bias_s * 1e12)

                    alias_arr = np.asarray(alias_flags, dtype=float)
                    tau_arr = np.asarray(tau_errors_ps, dtype=float)
                    deltaf_arr = np.asarray(deltaf_errors_hz, dtype=float)
                    coarse_arr = np.asarray(coarse_errors_ns, dtype=float)
                    carriers_arr = np.asarray(carrier_counts, dtype=float)

                    alias_success = alias_arr.mean() if alias_arr.size else np.nan
                    alias_fail_rate[i, j, k] = float(1.0 - alias_success) if alias_arr.size else float('nan')
                    tau_rmse_ps[i, j, k] = _rmse(tau_arr)
                    deltaf_rmse_hz[i, j, k] = _rmse(deltaf_arr)
                    coarse_rmse_ns[i, j, k] = _rmse(coarse_arr)
                    carriers_avg[i, j, k] = float(carriers_arr.mean()) if carriers_arr.size else float('nan')
                    tau_bias_ps[i, j, k] = float(np.mean(tau_arr)) if tau_arr.size else float('nan')
                    phase_bias_rad[i, j, k] = float(np.mean(phase_bias_vals)) if phase_bias_vals else float('nan')
                    reciprocity_bias_ps[i, j, k] = float(np.mean(reciprocity_vals_ps)) if reciprocity_vals_ps else float('nan')

        timestamp = datetime.now(timezone.utc).isoformat()
        metrics = {
            'alias_fail_rate': alias_fail_rate.tolist(),
            'tau_rmse_ps': tau_rmse_ps.tolist(),
            'tau_bias_ps': tau_bias_ps.tolist(),
            'deltaf_rmse_hz': deltaf_rmse_hz.tolist(),
            'coarse_rmse_ns': coarse_rmse_ns.tolist(),
            'avg_carriers_used': carriers_avg.tolist(),
            'phase_bias_rad': phase_bias_rad.tolist(),
            'channel_k_factor_db': self.config.channel_model.k_factor_db if self.config.channel_model else None,
            'reciprocity_bias_ps': reciprocity_bias_ps.tolist(),
        }
        with np.errstate(invalid='ignore'):
            bias_mean_raw = np.nanmean(reciprocity_bias_ps)
            bias_by_retune_raw = np.nanmean(reciprocity_bias_ps, axis=(1, 2))
            bias_by_snr_raw = np.nanmean(reciprocity_bias_ps, axis=(0, 1))
        bias_summary = {
            'calibration_mode': self.config.calib_mode,
            'delta_t_schedule_us': list(self.config.delta_t_us),
            'mean_bias_ps': _clean_numeric_value(None if np.isnan(bias_mean_raw) else float(bias_mean_raw)),
            'bias_by_retune_ps': _clean_numeric_list(
                [None if np.isnan(val) else float(val) for val in bias_by_retune_raw]
            ),
            'bias_by_snr_ps': _clean_numeric_list(
                [None if np.isnan(val) else float(val) for val in bias_by_snr_raw]
            ),
        }

        manifest = {
            'snr_db': snr_values,
            'retune_offsets_hz': retune_offsets,
            'coarse_bw_hz': coarse_bandwidths,
            'num_trials': num_trials,
            'metrics': metrics,
            'rng_seed': rng_seed,
            'timestamp': timestamp,
            'tdl_profile': self.config.tdl_profile,
            'delta_t_us': list(self.config.delta_t_us),
            'calib_mode': self.config.calib_mode,
            'loopback_cal_noise_ps': self.config.loopback_cal_noise_ps,
            'mac': (
                {
                    'preamble_len': self.config.mac_slots.preamble_len,
                    'narrowband_len': self.config.mac_slots.narrowband_len,
                    'guard_us': self.config.mac_slots.guard_us,
                    'guard_s': self.config.mac_slots.guard_seconds(),
                    'asymmetric': self.config.mac_slots.asymmetric,
                }
                if self.config.mac_slots
                else None
            ),
            'bias_diagnostics': bias_summary,
        }

        manifest_path = None
        csv_path = None
        plot_paths: List[str] = []

        if alias_dir:
            manifest_path = os.path.join(alias_dir, 'alias_map_manifest.json')
            save_json(manifest, manifest_path)

            csv_path = os.path.join(alias_dir, 'alias_map_metrics.csv')
            self._write_alias_csv(
                csv_path,
                retune_offsets,
                coarse_bandwidths,
                snr_values,
                alias_fail_rate,
                tau_bias_ps,
                tau_rmse_ps,
                deltaf_rmse_hz,
                coarse_rmse_ns,
                carriers_avg,
                phase_bias_rad,
                reciprocity_bias_ps,
            )

            if make_plots:
                plot_paths = self._plot_alias_heatmaps(
                    alias_dir,
                    retune_offsets,
                    coarse_bandwidths,
                    snr_values,
                    alias_fail_rate,
                    tau_rmse_ps,
                    deltaf_rmse_hz,
                    reciprocity_bias_ps,
                )
                bias_trend_path = self._plot_bias_trends(
                    alias_dir,
                    retune_offsets,
                    snr_values,
                    reciprocity_bias_ps,
                )
                if bias_trend_path:
                    plot_paths.append(bias_trend_path)

        return {
            'manifest': manifest,
            'manifest_path': manifest_path,
            'csv_path': csv_path,
            'plot_paths': plot_paths,
        }

    def _write_alias_csv(
        self,
        csv_path: str,
        retune_offsets: Sequence[float],
        coarse_bandwidths: Sequence[float],
        snr_values: Sequence[float],
        alias_fail_rate: np.ndarray,
        tau_bias_ps: np.ndarray,
        tau_rmse_ps: np.ndarray,
        deltaf_rmse_hz: np.ndarray,
        coarse_rmse_ns: np.ndarray,
        carriers_avg: np.ndarray,
        phase_bias_rad: np.ndarray,
        reciprocity_bias_ps: np.ndarray,
    ) -> None:
        fieldnames = [
            'retune_offset_hz',
            'coarse_bw_hz',
            'snr_db',
            'alias_fail_rate',
            'tau_rmse_ps',
            'tau_bias_ps',
            'deltaf_rmse_hz',
            'coarse_rmse_ns',
            'avg_carriers_used',
            'phase_bias_rad',
            'reciprocity_bias_ps',
        ]
        rows: List[Dict[str, Any]] = []
        for i, retune in enumerate(retune_offsets):
            for j, coarse in enumerate(coarse_bandwidths):
                for k, snr in enumerate(snr_values):
                    rows.append(
                        {
                            'retune_offset_hz': retune,
                            'coarse_bw_hz': coarse,
                            'snr_db': snr,
                            'alias_fail_rate': alias_fail_rate[i, j, k],
                            'tau_rmse_ps': tau_rmse_ps[i, j, k],
                            'tau_bias_ps': tau_bias_ps[i, j, k],
                            'deltaf_rmse_hz': deltaf_rmse_hz[i, j, k],
                            'coarse_rmse_ns': coarse_rmse_ns[i, j, k],
                            'avg_carriers_used': carriers_avg[i, j, k],
                            'phase_bias_rad': phase_bias_rad[i, j, k],
                            'reciprocity_bias_ps': reciprocity_bias_ps[i, j, k],
                        }
                    )
        write_csv(csv_path, fieldnames, rows)

    def _plot_alias_heatmaps(
        self,
        alias_dir: str,
        retune_offsets: Sequence[float],
        coarse_bandwidths: Sequence[float],
        snr_values: Sequence[float],
        alias_fail_rate: np.ndarray,
        tau_rmse_ps: np.ndarray,
        deltaf_rmse_hz: np.ndarray,
        reciprocity_bias_ps: np.ndarray,
    ) -> List[str]:
        plot_paths: List[str] = []
        metrics = {
            'alias_fail_rate': alias_fail_rate,
            'tau_rmse_ps': tau_rmse_ps,
            'deltaf_rmse_hz': deltaf_rmse_hz,
            'reciprocity_bias_ps': reciprocity_bias_ps,
        }
        label_map = {
            'alias_fail_rate': 'Alias Fail Rate',
            'tau_rmse_ps': 'τ RMSE (ps)',
            'deltaf_rmse_hz': 'Δf RMSE (Hz)',
            'reciprocity_bias_ps': 'Reciprocity Bias (ps)',
        }
        snr_labels = [str(s) for s in snr_values]
        coarse_labels = [f"{bw/1e6:.1f} MHz" for bw in coarse_bandwidths]

        for metric_name, data in metrics.items():
            for idx, offset in enumerate(retune_offsets):
                fig, ax = plt.subplots(figsize=(6, 4))
                matrix = np.asarray(data[idx], dtype=float)
                display_label = label_map.get(metric_name, metric_name.replace('_', ' ').title())
                title = f"{display_label} (retune {offset/1e6:.2f} MHz)"
                plot_heatmap(
                    ax,
                    matrix,
                    xticklabels=snr_labels,
                    yticklabels=coarse_labels,
                    title=title,
                    cbar_label=display_label,
                )
                ax.set_xlabel('SNR (dB)')
                ax.set_ylabel('Coarse BW')

                safe_offset = str(offset).replace('.', 'p')
                filename = f"{metric_name}_retune_{safe_offset}.png"
                plot_path = os.path.join(alias_dir, filename)
                save_figure(fig, plot_path)
                plot_paths.append(plot_path)

        return plot_paths

    def _plot_bias_trends(
        self,
        alias_dir: str,
        retune_offsets: Sequence[float],
        snr_values: Sequence[float],
        reciprocity_bias_ps: np.ndarray,
    ) -> Optional[str]:
        with np.errstate(invalid='ignore'):
            bias_by_coarse = np.nanmean(reciprocity_bias_ps, axis=1)
        valid_indices = [idx for idx in range(len(retune_offsets)) if not np.all(np.isnan(bias_by_coarse[idx]))]
        if not valid_indices:
            return None
        fig, ax = plt.subplots(figsize=(6, 4))
        for idx in valid_indices:
            series = bias_by_coarse[idx]
            ax.plot(snr_values, series, marker='o', label=f'{retune_offsets[idx]/1e6:.2f} MHz')
        ax.set_title('Reciprocity Bias vs SNR')
        ax.set_xlabel('SNR (dB)')
        ax.set_ylabel('Bias (ps)')
        if len(valid_indices) > 1:
            ax.legend(title='Retune Offset')
        schedule_text = ', '.join(f'{val:.2f}' for val in self.config.delta_t_us)
        annotation = f"Calibration: {self.config.calib_mode}\nΔt (µs): {schedule_text}"
        ax.text(
            0.02,
            0.98,
            annotation,
            transform=ax.transAxes,
            va='top',
            ha='left',
            fontsize=9,
            bbox={'facecolor': 'white', 'alpha': 0.8, 'edgecolor': 'none'},
        )
        bias_path = os.path.join(alias_dir, 'reciprocity_bias_trends.png')
        save_figure(fig, bias_path)
        return bias_path


def _rmse(values: np.ndarray) -> float:
    if values.size == 0:
        return float('nan')
    return float(np.sqrt(np.mean(np.square(values))))


def _clean_numeric_value(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return float(value)


def _clean_numeric_list(values: Sequence[Optional[float]]) -> List[Optional[float]]:
    cleaned: List[Optional[float]] = []
    for val in values:
        if val is None:
            cleaned.append(None)
            continue
        if isinstance(val, float) and (math.isnan(val) or math.isinf(val)):
            cleaned.append(None)
        else:
            cleaned.append(float(val))
    return cleaned


def _parse_float_list(arg: str) -> List[float]:
    if not arg:
        return []
    parts = [part.strip() for part in arg.split(',')]
    return [float(p) for p in parts if p]


def _parse_time_token(token: str) -> float:
    cleaned = token.strip().lower()
    suffixes = [
        ('ps', 1e-12),
        ('ns', 1e-9),
        ('us', 1e-6),
        ('ms', 1e-3),
        ('s', 1.0),
    ]
    for suffix, scale in suffixes:
        if cleaned.endswith(suffix):
            value = float(cleaned[: -len(suffix)])
            return value * scale
    return float(cleaned)


def _parse_tdl_profile(profile: str) -> TappedDelayLine:
    # First, check against the predefined profiles for convenience.
    if profile.upper() in TDL_PROFILES:
        # NOTE: This creates a single random realization of the channel. For a true
        # Monte Carlo simulation over channel fades, this function should be
        # called inside the trial loop with the simulation's RNG.
        rng = np.random.default_rng()
        return tdl_from_profile(profile, rng)

    tokens = [part.strip() for part in profile.split(':') if part.strip()]
    if not tokens:
        raise ValueError('TDL profile string is empty')
    mode = tokens[0].upper()
    if mode == 'EXPO':
        if len(tokens) < 2:
            raise ValueError('EXPO profile requires RMS delay token, e.g., EXPO:50ns')
        rms_delay = _parse_time_token(tokens[1])
        k_factor = None
        L = 5
        doppler = None
        for token in tokens[2:]:
            upper = token.upper()
            if upper.startswith('K='):
                value = upper.split('=', 1)[1]
                if value.endswith('DB'):
                    value = value[:-2]
                k_factor = float(value)
            elif upper.startswith('L='):
                L = int(upper.split('=', 1)[1])
            elif upper.startswith('FD='):
                doppler = float(upper.split('=', 1)[1])
        tdl = tdl_exponential(L=L, rms_delay_spread_s=rms_delay, k_factor_db=k_factor)
        if doppler is not None:
            tdl.doppler_hz = doppler
        return tdl
    raise ValueError(f'Unsupported TDL profile mode: {mode}')


def build_alias_map_cli(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        '--snr-db',
        type=str,
        default='20,10,0,-10,-20',
        help='Comma-separated SNR values in dB.',
    )
    parser.add_argument(
        '--retune-offsets-hz',
        type=str,
        default='1e6,5e6',
        help='Comma-separated retune offsets in Hz.',
    )
    parser.add_argument(
        '--coarse-bw-hz',
        type=str,
        default='20e6,40e6,80e6',
        help='Comma-separated coarse preamble bandwidths in Hz.',
    )
    parser.add_argument(
        '--num-trials',
        type=int,
        default=500,
        help='Monte Carlo trials per (offset, bandwidth, SNR) tuple.',
    )
    parser.add_argument(
        '--rng-seed',
        type=int,
        default=1337,
        help='Seed for the NumPy default RNG.',
    )
    parser.add_argument(
        '--results-dir',
        type=str,
        default='results/phase1',
        help='Output directory for manifests, CSV, and plots.',
    )
    parser.add_argument(
        '--make-plots',
        action='store_true',
        help='Generate heatmap plots for alias failure metrics.',
    )
    parser.add_argument(
        '--run-legacy-snr-sweep',
        action='store_true',
        help='Also execute the original Phase 1 SNR sweep.',
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Resolve configuration, echo it, then exit without running the simulation.',
    )
    parser.add_argument(
        '--echo-config',
        action='store_true',
        help='Print the resolved Phase 1 config (JSON) before execution.',
    )
    parser.add_argument(
        '--tdl-profile',
        type=str,
        default=None,
        help=(
            'TDL profile string, e.g., "EXPO:50ns:K=6dB:L=5". '
            'When provided, overrides the legacy two-ray stub.'
        ),
    )
    parser.add_argument(
        '--delta-t-us',
        type=str,
        default='0.0',
        help='Comma-separated schedule of delta-t values (microseconds).',
    )
    parser.add_argument(
        '--calib-mode',
        type=str,
        choices=['off', 'perfect', 'loopback'],
        default='off',
        help='Reciprocity calibration mode.',
    )
    parser.add_argument(
        '--loopback-cal-noise-ps',
        type=float,
        default=10.0,
        help='Std-dev of loopback calibration noise in picoseconds.',
    )
    parser.add_argument(
        '--preamble-len',
        type=int,
        default=1024,
        help='Preamble length (samples) used for coarse correlation.',
    )
    parser.add_argument(
        '--preamble-bw-hz',
        type=float,
        default=40e6,
        help='Coarse preamble bandwidth in Hz.',
    )
    parser.add_argument(
        '--guard-us',
        type=float,
        default=10.0,
        help='MAC guard interval in microseconds.',
    )
    parser.add_argument(
        '--mac-asymmetry',
        action='store_true',
        help='Allow asymmetric MAC slot durations.',
    )


def main() -> None:
    parser = argparse.ArgumentParser(description='Phase 1 alias-failure mapping tool')
    build_alias_map_cli(parser)
    parser.add_argument(
        '--atomic-mode',
        action='store_true',
        help='Run in atomic baseline mode using atomic_reconciliation.yaml',
    )
    parser.add_argument(
        '--atomic-config-path',
        type=str,
        default='sim/configs/atomic_reconciliation.yaml',
        help='Path to atomic reconciliation config file',
    )
    args = parser.parse_args()

    snr_values = _parse_float_list(args.snr_db)
    retune_offsets = _parse_float_list(args.retune_offsets_hz)
    coarse_bandwidths = _parse_float_list(args.coarse_bw_hz)
    channel_model = _parse_tdl_profile(args.tdl_profile) if args.tdl_profile else None
    delta_t_values = _parse_float_list(args.delta_t_us)
    if not delta_t_values:
        delta_t_values = [0.0]
    narrowband_len = max(1, int(Phase1Config.beat_duration_s * args.preamble_bw_hz))
    mac_slots = MacSlots(
        preamble_len=args.preamble_len,
        narrowband_len=narrowband_len,
        guard_us=args.guard_us,
        asymmetric=args.mac_asymmetry,
    )

    if not snr_values:
        raise ValueError('At least one SNR value must be provided via --snr-db')
    if not retune_offsets:
        raise ValueError('At least one retune offset must be provided via --retune-offsets-hz')
    if not coarse_bandwidths:
        raise ValueError('At least one coarse bandwidth must be provided via --coarse-bw-hz')

    cfg = Phase1Config(
        atomic_mode=args.atomic_mode,
        atomic_config_path=args.atomic_config_path,
        snr_values_db=snr_values,
        n_monte_carlo=min(args.num_trials, 200),
        retune_offsets_hz=(retune_offsets[0],),
        coarse_bandwidth_hz=args.preamble_bw_hz,
        save_results=True,
        plot_results=args.make_plots,
        results_dir=args.results_dir,
        channel_model=channel_model,
        tdl_profile=args.tdl_profile,
        delta_t_us=tuple(delta_t_values),
        calib_mode=args.calib_mode,
        loopback_cal_noise_ps=args.loopback_cal_noise_ps,
        coarse_duration_s=max(args.preamble_len / max(args.preamble_bw_hz, 1.0), 1e-6),
        mac_slots=mac_slots,
    )

    if args.echo_config or args.dry_run:
        print(echo_config(cfg, label='Phase1Config'))

    if args.dry_run:
        return

    simulator = Phase1Simulator(cfg)

    if cfg.save_results:
        dump_run_config(cfg.results_dir, cfg, filename='phase1_config.json')

    if cfg.atomic_mode:
        # Run atomic validation
        atomic_results = simulator.run_full_simulation()
        print("Atomic baseline simulation completed.")
    else:
        alias_summary = simulator.run_alias_failure_map(
            retune_offsets_hz=retune_offsets,
            coarse_bw_hz=coarse_bandwidths,
            snr_db=snr_values,
            num_trials=args.num_trials,
            rng_seed=args.rng_seed,
            make_plots=args.make_plots,
        )

        manifest_path = alias_summary['manifest_path']
        csv_path = alias_summary['csv_path']
        if manifest_path:
            print(f"Alias-map manifest written to {manifest_path}")
        if csv_path:
            print(f"Alias-map CSV written to {csv_path}")
        if alias_summary['plot_paths']:
            print('Generated plots:')
            for path in alias_summary['plot_paths']:
                print(f"  {path}")

        if args.run_legacy_snr_sweep:
            simulator.run_full_simulation()


if __name__ == '__main__':
    main()
