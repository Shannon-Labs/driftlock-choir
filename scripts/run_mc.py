#!/usr/bin/env python3
"""
Updated Monte Carlo harness for DriftLock simulations.

This version orchestrates Phase 1 alias-map sweeps and Phase 2 consensus runs
using the current simulator APIs. Jobs are described in a YAML file and their
outputs are collated under a timestamped results directory.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import yaml

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'sim'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from phase1 import Phase1Config, Phase1Simulator
from phase2 import Phase2Config, Phase2Simulation
from src.metrics.stats import StatisticalValidator, StatsParams
from mac.scheduler import MacSlots
from utils.io import ensure_directory, save_json

from ablation_sweeps import (
    load_ablations,
    generate_combinations,
    run_ablation_combo,
    aggregate_telemetries,
    save_summary_csv,
)


@dataclass
class MCRunConfig:
    simulation_type: str
    config_file: str
    output_dir: str = 'results'
    run_id: Optional[str] = None
    ablation_config: Optional[str] = None
    comparative: bool = False


class MonteCarloRunner:
    def __init__(self, cfg: MCRunConfig):
        self.cfg = cfg
        if cfg.ablation_config:
            self.ablation_params, self.n_monte_carlo = load_ablations(cfg.ablation_config)
            root = Path(cfg.output_dir) / 'ablations'
            if cfg.run_id:
                root = root / cfg.run_id
            else:
                root = root / f"ablation_{int(time.time())}"
            self.output_root = Path(ensure_directory(root))
        else:
            self.config = self._read_config(cfg.config_file)
            root = Path(cfg.output_dir)
            if cfg.run_id:
                root = root / cfg.run_id
            else:
                root = root / f"run_{int(time.time())}"
            self.output_root = Path(ensure_directory(root))
        self._persist_run_metadata()

    def _read_config(self, path: str) -> Dict[str, Any]:
        with open(path, 'r', encoding='utf-8') as handle:
            return yaml.safe_load(handle) or {}

    def _persist_run_metadata(self) -> None:
        payload = {
            'timestamp': time.time(),
            'simulation_type': self.cfg.simulation_type,
            'config_file': self.cfg.config_file,
            'ablation_config': self.cfg.ablation_config,
            'comparative': self.cfg.comparative,
        }
        if hasattr(self, 'config'):
            payload['config'] = self.config
        if hasattr(self, 'ablation_params'):
            payload['ablation_params'] = self.ablation_params
            payload['n_monte_carlo'] = self.n_monte_carlo
        save_json(payload, self.output_root / 'run_config.json')

    def run(self) -> Dict[str, Any]:
        summary: Dict[str, Any] = {}
        sim_type = self.cfg.simulation_type
        if sim_type in ('phase1', 'all'):
            summary['phase1'] = self._run_phase1_jobs()
        if sim_type in ('phase2', 'all'):
            summary['phase2'] = self._run_phase2_jobs()
        if self.cfg.ablation_config:
            summary['ablation'] = self._run_ablation_jobs()
        save_json(summary, self.output_root / 'final_results.json')
        report = self._compose_report(summary)
        (self.output_root / 'simulation_report.txt').write_text(report, encoding='utf-8')
        # Write a human‑friendly Markdown summary for quick sharing
        (self.output_root / 'SUMMARY.md').write_text(self._compose_markdown_summary(summary), encoding='utf-8')
        print(report)
        return summary

    def _run_phase1_jobs(self) -> List[Dict[str, Any]]:
        jobs = self.config.get('phase1_alias_map', [])
        results: List[Dict[str, Any]] = []
        for job in jobs:
            job_name = job.get('name', 'phase1_job')
            job_dir = Path(ensure_directory(self.output_root / 'phase1' / job_name))
            retune_offsets = [float(v) for v in job['retune_offsets_hz']]
            coarse_bw = [float(v) for v in job['coarse_bw_hz']]
            snr_values = [float(v) for v in job['snr_db']]
            delta_t_us = [float(v) for v in job.get('delta_t_us', [0.0])]
            mac_params = job.get('mac', {})
            preamble_bw_hz = float(mac_params.get('preamble_bw_hz', 40e6))
            mac = MacSlots(
                preamble_len=int(mac_params.get('preamble_len', 1024)),
                narrowband_len=int(mac_params.get('narrowband_len', max(1, int(Phase1Config.beat_duration_s * preamble_bw_hz)))),
                guard_us=float(mac_params.get('guard_us', 10.0)),
                asymmetric=bool(mac_params.get('asymmetric', False)),
            )
            hw = job.get('hardware_delays_ns', {})
            d_tx_ns = {int(k): float(v) for k, v in hw.get('d_tx', {}).items()}
            d_rx_ns = {int(k): float(v) for k, v in hw.get('d_rx', {}).items()}
            common = dict(
                snr_values_db=snr_values,
                n_monte_carlo=int(job.get('num_trials', 200)),
                plot_results=bool(job.get('make_plots', False)),
                save_results=True,
                delta_t_us=tuple(delta_t_us),
                loopback_cal_noise_ps=float(job.get('loopback_noise_ps', 5.0)),
                mac_slots=mac,
                d_tx_ns=d_tx_ns if d_tx_ns else None,
                d_rx_ns=d_rx_ns if d_rx_ns else None,
            )
            calib_modes = job.get('calibration_modes', ['off'])
            calib_summaries: List[Dict[str, Any]] = []
            for mode in calib_modes:
                calib_dir = Path(ensure_directory(job_dir / f"calib_{mode}"))
                cfg = Phase1Config(
                    **common,
                    results_dir=str(calib_dir),
                    calib_mode=str(mode),
                    retune_offsets_hz=(retune_offsets[0],),
                    coarse_bandwidth_hz=preamble_bw_hz,
                )
                sim = Phase1Simulator(cfg)
                alias_summary = sim.run_alias_failure_map(
                    retune_offsets_hz=retune_offsets,
                    coarse_bw_hz=coarse_bw,
                    snr_db=snr_values,
                    num_trials=int(job.get('num_trials', 200)),
                    rng_seed=int(job.get('rng_seed', 2024)),
                    make_plots=bool(job.get('make_plots', False)),
                )
                manifest = alias_summary['manifest']
                diag = manifest.get('bias_diagnostics', {})
                calib_summaries.append(
                    {
                        'calibration_mode': mode,
                        'mean_bias_ps': diag.get('mean_bias_ps'),
                        'bias_by_retune_ps': diag.get('bias_by_retune_ps'),
                        'bias_by_snr_ps': diag.get('bias_by_snr_ps'),
                        'manifest_path': alias_summary.get('manifest_path'),
                        'csv_path': alias_summary.get('csv_path'),
                        'plots': alias_summary.get('plot_paths', []),
                        'output_dir': str(calib_dir),
                    }
                )
            results.append(
                {
                    'job': job_name,
                    'retune_offsets_hz': retune_offsets,
                    'coarse_bw_hz': coarse_bw,
                    'snr_db': snr_values,
                    'delta_t_us': delta_t_us,
                    'calibration_summaries': calib_summaries,
                    'output_dir': str(job_dir),
                }
            )
        return results

    def _run_phase2_jobs(self) -> List[Dict[str, Any]]:
        jobs = self.config.get('phase2_runs', [])
        results: List[Dict[str, Any]] = []
        for job in jobs:
            job_name = job.get('name', 'phase2_job')
            job_dir = Path(ensure_directory(self.output_root / 'phase2' / job_name))
            density = job.get('density')
            comm_range = job.get('comm_range_m')
            if density is not None and comm_range is None:
                comm_range = _radius_from_density(float(density), float(job.get('area_size_m', Phase2Config.area_size_m)))
            cfg = Phase2Config(
                n_nodes=int(job.get('nodes', Phase2Config.n_nodes)),
                area_size_m=float(job.get('area_size_m', Phase2Config.area_size_m)),
                comm_range_m=float(comm_range or Phase2Config.comm_range_m),
                snr_db=float(job.get('snr_db', Phase2Config.snr_db)),
                weighting=str(job.get('weighting', Phase2Config.weighting)),
                target_rmse_ps=float(job.get('target_rmse_ps', Phase2Config.target_rmse_ps)),
                target_streak=int(job.get('target_streak', Phase2Config.target_streak)),
                max_iterations=int(job.get('max_iterations', Phase2Config.max_iterations)),
                timestep_s=float(job.get('timestep_ms', Phase2Config.timestep_s * 1e3)) * 1e-3,
                epsilon_override=job.get('epsilon'),
                spectral_margin=float(job.get('spectral_margin', Phase2Config.spectral_margin)),
                rng_seed=int(job.get('rng_seed', 42)),
                local_kf_enabled=_resolve_local_kf_flag(job.get('local_kf', 'auto'), Phase2Config.local_kf_enabled),
                local_kf_sigma_T_ps=float(job.get('local_kf_sigma_T_ps', Phase2Config.local_kf_sigma_T_ps)),
                local_kf_sigma_f_hz=float(job.get('local_kf_sigma_f_hz', Phase2Config.local_kf_sigma_f_hz)),
                local_kf_init_var_T_ps=float(job.get('local_kf_init_var_T_ps', Phase2Config.local_kf_init_var_T_ps)),
                local_kf_init_var_f_hz=float(job.get('local_kf_init_var_f_hz', Phase2Config.local_kf_init_var_f_hz)),
                local_kf_max_abs_ps=float(job.get('local_kf_max_abs_ps', Phase2Config.local_kf_max_abs_ps)),
                local_kf_max_abs_freq_hz=float(job.get('local_kf_max_abs_freq_hz', Phase2Config.local_kf_max_abs_freq_hz)),
                local_kf_clock_gain=float(job.get('local_kf_clock_gain', Phase2Config.local_kf_clock_gain)),
                local_kf_freq_gain=float(job.get('local_kf_freq_gain', Phase2Config.local_kf_freq_gain)),
                local_kf_iterations=int(job.get('local_kf_iterations', Phase2Config.local_kf_iterations)),
                retune_offsets_hz=tuple(float(v) for v in job.get('retune_offsets_hz', Phase2Config.retune_offsets_hz)),
                coarse_enabled=bool(job.get('coarse_enabled', Phase2Config.coarse_enabled)),
                coarse_bandwidth_hz=float(job.get('coarse_bw_hz', Phase2Config.coarse_bandwidth_hz)),
                coarse_duration_s=float(job.get('coarse_duration_us', Phase2Config.coarse_duration_s * 1e6)) * 1e-6,
                save_results=True,
                plot_results=bool(job.get('make_plots', False)),
                results_dir=str(job_dir),
            )
            sim = Phase2Simulation(cfg)
            telemetry = sim.run()
            consensus = telemetry['consensus']
            kf_metrics = telemetry['local_kf']
            results.append(
                {
                    'job': job_name,
                    'n_nodes': cfg.n_nodes,
                    'comm_range_m': cfg.comm_range_m,
                    'weighting': cfg.weighting,
                    'target_rmse_ps': cfg.target_rmse_ps,
                    'converged': consensus['converged'],
                    'convergence_iteration': consensus['convergence_iteration'],
                    'final_timing_rmse_ps': consensus['timing_rms_ps'][-1] if consensus['timing_rms_ps'] else None,
                    'local_kf': kf_metrics,
                    'output_dir': str(job_dir),
                }
            )

        # Aggregate statistical analysis across Phase 2 runs
        driftlock choir_rmse_tau = []
        baseline_rmse_tau = []
        for entry in results:
            jsonl_path = Path(entry['output_dir']) / 'phase2_runs.jsonl'
            if jsonl_path.exists():
                with open(jsonl_path, 'r') as f:
                    lines = [line.strip() for line in f if line.strip()]
                    if lines:
                        last_telemetry = json.loads(lines[-1])
                        stats = last_telemetry.get('statistics', {})
                        rmse_tau_point = stats.get('rmse_tau', {}).get('point_estimate')
                        if rmse_tau_point is not None:
                            if last_telemetry.get('baseline_mode', False):
                                baseline_rmse_tau.append(rmse_tau_point)
                            else:
                                driftlock choir_rmse_tau.append(rmse_tau_point)

        aggregated_stats = {}
        if driftlock choir_rmse_tau and baseline_rmse_tau:
            validator = StatisticalValidator(StatsParams(confidence_level=0.95, bootstrap_samples=1000, random_state=42))
            t_test = validator.paired_t_test(np.array(driftlock choir_rmse_tau), np.array(baseline_rmse_tau))
            effect = validator.effect_sizes(np.array(driftlock choir_rmse_tau), np.array(baseline_rmse_tau))
            boot_test = validator.bootstrap_hypothesis_test(np.array(driftlock choir_rmse_tau), np.array(baseline_rmse_tau))
            aggregated_stats = {
                't_test': t_test,
                'effect_sizes': effect,
                'bootstrap_test': boot_test,
                'n_driftlock choir_runs': len(driftlock choir_rmse_tau),
                'n_baseline_runs': len(baseline_rmse_tau),
                'mean_driftlock choir_rmse_tau_ps': np.mean(driftlock choir_rmse_tau),
                'mean_baseline_rmse_tau_ps': np.mean(baseline_rmse_tau),
            }

        # Add aggregated stats to the last result or as separate key
        if results:
            results[-1]['aggregated_statistics'] = aggregated_stats
        else:
            results.append({'aggregated_statistics': aggregated_stats})

        return results

    def _run_ablation_jobs(self) -> List[Dict[str, Any]]:
        """Run ablation sweeps using the provided ablation config."""
        keys = [
            'carrier_count',
            'delta_f_spacing_hz',
            'snr_db',
            'phase_noise_psd_dbc_hz',
            'consensus_gains',
        ]
        combinations = generate_combinations(self.ablation_params)
        combo_aggregates: List[Dict[str, Any]] = []
        results: List[Dict[str, Any]] = []
        base_seed = 2025
        max_iterations = 1000

        for combo_idx, combo in enumerate(combinations):
            (
                carrier_count,
                delta_f_spacing_hz,
                snr_db,
                phase_noise_psd_dbc_hz,
                consensus_gain,
            ) = combo
            combo_name = (
                f"carrier{carrier_count}_spacing{int(delta_f_spacing_hz/1000)}k_"
                f"snr{snr_db}db_psd{int(phase_noise_psd_dbc_hz)}dbc_gain{consensus_gain}"
            )
            combo_dir = self.output_root / combo_name
            combo_dir.mkdir(exist_ok=True, parents=True)

            if self.cfg.comparative:
                # Driftlock mode
                driftlock choir_dir = combo_dir / 'driftlock choir'
                driftlock choir_dir.mkdir(exist_ok=True, parents=True)
                telemetries_d = run_ablation_combo(
                    combo,
                    str(driftlock choir_dir),
                    self.n_monte_carlo,
                    base_seed,
                    max_iterations,
                    local_kf_enabled=True,
                    weighting='inverse_variance',
                )
                agg_d = aggregate_telemetries(telemetries_d)
                save_aggregated(str(driftlock choir_dir), agg_d)

                # Baseline mode
                baseline_dir = combo_dir / 'baseline'
                baseline_dir.mkdir(exist_ok=True, parents=True)
                telemetries_b = run_ablation_combo(
                    combo,
                    str(baseline_dir),
                    self.n_monte_carlo,
                    base_seed + 10000,  # Offset for baseline derivatives
                    max_iterations,
                    local_kf_enabled=False,
                    weighting='uniform',
                )
                agg_b = aggregate_telemetries(telemetries_b)
                save_aggregated(str(baseline_dir), agg_b)

                # Prepare rows
                row_d = dict(zip(keys, combo))
                row_d.update({'mode': 'driftlock choir', **agg_d})
                row_b = dict(zip(keys, combo))
                row_b.update({'mode': 'baseline', **agg_b})
                combo_aggregates.extend([row_d, row_b])

                # Comparative aggregates if both have data
                if agg_d and agg_b and 'mean_final_rmse_ps' in agg_d and 'mean_final_rmse_ps' in agg_b:
                    rmse_d = agg_d['mean_final_rmse_ps']
                    rmse_b = agg_b['mean_final_rmse_ps']
                    relative_improvement = ((rmse_b - rmse_d) / rmse_b) * 100 if rmse_b != 0 else 0
                    comparative_stats = {
                        'relative_improvement_pct': relative_improvement,
                        'mean_rmse_driftlock choir_ps': rmse_d,
                        'mean_rmse_baseline_ps': rmse_b,
                    }
                else:
                    comparative_stats = {}

                combo_summary = {
                    'combo_name': combo_name,
                    'driftlock choir': agg_d,
                    'baseline': agg_b,
                    'comparative': comparative_stats,
                    'output_dir': str(combo_dir),
                }
            else:
                # Single mode (Driftlock)
                telemetries = run_ablation_combo(
                    combo,
                    str(combo_dir),
                    self.n_monte_carlo,
                    base_seed,
                    max_iterations,
                )
                agg = aggregate_telemetries(telemetries)
                save_aggregated(str(combo_dir), agg)

                row = dict(zip(keys, combo))
                row.update(agg)
                combo_aggregates.append(row)

                combo_summary = {
                    'combo_name': combo_name,
                    'aggregates': agg,
                    'output_dir': str(combo_dir),
                }

            results.append(combo_summary)
            print(f"Completed ablation combination {combo_idx + 1}/{len(combinations)}: {combo_name}")

        # Save consolidated summary CSV
        summary_csv = self.output_root / 'ablation_summary.csv'
        save_summary_csv(str(summary_csv), combo_aggregates)

        return results

    def _compose_report(self, summary: Dict[str, Any]) -> str:
        lines = ["=== DriftLock Monte Carlo Summary ===", f"Output: {self.output_root}", ""]
        phase1 = summary.get('phase1', [])
        if phase1:
            lines.append('[Phase 1 alias-map]')
            for entry in phase1:
                lines.append(f"- {entry['job']}: offsets={entry['retune_offsets_hz']} SNR={entry['snr_db']}")
                for diag in entry['calibration_summaries']:
                    lines.append(
                        f"  • mode={diag['calibration_mode']} mean_bias_ps={diag['mean_bias_ps']} manifest={diag['manifest_path']}"
                    )
            lines.append('')
        phase2 = summary.get('phase2', [])
        if phase2:
            lines.append('[Phase 2 consensus]')
            for entry in phase2:
                lines.append(
                    f"- {entry['job']}: nodes={entry['n_nodes']} weighting={entry['weighting']} converged={entry['converged']} final_rmse={entry['final_timing_rmse_ps']}"
                )
            # Aggregated stats
            agg_stats = None
            for entry in phase2:
                if 'aggregated_statistics' in entry and entry['aggregated_statistics']:
                    agg_stats = entry['aggregated_statistics']
                    break
            if agg_stats:
                lines.append('[Statistical Validation (Driftlock vs Baseline)]')
                lines.append(f"  • Paired t-test p-value: {agg_stats['t_test']['p_value']:.4f} (significant: {agg_stats['t_test']['significant']})")
                lines.append(f"  • Cohen's d: {agg_stats['effect_sizes']['cohens_d']:.3f} ({agg_stats['effect_sizes']['cohens_d_interpretation']})")
                lines.append(f"  • Relative improvement: {agg_stats['effect_sizes']['relative_improvement_pct']:.1f}%")
                lines.append(f"  • Bootstrap p-value: {agg_stats['bootstrap_test']['p_value']:.4f} (significant: {agg_stats['bootstrap_test']['significant']})")
                lines.append(f"  • Runs: Driftlock={agg_stats['n_driftlock choir_runs']}, Baseline={agg_stats['n_baseline_runs']}")
            lines.append('')
        ablation = summary.get('ablation', [])
        if ablation:
            lines.append('[Ablation Sweeps]')
            for entry in ablation:
                lines.append(f"- {entry['combo_name']}:")
                if self.cfg.comparative:
                    agg_d = entry.get('driftlock choir', {})
                    agg_b = entry.get('baseline', {})
                    comp = entry.get('comparative', {})
                    lines.append(f"  • Driftlock RMSE: {agg_d.get('mean_final_rmse_ps', 'N/A')} ps")
                    lines.append(f"  • Baseline RMSE: {agg_b.get('mean_final_rmse_ps', 'N/A')} ps")
                    if comp:
                        lines.append(f"  • Improvement: {comp.get('relative_improvement_pct', 0):.1f}%")
                else:
                    agg = entry.get('aggregates', {})
                    lines.append(f"  • Mean RMSE: {agg.get('mean_final_rmse_ps', 'N/A')} ps")
                    lines.append(f"  • Convergence Rate: {agg.get('convergence_rate', 0):.2f}")
            lines.append(f"  Consolidated CSV: {self.output_root / 'ablation_summary.csv'}")
            lines.append('')
        lines.append('=== End Summary ===')
        return "\n".join(lines)

    def _compose_markdown_summary(self, summary: Dict[str, Any]) -> str:
        md: List[str] = []
        md.append(f"# Monte Carlo Summary\n\nOutput: `{self.output_root}`\n")
        # Phase 1
        phase1 = summary.get('phase1', [])
        if phase1:
            md.append("## Phase 1 — Alias Calibration\n")
            for entry in phase1:
                md.append(f"### Job: {entry.get('job')}\n")
                md.append(f"- Retune offsets (Hz): {entry.get('retune_offsets_hz')}\n")
                md.append(f"- Coarse BW (Hz): {entry.get('coarse_bw_hz')}\n")
                md.append(f"- SNR (dB): {entry.get('snr_db')}\n")
                md.append("\n| Calibration | Mean Bias (ps) | Manifest | CSV |\n|---|---:|---|---|\n")
                for diag in entry.get('calibration_summaries', []):
                    md.append(
                        f"| {diag.get('calibration_mode')} | {diag.get('mean_bias_ps')} | {diag.get('manifest_path')} | {diag.get('csv_path')} |\n"
                    )
                md.append("\n")
        # Phase 2
        phase2 = summary.get('phase2', [])
        if phase2:
            md.append("## Phase 2 — Consensus\n")
            md.append("\n| Job | Nodes | Weighting | Converged | Iteration | Final RMSE (ps) | Output |\n|---|---:|---|:---:|---:|---:|---|\n")
            for entry in phase2:
                md.append(
                    f"| {entry.get('job')} | {entry.get('n_nodes')} | {entry.get('weighting')} | {entry.get('converged')} | {entry.get('convergence_iteration')} | {entry.get('final_timing_rmse_ps')} | {entry.get('output_dir')} |\n"
                )
            # Aggregated stats
            agg_stats = None
            for entry in phase2:
                if 'aggregated_statistics' in entry and entry['aggregated_statistics']:
                    agg_stats = entry['aggregated_statistics']
                    break
            if agg_stats:
                md.append("## Statistical Validation (Driftlock vs Baseline)\n")
                md.append(f"- **Paired t-test**: p = {agg_stats['t_test']['p_value']:.4f} (significant: {agg_stats['t_test']['significant']})\n")
                md.append(f"- **Cohen's d**: {agg_stats['effect_sizes']['cohens_d']:.3f} ({agg_stats['effect_sizes']['cohens_d_interpretation']})\n")
                md.append(f"- **Relative improvement**: {agg_stats['effect_sizes']['relative_improvement_pct']:.1f}%\n")
                md.append(f"- **Bootstrap test**: p = {agg_stats['bootstrap_test']['p_value']:.4f} (significant: {agg_stats['bootstrap_test']['significant']})\n")
                md.append(f"- **Runs**: Driftlock = {agg_stats['n_driftlock choir_runs']}, Baseline = {agg_stats['n_baseline_runs']}\n")
                md.append(f"- **Means**: Driftlock RMSE τ = {agg_stats['mean_driftlock choir_rmse_tau_ps']:.2f} ps, Baseline = {agg_stats['mean_baseline_rmse_tau_ps']:.2f} ps\n")
            md.append("\n")
        # Ablation
        ablation = summary.get('ablation', [])
        if ablation:
            md.append("## Ablation Sweeps\n")
            if self.cfg.comparative:
                md.append("### Comparative Runs (Driftlock vs Baseline)\n")
                md.append("\n| Combo | Driftlock RMSE (ps) | Baseline RMSE (ps) | Improvement (%) | Output |\n|---|---:|---:|---:|---|\n")
                for entry in ablation:
                    agg_d = entry.get('driftlock choir', {})
                    agg_b = entry.get('baseline', {})
                    comp = entry.get('comparative', {})
                    rmse_d = agg_d.get('mean_final_rmse_ps', 'N/A')
                    rmse_b = agg_b.get('mean_final_rmse_ps', 'N/A')
                    imp = comp.get('relative_improvement_pct', 'N/A')
                    md.append(f"| {entry.get('combo_name')} | {rmse_d} | {rmse_b} | {imp} | {entry.get('output_dir')} |\n")
            else:
                md.append("### Single Mode (Driftlock)\n")
                md.append("\n| Combo | Mean RMSE (ps) | Convergence Rate | Output |\n|---|---:|---:|---|\n")
                for entry in ablation:
                    agg = entry.get('aggregates', {})
                    md.append(
                        f"| {entry.get('combo_name')} | {agg.get('mean_final_rmse_ps', 'N/A')} | "
                        f"{agg.get('convergence_rate', 'N/A')} | {entry.get('output_dir')} |\n"
                    )
            md.append(f"\nConsolidated summary: [{self.output_root / 'ablation_summary.csv'}]({self.output_root / 'ablation_summary.csv'})\n")
        md.append("---\nGenerated by `scripts/run_mc.py`.\n")
        return "".join(md)


def _radius_from_density(density: float, area_side_m: float) -> float:
    if density <= 0.0 or density > 1.0:
        raise ValueError('Density must lie in (0,1].')
    radius = math.sqrt(density) * area_side_m / math.sqrt(math.pi)
    return float(min(radius, area_side_m))


def _resolve_local_kf_flag(flag: str, default: bool) -> bool:
    if isinstance(flag, bool):
        return flag
    normalized = str(flag).lower()
    if normalized == 'auto':
        return default
    if normalized == 'on':
        return True
    if normalized in {'off', 'baseline'}:
        return False
    raise ValueError(f'Unknown local KF flag: {flag}')


def main() -> None:
    parser = argparse.ArgumentParser(description='DriftLock Monte Carlo runner (updated CLI)')
    parser.add_argument('simulation_type', choices=['phase1', 'phase2', 'all'], help='Simulations to execute')
    parser.add_argument('-c', '--config', required=True, help='YAML configuration describing jobs')
    parser.add_argument('-a', '--ablation-config', help='YAML configuration for ablation sweeps (e.g., sim/configs/ablations.yaml)')
    parser.add_argument('--comparative-runs', action='store_true', help='Run comparative Driftlock vs baseline modes for ablations')
    parser.add_argument('-o', '--output', default='results/mc_runs', help='Root output directory')
    parser.add_argument('-r', '--run-id', help='Optional run identifier appended to the output path')
    args = parser.parse_args()

    runner = MonteCarloRunner(
        MCRunConfig(
            args.simulation_type,
            args.config,
            args.output,
            args.run_id,
            ablation_config=args.ablation_config,
            comparative=args.comparative_runs,
        )
    )
    start = time.time()
    runner.run()
    duration = time.time() - start
    print(f"Total runtime: {duration:.2f} s")


if __name__ == '__main__':
    main()
