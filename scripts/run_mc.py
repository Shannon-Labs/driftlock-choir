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
from mac.scheduler import MacSlots
from utils.io import ensure_directory, save_json


@dataclass
class MCRunConfig:
    simulation_type: str
    config_file: str
    output_dir: str = 'results'
    run_id: Optional[str] = None


class MonteCarloRunner:
    def __init__(self, cfg: MCRunConfig):
        self.cfg = cfg
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
            'config': self.config,
        }
        save_json(payload, self.output_root / 'run_config.json')

    def run(self) -> Dict[str, Any]:
        summary: Dict[str, Any] = {}
        sim_type = self.cfg.simulation_type
        if sim_type in ('phase1', 'all'):
            summary['phase1'] = self._run_phase1_jobs()
        if sim_type in ('phase2', 'all'):
            summary['phase2'] = self._run_phase2_jobs()
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
            md.append("\n")
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
    parser.add_argument('-o', '--output', default='results/mc_runs', help='Root output directory')
    parser.add_argument('-r', '--run-id', help='Optional run identifier appended to the output path')
    args = parser.parse_args()

    runner = MonteCarloRunner(MCRunConfig(args.simulation_type, args.config, args.output, args.run_id))
    start = time.time()
    runner.run()
    duration = time.time() - start
    print(f"Total runtime: {duration:.2f} s")


if __name__ == '__main__':
    main()
