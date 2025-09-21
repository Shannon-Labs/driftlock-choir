#!/usr/bin/env python3
"""
Ablation sweep script for Phase 2 consensus simulations.
Generates parameter combinations from ablations.yaml, runs Monte Carlo simulations
for each combination, aggregates results, and outputs consolidated telemetry.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import yaml
from itertools import product

# Add sim/ and src/ to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'sim'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from phase2 import Phase2Config, Phase2Simulation  # type: ignore
from alg.chronometric_handshake import ChronometricHandshakeConfig  # type: ignore
from utils.io import ensure_directory  # type: ignore


@dataclass
class AblationConfig(Phase2Config):
    """Extended Phase2Config for ablations with additional parameters."""
    carrier_count: int = 1
    delta_f_spacing_hz: float = 100_000.0
    phase_noise_psd_dbc_hz: float = -80.0

    def handshake_config(self) -> ChronometricHandshakeConfig:
        """Override to include phase noise PSD from ablation config."""
        return ChronometricHandshakeConfig(
            beat_duration_s=20e-6,
            baseband_rate_factor=20.0,
            min_baseband_rate_hz=200_000.0,
            min_adc_rate_hz=20_000.0,
            filter_relative_bw=1.4,
            phase_noise_psd=self.phase_noise_psd_dbc_hz,  # Override with ablation value
            jitter_rms_s=1e-12,
            retune_offsets_hz=self.retune_offsets_hz,
            coarse_enabled=self.coarse_enabled,
            coarse_bandwidth_hz=self.coarse_bandwidth_hz,
            coarse_duration_s=self.coarse_duration_s,
            coarse_variance_floor_ps=self.coarse_variance_floor_ps,
            multipath_two_ray_alpha=self.multipath_two_ray_alpha,
            multipath_two_ray_delay_s=self.multipath_two_ray_delay_s,
        )


def load_ablations(config_path: str) -> Dict[str, Any]:
    """Load ablation parameters from YAML."""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        ablations = data.get('ablations', {})
        n_monte_carlo = ablations.get('n_monte_carlo', 20)
        params = {
            'carrier_count': ablations['carrier_count'],
            'delta_f_spacing_hz': ablations['delta_f_spacing_hz'],
            'snr_db': ablations['snr_db'],
            'phase_noise_psd_dbc_hz': ablations['phase_noise_psd_dbc_hz'],
            'consensus_gains': ablations['consensus_gains'],
        }
        return params, n_monte_carlo
    except (OSError, yaml.YAMLError, KeyError) as e:
        print(f"Error loading {config_path}: {e}", file=sys.stderr)
        sys.exit(1)


def generate_combinations(params: Dict[str, Any]) -> List[Tuple[Any, ...]]:
    """Generate all parameter combinations using itertools.product."""
    keys = ['carrier_count', 'delta_f_spacing_hz', 'snr_db', 'phase_noise_psd_dbc_hz', 'consensus_gains']
    values = [params[k] for k in keys]
    return list(product(*values))


def create_retune_offsets(carrier_count: int, spacing_hz: float) -> Tuple[float, ...]:
    """Create retune offsets centered around 0 with given spacing."""
    if carrier_count == 1:
        return (0.0,)
    half_span = (carrier_count - 1) / 2 * spacing_hz
    offsets = np.linspace(-half_span, half_span, carrier_count)
    return tuple(offsets)


def run_ablation_combo(
    combo: Tuple[Any, ...],
    combo_dir: str,
    n_monte_carlo: int = 20,
    base_seed: int = 2025,
    max_iterations: int = 1000,
    local_kf_enabled: bool = True,
    weighting: str = 'inverse_variance',
) -> List[Dict[str, Any]]:
    """Run Monte Carlo simulations for a single parameter combination."""
    (
        carrier_count,
        delta_f_spacing_hz,
        snr_db,
        phase_noise_psd_dbc_hz,
        consensus_gain,
    ) = combo

    retune_offsets_hz = create_retune_offsets(carrier_count, delta_f_spacing_hz)

    cfg = AblationConfig(
        n_nodes=50,
        area_size_m=500.0,
        comm_range_m=180.0,
        snr_db=float(snr_db),
        base_carrier_hz=2.4e9,
        freq_offset_span_hz=80e3,
        clock_bias_std_ps=25.0,
        clock_ppm_std=2.0,
        handshake_delta_f_hz=float(delta_f_spacing_hz),  # Intentional offset spacing
        retune_offsets_hz=retune_offsets_hz,
        max_iterations=max_iterations,
        timestep_s=1e-3,
        convergence_threshold_ps=100.0,
        asynchronous=False,
        save_results=True,
        plot_results=False,  # Disable plots for speed in ablations
        results_dir=combo_dir,
        epsilon_override=float(consensus_gain),  # Consensus gain as step size
        weighting=weighting,
        target_rmse_ps=100.0,
        target_streak=3,
        local_kf_enabled=local_kf_enabled,
        carrier_count=carrier_count,
        delta_f_spacing_hz=float(delta_f_spacing_hz),
        phase_noise_psd_dbc_hz=float(phase_noise_psd_dbc_hz),
    )

    telemetries: List[Dict[str, Any]] = []
    for i in range(n_monte_carlo):
        cfg.rng_seed = base_seed + len(telemetries) * n_monte_carlo + i  # Unique seed per run
        try:
            sim = Phase2Simulation(cfg)
            telemetry = sim.run()
            telemetries.append(telemetry)
        except Exception as e:
            print(f"Error in Monte Carlo run {i+1}/{n_monte_carlo} for combo {combo}: {e}", file=sys.stderr)
            continue

    return telemetries


def aggregate_telemetries(telemetries: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate key metrics from list of telemetries."""
    if not telemetries:
        return {}

    final_rmses_ps = []
    converged_flags = []
    convergence_iters = []
    crlb_ratios_tau = []

    for tel in telemetries:
        consensus = tel.get('consensus', {})
        edge_diag = tel.get('edge_diagnostics', {})
        final_rmse = consensus.get('timing_rms_ps', [])
        if final_rmse:
            final_rmses_ps.append(float(final_rmse[-1]))
        converged = consensus.get('converged', False)
        converged_flags.append(1.0 if converged else 0.0)
        conv_iter = consensus.get('convergence_iteration')
        convergence_iters.append(float(conv_iter) if conv_iter is not None else float('nan'))
        crlb_tau = edge_diag.get('crlb_ratio_tau')
        crlb_ratios_tau.append(float(crlb_tau) if crlb_tau is not None else float('nan'))

    aggregates = {
        'n_runs': len(telemetries),
        'n_successful': len([t for t in telemetries if t.get('consensus', {}).get('converged')]),
        'mean_final_rmse_ps': float(np.mean(final_rmses_ps)) if final_rmses_ps else float('nan'),
        'std_final_rmse_ps': float(np.std(final_rmses_ps)) if final_rmses_ps else float('nan'),
        'convergence_rate': float(np.mean(converged_flags)),
        'mean_convergence_iter': float(np.nanmean(convergence_iters)),
        'mean_crlb_ratio_tau': float(np.nanmean(crlb_ratios_tau)),
    }
    return aggregates


def save_aggregated(combo_dir: str, aggregates: Dict[str, Any]) -> None:
    """Save aggregated metrics for a combination to JSON."""
    aggregated_path = Path(combo_dir) / 'aggregated.json'
    with open(aggregated_path, 'w', encoding='utf-8') as f:
        json.dump(aggregates, f, indent=2)
    print(f"Aggregated results saved to {aggregated_path}")


def save_summary_csv(summary_path: str, combo_aggregates: List[Dict[str, Any]]) -> None:
    """Save summary CSV across all combinations."""
    if not combo_aggregates:
        return

    fieldnames = [
        'carrier_count', 'delta_f_spacing_hz', 'snr_db', 'phase_noise_psd_dbc_hz', 'consensus_gain',
        'n_runs', 'n_successful', 'mean_final_rmse_ps', 'std_final_rmse_ps',
        'convergence_rate', 'mean_convergence_iter', 'mean_crlb_ratio_tau',
    ]

    with open(summary_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(combo_aggregates)

    print(f"Summary CSV saved to {summary_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description='Run ablation sweeps for Phase 2 simulations')
    parser.add_argument('--config', default='sim/configs/ablations.yaml', help='Path to ablations YAML')
    parser.add_argument('--output', default='results/ablations', help='Output directory for ablations')
    parser.add_argument('--n-monte-carlo', type=int, default=20, help='Number of MC runs per combination')
    parser.add_argument('--base-seed', type=int, default=2025, help='Base seed for reproducibility')
    args = parser.parse_args()

    params, default_n_mc = load_ablations(args.config)
    n_monte_carlo = args.n_monte_carlo or default_n_mc

    combinations = generate_combinations(params)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / 'ablation_summary.csv'

    combo_aggregates: List[Dict[str, Any]] = []
    for combo_idx, combo in enumerate(combinations):
        carrier_count, delta_f_spacing_hz, snr_db, phase_noise_psd_dbc_hz, consensus_gain = combo
        combo_name = f"carrier{carrier_count}_spacing{int(delta_f_spacing_hz/1000)}k_snr{snr_db}db_psd{int(phase_noise_psd_dbc_hz)}dbc_gain{consensus_gain}"
        combo_dir = output_dir / combo_name
        combo_dir.mkdir(exist_ok=True)

        print(f"Running combination {combo_idx + 1}/{len(combinations)}: {combo_name}")
        telemetries = run_ablation_combo(
            combo, str(combo_dir), n_monte_carlo, args.base_seed
        )
        aggregates = aggregate_telemetries(telemetries)
        save_aggregated(str(combo_dir), aggregates)

        # Prepare row for summary CSV
        row = {
            'carrier_count': carrier_count,
            'delta_f_spacing_hz': delta_f_spacing_hz,
            'snr_db': snr_db,
            'phase_noise_psd_dbc_hz': phase_noise_psd_dbc_hz,
            'consensus_gain': consensus_gain,
            **aggregates,
        }
        combo_aggregates.append(row)

    save_summary_csv(str(summary_path), combo_aggregates)
    print(f"Ablation sweeps completed. Summary in {summary_path}")


if __name__ == "__main__":
    main()