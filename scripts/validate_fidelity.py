#!/usr/bin/env python3
"""
Fidelity validation script for DriftLock models.

Runs comprehensive checks on simulation telemetry against analytical bounds
using src/metrics/fidelity.py. Targets hw_emulation.yaml by default.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass, asdict, is_dataclass
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import yaml

REPO_ROOT = Path(__file__).parent.parent
sys.path.append(str(REPO_ROOT))
sys.path.append(str(REPO_ROOT / 'src'))


def sanitize_for_json(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {key: sanitize_for_json(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize_for_json(value) for value in obj]
    if is_dataclass(obj):
        return sanitize_for_json(asdict(obj))
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return float(obj)
    if isinstance(obj, Path):
        return str(obj)
    return obj

from src.metrics.fidelity import FidelityParams, FidelityValidator
from src.metrics.crlb import CRLBParams
from sim.phase1 import Phase1Config, Phase1Simulator
from sim.phase2 import Phase2Config, Phase2Simulation
from utils.io import ensure_directory  # Assume utility exists or implement simple


@dataclass
class ValidationConfig:
    """Configuration for fidelity validation run."""
    config_file: str
    output_dir: str = 'results/validate_fidelity'
    run_id: str = ''
    run_phase1: bool = True
    run_phase2: bool = False
    n_trials: int = 100
    rng_seed: int = 42
    log_level: str = 'INFO'


def load_yaml_config(config_path: str) -> Dict[str, Any]:
    """Load YAML configuration."""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f) or {}


def run_phase1_validation(
    config: Dict[str, Any],
    output_dir: Path,
    n_trials: int,
    rng_seed: int
) -> Dict[str, Any]:
    """Run Phase 1 simulation and extract metrics for validation."""
    # Extract params for Phase1Config (adapt from hw_emulation.yaml)
    common = {
        'snr_values_db': [config.get('channel', {}).get('awgn_snr_db', 25.0)],
        'n_monte_carlo': n_trials,
        'plot_results': False,
        'save_results': True,
        'delta_t_us': [config.get('truth', {}).get('tau_s', 100e-12) * 1e6],
        'loopback_cal_noise_ps': 5.0,
        'd_tx_ns': config.get('hardware', {}).get('d_tx_ns', {}),
        'd_rx_ns': config.get('hardware', {}).get('d_rx_ns', {}),
        'results_dir': str(output_dir / 'phase1'),
        'calib_mode': 'on',  # For fidelity check
        'retune_offsets_hz': (0.0,),
        'coarse_bandwidth_hz': config.get('sample_rate_hz', 20e6),
    }
    phase1_cfg = Phase1Config(**common)
    sim = Phase1Simulator(phase1_cfg)

    # Run simulation
    alias_summary = sim.run_alias_failure_map(
        retune_offsets_hz=[0.0],
        coarse_bw_hz=[config.get('sample_rate_hz', 20e6)],
        snr_db=[config.get('channel', {}).get('awgn_snr_db', 25.0)],
        num_trials=n_trials,
        rng_seed=rng_seed,
        make_plots=False
    )

    # Extract metrics from manifest or last run (assume structure)
    manifest_path = Path(alias_summary['manifest_path'])
    with open(manifest_path, 'r') as f:
        manifest = json.load(f)

    # Simulated RMSE and covariance (adapt to actual keys)
    stats = manifest.get('statistics', {})
    sim_rmse_delay = stats.get('rmse_tau', {}).get('point_estimate', np.nan)
    sim_rmse_freq = stats.get('rmse_df', {}).get('point_estimate', np.nan)
    sim_covariance = np.array(manifest.get('ls_covariance', [[1e-20, 0], [0, 1e-6]]))

    # CRLB params from config
    crlb_p = CRLBParams(
        snr_db=config.get('channel', {}).get('awgn_snr_db', 25.0),
        bandwidth=config.get('sample_rate_hz', 20e6),
        duration=config.get('duration_s', 0.01),
        carrier_freq=2.4e9,  # Assume from hw
        sample_rate=config.get('sample_rate_hz', 20e6)
    )

    return {
        'sim_rmse_delay': sim_rmse_delay,
        'sim_rmse_freq': sim_rmse_freq,
        'sim_covariance': sim_covariance,
        'crlb_params': crlb_p,
        'output_dir': str(output_dir / 'phase1'),
        'manifest_path': str(manifest_path)
    }


def run_phase2_validation(
    config: Dict[str, Any],
    output_dir: Path,
    n_trials: int,
    rng_seed: int
) -> Dict[str, Any]:
    """Run Phase 2 simulation if multi-node consensus needed."""
    # For hw_emulation, assume 2 nodes for simple consensus
    phase2_cfg = Phase2Config(
        n_nodes=2,
        area_size_m=10.0,
        comm_range_m=5.0,
        snr_db=config.get('channel', {}).get('awgn_snr_db', 25.0),
        weighting='inverse_variance',
        target_rmse_ps=100.0,
        target_streak=10,
        max_iterations=100,
        timestep_s=config.get('duration_s', 0.01),
        rng_seed=rng_seed,
        save_results=True,
        plot_results=False,
        results_dir=str(output_dir / 'phase2'),
        # Add other params from config as needed
    )
    sim = Phase2Simulation(phase2_cfg)
    telemetry = sim.run()

    # Extract consensus result (placeholder reconstruction)
    consensus = telemetry['consensus']
    consensus_result = ConsensusResult(  # Reconstruct for validation
        state_history=np.empty((1, 2, 2)),  # Placeholder
        timing_rms_ps=np.array(consensus.get('timing_rms_ps', [100.0])),
        frequency_rms_hz=np.array([0.0]),
        converged=consensus.get('converged', True),
        convergence_iteration=consensus.get('convergence_iteration'),
        epsilon=consensus.get('epsilon', 0.1),
        asynchronous=False,
        edge_residuals=consensus.get('edge_residuals', {}),
        lambda_max=1.0,
        lambda_2=consensus.get('spectral_gap', 0.1),
        spectral_gap=consensus.get('spectral_gap', 0.1)
    )

    # Graph for consensus
    import networkx as nx
    graph = nx.complete_graph(phase2_cfg.n_nodes)

    return {
        'consensus_result': consensus_result,
        'graph': graph,
        'output_dir': str(output_dir / 'phase2'),
        'telemetry_path': str(output_dir / 'phase2_runs.jsonl')
    }


def main():
    parser = argparse.ArgumentParser(description='Run DriftLock fidelity validation')
    parser.add_argument('config', nargs='?', default='sim/configs/hw_emulation.yaml',
                        help='YAML config file (default: hw_emulation.yaml)')
    parser.add_argument('-o', '--output', default='results/validate_fidelity',
                        help='Output directory')
    parser.add_argument('-r', '--run-id', help='Run identifier')
    parser.add_argument('--no-phase1', action='store_true', help='Skip Phase 1 run')
    parser.add_argument('--phase2', action='store_true', help='Include Phase 2 consensus')
    parser.add_argument('-n', '--n-trials', type=int, default=100, help='Number of trials')
    parser.add_argument('--seed', type=int, default=42, help='RNG seed')
    parser.add_argument('--log-level', default='INFO', choices=['DEBUG', 'INFO', 'WARNING'],
                        help='Logging level')
    parser.add_argument('--telemetry-path', type=str, default=None,
                        help='Path to real JSONL telemetry file (e.g., phase2_runs.jsonl); skips simulation if provided')
    parser.add_argument('--skip-hw-validation', action='store_true', help='Skip hardware-emulation validation stage.')
    parser.add_argument('--skip-crlb-validation', action='store_true', help='Skip CRLB cross-validation stage.')
    args = parser.parse_args()

    cfg = ValidationConfig(
        config_file=args.config,
        output_dir=args.output,
        run_id=args.run_id or f"fidelity_{int(time.time())}",
        run_phase1=not args.no_phase1,
        run_phase2=args.phase2,
        n_trials=args.n_trials,
        rng_seed=args.seed,
        log_level=args.log_level
    )

    # Setup logging
    logging.basicConfig(level=cfg.log_level, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    # Load config
    sim_config = load_yaml_config(cfg.config_file)
    output_root = Path(ensure_directory(cfg.output_dir)) / cfg.run_id
    output_root.mkdir(exist_ok=True, parents=True)

    # Fidelity validator
    fidelity_params = FidelityParams(
        confidence_level=0.95,
        bootstrap_samples=1000,
        discrepancy_threshold=2.0,
        consensus_tolerance_ps=100.0,
        rng_seed=cfg.rng_seed,
        log_level=cfg.log_level
    )
    validator = FidelityValidator(fidelity_params)

    validations: List[Dict[str, Any]] = []

    telemetry_path_arg = Path(args.telemetry_path) if args.telemetry_path else None
    use_real_telemetry = telemetry_path_arg is not None
    hw_telemetry_path: Optional[Path] = None
    skip_hw_validation = args.skip_hw_validation
    skip_crlb_validation = args.skip_crlb_validation

    if use_real_telemetry:
        logger.info(f"Loading real telemetry from {telemetry_path_arg}...")
        telemetry_data = load_phase2_telemetry(telemetry_path_arg, sim_config, logger)
        if not telemetry_data:
            logger.error("No telemetry entries could be parsed; aborting fidelity validation.")
            sys.exit(1)

        rmse_tau_ps = float(np.nanmean([entry['rmse_tau_ps'] for entry in telemetry_data]))
        rmse_df_hz = float(np.nanmean([entry['rmse_df_hz'] for entry in telemetry_data]))
        predicted_tau_std_ps = float(np.nanmean([entry['predicted_tau_std_ps'] for entry in telemetry_data]))
        predicted_df_std_hz = float(np.nanmean([entry['predicted_df_std_hz'] for entry in telemetry_data]))

        tau_std_ps = predicted_tau_std_ps if np.isfinite(predicted_tau_std_ps) else rmse_tau_ps
        df_std_hz = predicted_df_std_hz if np.isfinite(predicted_df_std_hz) else rmse_df_hz
        sim_covariance = np.array([
            [(tau_std_ps * 1e-12) ** 2, 0.0],
            [0.0, df_std_hz ** 2],
        ])

        first_entry = telemetry_data[0]
        crlb_params = CRLBParams(
            snr_db=first_entry['snr_db'],
            bandwidth=first_entry['bandwidth_hz'],
            duration=first_entry['duration_s'],
            carrier_freq=2.4e9,
            sample_rate=first_entry['sample_rate_hz'],
        )
        if not skip_crlb_validation:
            crlb_val = validator.validate_crlb_cross(
                rmse_tau_ps * 1e-12,
                rmse_df_hz,
                sim_covariance,
                crlb_params,
            )
            validations.append(crlb_val)
            logger.info(f"CRLB validation (real telemetry): {crlb_val['overall_pass']}")

        final_rmse_ps = float(np.nanmean([entry['final_rmse_ps'] for entry in telemetry_data]))
        converged = all(entry['converged'] for entry in telemetry_data)
        iterations = max((entry['convergence_iteration'] or 0) for entry in telemetry_data)
        tolerance_ps = first_entry['target_rmse_ps']
        if tolerance_ps is None or not np.isfinite(tolerance_ps):
            tolerance_ps = fidelity_params.consensus_tolerance_ps
        max_iterations = first_entry['max_iterations']
        if max_iterations is None:
            max_iterations = sim_config.get('algorithms', {}).get('max_iterations')
        consensus_val = validator.validate_consensus_metrics(
            final_rmse_ps,
            converged,
            iterations,
            tolerance_ps=tolerance_ps,
            max_iterations=max_iterations,
        )
        validations.append(consensus_val)
        logger.info(f"Consensus validation (real telemetry): {consensus_val['overall_pass']}")

        loaded_telemetry_file = output_root / 'loaded_phase2_telemetry.json'
        with open(loaded_telemetry_file, 'w', encoding='utf-8') as f:
            json.dump(sanitize_for_json(telemetry_data), f, indent=2)
        hw_telemetry_path = telemetry_path_arg
    else:
        # Run Phase 1 (handshake/CRLB)
        phase1_data: Dict[str, Any] = {}
        if cfg.run_phase1 and not skip_crlb_validation:
            logger.info("Running Phase 1 fidelity checks...")
            phase1_data = run_phase1_validation(sim_config, output_root, cfg.n_trials, cfg.rng_seed)
            crlb_val = validator.validate_crlb_cross(
                phase1_data['sim_rmse_delay'],
                phase1_data['sim_rmse_freq'],
                phase1_data['sim_covariance'],
                phase1_data['crlb_params']
            )
            validations.append(crlb_val)
            logger.info(f"Phase 1 CRLB validation: {crlb_val['overall_pass']}")
        elif cfg.run_phase1:
            logger.info("Skipping Phase 1 CRLB validation (flag set).")

        phase2_data: Dict[str, Any] = {}
        if cfg.run_phase2:
            logger.info("Running Phase 2 fidelity checks...")
            phase2_data = run_phase2_validation(sim_config, output_root, cfg.n_trials, cfg.rng_seed)
            consensus_val = validator.validate_consensus_convergence(
                phase2_data['consensus_result'],
                phase2_data['graph']
            )
            validations.append(consensus_val)
            logger.info(f"Phase 2 consensus validation: {consensus_val['overall_pass']}")

        if cfg.run_phase1 or cfg.run_phase2:
            combined_telemetry = {
                'phase1': phase1_data if cfg.run_phase1 else {},
                'phase2': phase2_data if cfg.run_phase2 else {},
                'config': sim_config
            }
            telemetry_file = output_root / 'validation_telemetry.json'
            with open(telemetry_file, 'w', encoding='utf-8') as f:
                json.dump(sanitize_for_json(combined_telemetry), f, indent=2)
            hw_telemetry_path = telemetry_file

    if not skip_hw_validation:
        if hw_telemetry_path is None:
            logger.error('No telemetry path available for hardware emulation validation.')
            sys.exit(1)
        hw_val = validator.validate_hw_emulation(hw_telemetry_path, sim_config)
        validations.append(hw_val)

    # Discrepancy analysis across all
    all_sim_metrics = {}
    all_theo_metrics = {}
    for val in validations:
        if 'simulated_rmse' in val:
            all_sim_metrics.update(val['simulated_rmse'])
        if 'theoretical_crlb' in val:
            theo = val['theoretical_crlb']
            all_theo_metrics.update({
                'rmse_delay': theo['delay_crlb_std'],
                'rmse_freq': theo['frequency_crlb_std']
            })
    if all_sim_metrics and all_theo_metrics:
        disc_val = validator.analyze_discrepancies(all_sim_metrics, all_theo_metrics)
        validations.append(disc_val)

    # Generate report
    report = validator.generate_report(validations, output_root / 'fidelity_report.json')
    summary_file = output_root / 'validation_summary.txt'
    with open(summary_file, 'w') as f:
        f.write(f"Fidelity Validation Report\n")
        f.write(f"Run ID: {cfg.run_id}\n")
        f.write(f"Config: {cfg.config_file}\n")
        f.write(f"Passed: {report['passed']}/{report['total_validations']}\n")
        f.write(f"Overall: {'PASS' if report['overall_pass'] else 'FAIL'}\n")
        if report['discrepancy_summary']:
            f.write("\nDiscrepancies:\n")
            for disc in report['discrepancy_summary']:
                f.write(f"- {disc['method']}: {disc['flags']}\n")

    logger.info(f"Validation complete. Report: {output_root / 'fidelity_report.json'}")
    if report['overall_pass']:
        sys.exit(0)
    else:
        sys.exit(1)


def load_phase2_telemetry(path: Path, config: Dict[str, Any], logger: logging.Logger) -> List[Dict[str, Any]]:
    """Parse Phase 2 JSONL telemetry into aggregate metrics."""
    records: List[Dict[str, Any]] = []
    channel_defaults = config.get('channel', {})
    duration_default = config.get('duration_s', 0.01)
    sample_rate_default = config.get('sample_rate_hz', 20e6)
    bandwidth_default = config.get('sample_rate_hz', 20e6)

    def _last(values: Optional[List[float]]) -> Optional[float]:
        if not values:
            return None
        return values[-1]

    def _first_non_none(*values: Optional[float]) -> Optional[float]:
        for value in values:
            if value is not None:
                return value
        return None

    with path.open('r', encoding='utf-8') as handle:
        for idx, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError:
                logger.warning(f"Skipping invalid JSONL line {idx} in {path}")
                continue

            config_block = raw.get('config', {})
            consensus = raw.get('consensus', {})
            stats = raw.get('statistics', {})
            edge_diag = raw.get('edge_diagnostics', {})

            rmse_tau_ps = _first_non_none(
                stats.get('rmse_tau', {}).get('point_estimate'),
                _last(consensus.get('timing_rms_ps')),
                edge_diag.get('measurement_rmse_tau_ps'),
            )
            rmse_df_hz = _first_non_none(
                stats.get('rmse_df', {}).get('point_estimate'),
                _last(consensus.get('frequency_rms_hz')),
                edge_diag.get('measurement_rmse_df_hz'),
            )
            predicted_tau_std_ps = edge_diag.get('predicted_tau_std_ps')
            predicted_df_std_hz = edge_diag.get('predicted_df_std_hz')
            final_rmse_ps = _first_non_none(
                _last(consensus.get('timing_rms_ps')),
                rmse_tau_ps,
            )

            records.append({
                'rmse_tau_ps': float(rmse_tau_ps) if rmse_tau_ps is not None else np.nan,
                'rmse_df_hz': float(rmse_df_hz) if rmse_df_hz is not None else np.nan,
                'predicted_tau_std_ps': float(predicted_tau_std_ps) if predicted_tau_std_ps is not None else np.nan,
                'predicted_df_std_hz': float(predicted_df_std_hz) if predicted_df_std_hz is not None else np.nan,
                'final_rmse_ps': float(final_rmse_ps) if final_rmse_ps is not None else np.nan,
                'converged': bool(consensus.get('converged', False)),
                'convergence_iteration': consensus.get('convergence_iteration'),
                'target_rmse_ps': consensus.get('target_rmse_ps'),
                'max_iterations': config_block.get('max_iterations'),
                'snr_db': float(config_block.get('snr_db', channel_defaults.get('awgn_snr_db', 25.0))),
                'bandwidth_hz': float(config_block.get('coarse_bandwidth_hz', bandwidth_default)),
                'duration_s': float(config_block.get('coarse_duration_s', duration_default)),
                'sample_rate_hz': float(config_block.get('min_baseband_rate_hz', sample_rate_default)),
            })

    logger.info(f"Loaded {len(records)} telemetry entries from {path}")
    return records


if __name__ == "__main__":
    main()