"""
Model fidelity validation metrics for DriftLock simulations.

This module provides cross-validation of simulation outputs against analytical
bounds (CRLB, consensus theory) and statistical discrepancy analysis.
Integrates with existing telemetry for automated flagging.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List
import numpy as np
from numpy.typing import NDArray

from dataclasses import dataclass

from .crlb import CRLBParams, JointCRLBCalculator
from .stats import StatisticalValidator, StatsParams
from ..alg.consensus import ConsensusOptions, ConsensusResult, DecentralizedChronometricConsensus
import networkx as nx


@dataclass
class FidelityParams:
    """Parameters for fidelity validation."""
    confidence_level: float = 0.95
    bootstrap_samples: int = 1000
    discrepancy_threshold: float = 2.0  # Max allowed CRLB ratio
    consensus_tolerance_ps: float = 100.0
    rng_seed: Optional[int] = 42
    log_level: str = 'INFO'


class FidelityValidator:
    """Core class for model fidelity checks."""

    def __init__(self, params: FidelityParams):
        self.params = params
        self.stats_validator = StatisticalValidator(
            StatsParams(
                confidence_level=params.confidence_level,
                bootstrap_samples=params.bootstrap_samples,
                random_state=params.rng_seed
            )
        )
        self.logger = logging.getLogger(__name__)

    def validate_crlb_cross(
        self,
        sim_rmse_delay: float,
        sim_rmse_freq: float,
        sim_covariance: np.ndarray,
        crlb_params: CRLBParams
    ) -> Dict[str, Any]:
        """
        Cross-validate simulated RMSE against analytical CRLB.

        Args:
            sim_rmse_delay: Simulated RMSE for delay (s)
            sim_rmse_freq: Simulated RMSE for frequency (Hz)
            sim_covariance: 2x2 covariance from LS fit
            crlb_params: Parameters for CRLB computation

        Returns:
            Dict with validation results, ratios, and pass/fail
        """
        calculator = JointCRLBCalculator(crlb_params)
        theoretical = calculator.compute_joint_crlb()
        crlb_delay_std = theoretical['delay_crlb_std']
        crlb_freq_std = theoretical['frequency_crlb_std']

        # Efficiency ratios (should be ~1 for efficient estimators)
        delay_ratio = sim_rmse_delay / crlb_delay_std if crlb_delay_std > 0 else np.inf
        freq_ratio = sim_rmse_freq / crlb_freq_std if crlb_freq_std > 0 else np.inf

        # LS covariance vs CRLB
        ls_delay_var = sim_covariance[0, 0]
        ls_freq_var = sim_covariance[1, 1]
        crlb_vs_ls_delay = np.sqrt(theoretical['delay_crlb_variance'] / ls_delay_var) if ls_delay_var > 0 else np.inf
        crlb_vs_ls_freq = np.sqrt(theoretical['frequency_crlb_variance'] / ls_freq_var) if ls_freq_var > 0 else np.inf

        # Pass/fail criteria: ratios within threshold, LS close to CRLB
        crlb_pass = (0.5 <= delay_ratio <= self.params.discrepancy_threshold and
                     0.5 <= freq_ratio <= self.params.discrepancy_threshold)
        ls_consistent = (0.5 <= crlb_vs_ls_delay <= 2.0 and 0.5 <= crlb_vs_ls_freq <= 2.0)
        overall_pass = crlb_pass and ls_consistent

        results = {
            'theoretical_crlb': theoretical,
            'simulated_rmse': {'delay': sim_rmse_delay, 'freq': sim_rmse_freq},
            'crlb_ratios': {'delay': float(delay_ratio), 'freq': float(freq_ratio)},
            'ls_vs_crlb_ratios': {'delay': float(crlb_vs_ls_delay), 'freq': float(crlb_vs_ls_freq)},
            'crlb_pass': crlb_pass,
            'ls_consistent': ls_consistent,
            'overall_pass': overall_pass,
            'discrepancy_flags': [],
            'method': 'crlb_cross_validation'
        }

        if not crlb_pass:
            if delay_ratio > self.params.discrepancy_threshold:
                results['discrepancy_flags'].append('delay_exceeds_crlb')
            if freq_ratio > self.params.discrepancy_threshold:
                results['discrepancy_flags'].append('freq_exceeds_crlb')
        if not ls_consistent:
            results['discrepancy_flags'].append('ls_crlb_mismatch')

        self.logger.info(f"CRLB validation: overall_pass={overall_pass}, ratios_delay/freq={delay_ratio:.2f}/{freq_ratio:.2f}")
        return results

    def validate_consensus_convergence(
        self,
        consensus_result: ConsensusResult,
        graph: nx.Graph,
        theoretical_rate: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Validate consensus convergence against theoretical predictions.

        Uses spectral gap for expected convergence rate; checks iteration count
        and final RMSE against tolerance.

        Args:
            consensus_result: Output from DecentralizedChronometricConsensus
            graph: NetworkX graph used in consensus
            theoretical_rate: Optional pre-computed convergence rate (1/lambda_2)

        Returns:
            Dict with convergence metrics, expected vs observed, pass/fail
        """
        if not consensus_result.converged:
            return {
                'converged': False,
                'final_rmse_ps': consensus_result.timing_rms_ps[-1],
                'iterations': consensus_result.convergence_iteration,
                'spectral_gap': consensus_result.spectral_gap,
                'expected_rate': np.inf,
                'observed_rate': np.inf,
                'pass': False,
                'discrepancy_flags': ['did_not_converge'],
                'method': 'consensus_convergence'
            }

        # Theoretical convergence rate from spectral gap (for linear consensus)
        if theoretical_rate is None:
            options = ConsensusOptions(epsilon=consensus_result.epsilon)
            consensus_solver = DecentralizedChronometricConsensus(graph, options)
            epsilon = consensus_solver._resolve_step_size()  # Internal, but for rate
            lambda_2 = consensus_result.lambda_2
            expected_rate = epsilon * lambda_2 if lambda_2 > 0 else 0.0
        else:
            expected_rate = theoretical_rate

        observed_rate = np.log(consensus_result.timing_rms_ps[0] / consensus_result.timing_rms_ps[-1]) / len(consensus_result.timing_rms_ps)
        rmse_pass = consensus_result.timing_rms_ps[-1] <= self.params.consensus_tolerance_ps
        rate_consistent = 0.5 <= (observed_rate / expected_rate) <= 2.0 if expected_rate > 0 else True

        overall_pass = rmse_pass and rate_consistent

        results = {
            'converged': True,
            'final_rmse_ps': float(consensus_result.timing_rms_ps[-1]),
            'iterations': consensus_result.convergence_iteration,
            'spectral_gap': float(consensus_result.spectral_gap),
            'expected_convergence_rate': float(expected_rate),
            'observed_convergence_rate': float(observed_rate),
            'rate_consistency_ratio': float(observed_rate / expected_rate) if expected_rate > 0 else 0.0,
            'rmse_pass': rmse_pass,
            'rate_consistent': rate_consistent,
            'overall_pass': overall_pass,
            'discrepancy_flags': [],
            'method': 'consensus_convergence'
        }

        if not rmse_pass:
            results['discrepancy_flags'].append('rmse_exceeds_tolerance')
        if not rate_consistent:
            results['discrepancy_flags'].append('convergence_rate_mismatch')

        self.logger.info(f"Consensus validation: pass={overall_pass}, rmse_ps={results['final_rmse_ps']:.1f}, rate_ratio={results['rate_consistency_ratio']:.2f}")
        return results

    def validate_consensus_metrics(
        self,
        final_rmse_ps: float,
        converged: bool,
        iterations: Optional[int] = None,
        tolerance_ps: Optional[float] = None,
        max_iterations: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Validate consensus results when only aggregate metrics are available."""
        tol = float(tolerance_ps if tolerance_ps is not None else self.params.consensus_tolerance_ps)
        flags: List[str] = []
        if not converged:
            flags.append('did_not_converge')
        if not np.isfinite(final_rmse_ps):
            flags.append('invalid_rmse')
        elif final_rmse_ps > tol:
            flags.append('rmse_exceeds_tolerance')
        if max_iterations is not None and iterations is not None and iterations > max_iterations:
            flags.append('exceeded_max_iterations')

        return {
            'method': 'consensus_metrics',
            'final_rmse_ps': float(final_rmse_ps),
            'converged': bool(converged),
            'iterations': iterations,
            'tolerance_ps': tol,
            'max_iterations': max_iterations,
            'overall_pass': len(flags) == 0,
            'discrepancy_flags': flags,
        }

    def analyze_discrepancies(
        self,
        sim_metrics: Dict[str, float],
        theoretical_metrics: Dict[str, float],
        n_samples: int = 1000
    ) -> Dict[str, Any]:
        """
        Perform statistical hypothesis testing on discrepancies.

        Uses bootstrap tests for mean differences and effect sizes.

        Args:
            sim_metrics: Dict of simulated metrics (e.g., {'rmse_delay': 1e-10})
            theoretical_metrics: Dict of theoretical metrics
            n_samples: Number of bootstrap samples

        Returns:
            Dict with test results, p-values, effect sizes, pass/fail
        """
        common_keys = set(sim_metrics) & set(theoretical_metrics)
        if not common_keys:
            return {'no_common_metrics': True, 'pass': False}

        test_results = {}
        overall_pass = True

        for key in common_keys:
            sim_vals = np.full(n_samples, sim_metrics[key])  # Approximate from point estimates
            theo_vals = np.full(n_samples, theoretical_metrics[key])
            boot_test = self.stats_validator.bootstrap_hypothesis_test(
                sim_vals, theo_vals, alternative='two-sided'
            )
            effect = self.stats_validator.effect_sizes(sim_vals, theo_vals)

            key_pass = boot_test['p_value'] > (1 - self.params.confidence_level)
            overall_pass = overall_pass and key_pass

            test_results[key] = {
                'bootstrap_test': boot_test,
                'effect_size': effect,
                'significant_discrepancy': not key_pass,
                'p_value': boot_test['p_value']
            }

        results = {
            'test_results': test_results,
            'overall_pass': overall_pass,
            'n_bootstrap_samples': n_samples,
            'discrepancy_count': sum(1 for r in test_results.values() if not r['significant_discrepancy']),
            'method': 'discrepancy_hypothesis_test'
        }

        if not overall_pass:
            results['discrepancy_flags'] = [k for k, r in test_results.items() if r['significant_discrepancy']]

        self.logger.info(f"Discrepancy analysis: overall_pass={overall_pass}, significant_discrepancies={len(results.get('discrepancy_flags', []))}")
        return results

    def validate_hw_emulation(
        self,
        telemetry_path: Path,
        config_params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Validate hardware emulation fidelity.

        Loads telemetry from hw_emulation run, extracts metrics, performs CRLB
        and consensus checks if applicable.

        Args:
            telemetry_path: Path to JSONL or JSON telemetry file
            config_params: Dict from hw_emulation.yaml (for CRLB params)

        Returns:
            Combined validation report
        """
        if not telemetry_path.exists():
            raise FileNotFoundError(f"Telemetry not found: {telemetry_path}")

        with open(telemetry_path, 'r') as f:
            if telemetry_path.suffix == '.jsonl':
                lines = [json.loads(line) for line in f if line.strip()]
                last_telemetry = lines[-1] if lines else {}
            else:
                last_telemetry = json.load(f)

        # Extract sim metrics (adapt to actual telemetry structure)
        stats = last_telemetry.get('statistics', {})
        sim_rmse_delay = stats.get('rmse_tau', {}).get('point_estimate', np.nan)
        sim_rmse_freq = stats.get('rmse_df', {}).get('point_estimate', np.nan)
        sim_cov = np.array(last_telemetry.get('ls_covariance', [[np.nan, 0], [0, np.nan]]))

        # CRLB params from config
        crlb_params = CRLBParams(
            snr_db=config_params.get('channel', {}).get('awgn_snr_db', 20.0),
            bandwidth=config_params.get('tx', {}).get('symbol_rate', 1e6),
            duration=config_params.get('duration_s', 0.01),
            carrier_freq=config_params.get('tx', {}).get('carrier_hz', 2.4e9),
            sample_rate=config_params.get('sample_rate_hz', 20e6)
        )

        crlb_results = self.validate_crlb_cross(
            sim_rmse_delay, sim_rmse_freq, sim_cov, crlb_params
        )

        # Consensus if present (for phase2-like)
        consensus_data = last_telemetry.get('consensus', {})
        if consensus_data:
            # Reconstruct ConsensusResult minimally
            mock_result = ConsensusResult(
                state_history=np.empty((1, 1, 2)),  # Placeholder
                timing_rms_ps=np.array([consensus_data.get('final_rmse_ps', 100.0)]),
                frequency_rms_hz=np.array([0.0]),
                converged=consensus_data.get('converged', True),
                convergence_iteration=consensus_data.get('iterations', 100),
                epsilon=consensus_data.get('epsilon', 0.1),
                asynchronous=False,
                edge_residuals={},
                lambda_max=1.0,
                lambda_2=consensus_data.get('spectral_gap', 0.1),
                spectral_gap=consensus_data.get('spectral_gap', 0.1)
            )
            # Need graph; assume from config or skip if not phase2
            graph = nx.complete_graph(config_params.get('n_nodes', 2))  # Fallback
            consensus_results = self.validate_consensus_convergence(mock_result, graph)
        else:
            consensus_results = {'skipped': 'no_consensus_data', 'pass': True}

        # Discrepancy analysis
        sim_metrics = {'rmse_delay': sim_rmse_delay, 'rmse_freq': sim_rmse_freq}
        theo_metrics = {
            'rmse_delay': crlb_results['theoretical_crlb']['delay_crlb_std'],
            'rmse_freq': crlb_results['theoretical_crlb']['frequency_crlb_std']
        }
        disc_results = self.analyze_discrepancies(sim_metrics, theo_metrics)

        report = {
            'crlb_validation': crlb_results,
            'consensus_validation': consensus_results,
            'discrepancy_analysis': disc_results,
            'hw_config_used': config_params,
            'telemetry_source': str(telemetry_path),
            'overall_pass': (crlb_results['overall_pass'] and
                             consensus_results.get('overall_pass', True) and
                             disc_results['overall_pass']),
            'discrepancy_flags': (crlb_results.get('discrepancy_flags', []) +
                                  consensus_results.get('discrepancy_flags', []) +
                                  disc_results.get('discrepancy_flags', [])),
            'method': 'hw_emulation_validation'
        }

        if report['overall_pass']:
            self.logger.info("HW emulation validation: PASSED")
        else:
            self.logger.warning(f"HW emulation validation: FAILED with flags: {report['discrepancy_flags']}")

        return report

    def generate_report(
        self,
        validations: List[Dict[str, Any]],
        output_path: Optional[Path] = None
    ) -> Dict[str, Any]:
        """
        Generate consolidated validation report.

        Args:
            validations: List of validation dicts (from various methods)
            output_path: Optional path to save JSON report

        Returns:
            Aggregated report with pass/fail summary
        """
        summary = {
            'total_validations': len(validations),
            'passed': sum(1 for v in validations if v.get('overall_pass', False)),
            'failed': len(validations) - sum(1 for v in validations if v.get('overall_pass', False)),
            'overall_pass': all(v.get('overall_pass', False) for v in validations),
            'discrepancy_summary': [],
            'detailed_results': validations
        }

        for val in validations:
            flags = val.get('discrepancy_flags', [])
            if flags:
                summary['discrepancy_summary'].extend([{'method': val['method'], 'flags': flags}])

        if output_path:
            with open(output_path, 'w') as f:
                json.dump(summary, f, indent=2, default=str)

        self.logger.info(f"Validation report: {summary['passed']}/{len(validations)} passed, overall={summary['overall_pass']}")
        return summary