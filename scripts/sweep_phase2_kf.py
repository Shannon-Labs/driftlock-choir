#!/usr/bin/env python3
"""Sweep Phase 2 local KF parameters across gains/seeds and report RMSE."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent / 'sim'))
sys.path.append(str(Path(__file__).resolve().parent.parent / 'src'))

from phase2 import Phase2Config, Phase2Simulation, _radius_from_density
from utils.io import ensure_directory


@dataclass
class SweepResult:
    gain_clock: float
    gain_freq: float
    iterations: int
    seed: int
    final_rmse_ps: float
    filtered_clock_rms_ps: float
    improvement_ps: float
    freq_improvement_hz: float
    converged: bool
    consensus_iterations: Optional[int]

    def to_dict(self) -> Dict[str, float | int | bool | None]:
        payload = asdict(self)
        # Cast floats for JSON cleanliness
        payload['final_rmse_ps'] = float(self.final_rmse_ps)
        payload['filtered_clock_rms_ps'] = float(self.filtered_clock_rms_ps)
        payload['improvement_ps'] = float(self.improvement_ps)
        payload['freq_improvement_hz'] = float(self.freq_improvement_hz)
        return payload


def _parse_float_list(text: str) -> List[float]:
    values: List[float] = []
    for chunk in text.split(','):
        chunk = chunk.strip()
        if not chunk:
            continue
        values.append(float(chunk))
    return values


def _parse_int_list(text: str) -> List[int]:
    values: List[int] = []
    for chunk in text.split(','):
        chunk = chunk.strip()
        if not chunk:
            continue
        values.append(int(chunk))
    return values


def run_sweep(
    gains: Sequence[float],
    freq_gains: Sequence[float],
    iterations: Sequence[int],
    seeds: Sequence[int],
    base_config: Phase2Config,
) -> List[SweepResult]:
    results: List[SweepResult] = []
    base_dict = asdict(base_config)
    for gain_clock in gains:
        for gain_freq in freq_gains:
            for iters in iterations:
                for seed in seeds:
                    cfg_dict = {
                        **base_dict,
                        'rng_seed': seed,
                        'local_kf_clock_gain': gain_clock,
                        'local_kf_freq_gain': gain_freq,
                        'local_kf_iterations': iters,
                    }
                    cfg = Phase2Config(**cfg_dict)
                    # Ensure we never persist artifacts during sweeps unless requested
                    cfg.save_results = base_config.save_results
                    cfg.plot_results = base_config.plot_results
                    sim = Phase2Simulation(cfg)
                    telemetry = sim.run()
                    kf = telemetry['local_kf']
                    consensus = telemetry['consensus']
                    results.append(
                        SweepResult(
                            gain_clock=gain_clock,
                            gain_freq=gain_freq,
                            iterations=iters,
                            seed=seed,
                            final_rmse_ps=float(consensus['timing_rms_ps'][-1]),
                            filtered_clock_rms_ps=float(kf['filtered_clock_rms_ps']),
                            improvement_ps=float(kf['clock_improvement_ps']),
                            freq_improvement_hz=float(kf['freq_improvement_hz']),
                            converged=bool(consensus['converged']),
                            consensus_iterations=consensus['convergence_iteration'],
                        )
                    )
    return results


def summarise(results: Sequence[SweepResult]) -> Dict[str, object]:
    if not results:
        raise ValueError('No sweep results available to summarise')
    arr = np.array([res.final_rmse_ps for res in results], dtype=float)
    improvements = np.array([res.improvement_ps for res in results], dtype=float)
    best = min(results, key=lambda r: r.final_rmse_ps)

    combo_groups: Dict[Tuple[float, float, int], List[SweepResult]] = {}
    for res in results:
        key = (res.gain_clock, res.gain_freq, res.iterations)
        combo_groups.setdefault(key, []).append(res)

    combo_stats: List[Dict[str, float]] = []
    for (gain_clock, gain_freq, iters), group in combo_groups.items():
        rmse_vals = np.array([entry.final_rmse_ps for entry in group], dtype=float)
        imp_vals = np.array([entry.improvement_ps for entry in group], dtype=float)
        combo_stats.append(
            {
                'gain_clock': float(gain_clock),
                'gain_freq': float(gain_freq),
                'iterations': int(iters),
                'mean_rmse_ps': float(np.mean(rmse_vals)),
                'std_rmse_ps': float(np.std(rmse_vals, ddof=1)) if len(rmse_vals) > 1 else 0.0,
                'min_rmse_ps': float(np.min(rmse_vals)),
                'max_rmse_ps': float(np.max(rmse_vals)),
                'mean_improvement_ps': float(np.mean(imp_vals)),
            }
        )

    best_combo_avg = min(combo_stats, key=lambda item: item['mean_rmse_ps']) if combo_stats else None

    summary = {
        'n_runs': len(results),
        'rmse_ps': {
            'min': float(np.min(arr)),
            'max': float(np.max(arr)),
            'mean': float(np.mean(arr)),
            'std': float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
        },
        'improvement_ps': {
            'min': float(np.min(improvements)),
            'max': float(np.max(improvements)),
            'mean': float(np.mean(improvements)),
        },
        'best': best.to_dict(),
        'combo_stats': combo_stats,
        'best_combo_avg': best_combo_avg,
    }
    return summary


def build_phase2_config(args: argparse.Namespace) -> Phase2Config:
    comm_range = args.comm_range_m
    if args.density is not None:
        if comm_range is not None:
            raise ValueError('Specify at most one of --density or --comm-range-m')
        comm_range = _radius_from_density(args.density, args.area_m)
    if comm_range is None:
        comm_range = Phase2Config.comm_range_m

    return Phase2Config(
        n_nodes=args.nodes,
        area_size_m=args.area_m,
        comm_range_m=comm_range,
        snr_db=args.snr_db,
        weighting=args.weighting,
        target_rmse_ps=args.target_rmse_ps,
        target_streak=args.target_streak,
        max_iterations=args.max_iterations,
        timestep_s=args.timestep_ms * 1e-3,
        epsilon_override=args.epsilon,
        spectral_margin=args.spectral_margin,
        rng_seed=args.seeds[0],  # placeholder; overwritten per run
        local_kf_enabled=True,
        local_kf_sigma_T_ps=args.local_kf_sigma_T_ps,
        local_kf_sigma_f_hz=args.local_kf_sigma_f_hz,
        local_kf_init_var_T_ps=args.local_kf_init_var_T_ps,
        local_kf_init_var_f_hz=args.local_kf_init_var_f_hz,
        local_kf_max_abs_ps=args.local_kf_max_abs_ps,
        local_kf_max_abs_freq_hz=args.local_kf_max_abs_f_hz,
        local_kf_clock_gain=args.gains[0],
        local_kf_freq_gain=args.freq_gains[0],
        local_kf_iterations=args.iters[0],
        save_results=args.save_results,
        plot_results=args.plot_results,
        retune_offsets_hz=tuple(args.retune_offsets_hz),
    )


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Sweep Phase 2 KF gains over multiple seeds')
    parser.add_argument('--nodes', type=int, default=64)
    parser.add_argument('--area-m', type=float, default=350.0)
    parser.add_argument('--density', type=float, default=None)
    parser.add_argument('--comm-range-m', type=float, default=None)
    parser.add_argument('--weighting', type=str, default='metropolis_var')
    parser.add_argument('--snr-db', type=float, default=20.0)
    parser.add_argument('--target-rmse-ps', type=float, default=90.0)
    parser.add_argument('--target-streak', type=int, default=3)
    parser.add_argument('--max-iterations', type=int, default=2000)
    parser.add_argument('--timestep-ms', type=float, default=0.5)
    parser.add_argument('--epsilon', type=float, default=0.02)
    parser.add_argument('--spectral-margin', type=float, default=0.8)
    parser.add_argument('--retune-offsets-hz', type=str, default='1e6')
    parser.add_argument('--local-kf-sigma-T-ps', type=float, default=5.0)
    parser.add_argument('--local-kf-sigma-f-hz', type=float, default=2.0)
    parser.add_argument('--local-kf-init-var-T-ps', type=float, default=5.0e3)
    parser.add_argument('--local-kf-init-var-f-hz', type=float, default=150.0)
    parser.add_argument('--local-kf-max-abs-ps', type=float, default=30.0)
    parser.add_argument('--local-kf-max-abs-f-hz', type=float, default=2000.0)
    parser.add_argument('--local-kf-freq-gain', type=float, default=0.05)
    parser.add_argument('--local-kf-iterations', type=int, default=1)
    parser.add_argument('--freq-gains', type=str, default=None, help='Comma-separated frequency gain values (defaults to --local-kf-freq-gain)')
    parser.add_argument('--iters', type=str, default=None, help='Comma-separated iteration counts (defaults to --local-kf-iterations)')
    parser.add_argument('--gains', type=str, required=True, help='Comma-separated clock gain values to evaluate')
    parser.add_argument('--seeds', type=str, required=True, help='Comma-separated RNG seeds to evaluate')
    parser.add_argument('--output-dir', type=str, default='results/kf_sweeps')
    parser.add_argument('--run-id', type=str, default=None)
    parser.add_argument('--save-results', action='store_true', help='Persist per-run Phase 2 outputs (defaults off for speed)')
    parser.add_argument('--plot-results', action='store_true', help='Draw plots for each Phase 2 run')
    parser.add_argument('--no-plot-results', dest='plot_results', action='store_false')
    parser.set_defaults(save_results=False, plot_results=False)
    parser.add_argument('--write-json', action='store_true', help='Persist sweep summary/results to JSON under output dir')
    parser.add_argument('--baseline', action='store_true', help='Run a no-KF baseline for comparison using first seed')
    parser.add_argument('--baseline-weighting', type=str, default='metropolis')
    args = parser.parse_args(argv)
    args.gains = _parse_float_list(args.gains)
    if not args.gains:
        raise ValueError('At least one gain must be specified')
    args.seeds = _parse_int_list(args.seeds)
    if not args.seeds:
        raise ValueError('At least one seed must be specified')
    args.retune_offsets_hz = _parse_float_list(args.retune_offsets_hz)
    if not args.retune_offsets_hz:
        args.retune_offsets_hz = [1e6]
    if args.freq_gains is not None:
        args.freq_gains = _parse_float_list(args.freq_gains)
    else:
        args.freq_gains = [args.local_kf_freq_gain]
    if not args.freq_gains:
        args.freq_gains = [args.local_kf_freq_gain]
    if args.iters is not None:
        args.iters = _parse_int_list(args.iters)
    else:
        args.iters = [args.local_kf_iterations]
    if not args.iters:
        args.iters = [args.local_kf_iterations]
    return args


def maybe_run_baseline(args: argparse.Namespace, base_cfg: Phase2Config) -> Optional[Dict[str, object]]:
    if not args.baseline:
        return None
    cfg = Phase2Config(
        **{
            **asdict(base_cfg),
            'rng_seed': args.seeds[0],
            'local_kf_enabled': False,
            'weighting': args.baseline_weighting,
        }
    )
    cfg.save_results = base_cfg.save_results
    cfg.plot_results = base_cfg.plot_results
    sim = Phase2Simulation(cfg)
    telemetry = sim.run()
    consensus = telemetry['consensus']
    return {
        'seed': args.seeds[0],
        'weighting': cfg.weighting,
        'final_rmse_ps': float(consensus['timing_rms_ps'][-1]),
        'converged': bool(consensus['converged']),
    }


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    base_cfg = build_phase2_config(args)

    results = run_sweep(args.gains, args.freq_gains, args.iters, args.seeds, base_cfg)
    summary = summarise(results)
    baseline = maybe_run_baseline(args, base_cfg)
    if baseline is not None:
        summary['baseline'] = baseline

    print('=== Phase 2 KF Sweep ===')
    print(f"Config: nodes={base_cfg.n_nodes} area={base_cfg.area_size_m}m density={args.density}" )
    print(f"Clock gains: {sorted(args.gains)}")
    print(f"Freq gains: {sorted(args.freq_gains)} | Iterations: {sorted(args.iters)}")
    print(f"Seeds: {sorted(args.seeds)}")
    print('---')
    print(f"RMSE ps -> min {summary['rmse_ps']['min']:.3f} | mean {summary['rmse_ps']['mean']:.3f} | std {summary['rmse_ps']['std']:.3f}")
    if baseline is not None:
        print(f"Baseline (KF off) rmse: {baseline['final_rmse_ps']:.3f} ps")
    best = summary['best']
    print(
        "Best single run: clock {gc:.3f} freq {gf:.3f} iters {it:d} (seed {seed}) -> {rmse:.3f} ps".format(
            gc=best['gain_clock'],
            gf=best['gain_freq'],
            it=best['iterations'],
            seed=best['seed'],
            rmse=best['final_rmse_ps'],
        )
    )
    best_avg = summary['best_combo_avg']
    if best_avg is not None:
        print(
            "Best mean combo: clock {gc:.3f} freq {gf:.3f} iters {it:d} -> {mean:.3f} ps (std {std:.3f})".format(
                gc=best_avg['gain_clock'],
                gf=best_avg['gain_freq'],
                it=best_avg['iterations'],
                mean=best_avg['mean_rmse_ps'],
                std=best_avg['std_rmse_ps'],
            )
        )

    if args.write_json:
        out_root = Path(ensure_directory(args.output_dir))
        if args.run_id:
            out_root = Path(ensure_directory(out_root / args.run_id))
        payload = {
            'config': {
                'base': asdict(base_cfg),
                'gains': list(args.gains),
                'freq_gains': list(args.freq_gains),
                'iterations': list(args.iters),
                'seeds': list(args.seeds),
            },
            'summary': summary,
            'results': [res.to_dict() for res in results],
        }
        out_path = out_root / 'kf_sweep_summary.json'
        out_path.write_text(json.dumps(payload, indent=2), encoding='utf-8')
        print(f"Wrote sweep summary to {out_path}")


if __name__ == '__main__':
    main()
