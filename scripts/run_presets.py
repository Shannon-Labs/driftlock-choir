#!/usr/bin/env python3
"""Convenience wrapper for common DriftLock simulation presets."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'sim'))

from phase1 import Phase1Config, Phase1Simulator
from phase2 import Phase2Config, Phase2Simulation
from mac.scheduler import MacSlots


def run_phase1_alias(args: argparse.Namespace) -> None:
    results_dir = Path(args.output or 'results/presets/phase1_alias')
    mac = MacSlots(preamble_len=1024, narrowband_len=512, guard_us=10.0)
    cfg = Phase1Config(
        snr_values_db=[20.0, 10.0, 0.0],
        n_monte_carlo=args.num_trials,
        retune_offsets_hz=(1e6,),
        coarse_bandwidth_hz=40e6,
        delta_t_us=(0.0, 1.5),
        calib_mode=args.calib_mode,
        loopback_cal_noise_ps=5.0,
        d_tx_ns={0: 24.0, 1: 31.0},
        d_rx_ns={0: 14.0, 1: 9.0},
        rng_seed=args.rng_seed,
        save_results=True,
        plot_results=args.make_plots,
        results_dir=str(results_dir),
        mac_slots=mac,
    )
    simulator = Phase1Simulator(cfg)
    summary = simulator.run_alias_failure_map(
        retune_offsets_hz=[1e6, 5e6],
        coarse_bw_hz=[20e6, 40e6],
        snr_db=[20.0, 10.0, 0.0],
        num_trials=args.num_trials,
        rng_seed=args.rng_seed,
        make_plots=args.make_plots,
    )
    print(f"Alias-map manifest: {summary['manifest_path']}")
    print(f"Bias diagnostics: {summary['manifest']['bias_diagnostics']}")


def run_phase2_consensus(args: argparse.Namespace) -> None:
    results_dir = Path(args.output or 'results/presets/phase2_consensus')
    cfg = Phase2Config(
        n_nodes=args.nodes,
        area_size_m=args.area_size_m,
        comm_range_m=args.comm_range_m,
        weighting=args.weighting,
        target_rmse_ps=args.target_rmse_ps,
        target_streak=args.target_streak,
        local_kf_enabled=(args.local_kf == 'on'),
        rng_seed=args.rng_seed,
        retune_offsets_hz=(1e6,),
        save_results=True,
        plot_results=args.make_plots,
        results_dir=str(results_dir),
    )
    sim = Phase2Simulation(cfg)
    telemetry = sim.run()
    final_rmse = telemetry['consensus']['timing_rms_ps'][-1]
    print(f"Final timing RMSE: {final_rmse:.2f} ps")
    print(f"Local KF metrics: {telemetry['local_kf']}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Run DriftLock canned simulations')
    sub = parser.add_subparsers(dest='preset', required=True)

    phase1 = sub.add_parser('phase1-alias', help='Run the alias-map sweep preset')
    phase1.add_argument('--num-trials', type=int, default=180)
    phase1.add_argument('--calib-mode', choices=['off', 'loopback', 'perfect'], default='loopback')
    phase1.add_argument('--rng-seed', type=int, default=2025)
    phase1.add_argument('--make-plots', action='store_true')
    phase1.add_argument('--output', type=str)
    phase1.set_defaults(func=run_phase1_alias)

    phase2 = sub.add_parser('phase2-consensus', help='Run the consensus preset')
    phase2.add_argument('--nodes', type=int, default=50)
    phase2.add_argument('--area-size-m', type=float, default=350.0)
    phase2.add_argument('--comm-range-m', type=float, default=180.0)
    phase2.add_argument('--weighting', type=str, default='metropolis_var')
    phase2.add_argument('--target-rmse-ps', type=float, default=120.0)
    phase2.add_argument('--target-streak', type=int, default=3)
    phase2.add_argument('--local-kf', choices=['on', 'off'], default='on')
    phase2.add_argument('--rng-seed', type=int, default=4242)
    phase2.add_argument('--make-plots', action='store_true')
    phase2.add_argument('--output', type=str)
    phase2.set_defaults(func=run_phase2_consensus)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
