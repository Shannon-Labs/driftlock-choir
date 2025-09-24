"""
Master script for rigorous Monte Carlo validation of the Driftlock Choir simulation.

This script runs the entire, hardened Phase 2 simulation across multiple
random seeds to eliminate the "lucky seed" problem and produce statistically
sound performance figures.
"""

import argparse
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd

# Add src/ and sim/ to import path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from sim.phase2 import Phase2Config, Phase2Simulation, build_phase2_cli, _radius_from_density, _resolve_local_kf_flag, _parse_float_sequence
from utils.io import echo_config


def run_monte_carlo(args: argparse.Namespace) -> None:
    """
    Orchestrates the Monte Carlo simulation runs.
    """
    base_seed = args.rng_seed
    num_runs = args.num_seeds
    results_dir = os.path.join(args.results_dir, f"mc_{datetime.now().strftime('%Y%m%d_%H%M%S')}")

    print(f"Starting Monte Carlo simulation with {num_runs} seeds.")
    print(f"Base seed: {base_seed}")
    print(f"Results will be saved in: {results_dir}")

    all_telemetry = []

    for i in range(num_runs):
        current_seed = base_seed + i
        print(f"\n--- Running Seed {i+1}/{num_runs} (RNG Seed: {current_seed}) ---")

        # Configure and run a single Phase2 simulation
        cfg = _build_config_from_args(args, current_seed, results_dir)

        if i == 0:
            print("\nUsing the following configuration for all runs:")
            print(echo_config(cfg, label='Phase2Config'))

        try:
            simulation = Phase2Simulation(cfg)
            telemetry = simulation.run()
            all_telemetry.append(telemetry)
            print(f"Seed {current_seed} complete. Final RMSE: {telemetry['consensus']['timing_rms_ps'][-1]:.2f} ps")
        except Exception as e:
            print(f"!!! ERROR running seed {current_seed}: {e}")
            print("Skipping this seed and continuing...")
            continue

    print("\n--- Monte Carlo Simulation Complete ---")
    summarize_results(results_dir)


def _build_config_from_args(args: argparse.Namespace, seed: int, results_dir: str) -> Phase2Config:
    """Helper to construct a Phase2Config from CLI arguments."""
    retune_offsets = _parse_float_sequence(args.retune_offsets_hz)
    if not retune_offsets:
        retune_offsets = (1e6,)

    comm_range = args.comm_range_m
    if args.density is not None:
        if comm_range is not None:
            raise ValueError('Specify at most one of --density or --comm-range-m')
        comm_range = _radius_from_density(args.density, args.area_m)
    if comm_range is None:
        # Fallback to a reasonable default if neither is provided
        comm_range = 180.0

    local_kf_enabled = _resolve_local_kf_flag(args.local_kf, True)

    # Build the config by passing all relevant args from the namespace
    return Phase2Config(
        n_nodes=args.nodes,
        area_size_m=args.area_m,
        comm_range_m=comm_range,
        snr_db=args.snr_db,
        results_dir=results_dir,
        save_results=True,
        plot_results=args.make_plots,
        spectral_margin=args.spectral_margin,
        epsilon_override=args.epsilon,
        weighting=args.weighting,
        target_rmse_ps=args.target_rmse_ps,
        target_streak=args.target_streak,
        max_iterations=args.max_iterations,
        timestep_s=args.timestep_ms * 1e-3,
        rng_seed=seed,
        retune_offsets_hz=retune_offsets,
        coarse_enabled=args.coarse_enabled,
        coarse_bandwidth_hz=args.coarse_bw_hz,
        coarse_duration_s=args.coarse_duration_us * 1e-6,
        channel_profile=args.channel_profile,
        num_timesteps=args.num_timesteps,
        consensus_mode=args.consensus_mode,
        consensus_iterations=args.consensus_iterations,
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
        baseline_mode=args.baseline_mode,
    )


def summarize_results(results_dir: str) -> None:
    """
    Reads the output CSV from all runs and prints a statistical summary.
    """
    csv_path = os.path.join(results_dir, 'phase2_runs.csv')
    if not os.path.exists(csv_path):
        print(f"\nERROR: Results file not found at {csv_path}. Cannot generate summary.")
        return

    df = pd.read_csv(csv_path)

    print("\n--- Statistical Summary ---")
    print(f"Total successful runs: {len(df)}")

    # Key performance indicators to summarize
    metrics_to_summarize = {
        'Final Timing RMSE (ps)': 'kf_filtered_clock_rms_ps',
        'Measurement RMSE (ps)': 'measurement_rmse_tau_ps',
        'Convergence Iterations (avg)': 'measured_iterations',
        'CRLB Ratio (tau)': 'crlb_ratio_tau',
    }

    summary_data = []
    for display_name, col_name in metrics_to_summarize.items():
        if col_name in df.columns and pd.api.types.is_numeric_dtype(df[col_name]):
            series = df[col_name].dropna()
            if not series.empty:
                summary_data.append({
                    "Metric": display_name,
                    "Mean": series.mean(),
                    "Median": series.median(),
                    "Std Dev": series.std(),
                    "Min": series.min(),
                    "Max": series.max(),
                })

    if not summary_data:
        print("\nNo numeric data found to summarize. Check the output CSV file.")
        return

    summary_df = pd.DataFrame(summary_data)
    print("\n" + summary_df.to_string(index=False, float_format="%.2f"))


def main():
    parser = argparse.ArgumentParser(description="Monte Carlo validation script for Driftlock Choir.")

    # --- Arguments for the Monte Carlo runner itself ---
    parser.add_argument('--num-seeds', type=int, default=100, help='Number of random seeds to run.')

    # --- Arguments to configure the underlying Phase2Simulation ---
    # We reuse the CLI builder from phase2.py to avoid duplicating arguments
    # We'll set a few defaults differently for MC runs.
    build_phase2_cli(parser)
    parser.set_defaults(
        save_results=True,
        make_plots=False, # Disable plotting for individual runs in a batch
        results_dir="results/monte_carlo"
    )

    args = parser.parse_args()
    run_monte_carlo(args)


if __name__ == "__main__":
    main()
