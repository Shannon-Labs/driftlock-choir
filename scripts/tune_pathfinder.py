#!/usr/bin/env python3
# Tune pathfinder parameters to minimize bias in challenging profiles.

import json
import subprocess
from pathlib import Path
from itertools import product
import time

def run_handshake_diag(params, profile="INDOOR_OFFICE", num_trials=16):
    """Run the handshake diagnostic and return absolute tau bias and runtime."""
    output_dir = Path("/Volumes/VIXinSSD/SHANNON-LABS-INC/apps/driftlock-choir/results/tuning_temp")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Remove any stale diagnostics so we only read the fresh run.
    for old_file in output_dir.glob(f"tdl_diag_{profile.lower()}_*.json"):
        old_file.unlink(missing_ok=True)

    cmd = [
        "python", "scripts/run_handshake_diag.py",
        "--channel-profile", profile,
        "--num-trials", str(num_trials),
        "--pathfinder-relative-threshold-db", str(params['relative_threshold_db']),
        "--pathfinder-noise-guard-multiplier", str(params['noise_guard_multiplier']),
        "--pathfinder-guard-interval-ns", str(params['aperture_duration_ns']),
        "--debug",
        "--output-dir", "results/tuning_temp"
    ]
    
    start_time = time.time()

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, 
                              cwd="/Volumes/VIXinSSD/SHANNON-LABS-INC/apps/driftlock-choir",
                              timeout=180)  # 3 minute timeout per run
        elapsed = time.time() - start_time

        if result.returncode == 0:
            # Find the output JSON file
            json_files = sorted(
                output_dir.glob(f"tdl_diag_{profile.lower()}_*.json"),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            if json_files:
                with open(json_files[0], 'r') as f:
                    data = json.load(f)
                
                # Extract the tau bias
                tau_bias = data.get('two_way_metrics', {}).get('tau_bias_ns', {}).get('mean', float('inf'))
                return abs(tau_bias), elapsed  # Return absolute bias and time taken
    except subprocess.TimeoutExpired:
        elapsed = time.time() - start_time
        print("  -> Timed out")
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"  -> Error: {e}")

    return float('inf'), elapsed  # Return infinity if there was an error


def main():
    print("Starting pathfinder parameter tuning for INDOOR_OFFICE profile...")
    
    # Define parameter ranges (reduced to make runtime more manageable)
    relative_threshold_db_range = [-20.0, -15.0, -12.0]  # Fewer values to test
    noise_guard_multiplier_range = [4.0, 5.0, 6.0, 7.0]  # Fewer values to test 
    aperture_duration_ns_range = [100.0, 150.0, 200.0, 250.0]  # Fewer values to test
    
    print(f"Testing {len(relative_threshold_db_range)} x {len(noise_guard_multiplier_range)} x {len(aperture_duration_ns_range)} = {len(relative_threshold_db_range) * len(noise_guard_multiplier_range) * len(aperture_duration_ns_range)} parameter combinations")
    
    best_results = []
    
    # Create results directory
    Path("results/tuning_temp").mkdir(parents=True, exist_ok=True)
    
    # Iterate through all parameter combinations
    total_combinations = len(relative_threshold_db_range) * len(noise_guard_multiplier_range) * len(aperture_duration_ns_range)
    for i, (rel_thresh, noise_mult, aperture_dur) in enumerate(product(relative_threshold_db_range, noise_guard_multiplier_range, aperture_duration_ns_range)):
        params = {
            'relative_threshold_db': rel_thresh,
            'noise_guard_multiplier': noise_mult,
            'aperture_duration_ns': aperture_dur
        }
        
        print(f"Testing combination {i+1}/{total_combinations}: {params}")
        
        # Run the diagnostic
        abs_bias, run_time = run_handshake_diag(params)
        
        if abs_bias != float('inf'):
            best_results.append((abs_bias, params, run_time))
            print(f"  -> Absolute bias: {abs_bias:.4f} ns, Time: {run_time:.2f}s")
        else:
            print(f"  -> Failed to run, Time: {run_time:.2f}s")
    
    # Sort by lowest absolute bias
    best_results.sort(key=lambda x: x[0])
    
    # Print top 5 results
    print("\n" + "="*80)
    print("TOP 5 PARAMETER SETS (lowest absolute bias):")
    print("="*80)
    
    for idx, (bias, params, run_time) in enumerate(best_results[:5], 1):
        print(f"{idx}. Absolute bias: {bias:.4f} ns, Run time: {run_time:.2f}s")
        print(f"   Parameters:")
        print(f"     - relative_threshold_db: {params['relative_threshold_db']}")
        print(f"     - noise_guard_multiplier: {params['noise_guard_multiplier']}")
        print(f"     - aperture_duration_ns: {params['aperture_duration_ns']}")
        print()
    
    # Also save the complete results
    results_file = Path("results/tuning_results.json")
    results_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(results_file, 'w') as f:
        json.dump({
            "profile": "INDOOR_OFFICE",
            "num_trials": 16,
            "top_5_results": [
                {"rank": i+1, "absolute_bias_ns": bias, "run_time_s": run_time, "parameters": params}
                for i, (bias, params, run_time) in enumerate(best_results[:5])
            ],
            "all_results": [
                {"absolute_bias_ns": bias, "run_time_s": run_time, "parameters": params}
                for bias, params, run_time in best_results
            ]
        }, f, indent=2)
    
    print(f"Complete results saved to: {results_file}")
    
    # Test the best parameters on URBAN_CANYON as well
    if best_results:
        print(f"\nBest parameters found: {best_results[0][1]}")
        print(f"Best bias: {best_results[0][0]:.4f} ns")
        
        print("\nTesting best parameters on URBAN_CANYON profile...")
        urban_bias, urban_time = run_handshake_diag(best_results[0][1], profile="URBAN_CANYON", num_trials=16)
        print(f"URBAN_CANYON bias with best INDOOR_OFFICE parameters: {urban_bias:.4f} ns, Time: {urban_time:.2f}s")


if __name__ == "__main__":
    main()
