#!/usr/bin/env python3
"""Comprehensive beacon performance sweep across TDL profiles and SNR ranges.

This script extends the spectrum beacon exploration to systematically test
performance across different channel conditions. It runs both handshake
diagnostics and beacon simulations for each profile to build a comprehensive
performance map.

Usage:

    python scripts/beacon_profile_sweep.py \
        --profiles IDEAL URBAN_CANYON INDOOR_OFFICE \
        --snr-ranges "15,25" "25,35" \
        --num-trials 512 \
        --num-receivers 3 \
        --base-output results/project_aperture_formant/profile_sweep

This will generate a structured output directory with results for each profile
and condition combination.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional


def run_command(cmd: List[str], capture_output: bool = True) -> subprocess.CompletedProcess:
    """Run a subprocess command with error handling."""
    print(f"Running: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, capture_output=capture_output, text=True, check=True)
        return result
    except subprocess.CalledProcessError as e:
        print(f"Command failed with exit code {e.returncode}")
        if e.stdout:
            print(f"stdout: {e.stdout}")
        if e.stderr:
            print(f"stderr: {e.stderr}")
        raise


def run_handshake_diagnostics(profile: str, output_path: Path, num_trials: int = 100) -> Dict[str, object]:
    """Run handshake diagnostics for clutter analysis."""
    cmd = [
        sys.executable, "scripts/run_handshake_diag.py",
        "--channel-profile", profile,
        "--num-trials", str(num_trials),
        "--output-json", str(output_path)
    ]
    
    run_command(cmd)
    
    # Load and return results
    with output_path.open() as f:
        return json.load(f)


def run_beacon_simulation(
    profile: str,
    output_path: Path,
    snr_min: float,
    snr_max: float,
    num_trials: int = 512,
    receiver_id: Optional[int] = None
) -> Dict[str, object]:
    """Run spectrum beacon simulation."""
    
    # Adjust output path for multiple receivers
    if receiver_id is not None:
        output_json = output_path.with_suffix(f'.rx{receiver_id}.json')
        output_trials = output_path.with_suffix(f'.rx{receiver_id}.trials.jsonl')
    else:
        output_json = output_path
        output_trials = output_path.with_suffix('.trials.jsonl')
    
    # Configure multipath parameters based on profile
    if profile == "IDEAL":
        max_extra_paths, max_delay_ns = 0, 0
    elif profile == "URBAN_CANYON":
        max_extra_paths, max_delay_ns = 4, 120
    elif profile == "INDOOR_OFFICE":
        max_extra_paths, max_delay_ns = 6, 200
    else:
        # Default to moderate multipath
        max_extra_paths, max_delay_ns = 2, 50
    
    cmd = [
        sys.executable, "scripts/run_spectrum_beacon_sim.py",
        "--profiles", "A", "E", "I", "O", "U",
        "--num-trials", str(num_trials),
        "--snr-db", str(snr_min), str(snr_max),
        "--max-extra-paths", str(max_extra_paths),
        "--max-delay-ns", str(max_delay_ns),
        "--empty-prob", "0.3",
        "--phase-jitter", "0.25",
        "--dump-trials", str(output_trials),
        "--output", str(output_json)
    ]
    
    # Add receiver-specific seed for diversity
    if receiver_id is not None:
        cmd.extend(["--rng-seed", str(2025 + receiver_id)])
    
    run_command(cmd)
    
    # Load and return results
    with output_json.open() as f:
        return json.load(f)


def aggregate_multi_receiver_votes(
    beacon_results: List[Path], 
    output_path: Path,
    strategy: str = "weighted"
) -> Dict[str, object]:
    """Aggregate votes from multiple receivers using enhanced voting."""
    
    trials_files = [p.with_suffix('.trials.jsonl') for p in beacon_results]
    existing_files = [f for f in trials_files if f.exists()]
    
    if not existing_files:
        return {}
    
    cmd = [
        sys.executable, "scripts/enhanced_beacon_votes.py",
        "--votes"] + [str(f) for f in existing_files] + [
        "--vote-strategy", strategy,
        "--vote-threshold", "2",
        "--missing-f0-tolerance-hz", "100",
        "--dominant-tolerance-hz", "1000", 
        "--score-variance-threshold", "0.5",
        "--output", str(output_path)
    ]
    
    run_command(cmd)
    
    # Load and return results
    with output_path.open() as f:
        return json.load(f)


def run_clutter_beacon_analysis(
    clutter_path: Path,
    beacon_trials_path: Path,
    output_path: Path
) -> Dict[str, object]:
    """Run integrated clutter-beacon analysis."""
    
    if not clutter_path.exists() or not beacon_trials_path.exists():
        return {}
    
    cmd = [
        sys.executable, "scripts/beacon_clutter_analysis.py",
        "--clutter-metrics", str(clutter_path),
        "--beacon-trials", str(beacon_trials_path),
        "--output", str(output_path)
    ]
    
    run_command(cmd)
    
    # Load and return results
    with output_path.open() as f:
        return json.load(f)


def sweep_profile_conditions(
    profile: str,
    snr_ranges: List[tuple],
    num_trials: int,
    num_receivers: int,
    output_dir: Path
) -> Dict[str, object]:
    """Run comprehensive sweep for a single profile across SNR conditions."""
    
    profile_dir = output_dir / profile
    profile_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%dT%H%M%SZ")
    
    results = {
        'profile': profile,
        'timestamp': timestamp,
        'num_receivers': num_receivers,
        'conditions': {}
    }
    
    # Run handshake diagnostics once per profile (for clutter analysis)
    print(f"\n=== Running handshake diagnostics for {profile} ===")
    clutter_path = profile_dir / f"{timestamp}_clutter_diag.json"
    clutter_results = run_handshake_diagnostics(profile, clutter_path, num_trials=100)
    results['clutter_metrics'] = clutter_results
    
    # Test each SNR condition
    for snr_min, snr_max in snr_ranges:
        condition_name = f"snr_{snr_min}_{snr_max}"
        print(f"\n=== Testing {profile} with SNR {snr_min}-{snr_max} dB ===")
        
        condition_results = {
            'snr_range': [snr_min, snr_max],
            'receivers': {},
            'aggregated': {}
        }
        
        # Run beacon simulation for each receiver
        beacon_paths = []
        for rx_id in range(num_receivers):
            print(f"  Running receiver {rx_id + 1}/{num_receivers}")
            beacon_path = profile_dir / f"{timestamp}_beacon_{condition_name}_rx{rx_id}.json"
            beacon_results = run_beacon_simulation(
                profile, beacon_path, snr_min, snr_max, 
                num_trials, receiver_id=rx_id
            )
            condition_results['receivers'][f'rx{rx_id}'] = beacon_results
            beacon_paths.append(beacon_path)
        
        # Aggregate votes using enhanced voting
        print(f"  Aggregating votes with enhanced strategy")
        vote_path = profile_dir / f"{timestamp}_votes_{condition_name}_enhanced.json"
        vote_results = aggregate_multi_receiver_votes(beacon_paths, vote_path, "weighted")
        condition_results['aggregated']['enhanced_voting'] = vote_results
        
        # Run basic voting for comparison
        basic_vote_path = profile_dir / f"{timestamp}_votes_{condition_name}_basic.json"
        basic_results = aggregate_multi_receiver_votes(beacon_paths, basic_vote_path, "basic")
        condition_results['aggregated']['basic_voting'] = basic_results
        
        # Correlate with clutter analysis (using first receiver's trial data)
        if beacon_paths:
            first_trials_path = beacon_paths[0].with_suffix('.trials.jsonl')
            correlation_path = profile_dir / f"{timestamp}_clutter_beacon_{condition_name}.json"
            correlation_results = run_clutter_beacon_analysis(
                clutter_path, first_trials_path, correlation_path
            )
            condition_results['clutter_correlation'] = correlation_results
        
        results['conditions'][condition_name] = condition_results
    
    # Save comprehensive results
    summary_path = profile_dir / f"{timestamp}_profile_summary.json"
    with summary_path.open('w') as f:
        json.dump(results, f, indent=2)
    
    return results


def generate_sweep_summary(all_results: List[Dict[str, object]], output_path: Path) -> None:
    """Generate a cross-profile comparison summary."""
    
    summary = {
        'timestamp': datetime.now().strftime("%Y%m%dT%H%M%SZ"),
        'profiles_tested': len(all_results),
        'profile_comparison': {}
    }
    
    for profile_result in all_results:
        profile = profile_result['profile']
        profile_summary = {
            'clutter_severity': 'unknown',
            'conditions': {}
        }
        
        # Extract clutter classification
        clutter_metrics = profile_result.get('clutter_metrics', {}).get('first_path_metrics', {})
        fp_neg_rate = clutter_metrics.get('first_path_negative_rate', 0)
        if fp_neg_rate > 0.25:
            profile_summary['clutter_severity'] = 'high'
        elif fp_neg_rate > 0.1:
            profile_summary['clutter_severity'] = 'medium'  
        else:
            profile_summary['clutter_severity'] = 'low'
        
        # Summarize performance across conditions
        for cond_name, cond_data in profile_result.get('conditions', {}).items():
            enhanced_voting = cond_data.get('aggregated', {}).get('enhanced_voting', {})
            basic_voting = cond_data.get('aggregated', {}).get('basic_voting', {})
            
            profile_summary['conditions'][cond_name] = {
                'enhanced': {
                    'detection_rate': enhanced_voting.get('detected_rate', 0),
                    'accuracy': enhanced_voting.get('label_accuracy', 0),
                    'false_positive_rate': enhanced_voting.get('false_positive_rate', 0),
                    'consistency_score': enhanced_voting.get('avg_consistency_score', 0),
                },
                'basic': {
                    'detection_rate': basic_voting.get('detected_rate', 0),
                    'accuracy': basic_voting.get('label_accuracy', 0),
                    'false_positive_rate': basic_voting.get('false_positive_rate', 0),
                }
            }
        
        summary['profile_comparison'][profile] = profile_summary
    
    # Save summary
    with output_path.open('w') as f:
        json.dump(summary, f, indent=2)
    
    # Print key findings
    print(f"\n{'='*60}")
    print("BEACON PROFILE SWEEP SUMMARY")
    print(f"{'='*60}")
    
    for profile, data in summary['profile_comparison'].items():
        print(f"\n{profile} (Clutter: {data['clutter_severity']})")
        for cond_name, metrics in data['conditions'].items():
            enhanced = metrics['enhanced']
            basic = metrics['basic']
            print(f"  {cond_name}:")
            print(f"    Enhanced: Det={enhanced['detection_rate']:.3f}, Acc={enhanced['accuracy']:.3f}, Cons={enhanced['consistency_score']:.3f}")
            print(f"    Basic:    Det={basic['detection_rate']:.3f}, Acc={basic['accuracy']:.3f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profiles', nargs='+', default=['IDEAL', 'URBAN_CANYON', 'INDOOR_OFFICE'],
                       help='TDL profiles to test')
    parser.add_argument('--snr-ranges', nargs='+', default=['15,25', '25,35'],
                       help='SNR ranges as "min,max" pairs')
    parser.add_argument('--num-trials', type=int, default=512,
                       help='Number of beacon trials per condition')
    parser.add_argument('--num-receivers', type=int, default=3,
                       help='Number of receiver instances for voting')
    parser.add_argument('--base-output', type=Path, default=Path('results/project_aperture_formant/profile_sweep'),
                       help='Base output directory')
    
    args = parser.parse_args()
    
    # Parse SNR ranges
    snr_ranges = []
    for snr_range_str in args.snr_ranges:
        snr_min, snr_max = map(float, snr_range_str.split(','))
        snr_ranges.append((snr_min, snr_max))
    
    print(f"Starting beacon profile sweep:")
    print(f"  Profiles: {args.profiles}")
    print(f"  SNR ranges: {snr_ranges}")
    print(f"  Trials per condition: {args.num_trials}")
    print(f"  Receivers per condition: {args.num_receivers}")
    print(f"  Output directory: {args.base_output}")
    
    # Create base output directory
    args.base_output.mkdir(parents=True, exist_ok=True)
    
    # Run sweep for each profile
    all_results = []
    for profile in args.profiles:
        print(f"\n{'#'*60}")
        print(f"PROFILE: {profile}")
        print(f"{'#'*60}")
        
        try:
            profile_results = sweep_profile_conditions(
                profile, snr_ranges, args.num_trials, 
                args.num_receivers, args.base_output
            )
            all_results.append(profile_results)
        except Exception as e:
            print(f"Error processing profile {profile}: {e}")
            continue
    
    # Generate cross-profile summary
    if all_results:
        timestamp = datetime.now().strftime("%Y%m%dT%H%M%SZ")
        summary_path = args.base_output / f"{timestamp}_sweep_summary.json"
        generate_sweep_summary(all_results, summary_path)
        print(f"\nSweep summary saved to {summary_path}")
    
    print(f"\nProfile sweep complete! Results in {args.base_output}")


if __name__ == '__main__':
    main()