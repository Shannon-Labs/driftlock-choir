#!/usr/bin/env python3
"""Demonstrate enhanced beacon consensus and clutter integration features.

This script showcases the new capabilities developed for spectrum beacon
consensus:

1. Enhanced voting with weighted decisions and consistency checks
2. Clutter correlation analysis  
3. Multi-receiver performance comparison

Usage:
    python scripts/beacon_enhancements_demo.py \
        --profile URBAN_CANYON \
        --output results/project_aperture_formant/enhancements_demo
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List


def run_command(cmd: List[str]) -> subprocess.CompletedProcess:
    """Run a subprocess command with error handling."""
    print(f"Running: {' '.join(cmd)}")
    return subprocess.run(cmd, capture_output=True, text=True, check=True)


def demo_enhanced_voting(profile: str, output_dir: Path) -> Dict[str, object]:
    """Demonstrate enhanced voting vs basic voting comparison."""
    
    print(f"=== Enhanced Voting Demo for {profile} ===")
    
    # First, generate multiple receiver datasets
    receiver_files = []
    for i in range(3):
        print(f"Generating receiver {i+1} data...")
        receiver_path = output_dir / f"demo_beacon_rx{i}.json"
        
        # Adjust parameters based on profile
        if profile == "IDEAL":
            max_paths, max_delay = 0, 0
        elif profile == "URBAN_CANYON":  
            max_paths, max_delay = 4, 120
        elif profile == "INDOOR_OFFICE":
            max_paths, max_delay = 6, 200
        else:
            max_paths, max_delay = 2, 50
            
        trials_path = receiver_path.with_suffix('.trials.jsonl')
        cmd = [
            sys.executable, "scripts/run_spectrum_beacon_sim.py",
            "--profiles", "A", "E", "I", "O", "U",
            "--num-trials", "200",
            "--snr-db", "20", "35", 
            "--max-extra-paths", str(max_paths),
            "--max-delay-ns", str(max_delay),
            "--empty-prob", "0.3",
            "--rng-seed", str(2025 + i),
            "--dump-trials", str(trials_path),
            "--output", str(receiver_path)
        ]
        
        try:
            run_command(cmd)
            receiver_files.append(receiver_path)
        except subprocess.CalledProcessError as e:
            print(f"Error running receiver {i}: {e}")
            continue
    
    if len(receiver_files) < 2:
        print("Need at least 2 receivers for voting comparison")
        return {}
    
    # Compare voting strategies
    results = {}
    
    # Test basic voting with threshold 1 and 2
    for threshold in [1, 2]:
        print(f"Testing basic voting with threshold {threshold}...")
        basic_path = output_dir / f"demo_basic_vote_thresh{threshold}.json"
        
        trials_files = [f.with_suffix('.trials.jsonl') for f in receiver_files[:2]]
        cmd = [
            sys.executable, "scripts/enhanced_beacon_votes.py",
            "--votes"] + [str(f) for f in trials_files] + [
            "--vote-strategy", "basic",
            "--vote-threshold", str(threshold),
            "--output", str(basic_path)
        ]
        
        try:
            run_command(cmd)
            with basic_path.open() as f:
                results[f'basic_threshold_{threshold}'] = json.load(f)
        except Exception as e:
            print(f"Error in basic voting (threshold {threshold}): {e}")
    
    # Test enhanced voting with different consistency parameters
    enhanced_configs = [
        {"name": "permissive", "f0_tol": 200, "dom_tol": 2000, "score_var": 0.8},
        {"name": "strict", "f0_tol": 50, "dom_tol": 500, "score_var": 0.3},
        {"name": "default", "f0_tol": 100, "dom_tol": 1000, "score_var": 0.5},
    ]
    
    for config in enhanced_configs:
        print(f"Testing enhanced voting ({config['name']} config)...")
        enhanced_path = output_dir / f"demo_enhanced_vote_{config['name']}.json"
        
        trials_files = [f.with_suffix('.trials.jsonl') for f in receiver_files[:2]]
        cmd = [
            sys.executable, "scripts/enhanced_beacon_votes.py",
            "--votes"] + [str(f) for f in trials_files] + [
            "--vote-strategy", "weighted",
            "--vote-threshold", "1",
            "--missing-f0-tolerance-hz", str(config['f0_tol']),
            "--dominant-tolerance-hz", str(config['dom_tol']),
            "--score-variance-threshold", str(config['score_var']),
            "--output", str(enhanced_path)
        ]
        
        try:
            run_command(cmd)
            with enhanced_path.open() as f:
                results[f'enhanced_{config["name"]}'] = json.load(f)
        except Exception as e:
            print(f"Error in enhanced voting ({config['name']}): {e}")
    
    return results


def demo_clutter_correlation(profile: str, output_dir: Path) -> Dict[str, object]:
    """Demonstrate clutter-beacon correlation analysis."""
    
    print(f"=== Clutter Correlation Demo for {profile} ===")
    
    # Generate handshake diagnostics
    print("Running handshake diagnostics...")
    clutter_path = output_dir / f"demo_clutter_{profile.lower()}.json"
    
    cmd = [
        sys.executable, "scripts/run_handshake_diag.py",
        "--channel-profile", profile,
        "--num-trials", "50",
        "--output-json", str(clutter_path)
    ]
    
    try:
        run_command(cmd)
    except subprocess.CalledProcessError as e:
        print(f"Error running handshake diagnostics: {e}")
        return {}
    
    # Generate beacon trials 
    print("Running beacon simulation...")
    beacon_path = output_dir / f"demo_beacon_{profile.lower()}.json"
    
    if profile == "IDEAL":
        max_paths, max_delay = 0, 0
    elif profile == "URBAN_CANYON":  
        max_paths, max_delay = 4, 120
    elif profile == "INDOOR_OFFICE":
        max_paths, max_delay = 6, 200
    else:
        max_paths, max_delay = 2, 50
    
    beacon_trials_path = beacon_path.with_suffix('.trials.jsonl')
    cmd = [
        sys.executable, "scripts/run_spectrum_beacon_sim.py",
        "--profiles", "A", "E", "I", "O", "U", 
        "--num-trials", "200",
        "--snr-db", "15", "35",
        "--max-extra-paths", str(max_paths),
        "--max-delay-ns", str(max_delay),
        "--empty-prob", "0.3",
        "--dump-trials", str(beacon_trials_path),
        "--output", str(beacon_path)
    ]
    
    try:
        run_command(cmd)
    except subprocess.CalledProcessError as e:
        print(f"Error running beacon simulation: {e}")
        return {}
    
    # Run correlation analysis
    print("Analyzing clutter-beacon correlation...")
    correlation_path = output_dir / f"demo_correlation_{profile.lower()}.json"
    beacon_trials_path = beacon_path.with_suffix('.trials.jsonl')
    
    cmd = [
        sys.executable, "scripts/beacon_clutter_analysis.py",
        "--clutter-metrics", str(clutter_path),
        "--beacon-trials", str(beacon_trials_path),
        "--output", str(correlation_path)
    ]
    
    try:
        run_command(cmd)
        with correlation_path.open() as f:
            return json.load(f)
    except subprocess.CalledProcessError as e:
        print(f"Error running correlation analysis: {e}")
        return {}


def generate_demo_report(voting_results: Dict, correlation_results: Dict, output_path: Path):
    """Generate a summary report of the demonstration."""
    
    report = {
        "timestamp": datetime.now().isoformat(),
        "voting_comparison": {},
        "clutter_analysis": correlation_results,
        "summary": {}
    }
    
    # Analyze voting results
    for strategy, data in voting_results.items():
        if not data:
            continue
            
        report["voting_comparison"][strategy] = {
            "detection_rate": data.get("detected_rate", 0),
            "accuracy": data.get("label_accuracy", 0), 
            "false_positive_rate": data.get("false_positive_rate", 0),
            "consistency_score": data.get("avg_consistency_score", 0),
        }
    
    # Generate insights
    insights = []
    
    # Voting strategy comparison
    enhanced_results = {k: v for k, v in report["voting_comparison"].items() if k.startswith("enhanced")}
    basic_results = {k: v for k, v in report["voting_comparison"].items() if k.startswith("basic")}
    
    if enhanced_results and basic_results:
        best_enhanced = max(enhanced_results.items(), key=lambda x: x[1]["accuracy"])
        best_basic = max(basic_results.items(), key=lambda x: x[1]["accuracy"])
        
        insights.append(
            f"Best enhanced strategy ({best_enhanced[0]}): "
            f"accuracy {best_enhanced[1]['accuracy']:.3f}, "
            f"consistency {best_enhanced[1]['consistency_score']:.3f}"
        )
        insights.append(
            f"Best basic strategy ({best_basic[0]}): "
            f"accuracy {best_basic[1]['accuracy']:.3f}"
        )
        
        if best_enhanced[1]["accuracy"] > best_basic[1]["accuracy"]:
            delta = best_enhanced[1]["accuracy"] - best_basic[1]["accuracy"]
            insights.append(f"Enhanced voting improves accuracy by {delta:.3f}")
    
    # Clutter insights
    if correlation_results:
        clutter_ind = correlation_results.get("clutter_indicators", {})
        beacon_perf = correlation_results.get("beacon_performance", {})
        
        insights.append(f"Clutter environment: {clutter_ind.get('clutter_severity', 'unknown')}")
        insights.append(f"Beacon detection rate: {beacon_perf.get('detection_rate', 0):.3f}")
        
        correlation_insights = correlation_results.get("correlation_insights", {})
        for insight in correlation_insights.values():
            insights.append(insight)
    
    report["summary"]["insights"] = insights
    
    # Save report
    with output_path.open('w') as f:
        json.dump(report, f, indent=2)
    
    # Print summary
    print(f"\n{'='*60}")
    print("BEACON ENHANCEMENTS DEMONSTRATION REPORT")
    print(f"{'='*60}")
    
    print("\nVoting Strategy Comparison:")
    for strategy, metrics in report["voting_comparison"].items():
        print(f"  {strategy}: Det={metrics['detection_rate']:.3f}, "
              f"Acc={metrics['accuracy']:.3f}, Cons={metrics.get('consistency_score', 0):.3f}")
    
    print("\nKey Insights:")
    for insight in insights:
        print(f"  • {insight}")
    
    print(f"\nFull report saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', default='URBAN_CANYON', 
                       help='TDL profile to test (IDEAL, URBAN_CANYON, INDOOR_OFFICE)')
    parser.add_argument('--output', type=Path, 
                       default=Path('results/project_aperture_formant/enhancements_demo'),
                       help='Output directory')
    
    args = parser.parse_args()
    
    # Create output directory
    args.output.mkdir(parents=True, exist_ok=True)
    
    print(f"Demonstrating enhanced beacon consensus features")
    print(f"Profile: {args.profile}")
    print(f"Output: {args.output}")
    
    # Run demonstrations
    voting_results = demo_enhanced_voting(args.profile, args.output)
    correlation_results = demo_clutter_correlation(args.profile, args.output)
    
    # Generate report
    report_path = args.output / f"demo_report_{args.profile.lower()}.json"
    generate_demo_report(voting_results, correlation_results, report_path)


if __name__ == '__main__':
    main()