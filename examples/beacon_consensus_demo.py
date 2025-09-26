#!/usr/bin/env python3
"""Example of enhanced beacon consensus usage.

This script demonstrates how to use the new spectrum beacon consensus
enhancements for improved reliability in multipath environments.
"""

import subprocess
import sys
from pathlib import Path


def run_demo():
    """Run a complete beacon consensus demonstration."""
    
    print("Spectrum Beacon Consensus Enhancement Demo")
    print("=" * 50)
    
    # Setup output directory
    output_dir = Path("results/project_aperture_formant/demo_consensus")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Output directory: {output_dir}")
    
    # 1. Generate beacon data from multiple receivers
    print("\n1. Generating multi-receiver beacon data...")
    
    receiver_configs = [
        {"seed": 2025, "snr": "25,35"},
        {"seed": 2026, "snr": "20,30"},  # Different SNR range for diversity
        {"seed": 2027, "snr": "25,35"},
    ]
    
    receiver_files = []
    for i, config in enumerate(receiver_configs):
        output_path = output_dir / f"receiver_{i}.json"
        trials_path = output_dir / f"receiver_{i}.trials.jsonl"
        
        cmd = [
            sys.executable, "scripts/run_spectrum_beacon_sim.py",
            "--profiles", "A", "E", "O", "U",  # Skip "I" due to known issues
            "--num-trials", "100",
            "--snr-db", *config["snr"].split(","),
            "--max-extra-paths", "4", 
            "--max-delay-ns", "120",
            "--empty-prob", "0.25",
            "--rng-seed", str(config["seed"]),
            "--dump-trials", str(trials_path),
            "--output", str(output_path)
        ]
        
        print(f"  Receiver {i+1}: SNR {config['snr']} dB, seed {config['seed']}")
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            receiver_files.append(trials_path)
        except subprocess.CalledProcessError as e:
            print(f"    Error: {e}")
            continue
    
    if len(receiver_files) < 2:
        print("Need at least 2 receivers for voting demo")
        return
    
    print(f"  Generated data from {len(receiver_files)} receivers")
    
    # 2. Compare voting strategies
    print("\n2. Comparing voting strategies...")
    
    voting_configs = [
        {
            "name": "basic_threshold_1",
            "strategy": "basic",
            "threshold": 1,
            "params": {}
        },
        {
            "name": "basic_threshold_2", 
            "strategy": "basic",
            "threshold": 2,
            "params": {}
        },
        {
            "name": "enhanced_default",
            "strategy": "weighted",
            "threshold": 1,
            "params": {
                "--missing-f0-tolerance-hz": "100",
                "--dominant-tolerance-hz": "1000", 
                "--score-variance-threshold": "0.5"
            }
        },
        {
            "name": "enhanced_strict",
            "strategy": "weighted", 
            "threshold": 1,
            "params": {
                "--missing-f0-tolerance-hz": "50",
                "--dominant-tolerance-hz": "500",
                "--score-variance-threshold": "0.3"
            }
        }
    ]
    
    results = {}
    for config in voting_configs:
        output_path = output_dir / f"vote_{config['name']}.json"
        
        cmd = [
            sys.executable, "scripts/enhanced_beacon_votes.py",
            "--votes", *[str(f) for f in receiver_files],
            "--vote-strategy", config["strategy"],
            "--vote-threshold", str(config["threshold"]),
            "--output", str(output_path)
        ]
        
        # Add strategy-specific parameters
        for param, value in config["params"].items():
            cmd.extend([param, value])
        
        print(f"  Testing {config['name']}...")
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            
            # Parse the output for key metrics
            lines = result.stdout.strip().split('\n')
            metrics = {}
            for line in lines:
                if 'Detection rate:' in line:
                    metrics['detection_rate'] = float(line.split(':')[1].strip())
                elif 'Label accuracy:' in line:
                    metrics['accuracy'] = float(line.split(':')[1].strip())
                elif 'Average consistency score:' in line:
                    metrics['consistency'] = float(line.split(':')[1].strip())
            
            results[config['name']] = metrics
            print(f"    Det: {metrics.get('detection_rate', 0):.3f}, "
                  f"Acc: {metrics.get('accuracy', 0):.3f}, "
                  f"Cons: {metrics.get('consistency', 0):.3f}")
            
        except subprocess.CalledProcessError as e:
            print(f"    Error: {e}")
            continue
    
    # 3. Performance analysis
    print("\n3. Analyzing performance...")
    
    if receiver_files:
        analysis_path = output_dir / "performance_analysis.json"
        summary_path = receiver_files[0].parent / "receiver_0.json"
        
        cmd = [
            sys.executable, "scripts/analyze_beacon_performance.py",
            "--beacon-summary", str(summary_path),
            "--beacon-trials", str(receiver_files[0]),
            "--output", str(analysis_path)
        ]
        
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            print("  Performance analysis completed")
            
            # Extract key insights
            lines = result.stdout.strip().split('\n')
            for line in lines:
                if 'Key Insights:' in line:
                    # Print subsequent insight lines
                    idx = lines.index(line)
                    for insight_line in lines[idx+1:]:
                        if insight_line.startswith('  • '):
                            print(f"    {insight_line[4:]}")
                        elif insight_line.strip() and not insight_line.startswith(' '):
                            break
            
        except subprocess.CalledProcessError as e:
            print(f"    Error in performance analysis: {e}")
    
    # 4. Summary and recommendations
    print("\n4. Summary and Recommendations")
    print("-" * 30)
    
    if results:
        # Find best strategy
        best_strategy = max(results.items(), key=lambda x: x[1].get('accuracy', 0))
        print(f"  Best performing strategy: {best_strategy[0]}")
        print(f"    Accuracy: {best_strategy[1].get('accuracy', 0):.1%}")
        
        # Compare enhanced vs basic
        enhanced_results = {k: v for k, v in results.items() if 'enhanced' in k}
        basic_results = {k: v for k, v in results.items() if 'basic' in k}
        
        if enhanced_results and basic_results:
            best_enhanced = max(enhanced_results.items(), key=lambda x: x[1].get('accuracy', 0))
            best_basic = max(basic_results.items(), key=lambda x: x[1].get('accuracy', 0))
            
            enhanced_acc = best_enhanced[1].get('accuracy', 0)
            basic_acc = best_basic[1].get('accuracy', 0)
            
            if enhanced_acc > basic_acc:
                improvement = enhanced_acc - basic_acc
                print(f"  Enhanced voting improves accuracy by {improvement:.1%}")
            else:
                print(f"  Basic voting performs similarly to enhanced")
            
            # Consistency advantage
            enhanced_cons = best_enhanced[1].get('consistency', 0)
            print(f"  Enhanced voting provides {enhanced_cons:.3f} consistency score")
    
    print(f"\n  Results saved to: {output_dir}")
    print("\nDemo completed! Check the output directory for detailed results.")


if __name__ == "__main__":
    run_demo()