#!/usr/bin/env python3
"""Integrate clutter metrics with spectrum beacon performance analysis.

This tool correlates first-path timing bias and negative hit rates from the 
handshake diagnostics with beacon detection reliability. The hypothesis is that
channels with heavy multipath clutter (high first_path_negative_rate, poor
first_path_within_10ns_rate) will also show degraded beacon consensus.

Usage:

    # First generate handshake clutter metrics
    python scripts/run_handshake_diag.py \
        --channel-profile URBAN_CANYON \
        --num-trials 100 \
        --output results/project_aperture_formant/URBAN_CANYON/clutter_diag.json

    # Then run beacon simulation with same profile  
    python scripts/run_spectrum_beacon_sim.py \
        --profiles A E I O U \
        --num-trials 512 \
        --snr-db 15 35 \
        --max-extra-paths 4 \
        --max-delay-ns 120 \
        --empty-prob 0.3 \
        --dump-trials \
        --output results/project_aperture_formant/URBAN_CANYON/beacon_sim.json

    # Finally correlate the two
    python scripts/beacon_clutter_analysis.py \
        --clutter-metrics results/project_aperture_formant/URBAN_CANYON/clutter_diag.json \
        --beacon-trials results/project_aperture_formant/URBAN_CANYON/beacon_sim.trials.jsonl \
        --output results/project_aperture_formant/URBAN_CANYON/clutter_beacon_correlation.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


def load_clutter_metrics(path: Path) -> Dict[str, object]:
    """Load handshake diagnostic results."""
    with path.open() as f:
        return json.load(f)


def load_beacon_trials(path: Path) -> List[dict]:
    """Load per-trial beacon results from JSONL."""
    trials = []
    with path.open() as f:
        for line in f:
            trials.append(json.loads(line))
    return trials


def analyze_beacon_by_snr_bins(trials: List[dict], num_bins: int = 4) -> Dict[str, object]:
    """Analyze beacon performance across SNR ranges."""
    snrs = [t['snr_db'] for t in trials if t.get('snr_db') is not None]
    if not snrs:
        return {}
    
    snr_min, snr_max = min(snrs), max(snrs)
    bin_edges = np.linspace(snr_min, snr_max, num_bins + 1)
    
    bins = {}
    for i in range(num_bins):
        bin_name = f"snr_{bin_edges[i]:.1f}_{bin_edges[i+1]:.1f}"
        bin_trials = [
            t for t in trials 
            if (t.get('snr_db') is not None and 
                bin_edges[i] <= t['snr_db'] < bin_edges[i+1])
        ]
        
        if not bin_trials:
            continue
            
        beacon_trials = [t for t in bin_trials if t.get('has_beacon')]
        detected_trials = [t for t in beacon_trials if t.get('predicted_label')]
        correct_trials = [
            t for t in detected_trials 
            if t.get('predicted_label') == t.get('true_label')
        ]
        
        bins[bin_name] = {
            'total_trials': len(bin_trials),
            'beacon_trials': len(beacon_trials),
            'detected_trials': len(detected_trials),
            'correct_trials': len(correct_trials),
            'detection_rate': len(detected_trials) / len(beacon_trials) if beacon_trials else 0,
            'accuracy_rate': len(correct_trials) / len(beacon_trials) if beacon_trials else 0,
            'snr_range': [bin_edges[i], bin_edges[i+1]],
            'avg_snr': np.mean([t['snr_db'] for t in bin_trials]),
        }
    
    return bins


def compute_beacon_reliability_metrics(trials: List[dict]) -> Dict[str, float]:
    """Compute various beacon reliability metrics."""
    beacon_trials = [t for t in trials if t.get('has_beacon')]
    empty_trials = [t for t in trials if not t.get('has_beacon')]
    
    if not trials:
        return {}
    
    # Detection metrics
    detected_beacons = [t for t in beacon_trials if t.get('predicted_label')]
    correct_detections = [
        t for t in detected_beacons 
        if t.get('predicted_label') == t.get('true_label')
    ]
    false_positives = [t for t in empty_trials if t.get('predicted_label')]
    
    # Score statistics for successful detections
    success_scores = [t.get('score', 0) for t in correct_detections if t.get('score')]
    failure_scores = [
        t.get('score', 0) for t in beacon_trials 
        if t.get('predicted_label') and t.get('predicted_label') != t.get('true_label') and t.get('score')
    ]
    
    metrics = {
        'total_trials': len(trials),
        'beacon_trials': len(beacon_trials),
        'empty_trials': len(empty_trials),
        'detection_rate': len(detected_beacons) / len(beacon_trials) if beacon_trials else 0,
        'accuracy_rate': len(correct_detections) / len(beacon_trials) if beacon_trials else 0,
        'false_positive_rate': len(false_positives) / len(empty_trials) if empty_trials else 0,
        'precision': len(correct_detections) / len(detected_beacons) if detected_beacons else 0,
        'recall': len(correct_detections) / len(beacon_trials) if beacon_trials else 0,
    }
    
    # Score-based metrics
    if success_scores:
        metrics['success_score_mean'] = float(np.mean(success_scores))
        metrics['success_score_std'] = float(np.std(success_scores))
    
    if failure_scores:
        metrics['failure_score_mean'] = float(np.mean(failure_scores))
        metrics['failure_score_std'] = float(np.std(failure_scores))
    
    return metrics


def correlate_clutter_beacon(
    clutter_data: Dict[str, object], 
    beacon_trials: List[dict]
) -> Dict[str, object]:
    """Correlate clutter metrics with beacon performance."""
    
    # Extract clutter metrics
    clutter_metrics = clutter_data.get('first_path_metrics', {})
    fp_negative_rate = clutter_metrics.get('first_path_negative_rate', 0)
    fp_within_5ns_rate = clutter_metrics.get('first_path_within_5ns_rate', 0)
    fp_within_10ns_rate = clutter_metrics.get('first_path_within_10ns_rate', 0)
    
    # Compute beacon reliability
    beacon_metrics = compute_beacon_reliability_metrics(beacon_trials)
    
    # SNR binning analysis
    snr_bins = analyze_beacon_by_snr_bins(beacon_trials)
    
    # Correlation analysis
    correlation = {
        'clutter_indicators': {
            'first_path_negative_rate': fp_negative_rate,
            'first_path_within_5ns_rate': fp_within_5ns_rate,
            'first_path_within_10ns_rate': fp_within_10ns_rate,
            'clutter_severity': 'high' if fp_negative_rate > 0.25 else 'medium' if fp_negative_rate > 0.1 else 'low'
        },
        'beacon_performance': beacon_metrics,
        'snr_analysis': snr_bins,
        'correlation_insights': {}
    }
    
    # Generate insights
    insights = correlation['correlation_insights']
    
    # Clutter impact on detection
    if fp_negative_rate > 0.3 and beacon_metrics.get('detection_rate', 0) < 0.7:
        insights['high_clutter_low_detection'] = (
            f"High clutter environment (negative rate: {fp_negative_rate:.2f}) "
            f"correlates with reduced beacon detection ({beacon_metrics['detection_rate']:.2f})"
        )
    
    # SNR resilience in cluttered environments  
    if snr_bins:
        high_snr_bins = [b for name, b in snr_bins.items() if b['avg_snr'] > 30]
        low_snr_bins = [b for name, b in snr_bins.items() if b['avg_snr'] < 20]
        
        if high_snr_bins and low_snr_bins:
            high_snr_detection = np.mean([b['detection_rate'] for b in high_snr_bins])
            low_snr_detection = np.mean([b['detection_rate'] for b in low_snr_bins])
            
            detection_delta = high_snr_detection - low_snr_detection
            
            if fp_negative_rate > 0.2 and detection_delta < 0.3:
                insights['clutter_snr_resilience'] = (
                    f"Multipath clutter reduces SNR resilience: high SNR detection "
                    f"({high_snr_detection:.2f}) vs low SNR ({low_snr_detection:.2f}) "
                    f"delta only {detection_delta:.2f}"
                )
    
    # False positive correlation
    if fp_negative_rate > 0.25 and beacon_metrics.get('false_positive_rate', 0) > 0.05:
        insights['clutter_false_positives'] = (
            f"Clutter environment may be contributing to false positives: "
            f"negative rate {fp_negative_rate:.2f}, FP rate {beacon_metrics['false_positive_rate']:.3f}"
        )
    
    return correlation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--clutter-metrics', type=Path, required=True, 
                       help='JSON file from run_handshake_diag.py')
    parser.add_argument('--beacon-trials', type=Path, required=True,
                       help='JSONL file from run_spectrum_beacon_sim.py --dump-trials')
    parser.add_argument('--output', type=Path, required=True)
    
    args = parser.parse_args()
    
    # Load data
    clutter_data = load_clutter_metrics(args.clutter_metrics)
    beacon_trials = load_beacon_trials(args.beacon_trials)
    
    # Perform correlation analysis
    correlation_result = correlate_clutter_beacon(clutter_data, beacon_trials)
    
    # Save results
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('w') as f:
        json.dump(correlation_result, f, indent=2)
    
    print(f"Clutter-beacon correlation saved to {args.output}")
    
    # Print summary
    clutter_ind = correlation_result['clutter_indicators']
    beacon_perf = correlation_result['beacon_performance']
    
    print(f"\nClutter Environment: {clutter_ind['clutter_severity']}")
    print(f"  First-path negative rate: {clutter_ind['first_path_negative_rate']:.3f}")
    print(f"  First-path within 10ns: {clutter_ind['first_path_within_10ns_rate']:.3f}")
    
    print(f"\nBeacon Performance:")
    print(f"  Detection rate: {beacon_perf.get('detection_rate', 0):.3f}")
    print(f"  Accuracy rate: {beacon_perf.get('accuracy_rate', 0):.3f}")
    print(f"  False positive rate: {beacon_perf.get('false_positive_rate', 0):.3f}")
    
    insights = correlation_result.get('correlation_insights', {})
    if insights:
        print(f"\nCorrelation Insights:")
        for key, insight in insights.items():
            print(f"  • {insight}")


if __name__ == '__main__':
    main()