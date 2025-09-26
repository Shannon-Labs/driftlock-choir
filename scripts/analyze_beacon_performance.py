#!/usr/bin/env python3
"""Analyze existing beacon performance data and generate insights.

This script loads existing spectrum beacon results and performs detailed
analysis including per-vowel performance, SNR dependencies, score distributions,
and false positive analysis.

Usage:
    python scripts/analyze_beacon_performance.py \
        --beacon-summary results/project_aperture_formant/URBAN_CANYON/20250925T212300Z_spectrum_beacon_tol.json \
        --beacon-trials results/project_aperture_formant/URBAN_CANYON/20250925T212300Z_spectrum_beacon_tol.trials.jsonl \
        --output results/project_aperture_formant/URBAN_CANYON/performance_analysis.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


def load_beacon_data(summary_path: Path, trials_path: Path) -> tuple[Dict, List[Dict]]:
    """Load beacon summary and per-trial data."""
    with summary_path.open() as f:
        summary = json.load(f)
    
    trials = []
    if trials_path.exists():
        with trials_path.open() as f:
            for line in f:
                trials.append(json.loads(line))
    
    return summary, trials


def analyze_per_vowel_performance(trials: List[Dict]) -> Dict[str, Dict]:
    """Analyze performance broken down by vowel label."""
    vowel_analysis = {}
    
    # Group trials by true label
    vowel_trials = {}
    for trial in trials:
        if trial.get('has_beacon') and trial.get('true_label'):
            label = trial['true_label']
            if label not in vowel_trials:
                vowel_trials[label] = []
            vowel_trials[label].append(trial)
    
    # Analyze each vowel
    for vowel, vowel_data in vowel_trials.items():
        detected_trials = [t for t in vowel_data if t.get('predicted_label')]
        correct_trials = [t for t in detected_trials if t['predicted_label'] == vowel]
        
        # Score statistics
        all_scores = [t.get('score', 0) for t in vowel_data if t.get('score')]
        success_scores = [t.get('score', 0) for t in correct_trials if t.get('score')]
        failure_scores = [t.get('score', 0) for t in detected_trials if t['predicted_label'] != vowel and t.get('score')]
        
        # Missing f0 and dominant frequency stats for successful detections
        success_f0s = [t.get('missing_f0_hz', 0) for t in correct_trials if t.get('missing_f0_hz')]
        success_doms = [t.get('dominant_hz', 0) for t in correct_trials if t.get('dominant_hz')]
        
        # SNR analysis
        snrs = [t.get('snr_db', 0) for t in vowel_data if t.get('snr_db')]
        success_snrs = [t.get('snr_db', 0) for t in correct_trials if t.get('snr_db')]
        
        vowel_analysis[vowel] = {
            'total_trials': len(vowel_data),
            'detected_trials': len(detected_trials),
            'correct_trials': len(correct_trials),
            'detection_rate': len(detected_trials) / len(vowel_data) if vowel_data else 0,
            'accuracy_rate': len(correct_trials) / len(vowel_data) if vowel_data else 0,
            'precision': len(correct_trials) / len(detected_trials) if detected_trials else 0,
            'score_stats': {
                'all_mean': float(np.mean(all_scores)) if all_scores else 0,
                'all_std': float(np.std(all_scores)) if all_scores else 0,
                'success_mean': float(np.mean(success_scores)) if success_scores else 0,
                'success_std': float(np.std(success_scores)) if success_scores else 0,
                'failure_mean': float(np.mean(failure_scores)) if failure_scores else 0,
                'failure_std': float(np.std(failure_scores)) if failure_scores else 0,
            },
            'spectral_stats': {
                'missing_f0_mean': float(np.mean(success_f0s)) if success_f0s else 0,
                'missing_f0_std': float(np.std(success_f0s)) if success_f0s else 0,
                'dominant_mean': float(np.mean(success_doms)) if success_doms else 0,
                'dominant_std': float(np.std(success_doms)) if success_doms else 0,
            },
            'snr_analysis': {
                'all_snr_mean': float(np.mean(snrs)) if snrs else 0,
                'success_snr_mean': float(np.mean(success_snrs)) if success_snrs else 0,
                'snr_sensitivity': float(np.mean(success_snrs) - np.mean(snrs)) if snrs and success_snrs else 0,
            }
        }
    
    return vowel_analysis


def analyze_snr_dependence(trials: List[Dict], num_bins: int = 4) -> Dict[str, Dict]:
    """Analyze beacon performance across SNR ranges."""
    beacon_trials = [t for t in trials if t.get('has_beacon')]
    snrs = [t['snr_db'] for t in beacon_trials if t.get('snr_db') is not None]
    
    if not snrs:
        return {}
    
    snr_min, snr_max = min(snrs), max(snrs)
    bin_edges = np.linspace(snr_min, snr_max, num_bins + 1)
    
    snr_analysis = {}
    for i in range(num_bins):
        bin_name = f"snr_{bin_edges[i]:.1f}_{bin_edges[i+1]:.1f}"
        bin_trials = [
            t for t in beacon_trials 
            if bin_edges[i] <= t['snr_db'] < bin_edges[i+1]
        ]
        
        if not bin_trials:
            continue
        
        detected = [t for t in bin_trials if t.get('predicted_label')]
        correct = [t for t in detected if t['predicted_label'] == t.get('true_label')]
        
        # Score statistics for this SNR range
        scores = [t.get('score', 0) for t in bin_trials if t.get('score')]
        
        snr_analysis[bin_name] = {
            'snr_range': [float(bin_edges[i]), float(bin_edges[i+1])],
            'total_trials': len(bin_trials),
            'detected_trials': len(detected),
            'correct_trials': len(correct),
            'detection_rate': len(detected) / len(bin_trials),
            'accuracy_rate': len(correct) / len(bin_trials),
            'precision': len(correct) / len(detected) if detected else 0,
            'avg_snr': float(np.mean([t['snr_db'] for t in bin_trials])),
            'avg_score': float(np.mean(scores)) if scores else 0,
            'score_std': float(np.std(scores)) if scores else 0,
        }
    
    return snr_analysis


def analyze_confusion_matrix(trials: List[Dict]) -> Dict[str, Dict]:
    """Analyze label confusion patterns."""
    beacon_trials = [t for t in trials if t.get('has_beacon') and t.get('true_label')]
    
    # Count true vs predicted label combinations
    confusion_counts = {}
    true_labels = set(t['true_label'] for t in beacon_trials)
    
    for true_label in true_labels:
        confusion_counts[true_label] = {}
        true_trials = [t for t in beacon_trials if t['true_label'] == true_label]
        
        for trial in true_trials:
            pred_label = trial.get('predicted_label', 'NONE')
            if pred_label not in confusion_counts[true_label]:
                confusion_counts[true_label][pred_label] = 0
            confusion_counts[true_label][pred_label] += 1
    
    # Convert to rates
    confusion_matrix = {}
    for true_label, pred_counts in confusion_counts.items():
        total = sum(pred_counts.values())
        confusion_matrix[true_label] = {
            pred_label: count / total for pred_label, count in pred_counts.items()
        }
    
    # Identify most common confusions
    confusions = []
    for true_label, pred_rates in confusion_matrix.items():
        for pred_label, rate in pred_rates.items():
            if pred_label != true_label and pred_label != 'NONE' and rate > 0.05:  # 5% threshold
                confusions.append({
                    'true': true_label,
                    'predicted': pred_label,
                    'rate': rate
                })
    
    confusions.sort(key=lambda x: x['rate'], reverse=True)
    
    return {
        'confusion_matrix': confusion_matrix,
        'top_confusions': confusions[:10],  # Top 10 confusion pairs
    }


def analyze_false_positive_patterns(trials: List[Dict]) -> Dict[str, object]:
    """Analyze false positive detection patterns."""
    empty_trials = [t for t in trials if not t.get('has_beacon')]
    false_positives = [t for t in empty_trials if t.get('predicted_label')]
    
    if not false_positives:
        return {'false_positive_rate': 0, 'patterns': {}}
    
    # Analyze false positive labels
    fp_labels = [t['predicted_label'] for t in false_positives]
    label_counts = {label: fp_labels.count(label) for label in set(fp_labels)}
    
    # Score analysis of false positives
    fp_scores = [t.get('score', 0) for t in false_positives if t.get('score')]
    
    # Compare with true positive scores
    beacon_trials = [t for t in trials if t.get('has_beacon')]
    correct_detections = [t for t in beacon_trials if t.get('predicted_label') == t.get('true_label')]
    tp_scores = [t.get('score', 0) for t in correct_detections if t.get('score')]
    
    patterns = {
        'false_positive_rate': len(false_positives) / len(empty_trials),
        'total_false_positives': len(false_positives),
        'label_distribution': label_counts,
        'score_stats': {
            'fp_mean': float(np.mean(fp_scores)) if fp_scores else 0,
            'fp_std': float(np.std(fp_scores)) if fp_scores else 0,
            'tp_mean': float(np.mean(tp_scores)) if tp_scores else 0,
            'tp_std': float(np.std(tp_scores)) if tp_scores else 0,
        }
    }
    
    # Score threshold analysis
    if fp_scores and tp_scores:
        # Find threshold that minimizes false positives while preserving true positives
        all_scores = sorted(fp_scores + tp_scores)
        thresholds = []
        
        for threshold in all_scores[::len(all_scores)//20]:  # Sample ~20 thresholds
            fp_above = sum(1 for s in fp_scores if s >= threshold)
            tp_above = sum(1 for s in tp_scores if s >= threshold)
            
            fp_rate = fp_above / len(fp_scores)
            tp_rate = tp_above / len(tp_scores)
            
            thresholds.append({
                'threshold': float(threshold),
                'false_positive_rate': fp_rate,
                'true_positive_rate': tp_rate,
                'f1_score': 2 * tp_rate / (2 * tp_rate + fp_rate) if (tp_rate + fp_rate) > 0 else 0
            })
        
        # Find best threshold by F1 score
        best_threshold = max(thresholds, key=lambda x: x['f1_score'])
        patterns['threshold_analysis'] = {
            'best_threshold': best_threshold,
            'threshold_sweep': thresholds
        }
    
    return patterns


def generate_performance_insights(analysis: Dict) -> List[str]:
    """Generate human-readable insights from the analysis."""
    insights = []
    
    # Overall performance insight
    vowel_data = analysis.get('per_vowel_analysis', {})
    if vowel_data:
        avg_accuracy = np.mean([v['accuracy_rate'] for v in vowel_data.values()])
        avg_detection = np.mean([v['detection_rate'] for v in vowel_data.values()])
        insights.append(f"Overall: {avg_accuracy:.1%} accuracy, {avg_detection:.1%} detection rate across vowels")
        
        # Best/worst performing vowels
        best_vowel = max(vowel_data.items(), key=lambda x: x[1]['accuracy_rate'])
        worst_vowel = min(vowel_data.items(), key=lambda x: x[1]['accuracy_rate'])
        insights.append(f"Best vowel: {best_vowel[0]} ({best_vowel[1]['accuracy_rate']:.1%} accuracy)")
        insights.append(f"Worst vowel: {worst_vowel[0]} ({worst_vowel[1]['accuracy_rate']:.1%} accuracy)")
    
    # SNR dependence
    snr_data = analysis.get('snr_analysis', {})
    if len(snr_data) >= 2:
        snr_bins = sorted(snr_data.items(), key=lambda x: x[1]['snr_range'][0])
        low_snr = snr_bins[0][1]
        high_snr = snr_bins[-1][1]
        
        snr_improvement = high_snr['accuracy_rate'] - low_snr['accuracy_rate']
        insights.append(f"SNR impact: {snr_improvement:+.1%} accuracy gain from low to high SNR")
    
    # Confusion patterns
    confusion_data = analysis.get('confusion_analysis', {})
    top_confusions = confusion_data.get('top_confusions', [])
    if top_confusions:
        worst_confusion = top_confusions[0]
        insights.append(f"Top confusion: {worst_confusion['true']} → {worst_confusion['predicted']} ({worst_confusion['rate']:.1%})")
    
    # False positive analysis
    fp_data = analysis.get('false_positive_analysis', {})
    fp_rate = fp_data.get('false_positive_rate', 0)
    if fp_rate > 0:
        insights.append(f"False positive rate: {fp_rate:.1%}")
        
        threshold_data = fp_data.get('threshold_analysis', {})
        if threshold_data:
            best_thresh = threshold_data.get('best_threshold', {})
            insights.append(f"Optimal threshold: {best_thresh.get('threshold', 0):.2e} (F1: {best_thresh.get('f1_score', 0):.3f})")
    else:
        insights.append("No false positives detected")
    
    return insights


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--beacon-summary', type=Path, required=True,
                       help='JSON summary from run_spectrum_beacon_sim.py')
    parser.add_argument('--beacon-trials', type=Path, required=True,
                       help='JSONL trials from run_spectrum_beacon_sim.py --dump-trials')
    parser.add_argument('--output', type=Path, required=True,
                       help='Output analysis JSON file')
    
    args = parser.parse_args()
    
    # Load data
    summary, trials = load_beacon_data(args.beacon_summary, args.beacon_trials)
    
    print(f"Analyzing {len(trials)} beacon trials...")
    
    # Perform analyses
    analysis = {
        'timestamp': summary.get('timestamp', 'unknown'),
        'config': summary.get('config', {}),
        'overall_metrics': {
            'num_trials': summary.get('num_trials', 0),
            'detected_rate': summary.get('detected_rate', 0),
            'label_accuracy': summary.get('label_accuracy', 0),
            'false_positive_rate': summary.get('false_positive_rate', 0),
        },
        'per_vowel_analysis': analyze_per_vowel_performance(trials),
        'snr_analysis': analyze_snr_dependence(trials),
        'confusion_analysis': analyze_confusion_matrix(trials),
        'false_positive_analysis': analyze_false_positive_patterns(trials),
    }
    
    # Generate insights
    analysis['insights'] = generate_performance_insights(analysis)
    
    # Save analysis
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('w') as f:
        json.dump(analysis, f, indent=2)
    
    print(f"Analysis saved to {args.output}")
    
    # Print summary
    print(f"\n{'='*50}")
    print("BEACON PERFORMANCE ANALYSIS")
    print(f"{'='*50}")
    
    overall = analysis['overall_metrics']
    print(f"Trials: {overall['num_trials']}")
    print(f"Detection Rate: {overall['detected_rate']:.1%}")
    print(f"Label Accuracy: {overall['label_accuracy']:.1%}")
    print(f"False Positive Rate: {overall['false_positive_rate']:.1%}")
    
    print("\nKey Insights:")
    for insight in analysis['insights']:
        print(f"  • {insight}")
    
    print(f"\nPer-Vowel Performance:")
    for vowel, data in analysis['per_vowel_analysis'].items():
        print(f"  {vowel}: {data['accuracy_rate']:.1%} accuracy, {data['detection_rate']:.1%} detection")


if __name__ == '__main__':
    main()