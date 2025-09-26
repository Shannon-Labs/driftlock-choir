#!/usr/bin/env python3
"""Enhanced spectrum beacon aggregation with weighted voting and consistency checks.

This script extends the basic voting aggregation in `aggregate_beacon_votes.py`
with additional fusion strategies:

1. **Weighted Voting** - Receivers with higher scores or lower score variance get
   more weight in the final decision
2. **Consistency Checks** - Agreement on missing-f0 and dominant harmonics across
   receivers strengthens confidence
3. **Score Variance** - Reject decisions when receivers report wildly different
   scores for the same label
4. **Harmonic Agreement** - Ensure dominant frequencies are within tolerance

Example:

    python scripts/enhanced_beacon_votes.py \
        --votes results/project_aperture_formant/URBAN_CANYON/*.trials.jsonl \
        --vote-strategy weighted \
        --missing-f0-tolerance-hz 100 \
        --dominant-tolerance-hz 1000 \
        --score-variance-threshold 0.5 \
        --output results/project_aperture_formant/URBAN_CANYON/enhanced_vote.json
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np


def load_trials(path: Path) -> List[dict]:
    records: List[dict] = []
    with path.open() as handle:
        for line in handle:
            records.append(json.loads(line))
    return records


def basic_vote(records: Iterable[str], threshold: int) -> Optional[str]:
    """Original simple majority voting."""
    counter = Counter(rec for rec in records if rec)
    if not counter:
        return None
    label, count = counter.most_common(1)[0]
    if count >= threshold:
        return label
    return None


def weighted_vote(
    detections: List[dict], 
    threshold: int,
    missing_f0_tolerance_hz: float = 100.0,
    dominant_tolerance_hz: float = 1000.0,
    score_variance_threshold: float = 0.5
) -> Tuple[Optional[str], dict]:
    """Enhanced voting with weights and consistency checks.
    
    Returns:
        (winning_label, metadata_dict)
    """
    # Filter out null detections
    valid_detections = [d for d in detections if d.get('predicted_label')]
    
    if not valid_detections:
        return None, {'reason': 'no_valid_detections', 'consistency_score': 0.0}
    
    # Group by predicted label
    label_groups = {}
    for det in valid_detections:
        label = det['predicted_label']
        if label not in label_groups:
            label_groups[label] = []
        label_groups[label].append(det)
    
    # Calculate weighted scores for each label
    label_scores = {}
    for label, group in label_groups.items():
        scores = [d.get('score', 0) for d in group]
        missing_f0s = [d.get('missing_f0_hz', 0) for d in group if d.get('missing_f0_hz')]
        dominants = [d.get('dominant_hz', 0) for d in group if d.get('dominant_hz')]
        
        # Weight calculation
        weight_sum = 0.0
        consistency_penalties = []
        
        for det in group:
            # Base weight from score magnitude (higher is better, but normalize)
            score = det.get('score', 0)
            base_weight = np.log10(max(score, 1e6)) / 15.0  # Rough normalization
            
            # Penalty for missing key metrics
            if not det.get('missing_f0_hz') or not det.get('dominant_hz'):
                base_weight *= 0.5
            
            weight_sum += base_weight
        
        # Consistency checks
        consistency_score = 1.0
        
        # Missing f0 consistency
        if len(missing_f0s) > 1:
            f0_std = np.std(missing_f0s)
            f0_mean = np.mean(missing_f0s)
            if f0_std > missing_f0_tolerance_hz:
                consistency_score *= 0.5  # Penalize f0 disagreement
        
        # Dominant frequency consistency  
        if len(dominants) > 1:
            dom_std = np.std(dominants)
            if dom_std > dominant_tolerance_hz:
                consistency_score *= 0.5  # Penalize dominant disagreement
        
        # Score variance check
        if len(scores) > 1:
            score_cv = np.std(scores) / (np.mean(scores) + 1e-12)  # Coefficient of variation
            if score_cv > score_variance_threshold:
                consistency_score *= 0.3  # Heavy penalty for score disagreement
        
        final_weight = weight_sum * consistency_score
        label_scores[label] = {
            'weight': final_weight,
            'count': len(group),
            'consistency_score': consistency_score,
            'score_mean': np.mean(scores) if scores else 0,
            'score_std': np.std(scores) if len(scores) > 1 else 0,
            'missing_f0_mean': np.mean(missing_f0s) if missing_f0s else None,
            'missing_f0_std': np.std(missing_f0s) if len(missing_f0s) > 1 else 0,
            'dominant_mean': np.mean(dominants) if dominants else None,
            'dominant_std': np.std(dominants) if len(dominants) > 1 else 0,
        }
    
    # Find the winning label (highest weighted score + minimum threshold)
    best_label = None
    best_weight = 0
    best_metadata = {}
    
    for label, metrics in label_scores.items():
        if metrics['count'] >= threshold and metrics['weight'] > best_weight:
            best_label = label
            best_weight = metrics['weight']
            best_metadata = metrics
    
    if best_label is None:
        return None, {
            'reason': 'threshold_not_met', 
            'candidates': label_scores,
            'consistency_score': 0.0
        }
    
    return best_label, best_metadata


def aggregate_enhanced(
    paths: List[Path], 
    threshold: int,
    vote_strategy: str = "basic",
    missing_f0_tolerance_hz: float = 100.0,
    dominant_tolerance_hz: float = 1000.0,
    score_variance_threshold: float = 0.5
) -> Dict[str, object]:
    """Enhanced aggregation with multiple voting strategies."""
    
    trials_per_path = [load_trials(path) for path in paths]
    num_receivers = len(trials_per_path)
    num_trials = min(len(trials) for trials in trials_per_path)

    results = []
    for idx in range(num_trials):
        beacon = any(trials[idx]['has_beacon'] for trials in trials_per_path)
        true_label = None
        true_labels = {trials[idx]['true_label'] for trials in trials_per_path if trials[idx]['true_label']}
        if true_labels:
            true_label = sorted(true_labels)[0]
        
        # Collect all detection data for this trial
        detections = []
        for trials in trials_per_path:
            trial_data = trials[idx]
            detections.append({
                'predicted_label': trial_data.get('predicted_label'),
                'score': trial_data.get('score'),
                'missing_f0_hz': trial_data.get('missing_f0_hz'),
                'dominant_hz': trial_data.get('dominant_hz'),
            })
        
        # Apply voting strategy
        if vote_strategy == "weighted":
            winning_label, vote_metadata = weighted_vote(
                detections, threshold, missing_f0_tolerance_hz, 
                dominant_tolerance_hz, score_variance_threshold
            )
        else:  # basic
            votes = [d['predicted_label'] for d in detections]
            winning_label = basic_vote(votes, threshold)
            vote_metadata = {'strategy': 'basic'}
        
        results.append({
            'trial': idx,
            'has_beacon': beacon,
            'true_label': true_label,
            'winning_label': winning_label,
            'all_votes': [d['predicted_label'] for d in detections],
            'vote_metadata': vote_metadata,
            'detections': detections,
        })

    # Calculate overall metrics
    correct = sum(1 for row in results if row['has_beacon'] and row['winning_label'] == row['true_label'])
    detected = sum(1 for row in results if row['winning_label'] is not None)
    total_beacons = sum(1 for row in results if row['has_beacon'])
    total_empty = sum(1 for row in results if not row['has_beacon'])
    false_positive = sum(1 for row in results if not row['has_beacon'] and row['winning_label'] is not None)

    # Enhanced metrics
    consistency_scores = [
        row['vote_metadata'].get('consistency_score', 0) 
        for row in results 
        if row['winning_label'] is not None
    ]
    avg_consistency = np.mean(consistency_scores) if consistency_scores else 0.0

    summary = {
        'num_receivers': num_receivers,
        'num_trials': num_trials,
        'vote_threshold': threshold,
        'vote_strategy': vote_strategy,
        'detected_rate': detected / num_trials if num_trials else None,
        'label_accuracy': correct / total_beacons if total_beacons else None,
        'false_positive_rate': false_positive / total_empty if total_empty else None,
        'avg_consistency_score': avg_consistency,
        'config': {
            'missing_f0_tolerance_hz': missing_f0_tolerance_hz,
            'dominant_tolerance_hz': dominant_tolerance_hz,
            'score_variance_threshold': score_variance_threshold,
        },
        'results': results,
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--votes', type=Path, nargs='+', required=True, help='JSONL files (per receiver) from --dump-trials')
    parser.add_argument('--vote-threshold', type=int, default=2, help='Minimum agreeing receivers to declare a label')
    parser.add_argument('--vote-strategy', choices=['basic', 'weighted'], default='weighted', help='Voting strategy')
    parser.add_argument('--missing-f0-tolerance-hz', type=float, default=100.0, help='Missing f0 agreement tolerance')
    parser.add_argument('--dominant-tolerance-hz', type=float, default=1000.0, help='Dominant frequency agreement tolerance')
    parser.add_argument('--score-variance-threshold', type=float, default=0.5, help='Score coefficient of variation threshold')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()

    summary = aggregate_enhanced(
        args.votes, args.vote_threshold, args.vote_strategy,
        args.missing_f0_tolerance_hz, args.dominant_tolerance_hz,
        args.score_variance_threshold
    )
    
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2), encoding='utf-8')
    
    print(f'Enhanced vote summary saved to {args.output}')
    print(f'Strategy: {args.vote_strategy}')
    print(f'Detection rate: {summary["detected_rate"]:.3f}')
    print(f'Label accuracy: {summary["label_accuracy"]:.3f}')
    print(f'False positive rate: {summary["false_positive_rate"]:.3f}')
    print(f'Average consistency score: {summary["avg_consistency_score"]:.3f}')


if __name__ == '__main__':
    main()