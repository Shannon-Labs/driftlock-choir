#!/usr/bin/env python3
"""Aggregate spectrum beacon detections from multiple receivers.

Each input is a JSON lines file produced by
`scripts/run_spectrum_beacon_sim.py --dump-trials ...`.  We align trials by
index (i.e., assume every receiver simulated the same sequence with the same
RNG seed) and apply a voting rule.  The default rule marks a beacon present when
at least `--vote-threshold` receivers concur on the same vowel label.

Example:

    python scripts/aggregate_beacon_votes.py \
        --votes results/project_aperture_formant/URBAN_CANYON/*.trials.jsonl \
        --vote-threshold 2 \
        --output results/project_aperture_formant/URBAN_CANYON/vote_summary.json
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional


def load_trials(path: Path) -> List[dict]:
    records: List[dict] = []
    with path.open() as handle:
        for line in handle:
            records.append(json.loads(line))
    return records


def summarize_vote(records: Iterable[str], threshold: int) -> Optional[str]:
    counter = Counter(rec for rec in records if rec)
    if not counter:
        return None
    label, count = counter.most_common(1)[0]
    if count >= threshold:
        return label
    return None


def aggregate(paths: List[Path], threshold: int) -> Dict[str, object]:
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
        votes = [trials[idx]['predicted_label'] if trials[idx]['predicted_label'] else None for trials in trials_per_path]
        winning_label = summarize_vote(votes, threshold)
        results.append({
            'trial': idx,
            'has_beacon': beacon,
            'true_label': true_label,
            'winning_label': winning_label,
            'all_votes': votes,
        })

    correct = sum(1 for row in results if row['has_beacon'] and row['winning_label'] == row['true_label'])
    detected = sum(1 for row in results if row['winning_label'] is not None)
    total_beacons = sum(1 for row in results if row['has_beacon'])
    total_empty = sum(1 for row in results if not row['has_beacon'])
    false_positive = sum(1 for row in results if not row['has_beacon'] and row['winning_label'] is not None)

    summary = {
        'num_receivers': num_receivers,
        'num_trials': num_trials,
        'vote_threshold': threshold,
        'detected_rate': detected / num_trials if num_trials else None,
        'label_accuracy': correct / total_beacons if total_beacons else None,
        'false_positive_rate': false_positive / total_empty if total_empty else None,
        'results': results,
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--votes', type=Path, nargs='+', required=True, help='JSONL files (per receiver) from --dump-trials')
    parser.add_argument('--vote-threshold', type=int, default=2, help='Minimum agreeing receivers to declare a label')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()

    summary = aggregate(args.votes, args.vote_threshold)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(f'Saved vote summary to {args.output}')


if __name__ == '__main__':
    main()
