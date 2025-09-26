#!/usr/bin/env python3
"""Summarize Pathfinder guard clutter metrics from sweep JSON artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List


def load_rows(root: Path) -> List[dict]:
    rows: List[dict] = []
    for path in sorted(root.rglob('*.json')):
        try:
            data = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        if 'directional_metrics' not in data:
            continue
        forward = data['directional_metrics']['forward']
        reverse = data['directional_metrics']['reverse']
        if forward.get('first_path_negative_rate') is None:
            continue
        rows.append({
            'name': path.stem,
            'path': path,
            'guard': data.get('pathfinder_guard_interval_ns'),
            'blend': data.get('pathfinder_first_path_blend'),
            'missing': data.get('coarse_formant_missing_fundamental'),
            'within10_fwd': forward.get('first_path_within_10ns_rate'),
            'within10_rev': reverse.get('first_path_within_10ns_rate'),
            'neg_fwd': forward.get('first_path_negative_rate'),
            'neg_rev': reverse.get('first_path_negative_rate'),
            'bias': data['two_way_metrics']['tau_bias_ns']['mean'],
        })
    rows.sort(key=lambda r: (r['guard'] or 0, r['blend'] or 0, r['name']))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path, help='Directory containing handshake JSON summaries')
    args = parser.parse_args()

    rows = load_rows(args.root)
    if not rows:
        raise SystemExit('No compatible JSON files found')

    print('| scenario | guard_ns | blend | missing_f0 | f_within10 | r_within10 | f_negative | r_negative | tau_bias_ns |')
    print('| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |')
    for row in rows:
        blend = '-' if row['blend'] is None else f"{row['blend']:.2f}"
        missing = 'on' if row['missing'] else 'off'
        print(
            f"| {row['name']} | {row['guard'] or '-'} | {blend} | {missing} | "
            f"{row['within10_fwd']:.3f} | {row['within10_rev']:.3f} | "
            f"{row['neg_fwd']:.3f} | {row['neg_rev']:.3f} | {row['bias']:.2f} |"
        )


if __name__ == '__main__':
    main()
