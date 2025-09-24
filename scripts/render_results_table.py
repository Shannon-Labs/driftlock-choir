#!/usr/bin/env python3
"""Render README-ready bullet and table from TDL profile manifests."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


def load_manifest(path: Path) -> Dict[str, any]:
    with path.open('r', encoding='utf-8') as handle:
        return json.load(handle)


def collect_manifests(paths: Iterable[Path]) -> List[Dict[str, any]]:
    manifests: List[Dict[str, any]] = []
    for path in paths:
        if path.is_dir():
            manifests.extend(collect_manifests(sorted(path.glob('*.json'))))
        elif path.suffix.lower() == '.json':
            manifests.append(load_manifest(path))
    return manifests


def format_number(value: float, precision: int = 2) -> str:
    return f"{value:.{precision}f}" if value is not None else "N/A"


def short_sha(sha: str | None) -> str:
    return sha[:7] if sha else 'unknown'


def build_bullet(manifests: List[Dict[str, any]]) -> str:
    if not manifests:
        return "No manifests supplied."
    manifests_sorted = sorted(manifests, key=lambda m: m.get('profile', ''))
    ref = manifests_sorted[0]
    snr = ref.get('snr_db')
    seed = ref.get('rng_seed')
    sha = short_sha(ref.get('git_sha'))

    segments = []
    for manifest in manifests_sorted:
        profile = manifest.get('profile', 'UNKNOWN')
        two_way = manifest.get('two_way_metrics', {})
        bias = two_way.get('tau_bias_ns', {}).get('mean')
        rmse = two_way.get('tau_rmse_ns')
        df_bias = two_way.get('deltaf_bias_hz', {}).get('mean')
        ratio = two_way.get('rmse_over_crlb')
        status = manifest.get('status', 'OK')
        segments.append(
            f"{profile}: {format_number(bias)} ns bias (RMSE {format_number(rmse)}, Δf bias {format_number(df_bias, 1)} Hz, CRLB ratio {format_number(ratio, 2)})[{status}]"
        )

    header = f"**TDL Sweep @ {snr:.0f} dB (seed {seed}, commit {sha})**: "
    return header + "; ".join(segments)


def build_table(manifests: List[Dict[str, any]]) -> str:
    rows = []
    header = "| Date | Commit | Profile | Bias (ns) | RMSE (ns) | Δf bias (Hz) | CRLB (ns) | RMSE/CRLB | Coarse Lock | Guard Hit | Status |\n"
    separator = "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |\n"
    rows.append(header)
    rows.append(separator)

    def manifest_date(manifest: Dict[str, any]) -> str:
        timestamp = manifest.get('timestamp')
        if not timestamp:
            return ''
        try:
            dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            return dt.date().isoformat()
        except ValueError:
            return timestamp

    for manifest in sorted(manifests, key=lambda m: manifest_date(m)):
        two_way = manifest.get('two_way_metrics', {})
        row = "| {date} | {sha} | {profile} | {bias} | {rmse} | {dfbias} | {crlb} | {ratio} | {locked} | {guard} | {status} |\n".format(
            date=manifest_date(manifest),
            sha=short_sha(manifest.get('git_sha')),
            profile=manifest.get('profile', 'UNKNOWN'),
            bias=format_number(two_way.get('tau_bias_ns', {}).get('mean')),
            rmse=format_number(two_way.get('tau_rmse_ns')),
            dfbias=format_number(two_way.get('deltaf_bias_hz', {}).get('mean'), 1),
            crlb=format_number(two_way.get('crlb_tau_ns') or 0.0),
            ratio=format_number(two_way.get('rmse_over_crlb'), 2),
            locked='✔' if manifest.get('coarse_locked') else '✖',
            guard='✔' if manifest.get('guard_hit') else '✖',
            status=manifest.get('status', 'OK'),
        )
        rows.append(row)
    return ''.join(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description='Render README snippet from TDL profile manifests.')
    parser.add_argument('paths', nargs='+', type=Path, help='Manifest files or directories.')
    parser.add_argument('--bullet-only', action='store_true', help='Print only the bullet line.')
    args = parser.parse_args()

    manifests = collect_manifests(args.paths)
    if not manifests:
        print('No manifests found.', flush=True)
        return

    bullet = build_bullet(manifests)
    table = build_table(manifests)

    if args.bullet_only:
        print(bullet)
    else:
        print(bullet)
        print()
        print(table)


if __name__ == '__main__':
    main()
