#!/usr/bin/env python3
"""Run joint Project Aperture × Formant sweeps.

This helper orchestrates Chronometric handshake diagnostics across a grid of
Pathfinder pre-guard/aperture settings and vowel-coded coarse preambles.
Results are written to timestamped directories under
``results/project_aperture_formant/<profile>/<experiment_tag>/`` alongside a
Markdown summary capturing the key metrics requested by the exploration brief.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional


@dataclass(frozen=True)
class SweepScenario:
    """Parameter bundle for a single Aperture/Formant experiment."""

    name: str
    pathfinder_pre_guard_ns: float
    pathfinder_guard_interval_ns: float
    pathfinder_aperture_duration_ns: float
    coarse_formant_profile: str
    coarse_formant_fundamental_hz: float
    coarse_formant_harmonic_count: int
    coarse_formant_include_fundamental: bool = False
    coarse_formant_scale: float = 1_000.0
    coarse_formant_phase_jitter: float = 0.0
    missing_fundamental_enabled: bool = True
    pathfinder_relative_threshold_db: Optional[float] = None
    pathfinder_noise_guard_multiplier: Optional[float] = None
    pathfinder_first_path_blend: Optional[float] = None
    pathfinder_use_simple_search: Optional[bool] = None


# Default sweep grid exploring a small set of aperture/formant trade-offs.
DEFAULT_SCENARIOS: List[SweepScenario] = [
    SweepScenario(
        name="baseline_guarded_formant25k",
        pathfinder_pre_guard_ns=0.0,
        pathfinder_guard_interval_ns=30.0,
        pathfinder_aperture_duration_ns=100.0,
        coarse_formant_profile="A",
        coarse_formant_fundamental_hz=25_000.0,
        coarse_formant_harmonic_count=12,
    ),
    SweepScenario(
        name="pre_guard20_aperture120_fund22k",
        pathfinder_pre_guard_ns=20.0,
        pathfinder_guard_interval_ns=45.0,
        pathfinder_aperture_duration_ns=120.0,
        coarse_formant_profile="E",
        coarse_formant_fundamental_hz=22_000.0,
        coarse_formant_harmonic_count=10,
        coarse_formant_phase_jitter=0.15,
    ),
    SweepScenario(
        name="pre_guard40_aperture160_fund30k",
        pathfinder_pre_guard_ns=40.0,
        pathfinder_guard_interval_ns=60.0,
        pathfinder_aperture_duration_ns=160.0,
        coarse_formant_profile="I",
        coarse_formant_fundamental_hz=30_000.0,
        coarse_formant_harmonic_count=14,
    ),
    SweepScenario(
        name="tight_guard_include_f0_fund18k",
        pathfinder_pre_guard_ns=10.0,
        pathfinder_guard_interval_ns=25.0,
        pathfinder_aperture_duration_ns=70.0,
        coarse_formant_profile="O",
        coarse_formant_fundamental_hz=18_000.0,
        coarse_formant_harmonic_count=8,
        coarse_formant_include_fundamental=True,
    ),
    SweepScenario(
        name="pre_guard15_guard35_fund24k",
        pathfinder_pre_guard_ns=15.0,
        pathfinder_guard_interval_ns=35.0,
        pathfinder_aperture_duration_ns=110.0,
        coarse_formant_profile="A",
        coarse_formant_fundamental_hz=24_000.0,
        coarse_formant_harmonic_count=12,
    ),
    SweepScenario(
        name="pre_guard30_guard50_aperture140_fund24k",
        pathfinder_pre_guard_ns=30.0,
        pathfinder_guard_interval_ns=50.0,
        pathfinder_aperture_duration_ns=140.0,
        coarse_formant_profile="U",
        coarse_formant_fundamental_hz=24_000.0,
        coarse_formant_harmonic_count=12,
    ),
]


def _run_handshake(
    profile: str,
    scenario: SweepScenario,
    num_trials: int,
    rng_seed: int,
    output_json: Path,
    python_executable: str,
    extra_args: Optional[List[str]] = None,
) -> Dict[str, object]:
    """Invoke ``run_handshake_diag.py`` with the scenario parameters."""

    cmd: List[str] = [
        python_executable,
        "scripts/run_handshake_diag.py",
        "--channel-profile",
        profile,
        "--num-trials",
        str(num_trials),
        "--rng-seed",
        str(rng_seed),
        "--coarse-preamble-mode",
        "formant",
        "--coarse-formant-profile",
        scenario.coarse_formant_profile,
        "--coarse-formant-fundamental-hz",
        f"{scenario.coarse_formant_fundamental_hz}",
        "--coarse-formant-harmonic-count",
        str(scenario.coarse_formant_harmonic_count),
        "--coarse-formant-scale",
        f"{scenario.coarse_formant_scale}",
        "--coarse-formant-phase-jitter",
        f"{scenario.coarse_formant_phase_jitter}",
        "--pathfinder-pre-guard-ns",
        f"{scenario.pathfinder_pre_guard_ns}",
        "--pathfinder-guard-interval-ns",
        f"{scenario.pathfinder_guard_interval_ns}",
        "--pathfinder-aperture-duration-ns",
        f"{scenario.pathfinder_aperture_duration_ns}",
        "--output-json",
        str(output_json),
    ]

    if scenario.coarse_formant_include_fundamental:
        cmd.append("--coarse-formant-include-fundamental")
    if not scenario.missing_fundamental_enabled:
        cmd.append("--disable-formant-missing-fundamental")
    if scenario.pathfinder_relative_threshold_db is not None:
        cmd.extend([
            "--pathfinder-relative-threshold-db",
            f"{scenario.pathfinder_relative_threshold_db}",
        ])
    if scenario.pathfinder_noise_guard_multiplier is not None:
        cmd.extend([
            "--pathfinder-noise-guard-multiplier",
            f"{scenario.pathfinder_noise_guard_multiplier}",
        ])
    if scenario.pathfinder_first_path_blend is not None:
        cmd.extend([
            "--pathfinder-first-path-blend",
            f"{scenario.pathfinder_first_path_blend}",
        ])
    if scenario.pathfinder_use_simple_search is False:
        cmd.append("--pathfinder-disable-simple-search")
    if extra_args:
        cmd.extend(extra_args)

    subprocess.run(cmd, check=True)

    with output_json.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _get_stat(stats: Optional[Dict[str, float]], key: str) -> Optional[float]:
    if not stats:
        return None
    return stats.get(key)


def _format_float(value: Optional[float], digits: int = 3) -> str:
    if value is None or (isinstance(value, float) and not (value == value)):
        return "-"
    return f"{value:.{digits}f}"


def _format_distribution(dist: Optional[Dict[str, int]]) -> str:
    if not dist:
        return "-"
    total = sum(dist.values())
    if total <= 0:
        return "-"
    parts = []
    for label, count in sorted(dist.items(), key=lambda item: (-item[1], item[0])):
        fraction = 100.0 * count / total
        parts.append(f"{label}:{fraction:.0f}%")
    return " ".join(parts)


def _append_markdown_table(
    lines: List[str],
    profile: str,
    experiment_tag: str,
    scenarios: List[SweepScenario],
    summaries: List[Dict[str, object]],
) -> None:
    lines.append(f"# Project Aperture × Formant Sweep – {profile} – {experiment_tag}")
    lines.append("")
    header = (
        "| scenario | pre_guard_ns | guard_ns | aperture_ns | fundamental_hz | harmonics | include_f0 | missing_f0_mode | first_path_blend | noise_guard_mult | rel_threshold_db | "
        "forward_tau_bias_ns | reverse_tau_bias_ns | two_way_tau_bias_ns | "
        "forward_first_path_ns | reverse_first_path_ns | forward_coarse_lock | reverse_coarse_lock | "
        "forward_guard_hit | reverse_guard_hit | forward_alias | reverse_alias | "
        "forward_formant_labels | reverse_formant_labels | forward_missing_f0_hz | reverse_missing_f0_hz |"
    )
    lines.append(header)
    lines.append(
        "| --- | ---: | ---: | ---: | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | "
        "---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | ---: | ---: |"
    )

    for scenario, summary in zip(scenarios, summaries):
        forward = summary["directional_metrics"]["forward"]
        reverse = summary["directional_metrics"]["reverse"]
        two_way = summary["two_way_metrics"]

        row = "| {scenario} | {pre_guard} | {guard} | {aperture} | {fundamental} | {harmonics} | {include_f0} | " \
            "{missing_mode} | {first_path_blend} | {noise_guard} | {rel_threshold} | {f_tau_bias} | {r_tau_bias} | {tw_tau_bias} | " \
            "{f_first_path} | {r_first_path} | {f_coarse_lock} | " \
            "{r_coarse_lock} | {f_guard} | {r_guard} | {f_alias} | {r_alias} | {f_labels} | {r_labels} | " \
            "{f_missing} | {r_missing} |".format(
                scenario=scenario.name,
                pre_guard=_format_float(scenario.pathfinder_pre_guard_ns, 1),
                guard=_format_float(scenario.pathfinder_guard_interval_ns, 1),
                aperture=_format_float(scenario.pathfinder_aperture_duration_ns, 1),
                fundamental=_format_float(scenario.coarse_formant_fundamental_hz, 1),
                harmonics=scenario.coarse_formant_harmonic_count,
                include_f0="yes" if scenario.coarse_formant_include_fundamental else "no",
                missing_mode="enabled" if scenario.missing_fundamental_enabled else "disabled",
                first_path_blend=_format_float(scenario.pathfinder_first_path_blend),
                noise_guard=_format_float(scenario.pathfinder_noise_guard_multiplier),
                rel_threshold=_format_float(scenario.pathfinder_relative_threshold_db),
                f_tau_bias=_format_float(_get_stat(forward["tau_bias_ns"], "mean")),
                r_tau_bias=_format_float(_get_stat(reverse["tau_bias_ns"], "mean")),
                tw_tau_bias=_format_float(_get_stat(two_way["tau_bias_ns"], "mean")),
                f_first_path=_format_float(_get_stat(forward["first_path_error_ns"], "mean")),
                r_first_path=_format_float(_get_stat(reverse["first_path_error_ns"], "mean")),
                f_coarse_lock=_format_float(forward.get("coarse_locked_rate")),
                r_coarse_lock=_format_float(reverse.get("coarse_locked_rate")),
                f_guard=_format_float(forward.get("guard_hit_rate")),
                r_guard=_format_float(reverse.get("guard_hit_rate")),
                f_alias=_format_float(forward.get("alias_success_rate")),
                r_alias=_format_float(reverse.get("alias_success_rate")),
                f_labels=_format_distribution(forward.get("pathfinder_formant_labels")),
                r_labels=_format_distribution(reverse.get("pathfinder_formant_labels")),
                f_missing=_format_float(_get_stat(forward.get("pathfinder_missing_fundamental_hz"), "mean")),
                r_missing=_format_float(_get_stat(reverse.get("pathfinder_missing_fundamental_hz"), "mean")),
            )
        lines.append(row)

    lines.append("")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profiles",
        nargs="+",
        default=["IDEAL", "URBAN_CANYON"],
        help="TDL profiles to evaluate (default: IDEAL URBAN_CANYON)",
    )
    parser.add_argument(
        "--scenario",
        dest="scenario_names",
        action="append",
        default=None,
        help="Restrict the sweep to one or more named scenarios (may be repeated).",
    )
    parser.add_argument(
        "--grid-file",
        type=Path,
        default=None,
        help="Optional JSON file describing custom scenarios.",
    )
    parser.add_argument(
        "--num-trials",
        type=int,
        default=120,
        help="Number of Monte Carlo trials fed to each handshake diagnostic.",
    )
    parser.add_argument(
        "--rng-seed",
        type=int,
        default=2025,
        help="Base RNG seed shared across scenarios.",
    )
    parser.add_argument(
        "--experiment-tag",
        type=str,
        default=None,
        help="Override the ISO8601 experiment tag used for artifact directories.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("results") / "project_aperture_formant",
        help="Root directory for sweep outputs.",
    )
    parser.add_argument(
        "--python",
        type=str,
        default=sys.executable,
        help="Python interpreter used to invoke run_handshake_diag.py.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Log planned commands without executing them.",
    )
    parser.add_argument(
        "--extra-arg",
        dest="extra_args",
        action="append",
        default=None,
        help="Additional CLI arguments forwarded to run_handshake_diag.py (may be repeated).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    scenarios: List[SweepScenario]

    if args.grid_file is not None:
        payload = json.loads(args.grid_file.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise SystemExit("Grid file must contain a JSON list of scenarios")

        scenarios = []
        for entry in payload:
            if not isinstance(entry, dict):
                raise SystemExit("Each grid entry must be a JSON object")
            scenarios.append(SweepScenario(**entry))
    else:
        scenarios = DEFAULT_SCENARIOS

    if args.scenario_names:
        requested = set(args.scenario_names)
        available = {scenario.name: scenario for scenario in scenarios}
        missing = sorted(requested.difference(available))
        if missing:
            raise SystemExit(f"Unknown scenario(s): {', '.join(missing)}")
        scenarios = [scenario for scenario in scenarios if scenario.name in requested]

    if not scenarios:
        raise SystemExit("No scenarios selected for sweep")


    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    experiment_tag = args.experiment_tag or f"{timestamp}_aperture_formant"

    extra_args = args.extra_args or []

    for profile in args.profiles:
        profile_dir = args.output_root / profile / experiment_tag
        profile_dir.mkdir(parents=True, exist_ok=True)

        summaries: List[Dict[str, object]] = []
        for scenario in scenarios:
            output_json = profile_dir / f"{scenario.name}.json"
            if args.dry_run:
                print(f"DRY RUN: would execute sweep for {profile} :: {scenario.name} -> {output_json}")
                summaries.append({})
                continue

            summary = _run_handshake(
                profile=profile,
                scenario=scenario,
                num_trials=args.num_trials,
                rng_seed=args.rng_seed,
                output_json=output_json,
                python_executable=args.python,
                extra_args=extra_args,
            )
            summaries.append(summary)
            print(f"Saved {profile} :: {scenario.name} -> {output_json}")

        if args.dry_run:
            continue

        summary_path = profile_dir / "summary.md"
        lines: List[str] = []
        _append_markdown_table(lines, profile=profile, experiment_tag=experiment_tag, scenarios=scenarios, summaries=summaries)
        summary_path.write_text("\n".join(lines), encoding="utf-8")
        print(f"Wrote summary to {summary_path}")


if __name__ == "__main__":
    main()
