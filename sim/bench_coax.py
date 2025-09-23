"""Coax bench emulation for quick σ/μ, Allan deviation, and reciprocity sweeps."""

from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from alg.chronometric_handshake import (
    ChronometricHandshakeConfig,
    ChronometricHandshakeSimulator,
    ChronometricNode,
    ChronometricNodeConfig,
)


@dataclass
class PairStatistics:
    pair: Tuple[int, int]
    mean_ps: float
    std_ps: float
    allan_dev_ps: float
    reciprocity_bias_ps: float

    def to_dict(self) -> Dict[str, object]:
        return {
            "pair": list(self.pair),
            "mean_ps": self.mean_ps,
            "std_ps": self.std_ps,
            "allan_dev_ps": self.allan_dev_ps,
            "reciprocity_bias_ps": self.reciprocity_bias_ps,
        }


def _allan_deviation(values: Sequence[float]) -> float:
    if len(values) < 2:
        return float("nan")
    diffs = np.diff(values)
    return float(np.sqrt(np.mean(diffs ** 2) / 2.0))


def _build_nodes(n_nodes: int, rng: np.random.Generator) -> Dict[int, ChronometricNode]:
    base_freq = 2.4e9
    nodes: Dict[int, ChronometricNode] = {}
    for node_id in range(n_nodes):
        ppm = rng.normal(0.0, 0.2)
        clock_bias = rng.normal(0.0, 15e-12)
        phase = rng.uniform(0.0, 2.0 * np.pi)
        nodes[node_id] = ChronometricNode(
            ChronometricNodeConfig(
                node_id=node_id,
                carrier_freq_hz=base_freq,
                phase_offset_rad=phase,
                clock_bias_s=clock_bias,
                freq_error_ppm=ppm,
            )
        )
    return nodes


def _with_offset(node: ChronometricNode, offset_hz: float) -> ChronometricNode:
    cfg = node.config
    return ChronometricNode(
        ChronometricNodeConfig(
            node_id=cfg.node_id,
            carrier_freq_hz=cfg.carrier_freq_hz + offset_hz,
            phase_offset_rad=cfg.phase_offset_rad,
            clock_bias_s=cfg.clock_bias_s,
            freq_error_ppm=cfg.freq_error_ppm,
        )
    )


def _run_pair_trials(
    simulator: ChronometricHandshakeSimulator,
    node_a: ChronometricNode,
    node_b: ChronometricNode,
    rng: np.random.Generator,
    trials: int,
    delta_f_min: float,
    delta_f_max: float,
    snr_db: float,
) -> PairStatistics:
    errors_ps: List[float] = []
    reciprocity_ps: List[float] = []

    for _ in range(trials):
        delta_f = rng.uniform(delta_f_min, delta_f_max)
        offset = delta_f / 2.0
        trial_a = _with_offset(node_a, -offset)
        trial_b = _with_offset(node_b, offset)
        result, _ = simulator.run_two_way(
            node_a=trial_a,
            node_b=trial_b,
            distance_m=0.0,
            snr_db=snr_db,
            rng=rng,
        )
        errors_ps.append((result.tof_est_s - result.tof_true_s) * 1e12)
        reciprocity_ps.append(result.reciprocity_bias_s * 1e12)

    errors_arr = np.asarray(errors_ps)
    reciprocity_arr = np.asarray(reciprocity_ps)
    std_ps = float(np.std(errors_arr, ddof=1)) if errors_arr.size > 1 else 0.0

    return PairStatistics(
        pair=(int(node_a.node_id), int(node_b.node_id)),
        mean_ps=float(np.mean(errors_arr)),
        std_ps=std_ps,
        allan_dev_ps=_allan_deviation(errors_arr),
        reciprocity_bias_ps=float(np.mean(reciprocity_arr)),
    )


def _format_table(stats: Iterable[PairStatistics]) -> str:
    rows = ["pair       μ_ps    σ_ps    allan_ps  bias_ps"]
    for entry in stats:
        rows.append(
            f"{entry.pair!s:<10} "
            f"{entry.mean_ps:6.2f} "
            f"{entry.std_ps:6.2f} "
            f"{entry.allan_dev_ps:8.2f} "
            f"{entry.reciprocity_bias_ps:7.2f}"
        )
    return "\n".join(rows)


def _build_handshake_config(observation_ms: float) -> ChronometricHandshakeConfig:
    observation_s = max(observation_ms, 1.0) / 1_000.0
    beat_duration = min(observation_s / 10.0, 1e-3)
    return ChronometricHandshakeConfig(
        beat_duration_s=beat_duration,
        baseband_rate_factor=8.0,
        min_baseband_rate_hz=50_000.0,
        min_adc_rate_hz=50_000.0,
        filter_relative_bw=1.2,
        phase_noise_psd=-70.0,
        jitter_rms_s=2e-12,
        retune_offsets_hz=(),
        coarse_enabled=False,
        use_phase_slope_fit=False,
        use_theoretical_variance=False,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Coax bench emulation for small-node truthing")
    parser.add_argument("--nodes", type=int, default=4, help="Number of nodes (2–7 recommended)")
    parser.add_argument("--delta-f-min", type=float, default=0.5e6, help="Minimum intentional Δf (Hz)")
    parser.add_argument("--delta-f-max", type=float, default=2.0e6, help="Maximum intentional Δf (Hz)")
    parser.add_argument("--observation-ms", type=float, default=100.0, help="Observation window in milliseconds")
    parser.add_argument("--trials", type=int, default=40, help="Monte Carlo trials per link")
    parser.add_argument("--snr-db", type=float, default=35.0, help="SNR for the coax bench scenario")
    parser.add_argument("--seed", type=int, default=2025, help="RNG seed")
    parser.add_argument("--output", type=Path, help="Optional JSON summary path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.nodes < 2:
        raise ValueError("At least two nodes required for the bench test")

    rng = np.random.default_rng(args.seed)
    nodes = _build_nodes(args.nodes, rng)
    handshake_cfg = _build_handshake_config(args.observation_ms)
    simulator = ChronometricHandshakeSimulator(handshake_cfg)

    stats: List[PairStatistics] = []
    for idx_a, idx_b in itertools.combinations(nodes.keys(), 2):
        stats.append(
            _run_pair_trials(
                simulator,
                nodes[idx_a],
                nodes[idx_b],
                rng,
                args.trials,
                args.delta_f_min,
                args.delta_f_max,
                args.snr_db,
            )
        )

    report = _format_table(stats)
    delta_min_mhz = args.delta_f_min / 1e6
    delta_max_mhz = args.delta_f_max / 1e6
    print(f"Nodes: {args.nodes} | Trials per link: {args.trials}")
    print(f"Δf sweep: {delta_min_mhz:.2f}–{delta_max_mhz:.2f} MHz | Observation window: {args.observation_ms:.1f} ms")
    print(report)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "nodes": args.nodes,
            "trials": args.trials,
            "delta_f_range_hz": [args.delta_f_min, args.delta_f_max],
            "observation_ms": args.observation_ms,
            "snr_db": args.snr_db,
            "seed": args.seed,
            "statistics": [entry.to_dict() for entry in stats],
        }
        args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Saved summary to {args.output}")


if __name__ == "__main__":
    main()
