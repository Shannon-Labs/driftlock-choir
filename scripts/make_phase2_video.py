#!/usr/bin/env python3
"""
Make a convergence video from a Phase 2 consensus run.

Two modes:
  1) Replay a recorded run config and positions, re-run consensus to obtain
     per-iteration states, and animate node clock offsets.
  2) If only JSONL is provided, select a record (best or by index).

Outputs MP4 (if ffmpeg available) or GIF fallback under results/videos/.

Usage examples:
  python scripts/make_phase2_video.py \
    --source-jsonl results/phase2/phase2_runs.jsonl --find-best

  python scripts/make_phase2_video.py \
    --source-jsonl results/phase2/phase2_runs.jsonl --record-index 1 \
    --out results/videos/phase2_best.mp4
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation

# Ensure local src/ is importable
import sys
# Ensure both repo root and src/ are importable
_HERE = os.path.dirname(__file__)
sys.path.append(os.path.abspath(os.path.join(_HERE, '..')))
sys.path.append(os.path.abspath(os.path.join(_HERE, '..', 'src')))

from alg.consensus import ConsensusOptions, DecentralizedChronometricConsensus
from sim.phase2 import Phase2Config, Phase2Simulation


def _load_record(path: str, idx: int | None, find_best: bool) -> Dict[str, Any]:
    with open(path, 'r', encoding='utf-8') as f:
        lines = [ln for ln in f if ln.strip()]
    if not lines:
        raise RuntimeError(f'No records in {path}')

    if find_best:
        best = None
        best_i = None
        for i, ln in enumerate(lines):
            try:
                d = json.loads(ln)
            except json.JSONDecodeError:
                continue
            rms = d.get('consensus', {}).get('timing_rms_ps', [])
            if rms:
                final = float(rms[-1])
                if best is None or final < best:
                    best = final
                    best_i = i
        if best_i is None:
            raise RuntimeError('Could not find a valid record with timing_rms_ps')
        return json.loads(lines[best_i])

    if idx is None:
        idx = 0
    if idx < 0 or idx >= len(lines):
        raise IndexError(f'--record-index {idx} out of range [0, {len(lines)-1}]')
    return json.loads(lines[idx])


def _reconstruct_and_run(record: Dict[str, Any], *, epsilon: float | None = None, tolerance_ps: float | None = None, max_iters: int | None = None, spectral_margin: float | None = None, init_clock_ps: float | None = None, init_freq_hz: float | None = None):
    cfg_data = record['config']
    # Build Phase2Config from saved fields (use defaults where absent)
    cfg = Phase2Config(
        n_nodes=int(cfg_data.get('n_nodes', 50)),
        area_size_m=float(cfg_data.get('area_size_m', 500.0)),
        comm_range_m=float(cfg_data.get('comm_range_m', 180.0)),
        snr_db=float(cfg_data.get('snr_db', 20.0)),
        base_carrier_hz=float(cfg_data.get('base_carrier_hz', 2.4e9)),
        freq_offset_span_hz=float(cfg_data.get('freq_offset_span_hz', 80e3)),
        handshake_delta_f_hz=float(cfg_data.get('handshake_delta_f_hz', 100e3)),
        retune_offsets_hz=tuple(cfg_data.get('retune_offsets_hz', (1e6,))),
        coarse_enabled=bool(cfg_data.get('coarse_enabled', True)),
        coarse_bandwidth_hz=float(cfg_data.get('coarse_bandwidth_hz', 20e6)),
        coarse_duration_s=float(cfg_data.get('coarse_duration_s', 5e-6)),
        coarse_variance_floor_ps=float(cfg_data.get('coarse_variance_floor_ps', 50.0)),
        max_iterations=int(cfg_data.get('max_iterations', 1000)),
        timestep_s=float(cfg_data.get('timestep_s', 1e-3)),
        convergence_threshold_ps=float(cfg_data.get('convergence_threshold_ps', 100.0)),
        asynchronous=bool(cfg_data.get('asynchronous', False)),
        rng_seed=int(cfg_data.get('rng_seed', 42)) if cfg_data.get('rng_seed') is not None else 42,
        spectral_margin=float(cfg_data.get('spectral_margin', 0.8)),
        epsilon_override=(
            None if cfg_data.get('epsilon_override') in (None, 'null') else float(cfg_data.get('epsilon_override'))
        ),
        weighting=str(cfg_data.get('weighting', 'inverse_variance')),
        results_dir=str(cfg_data.get('results_dir', 'results/phase2')),
        save_results=False,
        plot_results=False,
        local_kf_enabled=bool(cfg_data.get('local_kf_enabled', True)),
        local_kf_sigma_T_ps=float(cfg_data.get('local_kf_sigma_T_ps', 10.0)),
        local_kf_sigma_f_hz=float(cfg_data.get('local_kf_sigma_f_hz', 5.0)),
        local_kf_init_var_T_ps=float(cfg_data.get('local_kf_init_var_T_ps', 1e5)),
        local_kf_init_var_f_hz=float(cfg_data.get('local_kf_init_var_f_hz', 1e3)),
        local_kf_max_abs_ps=float(cfg_data.get('local_kf_max_abs_ps', 500.0)),
        local_kf_max_abs_freq_hz=float(cfg_data.get('local_kf_max_abs_freq_hz', 1e5)),
        local_kf_clock_gain=float(cfg_data.get('local_kf_clock_gain', 0.18)),
        local_kf_freq_gain=float(cfg_data.get('local_kf_freq_gain', 0.05)),
        local_kf_iterations=int(cfg_data.get('local_kf_iterations', 1)),
        baseline_mode=bool(record.get('baseline_mode', False)),
    )

    # Apply CLI overrides to encourage more iterations if requested
    if max_iters is not None:
        cfg.max_iterations = int(max_iters)
    if tolerance_ps is not None:
        cfg.convergence_threshold_ps = float(tolerance_ps)
    if spectral_margin is not None:
        cfg.spectral_margin = float(spectral_margin)
    if epsilon is not None:
        cfg.epsilon_override = float(epsilon)

    sim = Phase2Simulation(cfg)

    # Use recorded positions if available for visual continuity.
    positions_raw = record.get('network', {}).get('positions')
    if not positions_raw:
        positions = sim._sample_positions()
    else:
        positions = {int(k): np.array(v, dtype=float) for k, v in positions_raw.items()}

    graph = sim._build_graph(positions)
    nodes = sim._build_nodes()
    sim._populate_measurements(graph, nodes, positions)
    true_state = sim._oracle_state(nodes)
    initial_state, _ = sim._prepare_initial_state(graph, true_state)
    # Optionally inflate the initial state to show more activity
    if (init_clock_ps and init_clock_ps > 0) or (init_freq_hz and init_freq_hz > 0):
        rng = np.random.default_rng(999)
        N = initial_state.shape[0]
        if init_clock_ps and init_clock_ps > 0:
            initial_state[:, 0] = (rng.normal(0.0, init_clock_ps * 1e-12, size=N))
        if init_freq_hz and init_freq_hz > 0:
            initial_state[:, 1] = (rng.normal(0.0, init_freq_hz, size=N))
        # Enforce zero-mean gauge
        initial_state[:, 0] -= np.mean(initial_state[:, 0])
        initial_state[:, 1] -= np.mean(initial_state[:, 1])

    options = ConsensusOptions(
        max_iterations=cfg.max_iterations,
        epsilon=cfg.epsilon_override,
        tolerance_ps=cfg.convergence_threshold_ps,
        asynchronous=cfg.asynchronous,
        rng_seed=None if cfg.rng_seed is None else cfg.rng_seed + 1,
        enforce_zero_mean=True,
        spectral_margin=cfg.spectral_margin,
    )
    solver = DecentralizedChronometricConsensus(graph, options)
    result = solver.run(initial_state, true_state)
    # Returns: state_history [T, N, 2], timing_rms_ps [T]
    return {
        'graph': graph,
        'positions': positions,
        'state_history': result.state_history,
        'timing_rms_ps': result.timing_rms_ps,
        'timestep_s': cfg.timestep_s,
    }


def _select_writer():
    # Prefer ffmpeg if available
    from shutil import which
    if which('ffmpeg'):
        try:
            return animation.FFMpegWriter(fps=10, bitrate=2000)
        except Exception:
            pass
    try:
        from matplotlib.animation import PillowWriter
        return PillowWriter(fps=10)
    except Exception as e:
        raise RuntimeError('No video writer available (ffmpeg or Pillow).') from e


def _animate(payload: Dict[str, Any], out_path: str, dpi: int = 150, fps: int = 10, duration_s: float = 5.0) -> str:
    graph = payload['graph']
    positions = payload['positions']
    states = payload['state_history']  # [T, N, 2]
    rms = payload['timing_rms_ps']     # [T]
    T, N, _ = states.shape

    coords = np.array([positions[i] for i in range(N)], dtype=float)
    edges = list(graph.edges())

    # Color scale from full history in ps
    clocks_ps = states[:, :, 0] * 1e12
    vabs = float(np.max(np.abs(clocks_ps))) if clocks_ps.size else 1.0
    vlim = max(vabs, 1.0)

    # Build interpolated timeline to fill requested duration
    total_frames = max(int(fps * max(duration_s, 0.1)), 1)
    if T <= 1:
        interp_clocks = np.repeat(clocks_ps, total_frames, axis=0)
        interp_rms = np.repeat(rms, total_frames)
        iter_x = np.zeros(total_frames)
    else:
        steps = T - 1
        frames_per_step = max(int(np.ceil(total_frames / steps)), 1)
        total_frames = frames_per_step * steps + 1
        interp_clocks = np.zeros((total_frames, N))
        interp_rms = np.zeros(total_frames)
        iter_x = np.zeros(total_frames)
        idx = 0
        for s in range(steps):
            a0 = clocks_ps[s]
            a1 = clocks_ps[s + 1]
            r0 = rms[s]
            r1 = rms[s + 1]
            for k in range(frames_per_step):
                alpha = k / frames_per_step
                interp_clocks[idx] = (1 - alpha) * a0 + alpha * a1
                interp_rms[idx] = (1 - alpha) * r0 + alpha * r1
                iter_x[idx] = s + alpha
                idx += 1
        # final keyframe
        interp_clocks[idx] = clocks_ps[-1]
        interp_rms[idx] = rms[-1]
        iter_x[idx] = float(T - 1)

    fig = plt.figure(figsize=(10, 5), dpi=dpi)
    gs = fig.add_gridspec(1, 2, width_ratios=[2.0, 1.2])
    ax_net = fig.add_subplot(gs[0, 0])
    ax_rmse = fig.add_subplot(gs[0, 1])

    # Static edges
    for u, v in edges:
        xy = np.vstack((coords[u], coords[v]))
        ax_net.plot(xy[:, 0], xy[:, 1], color='lightgray', linewidth=0.7, zorder=1)
    sc = ax_net.scatter(coords[:, 0], coords[:, 1], c=interp_clocks[0], cmap='coolwarm', vmin=-vlim, vmax=vlim, s=36, zorder=2, edgecolors='k', linewidths=0.2)
    cb = fig.colorbar(sc, ax=ax_net, fraction=0.046, pad=0.04)
    cb.set_label('Clock offset (ps)')
    ax_net.set_title('Consensus Convergence (clock offsets)')
    ax_net.set_xlabel('x (m)')
    ax_net.set_ylabel('y (m)')
    ax_net.set_aspect('equal', adjustable='box')
    # Big RMSE overlay
    rmse_text = ax_net.text(0.02, 0.98, f"RMSE: {interp_rms[0]:.2f} ps", transform=ax_net.transAxes,
                            va='top', ha='left', fontsize=16,
                            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7, edgecolor='none'))

    # RMSE trace
    iters = np.arange(len(rms))
    line_rmse, = ax_rmse.plot(iters, rms, marker='o', color='tab:blue')
    marker, = ax_rmse.plot([0], [rms[0]], 'o', color='tab:red')
    ax_rmse.set_title('Timing RMSE vs iteration')
    ax_rmse.set_xlabel('Iteration')
    ax_rmse.set_ylabel('RMSE (ps)')
    ax_rmse.grid(True, alpha=0.3)

    def update(i: int):
        sc.set_array(interp_clocks[i])
        rmse_text.set_text(f"RMSE: {interp_rms[i]:.2f} ps")
        # Move marker along true iterations
        marker.set_data([iter_x[i]], [interp_rms[i]])
        return sc, marker, rmse_text

    ani = animation.FuncAnimation(fig, update, frames=len(interp_clocks), interval=1000 // max(fps, 1), blit=False)
    writer = _select_writer()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    ani.save(out_path, writer=writer, dpi=dpi)
    plt.close(fig)
    return out_path


def main():
    parser = argparse.ArgumentParser(description='Create a Phase 2 consensus convergence video')
    parser.add_argument('--source-jsonl', type=str, default='results/phase2/phase2_runs.jsonl', help='Path to phase2 JSONL telemetry')
    parser.add_argument('--record-index', type=int, default=None, help='Record index to use (0-based)')
    parser.add_argument('--find-best', action='store_true', help='Pick record with the lowest final timing RMSE')
    parser.add_argument('--out', type=str, default=None, help='Output video path (.mp4 or .gif). Default auto under results/videos/.')
    parser.add_argument('--duration-s', type=float, default=5.0, help='Target video duration in seconds (approx).')
    parser.add_argument('--dpi', type=int, default=150, help='Figure DPI')
    parser.add_argument('--fps', type=int, default=10, help='Frames per second')
    parser.add_argument('--epsilon', type=float, default=None, help='Override consensus step size (smaller -> slower convergence)')
    parser.add_argument('--tolerance-ps', type=float, default=None, help='Override convergence tolerance (set negative to disable early stop)')
    parser.add_argument('--max-iters', type=int, default=None, help='Override maximum iterations for consensus')
    parser.add_argument('--spectral-margin', type=float, default=None, help='Override spectral margin used to compute epsilon when not set')
    parser.add_argument('--init-clock-ps', type=float, default=None, help='Set initial per-node clock offsets RMS (ps) for a more dynamic video')
    parser.add_argument('--init-freq-hz', type=float, default=None, help='Set initial per-node frequency offsets RMS (Hz) for a more dynamic video')
    args = parser.parse_args()

    record = _load_record(args.source_jsonl, args.record_index, args.find_best)
    pack = _reconstruct_and_run(
        record,
        epsilon=args.epsilon,
        tolerance_ps=args.tolerance_ps,
        max_iters=args.max_iters,
        spectral_margin=args.spectral_margin,
        init_clock_ps=args.init_clock_ps,
        init_freq_hz=args.init_freq_hz,
    )

    # Compose default out path
    final_rmse = float(record.get('consensus', {}).get('timing_rms_ps', [0])[-1]) if record.get('consensus') else None
    suffix = 'mp4'
    try:
        _ = _select_writer()
        # If PillowWriter selected, produce GIF instead
        from matplotlib.animation import PillowWriter
        if isinstance(_, PillowWriter):
            suffix = 'gif'
    except Exception:
        suffix = 'gif'

    out_path = args.out or os.path.join('results', 'videos', f'phase2_run_{final_rmse:.1f}ps.{suffix}' if final_rmse is not None else f'phase2_run.{suffix}')
    saved = _animate(pack, out_path, dpi=args.dpi, fps=args.fps, duration_s=args.duration_s)
    print(saved)


if __name__ == '__main__':
    main()
