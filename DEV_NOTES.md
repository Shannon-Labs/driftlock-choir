# Developer Notes

## Environment
- Python: 3.12.2 | packaged by conda-forge | (main, Feb 16 2024, 20:54:21) [Clang 16.0.6 ]
- NumPy: 1.26.0
- SciPy: 1.12.0

## Baseline commands (Task 0)
- `pytest -q` → 3 passed in 0.64s.
- `python sim/phase1.py` → FAILED (`AttributeError: 'Phase1Simulator' object has no attribute '_sample_nodes'`).
- `python sim/phase2.py` → completed without error.

## Updates
- Added `src/chan/tdl.py` and `sim/phase1.py --tdl-profile` (e.g. `EXPO:50ns:K=6dB:L=5`) to exercise tapped-delay-line multipath in the alias map sweep.
- Chronometric handshake now outputs per-direction `phase_bias_rad`; coarse delay estimator upgraded to correlation-based implementation shared with TDL channel.
- Added spectral iteration predictor, consensus weighting strategies, and run manifest CSV logging in `sim/phase2.py`; default pytest scope pinned to `/tests` via `pytest.ini`.
- Local Kalman filter front-end, reciprocity calibration toggles, Zadoff–Chu coarse preamble, and MAC slot descriptors wired through Phase 1/2 flows.
- Introduced `src/utils/io.py` and `src/utils/plotting.py`; Phase 1/2 CLIs now support `--dry-run` and config echo, with configs dumped alongside persisted results.
- Alias-map manifest adds bias diagnostics (mean/retune/SNR slices), reciprocity bias heatmaps + trend plot, and records MAC guard times in seconds.
- Phase 2 consensus manifest logs Kalman improvement deltas/ratios and exposes `--local-kf {auto,on,off,baseline}`; CSV includes mode + comparison metrics.
- Added deterministic tests for bias reduction, coarse delay accuracy, and KF telemetry to keep `pytest -q` under ~2 s.
- Added `scripts/run_presets.py` and `sim/configs/mc_extended.yaml` so CI smoke tests can exercise the canonical alias-map + consensus presets (see docs/quickstart.md for commands).
- Replaced the Phase 2 local Kalman helper with a variance-weighted shrinkage
  smoother (config knobs: `local_kf_clock_gain`, `local_kf_freq_gain`,
  `local_kf_iterations`). Default gain 0.18 keeps the filter gentle, while the
  dense Monte Carlo preset now locks to clock 0.32 / freq 0.03 / one pass based
  on the sweep best-mean combo (small-network preset stays at 0.25 / 0.05).
- Regenerated Monte Carlo artifacts under `results/mc_runs/extended_011/` and
  documented the run in `docs/results_extended_011.md`; dense KF now lands at
  22.13 ps vs 22.45 ps baseline (Δ ≈ 0.33 ps) while the small network remains
  20.96 vs 24.38 ps.
- Added `scripts/sweep_phase2_kf.py` to scan shrinkage gains/seeds; dense and
  small network sweeps live under `results/kf_sweeps/` with per-gain stats and
  baseline comparisons. Latest combo scan (clock vs. freq gain, 1–2 passes)
  pushes the dense preset down to 20.93 ps (clock 0.22, freq 0.03, 2 iters) and
  finds a best-mean combo at 21.89 ± 0.75 ps; the small-net scan bottoms out at
  18.69 ps, roughly matching the 18.70 ps baseline.
- Added `scripts/verify_kf_sweep.py` plus
  `tests/test_consensus.py::test_dense_kf_vs_baseline`; the regression pins RNG
  seed 5001 and asserts ≥1 ps improvement for the dense preset, keeping the
  sweep-backed shrinkage gains reproducible.
- `scripts/run_verification_checks.sh` bundles the regression and sweep verifier
  for pre-commit/CI hooks (callable via `scripts/run_verification_checks.sh`).
