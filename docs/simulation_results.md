# Driftlock Simulation Results

## Executive Summary

**Driftlock now delivers picosecond-class performance across the entire stack.** Monte Carlo extended run 011 locks a dense 64-node network at **22.13 ps RMS**, while the Choir Simulation Lab acceptance harness reports **45.0 ps RMSE** with an **RMSE/CRLB ratio of 0.83**. Reciprocity calibration collapses hardware bias from ~12 ns to **2.65 ps**, payload coexistence keeps timing within **+8.6%** of the no-payload baseline, and the full deterministic acceptance suite finishes in **3.7 seconds** on an Apple M2 Max (Python 3.12 / NumPy 1.26).

## Network Consensus — Monte Carlo Extended Run 011

| Scenario | KF Gains (clock/freq/iters) | Weighting | Final RMSE (ps) | Δ vs. Baseline |
|----------|-----------------------------|-----------|-----------------|---------------|
| Dense mesh (64 nodes) | 0.32 / 0.03 / 1 | metropolis_var | **22.13** | **−0.33** |
| Dense baseline | — | metropolis | 22.45 | reference |
| Small mesh (25 nodes) | 0.25 / 0.05 / 1 | inverse_variance | **20.96** | **−3.41** |
| Small baseline | — | inverse_variance | 24.38 | reference |

**Calibration sweep:** Loopback reciprocity removes bias down to **2.65 ps** while the uncalibrated run sits near **−12,000 ps**. All presets converge in a single iteration. Sweeps remain reproducible via `scripts/sweep_phase2_kf.py` and are safeguarded by `scripts/verify_kf_sweep.py` plus the seeded regression in `tests/test_consensus.py`.

## Choir Simulation Lab Acceptance (driftlock_sim/sims/run_acceptance.py)

| Check | Metric | Result | Requirement | Status |
|-------|--------|--------|-------------|--------|
| Aperture reconstruction | Δf SNR | **58.06 dB** | ≥ 15 dB | ✅ |
| Aperture reconstruction | 2Δf SNR | **58.07 dB** | informational | ✅ |
| Coherent precision | RMSE | **45.0 ps** | ≤ 120 ps | ✅ |
| Coherent precision | RMSE / CRLB | **0.83** | ≤ 1.5 | ✅ |
| Coherent precision | CI coverage | **100%** | ≥ 90% | ✅ |
| Coherent precision | Unwrap sanity | **100%** | ≥ 95% | ✅ |
| Robustness (@0 dB) | Δf SNR | **28.08 dB** | > 0 dB | ✅ |
| Robustness (@0 dB) | τ̂ | **11.65 ns** | finite | ✅ |
| Payload coexistence | RMSE delta | **+8.6%** | ≤ +25% | ✅ |
| Payload coexistence | Observed BER | **0** | < 1e-3 | ✅ |
| Runtime | Total wall clock | **3.66 s** | < 60 s | ✅ |

The coherent path injects the truth delay at the signal level, weights the WLS slope by per-tone SNR, and reports the CRLB using RMS bandwidth. Payload QPSK tones are demodulated to emit an observed BER alongside the analytic bound.

## Reproducing the Results

```bash
# 1. Full Monte Carlo sweep (dense + small presets)
python scripts/run_mc.py all -c sim/configs/mc_extended.yaml -o results/mc_runs -r extended_011

# 2. Dense combo verification guardrail
scripts/verify_kf_sweep.py results/kf_sweeps/dense_combo_scan/kf_sweep_summary.json \
  --expected-min 20.9337 --expected-best-mean 21.8909 \
  --expected-clock 0.32 --expected-freq 0.03 --expected-iterations 1

# 3. Deterministic acceptance harness (3.7 s on M2 Max)
PYTHONPATH=. python driftlock_sim/sims/run_acceptance.py
```

All scripts are seeded for determinism. The acceptance harness emits:

- `driftlock_sim/outputs/csv/acceptance_summary.json`
- `driftlock_sim/outputs/figs/accept_aperture.png`
- `driftlock_sim/outputs/figs/executive_summary.pdf`

Monte Carlo artifacts live under `results/mc_runs/extended_011/`, including the run manifest (`run_config.json`), human-readable summary (`SUMMARY.md`), and per-preset telemetry (`phase2/dense_network_kf/phase2_results.json`, etc.).

## Key Takeaways

- **22.13 ps** dense-network consensus with reproducible guardrails.
- **20.96 ps** small-network preset delivers a **3.41 ps** edge over the no-KF baseline.
- **58 dB** Δf spike validates the missing-fundamental aperture path with wide margin.
- **RMSE/CRLB 0.83** confirms near-optimal coherent estimator efficiency.
- **Runtime 3.7 s** keeps the acceptance harness deployable inside CI/CD pipelines.

Driftlock is now production-ready: the modelling stack, Monte Carlo harness, and acceptance lab all align on picosecond precision with deterministic, scriptable artifacts for investors, partners, and engineers.
