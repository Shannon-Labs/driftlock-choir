# Slide 10 — Evidence: Phase 1 Handshakes

Title: Estimation holds the line across harsh channels

Bullets
- Plot: `driftlock_choir_sim/outputs/phase1_enhancements/figures/wls_performance_improvement.png` — weighted LS matches CRLB within 1.3× from -5 to 25 dB SNR.
- Alias resolution >95% in Monte Carlo sweeps; tau variance floor pinned at 50 ps with coarse preambles.
- Two-ray and jitter injections show graceful degradation; confidence bands via 500-sample bootstrap.
- Config reproducibility: `sim/configs/default.yaml` (phase1 block) with deterministic RNG seeds.

Notes for presenter
- Bring up the plot; call out how we bound variance even in low SNR ranges.
- Mention telemetry exports for diligence (`results/smoke_phase1_atomic/phase1_results.json`).
- Tie back to API roadmap: these estimators will power automated remediation suggestions.
