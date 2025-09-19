# Results Overview

This repository emits artifacts into `results/phase*/` when the Phase simulators are executed with `save_results=True` (default). The table below summarizes what to expect.

| Phase | Directory | Files | Description |
|-------|-----------|-------|-------------|
| Phase 1 | `results/phase1/` | `phase1_results.json` | SNR sweep telemetry (RMSE, alias success, coarse delay accuracy). |
|       |           | `phase1_waveforms.png` | Example beat/phase plots when trace capture is enabled. |
|       |           | `phase1_errors.png` | RMSE vs. SNR plot for ToF/clock/Δf. |
| Phase 2 | `results/phase2/` | `phase2_results.json` | Consensus telemetry (λ
d, RMS vs. iteration, residuals, diagnostic summary). |
|       |           | `phase2_topology.png` | Topology heatmap with clock-offset coloration. |
|       |           | `phase2_convergence.png` | Timing/frequency RMS trajectories. |
|       |           | `phase2_residuals.png` | Histograms of per-edge residuals. |

To regenerate the default artifacts:

```bash
# Phase 1: quick LOS sweep (disable plots with --plot-results False)
python sim/phase1.py

# Phase 2: 20-node sandbox saving PNG/JSON outputs
python sim/phase2.py
```

Set `save_results=False` or `plot_results=False` in the corresponding `Phase*Config` if you want to suppress file creation.
