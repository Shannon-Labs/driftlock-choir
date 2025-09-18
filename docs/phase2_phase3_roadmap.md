# Phase 2 & Phase 3 Roadmap

## Phase 2 – Network Consensus Demonstration

### Objectives
- Simulate decentralized convergence for `N≈50` nodes using the Chronometric Interferometry handshake results as pairwise measurements.
- Implement Equation (4) consensus update with variance-aware weights and validate sub-100 ps RMS convergence within 5 ms.

### Architecture Changes
1. **Reusable Handshake Export**
   - Expose `simulate_handshake_pair(node_i, node_j, snr_db)` from Phase 1 to return `(τ̂_ij, Δf̂_ij, σ²_τ, σ²_Δf)` for any distance/noise.
   - Cache measurement statistics for repeated use in Monte Carlo loops.
2. **Network Model Integration**
   - Replace ad-hoc adjacency handling with NetworkX `random_geometric_graph` (while keeping existing utilities for plotting).
   - Annotate each edge with ground-truth `d_ij`, measurement noise covariance, and the most recent handshake estimate.
3. **Consensus Core**
   - Implement `DecentralizedChronometricConsensus` in `src/alg/consensus.py`:
     - State per node: `[ΔT_i, Δf_i]`.
     - Edge messages: `d_ij = [τ̂_ij - τ̂_ji, Δf̂_ij - Δf̂_ji]` per Equation (4).
     - Adaptive stepsize `ε` derived from Laplacian spectral radius (compute via sparse eigs).
     - Weight matrix `W_ij = diag(1/σ²_τ, 1/σ²_Δf)` (inverse measurement covariance).
   - Support synchronous (vectorized) and asynchronous (random node updates) modes.
4. **Simulation Harness (`sim/phase2.py`)**
   - Scenario generator: positions, oscillator ppm draws, initial clock offsets/drifts.
   - Loop over iterations (target 1 kHz rate → `Δt = 1 ms`) for up to 1000 iterations.
   - Metrics:
     - RMS timing error vs. iteration relative to oracle midline.
     - Frequency skew RMSE.
     - Convergence detection (stops when RMS < 100 ps).
   - Visualizations: topology scatter with edge weights, RMS convergence curves, histogram of residuals.
5. **Outputs**
   - JSON: timing of convergence, final RMS, iterations, success rate over Monte Carlo trials.
   - PNG/SVG plots saved to `results/phase2/`.

### Milestones
- [ ] Port handshake API to reusable helper.
- [ ] Implement consensus update with variance weighting.
- [ ] Validate on static network (N=20) with deterministic seed.
- [ ] Scale to N=50, run 100-trial Monte Carlo, ensure runtime < 5 minutes.

## Phase 3 – Robustness & Scenario Testing

### Objectives
- Quantify performance under mobility, network scaling, and degraded oscillator quality.
- Produce summary charts (accuracy vs. mobility, convergence vs. network size, coherence vs. ppm).

### Enhancements
1. **Mobility & Doppler**
   - Introduce node trajectories (constant velocity up to 20 m/s) and recompute distances each iteration.
   - Update handshake to include Doppler shift: modify beat phase with `exp(j2π f_D t)` where `f_D = v_rel/λ`.
   - Extend consensus loop to handle time-varying measurements; optionally incorporate simple Kalman smoothing.
2. **Scaling Studies**
   - Parameter sweep for `N ∈ {10, 20, 50, 100, 200, 500}`.
   - Track convergence iterations, RMS floor, and communication load (edge-count × iterations).
   - Implement batching + progress logging to keep runs tractable.
3. **Oscillator Quality Sweep**
   - Vary `clock_ppm_std` (2 ppm TCXO → 20 ppm XO).
   - Evaluate resulting steady-state coherence after consensus; plot RMS vs. ppm.
4. **Reporting Utilities**
   - Create `sim/reporting.py` helpers to collate phase summaries into Markdown / JSON tables.
   - Provide `phase3.py` entry point to generate combined PDF/PNG charts.

### Deliverables
- `results/phase3/mobility_sweep.json` + `mobility_plot.png` (RMS vs. velocity).
- `results/phase3/scale_sweep.json` + `scale_plot.png` (iterations vs. N).
- `results/phase3/oscillator_sweep.json` + `oscillator_plot.png` (RMS vs. ppm).
- Narrative `results/phase3/summary.md` capturing performance bounds.

### Milestones
- [ ] Implement Doppler-aware handshake variant.
- [ ] Add mobility + scaling drivers in `phase3.py`.
- [ ] Automate plotting/report generation.
- [ ] Document success criteria validation (sub-100 ps in 5 ms for N=50 TCXO case).
