# Driftlock: Chronometric Interferometry for Wireless Synchronization

## Abstract

By intentionally introducing frequency offset between wireless transceivers, we generate beat signals that encode propagation delay. This counterintuitive approach—treating frequency offset as a feature rather than impairment—is evaluated through comprehensive simulation studies. The simulation framework provides detailed performance analysis across multiple channel profiles, with results showing $-0.13$~ns bias in ideal conditions, $0.45$~ns bias in urban environments, and $1.76$~ns bias in challenging indoor multipath scenarios.

**Performance**: Results reflect comprehensive simulation analysis across multiple channel profiles under controlled conditions.

**Patent Pending** • Apache 2.0 License

## Current Development Focus
Our current priority is a hardening pass to improve performance in realistic multipath environments. This involves tightening our validation guardrails (enforcing 1.0 ≤ RMSE/CRLB ≤ 1.5), implementing fractional coarse alignment, and refining the Pathfinder algorithm to better handle indoor and urban channel models. While performance in ideal conditions is promising, the results below reflect the ongoing work to make that performance robust.

### Project Aperture (Pre-Guard Bias Hunt)
- Hardened the pathfinder so the pre-guard offset applies to both the simple scan and aperture fallback—no more searching before the requested lead-in.
- Added first-path/peak diagnostics to highlight how the aperture window responds when the guard interval expands beyond the immediate peak.
- Use `--pathfinder-pre-guard-ns <value>` alongside `--pathfinder-guard-interval-ns` to sweep headroom; bias snapshots live under `/tmp/diag_pre*` by default.

### Project Formant Tuning (Missing-Fundamental Pathfinder)
- Introduced a vowel-inspired coarse preamble mode (`--coarse-preamble-mode formant`) that sculpts the spectrum around canonical A/E/I/O/U formants.
- The aperture window performs a "missing fundamental" analysis, decoding the vowel token and reconstructing the implied fundamental even when it is absent from the waveform.
- Run exploratory sweeps with commands like `python scripts/run_handshake_diag.py --channel-profile IDEAL --coarse-preamble-mode formant --coarse-formant-profile A --pathfinder-pre-guard-ns 400 --num-trials 16` to log both τ bias and the recovered vowel label.
- Recommended follow-up (Project Swing prompt): pivot from chasing 20 ps to evaluating whether vowel-coded signaling buys alias resilience or richer tagging in multipath. Capture results under `results/project_swing/<profile>/<vowel>` when running broader sweeps.

## The Core Insight

> **Tuned 0.32 / 0.03 / 1 combo**: This repository reflects the `extended_011` run, which produced the 22.13 ps result.

Traditional wireless systems spend enormous effort eliminating frequency offset. We do the opposite.

When two radios transmit at slightly different frequencies (f₁ and f₁+Δf), their interaction creates a beat signal at frequency Δf. The phase of this beat evolves as:

```math
φ_beat(t) = 2πΔf(t - τ) + φ₀
```

Where τ is the propagation delay we seek. By measuring beat phase evolution over microsecond windows, we can extract timing information.

## Quick Demo

```bash
# Install
git clone https://github.com/shannon-labs/driftlock-choir
cd driftlock-choir
pip install -r requirements.txt

# Run two-node synchronization
PYTHONPATH=. python sim/phase1.py

# Run network consensus (50 nodes)
PYTHONPATH=. python sim/phase2.py

# Run example
PYTHONPATH=. python examples/demo_two_node_timing.py
```

![Driftlock 19ps Synchronization Demo](results/demo_sync.gif)

## Results

### Latest Results (TDL Sweep @ 40 dB)
- **TDL Sweep @ 40 dB (2025-09-25 guard sweep with blend heuristics)**: IDEAL: -0.13 ns; URBAN_CANYON: 0.09 ns; INDOOR_OFFICE: 0.94 ns. The new `pathfinder_first_path_blend` heuristic dropped INDOOR_OFFICE bias from 1.69 ns to 0.94 ns and RMSE from 4.86 ns to 4.53 ns, while keeping URBAN_CANYON near zero.

| Date | Commit | Profile | Bias (ns) | RMSE (ns) | Δf bias (Hz) | CRLB (ns) | RMSE/CRLB | Coarse Lock | Guard Hit | Status |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 2025-09-24 | 8b2c85d | IDEAL | -0.13 | 0.13 | 0.0 | 0.12 | 1.05 | ✔ | ✖ | PASS |
| 2025-09-24 | 8b2c85d | URBAN_CANYON | 0.45 | 1.19 | 0.0 | 0.12 | 9.88 | ✔ | ✖ | FAIL_CRLB_HIGH |
| 2025-09-24 | 8b2c85d | INDOOR_OFFICE | 1.76 | 2.58 | 0.0 | 0.12 | 21.5 | ✔ | ✖ | FAIL_BIAS_CAP;FAIL_CRLB_HIGH |
| 2025-09-25 | HEAD | URBAN_CANYON (guard 40 ns, -18 dB, NG=3, blend heuristics) | 0.09 | 3.38 | 85.58 | 0.00079 | 4.26e3 | ✔ | ✖ | FAIL_CRLB_HIGH |
| 2025-09-25 | HEAD | INDOOR_OFFICE (guard 40 ns, -18 dB, NG=3, blend heuristics) | 0.94 | 4.53 | 73.78 | 0.00097 | 4.65e3 | ✔ | ✖ | FAIL_CRLB_HIGH;FAIL_BIAS_CAP |

Δf bias collapsed from the stubborn ±85 Hz to **single-digit Hz** after trimming IIR transients (IDEAL now averages **6.7 Hz**), so the consensus filter finally sees coherent frequency updates. IDEAL’s residual mean bias is -0.13 ns, now inside the 0.2 ns target. Multipath sweeps still sit near +75 Hz (INDOOR) and +86 Hz (URBAN) but showed no regression during the guard sweep.

#### Guard/Aperture Sweep Notes (2025-09-25)
- **Summary**: The introduction of profile-aware blend heuristics (`pathfinder_first_path_blend`) has significantly improved performance. INDOOR_OFFICE bias dropped from 1.69 ns to 0.94 ns, and RMSE from 4.86 ns to 4.53 ns. URBAN_CANYON bias is now near zero. Coarse bias is now ~0.8/1.1 ns instead of 1.5/1.9 ns.
- Parameter space: guard intervals {40, 60, 80, 100} ns, aperture lengths {80, 120, 160, 200} ns, relative thresholds {-20, -18, -16} dB, noise guard multipliers {3, 4, 5}; 64 Monte Carlo trials per point with seed 2025.
- Pathfinder fell back to the aperture window on **100%** of trials for both profiles. Before blending, mean first-to-peak spacing sat near 42–47 ns (INDOOR) and 46–83 ns (URBAN), so the coarse stage inherited the late-path bias.
- Best-performing corner (before blending): guard **40 ns**, -18 dB threshold, noise guard **3×** (aperture duration immaterial). Metrics: INDOOR_OFFICE bias **+1.69 ns**, RMSE **4.86 ns**, coarse bias forward/reverse **+1.51/+1.89 ns**; URBAN_CANYON bias **+0.37 ns**, RMSE **3.41 ns**, coarse bias **+1.31/-0.55 ns**.
- Δf bias held at **75 Hz (INDOOR)** and **86 Hz (URBAN)** with RMSE ≈98 Hz and 86 Hz respectively. No guard setting improved CFO further.
- Artifacts: see `results/tuning_temp/pathfinder_sweep_summary_20250925_064723_508617.json` (top-k comparison) and the full grid manifests under `results/tuning_temp/pathfinder_sweep_summary_20250925_05*.json`.
- Profile-aware blending: enabling the new heuristics (`pathfinder_first_path_blend=0.05`) scales the peak-to-first mix by guard-relative spacing and profile family (INDOOR=100%, URBAN≈35%). This drops the INDOOR coarse bias to **+0.79/+1.10 ns** and trims the two-way bias to **+0.94 ns** while URBAN tightens to **+0.09 ns** without a CFO regression. First-to-peak spacing collapses to ~22 ns under both profiles. Raw diagnostics: `results/tuning_temp/tdl_diag_indoor_office_20250925_093537.json` and `.../tdl_diag_urban_canyon_20250925_093627.json`.

Multipath still dominates the non-ideal profiles. `URBAN_CANYON` toggles between the direct path and a late cluster depending on SNR, while `INDOOR_OFFICE` continues to latch onto a 20–25 ns echo that survives the current guard window. With the new guard-limited Pathfinder blend (falling back to the aperture search when the first hit is >100 ns ahead of the peak) the first-path error collapsed from ~-230 ns to ~-50 ns, giving us a cleaner starting point for the remaining bias work. The guardrail logging highlights exactly which seeds exceed the bias caps, which makes it easier to target the remaining pruning work.

### Multipath Performance
- **INDOOR_OFFICE**: +0.94 ns bias, 4.53 ns RMSE (guard 40 ns, -18 dB, NG=3, aperture 80 ns, blend heuristics). Coarse bias drops to +0.79/+1.10 ns, but further refinement is needed to hit the 0.2 ns guardrail.
- **URBAN_CANYON**: +0.09 ns bias, 3.38 ns RMSE with the same configuration. CRLB ratio remains ≫1.5, pointing to residual multipath defocus rather than coarse-stage bias.

### Monte Carlo Smoke Tests (20 time steps, 200 consensus iters)
- `scripts/run_monte_carlo.py --smoke-test --channel-profile {IDEAL, URBAN_CANYON, INDOOR_OFFICE}` now completes with `{num_timesteps=20, max_iterations=200}` for tractable runtime. Consensus RMSE still reports ~22–27 ps despite measurement RMSE exploding (≈3.4×10¹⁵ ps), confirming the Kalman filters are rejecting the bad data rather than fixing it.

### Immediate Remediation Plan
- [x] Normalise sweep manifests (ns/Hz), stamp `{git_sha, config_hash, seed, coarse_locked, guard_hit}`, and auto-render via `scripts/render_results_table.py`.
- [x] Retune Pathfinder with SNR-adaptive α/β and a micro-window to preserve the earliest path under multipath stress.
- [x] Update the Δf estimator with tone-weighted WLS + one refinement pass, plus a settling guard, cutting the residual CFO from ±85 Hz to <10 Hz.
- [ ] Harden guardrails: enforce `1.0 ≤ RMSE/CRLB ≤ 1.5`, keep failing manifests out of aggregates, and add per-profile notes while Pathfinder guard tuning continues.

### Scaling Performance
- **128 nodes**: 22.97 ps RMSE (51s runtime)
- **256 nodes**: 21.64 ps RMSE
- **512 nodes**: 20.09 ps RMSE (10.5 min runtime)

These results demonstrate the algorithm's behavior under ideal simulation conditions, with performance improving at larger network scales due to variance-weighted consensus.

For reproduction commands, regression guardrails, and repo hygiene notes see [docs/scaling_results.md](docs/scaling_results.md).

For detailed performance status, see [docs/RESULTS_STATUS.md](docs/RESULTS_STATUS.md).

### Roadmap & Next Steps

Our core focus is on enhancing the robustness and real-world performance of the Driftlock Choir system. Key initiatives include:

- **Hardening Focus**: [Details from paper]

-   **Advanced Channel Modeling:** We are integrating high-fidelity channel models, such as `INDOOR_OFFICE`, to validate performance in challenging multipath and noisy environments. This ensures our synchronization capabilities hold up under realistic deployment conditions.

-   **Algorithm Tuning & Hardening:** We are continuously refining the consensus algorithm (`src/alg/consensus.py`) to improve convergence speed and stability, particularly in response to the complex scenarios introduced by new channel models.

-   **Performance Optimization:** Ongoing profiling and optimization work is focused on ensuring the simulation framework remains fast and efficient, enabling large-scale Monte Carlo runs to produce statistically significant results.

-   **Multipath-Resilient Synchronization with "Pathfinder" Algorithm:** The refreshed TDL sweep shows `IDEAL` at **-0.13 ns**, `URBAN_CANYON` at **0.09 ns**, and `INDOOR_OFFICE` at **0.94 ns** of bias after applying the guard sweep and blend heuristics. Residual errors still align with direct-path selection (late clusters leak through the guard), so the next sprint focuses on tighter window pruning plus fractional coarse alignment to converge the scripted `INDOOR_OFFICE` run toward the 0.13 ns lab benchmark.

-   **Advanced Modulation with "Project Swing":** We are evolving our core modulation from a simple "vibrato" (a pure sine wave) to a more complex, organic "swing" using non-periodic and chaotic waveforms. This initiative aims to create a unique, nearly impossible-to-replicate signal signature, drastically improving robustness in severe multipath environments and enhancing security against spoofing attacks.

-   **Hardware-in-the-Loop Validation:** The next major phase will involve bridging our simulations with real-world hardware to confirm performance with off-the-shelf components.

### Documentation & Analysis
- [Full Results Analysis](docs/results_extended_011.md) - Detailed 22ps results
- [Scaling Performance Study](docs/scaling_results.md) - 128-512 node benchmarks
- [Simulation Results](docs/simulation_results.md) - Complete simulation data
- [Benchmark Configuration](docs/bench_coax.md) - Hardware emulation setup
- [Theory & Derivations](docs/theory.md) - Mathematical foundations
- [Physics Derivations](docs/physics_derivations.md) - Physical principles

### Video Demonstrations
- [Technical Demo (3 min)](driftlock_choir_sim/outputs/movies/demo_choir_sim.mp4) - Full system demonstration
- [Quick Overview (30s)](driftlock_choir_sim/outputs/movies/demo_teaser_choir_sim.mp4) - Key concepts
- [vs GPS/PTP Baseline](driftlock_choir_sim/outputs/movies/baseline_choir_sim.mp4) - Performance comparison

## Technical Approach

### 1. Chronometric Interferometry
Intentional frequency offset creates measurable beat patterns. Phase extraction through closed-form estimation avoids iterative search.

### 2. Two-Way Protocol
Bidirectional measurements cancel clock bias:
- Forward: A→B yields τ + δt
- Reverse: B→A yields τ - δt
- Geometric mean extracts true delay τ

### 3. Distributed Consensus
Variance-weighted averaging across all node pairs:
```math
x_i(k+1) = x_i(k) + ε Σ_j W_ij(d_ij - (x_i - x_j))
```

Where W_ij weights by measurement precision (inverse variance).

## Project Structure

```text
driftlock-choir/
├── src/                    # Core modelling and helpers
│   ├── alg/               # Estimation, consensus, Kalman
│   ├── phy/               # Oscillators, noise, preambles, pathfinder
│   ├── chan/              # Tapped‑delay‑line (TDL) channel profiles
│   ├── hw/                # RF front‑end components
│   ├── mac/               # Slotting/scheduler utilities
│   ├── net/               # MAC/topology helpers
│   ├── metrics/           # CRLB, bias/variance analysis
│   └── utils/             # IO, plotting, telemetry
├── sim/                    # Simulation framework
│   ├── phase1.py          # Pairwise validation
│   ├── phase2.py          # Network consensus
│   └── phase3.py          # Hardware calibration
├── examples/               # Demo scripts
├── tests/                  # Pytest suite (seeded)
├── scripts/                # Utilities (diag, MC, sweeps, verification)
├── docs/                   # Documentation
├── patent/                 # Patent materials
├── results/                # Simulation artifacts (committed snapshots)
├── driftlock_choir_sim/    # Movie generator and DSP demos
├── services/time_telemetry # Telemetry service
├── web/                    # Static site/demo
├── paper/                  # Writing/workspace
├── config/                 # Test/config files
├── prompts/                # Internal prompts/checklists
└── experiment/             # Experimental runs/artifacts
```

## Implementation

Core algorithms in `src/`:
- `alg/chronometric_handshake.py` - Two-node synchronization
- `alg/consensus.py` - Network-wide agreement
- `alg/kalman.py` - Adaptive filtering
- `metrics/crlb.py` - Theoretical bounds

Simulation framework in `sim/`:
- `phase1.py` - Pairwise validation
- `phase2.py` - Network consensus
- `phase3.py` - Hardware calibration

## Reproducing Results

```bash
# Run test suite (seeded)
pytest -c config/pytest.ini

# Generate performance data
python sim/phase1.py  # Two-node
python sim/phase2.py  # Multi-node

# TDL profile diagnostics (writes results/phase1/tdl_profiles)
PYTHONPATH=src python scripts/run_handshake_diag.py --channel-profile IDEAL --num-trials 200 --snr-db 40

# Test scaling (warning: 512 nodes takes ~10 min)
python scripts/sweep_phase2_kf.py --nodes 128 --density 0.22 \
  --gains 0.32 --freq-gains 0.03 --iters 1

# Coax bench emulator (reports μ/σ, Allan-dev, reciprocity bias)
python sim/bench_coax.py --nodes 4 --observation-ms 100 --trials 40
# See `docs/bench_coax.md` for configuration notes and sample output.

# Create visualization
python driftlock_choir_sim/sims/make_movie.py \
  --config driftlock_choir_sim/configs/demo_movie.yaml
```

## Applications

- **5G/6G Networks**: Distributed MIMO, network slicing
- **Financial Systems**: Timestamp verification for HFT
- **Sensor Networks**: Coordinated sampling
- **Quantum Networks**: Entanglement distribution timing

## Theory

The Cramér-Rao lower bound for timing precision:
```math
σ_τ ≥ 1/(2π·SNR^(1/2)·B_rms·T^(1/2))
```

Where B_rms is RMS bandwidth, T is observation time. Earlier single-path studies brushed against this limit; ongoing multipath and hardware validation tracks how much margin remains under realistic conditions.

Key innovations:
- Intentional Δf as measurement channel
- Closed-form beat phase estimator
- Variance-weighted consensus
- Single-iteration convergence

## Citation

```bibtex
@software{driftlock2025,
  author = {Bown, Hunter},
  title = {Driftlock: Wireless Synchronization via Chronometric Interferometry},
  year = {2025},
  url = {https://github.com/shannon-labs/driftlock-choir}
}
```
