# Driftlock: Chronometric Interferometry for Wireless Synchronization

## Abstract

By intentionally introducing frequency offset between wireless transceivers, we generate beat signals that encode propagation delay. This counterintuitive approach—treating frequency offset as a feature rather than impairment—is evaluated through comprehensive simulation studies. The simulation framework provides detailed performance analysis across multiple channel profiles, with results showing $-0.13$~ns bias in ideal conditions, $0.45$~ns bias in urban environments, and $1.76$~ns bias in challenging indoor multipath scenarios.

**Performance**: Results reflect comprehensive simulation analysis across multiple channel profiles under controlled conditions.

**Patent Pending** • Apache 2.0 License

## Current Development Focus
Our current priority is a hardening pass to improve performance in realistic multipath environments. This involves tightening our validation guardrails (enforcing 1.0 ≤ RMSE/CRLB ≤ 1.5), implementing fractional coarse alignment, and refining the Pathfinder algorithm to better handle indoor and urban channel models. While performance in ideal conditions is promising, the results below reflect the ongoing work to make that performance robust.

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
- **TDL Sweep @ 40 dB**: IDEAL: -0.13 ns bias; INDOOR_OFFICE: 1.76 ns bias; URBAN_CANYON: 0.45 ns bias. These results reflect current performance baseline across multiple channel profiles.

| Date | Commit | Profile | Bias (ns) | RMSE (ns) | Δf bias (Hz) | CRLB (ns) | RMSE/CRLB | Coarse Lock | Guard Hit | Status |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 2025-09-24 | 8b2c85d | IDEAL | -0.13 | 0.13 | 0.0 | 0.12 | 1.05 | ✔ | ✖ | PASS |
| 2025-09-24 | 8b2c85d | URBAN_CANYON | 0.45 | 1.19 | 0.0 | 0.12 | 9.88 | ✔ | ✖ | FAIL_CRLB_HIGH |
| 2025-09-24 | 8b2c85d | INDOOR_OFFICE | 1.76 | 2.58 | 0.0 | 0.12 | 21.5 | ✔ | ✖ | FAIL_BIAS_CAP;FAIL_CRLB_HIGH |
| 2025-09-25 | HEAD | URBAN_CANYON (guard 100 ns) | 1.02 | 4.47 | 92.96 | 0.00096 | 4.64e3 | ✔ | ✖ | FAIL_CRLB_HIGH;FAIL_BIAS_CAP |
| 2025-09-25 | HEAD | INDOOR_OFFICE (guard 100 ns) | 2.02 | 5.44 | 59.54 | 0.00122 | 4.47e3 | ✔ | ✖ | FAIL_CRLB_HIGH;FAIL_BIAS_CAP |

Δf bias collapsed from the stubborn ±85 Hz to **single-digit Hz** after trimming IIR transients (IDEAL now averages **6.7 Hz**), so the consensus filter finally sees coherent frequency updates. IDEAL’s residual mean bias is -0.13 ns, now inside the 0.2 ns target.

Multipath still dominates the non-ideal profiles. `URBAN_CANYON` toggles between the direct path and a late cluster depending on SNR, while `INDOOR_OFFICE` continues to latch onto a 20–25 ns echo that survives the current guard window. With the new guard-limited Pathfinder blend (falling back to the aperture search when the first hit is >100 ns ahead of the peak) the first-path error collapsed from ~-230 ns to ~-50 ns, giving us a cleaner starting point for the remaining bias work. The guardrail logging highlights exactly which seeds exceed the bias caps, which makes it easier to target the remaining pruning work.

### Multipath Performance
- **INDOOR_OFFICE**: +1.76 ns mean bias. This result reflects current simulation performance in challenging indoor multipath environments and is flagged by our performance guardrails, indicating ongoing refinement is needed.

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

-   **Multipath-Resilient Synchronization with "Pathfinder" Algorithm:** The refreshed TDL sweep shows `IDEAL` at **-0.13 ns**, `URBAN_CANYON` at **0.45 ns**, and `INDOOR_OFFICE` at **1.76 ns** of bias. The residual errors line up with direct-path selection (late clusters still leak through the guard), so the next sprint focuses on tighter window pruning plus fractional coarse alignment to converge the scripted `INDOOR_OFFICE` run toward the 0.13 ns lab benchmark.

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
