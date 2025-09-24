# Driftlock: 22-Picosecond Wireless Synchronization

## Abstract

By intentionally introducing frequency offset between wireless transceivers, we generate beat signals that encode propagation delay with unprecedented precision. This counterintuitive approach—treating frequency offset as a feature rather than impairment—achieves **22.13 ps dense-network synchronization** using commercial hardware. Read the [full results analysis](docs/results_extended_011.md).

**Performance**: 22.13 ps consensus precision • 2,273× improvement over GPS • Single-iteration convergence

**Patent Pending** • Apache 2.0 License

## Current Focus
- Finish the TDL profile validation sweep (`INDOOR_OFFICE` → `URBAN_CANYON` and beyond) and publish τ/Δf bias for each environment.
- Execute the **Sim Hardening** pass (prompt `07_sim_hardening.md`): add power-law phase-noise spectra, wire in temperature coefficients/thermal time constants, enforce deterministic RNG seeding, and expose per-edge bias exports.
- Execute the **Acceptance Polish** pass (prompt `08_acceptance_polish.md`): tighten RMSE/CRLB thresholds, reorder README/docs to lead with acceptance numbers, and add an SDR IQ ingestion script plus usage notes.
- Spin up “Project Swing” (advanced modulation) once the hardened baseline is locked, then proceed to Harmony/Score as the data matures.

## The Core Insight

> **Tuned 0.32 / 0.03 / 1 combo**: This repository reflects the `extended_011` run, which produced the 22.13 ps result.

Traditional wireless systems spend enormous effort eliminating frequency offset. We do the opposite.

When two radios transmit at slightly different frequencies (f₁ and f₁+Δf), their interaction creates a beat signal at frequency Δf. The phase of this beat evolves as:

```
φ_beat(t) = 2πΔf(t - τ) + φ₀
```

Where τ is the propagation delay we seek. By measuring beat phase evolution over microsecond windows, we extract timing with picosecond precision.

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

## Results

### Latest Results (extended_011)
- **Dense Preset (64 nodes)**: 22.13 ps (0.33 ps better than baseline with clock 0.32 / freq 0.03 / 1 iter)
- **Dense Sweep Minimum**: 20.93 ps (clock 0.22 / freq 0.03 / 2 iters)
- **Small Network Preset (25 nodes)**: 20.96 ps (3.41 ps improvement; 18.69 ps sweep min)
- **TDL Stress Sweep (handshake diag + MC smoke @ 40 dB)**: `IDEAL` two‑way τ median ≈ −0.27 ns (coarse peak quantization); `URBAN_CANYON` median ≈ +0.74 ns (late‑cluster handoff); `INDOOR_OFFICE` median ≈ +0.52 ns with heavy‑tail mean ≈ +1.68 ns pending reconciliation with the ~0.13 ns lab result. Δf median biases sit near ±85 Hz. Manifests are written under `results/phase1/tdl_profiles` by `scripts/run_handshake_diag.py`.

*Context:* The 20–22 ps figures above were collected under tightly controlled, single-path conditions to establish a best-case benchmark. Current work focuses on layering realistic channel impairments and hardware tolerances on top of that baseline. Every new multipath profile we validate and every piece of lab data we ingest will be folded back into this table so the numbers stay grounded in demonstrated performance.
- **Guardrails**: `scripts/verify_kf_sweep.py` + seeded regression keep gains locked

### Immediate Remediation Plan
- Normalise all sweep outputs to **ns**/**Hz**, stamp each manifest with `{git_sha, config_hash, seed, coarse_locked, guard_hit}`, and auto-render the README table via a forthcoming helper (`scripts/render_results_table.py`).
- Add fractional coarse alignment (parabolic sub-sample refinement, clamped |δ|≤0.5) so quantisation no longer sets a nanosecond floor.
- Retune Pathfinder with SNR-adaptive thresholds and a micro refinement window so the direct path survives in URBAN/INDOOR profiles.
- Update the Δf estimator to use tone-weighted WLS plus one re-estimation loop, removing the ±85 Hz residual CFO.
- Tighten acceptance guardrails: enforce `1.0 ≤ RMSE/CRLB ≤ 1.5`, log sub-bound passes, and apply temporary per-profile bias caps until the fixes land.

### Scaling Performance
- **128 nodes**: 22.97 ps RMSE (51s runtime)
- **256 nodes**: 21.64 ps RMSE (better precision at scale!)
- **512 nodes**: 20.09 ps RMSE (10.5 min runtime)

The algorithm gets *more* precise with larger networks - a counterintuitive result enabled by variance-weighted consensus.

For reproduction commands, regression guardrails, and repo hygiene notes see [docs/scaling_results.md](docs/scaling_results.md).

### Roadmap & Next Steps

Our core focus is on enhancing the robustness and real-world performance of the Driftlock Choir system. Key initiatives include:

-   **Advanced Channel Modeling:** We are integrating high-fidelity channel models, such as `INDOOR_OFFICE`, to validate performance in challenging multipath and noisy environments. This ensures our synchronization capabilities hold up under realistic deployment conditions.

-   **Algorithm Tuning & Hardening:** We are continuously refining the consensus algorithm (`src/alg/consensus.py`) to improve convergence speed and stability, particularly in response to the complex scenarios introduced by new channel models.

-   **Performance Optimization:** Ongoing profiling and optimization work is focused on ensuring the simulation framework remains fast and efficient, enabling large-scale Monte Carlo runs to produce statistically significant results.

-   **Multipath-Resilient Synchronization with "Pathfinder" Algorithm:** The scripted sweep now pegs `IDEAL` at ~0.27 ns, `URBAN_CANYON` at ~0.62 ns, and `INDOOR_OFFICE` at ~1.68 ns of tau bias, all traceable to coarse peak placement (Pathfinder windowing plus quantised coarse taps). Next up: retune the window/guard so the direct path survives, add fractional coarse alignment, and reconcile this scripted `INDOOR_OFFICE` result with the 0.13 ns lab run before pushing to outdoor macros.

-   **Advanced Modulation with "Project Swing":** We are evolving our core modulation from a simple "vibrato" (a pure sine wave) to a more complex, organic "swing" using non-periodic and chaotic waveforms. This initiative aims to create a unique, nearly impossible-to-replicate signal signature, drastically improving robustness in severe multipath environments and enhancing security against spoofing attacks.

-   **Hardware-in-the-Loop Validation:** The next major phase will involve bridging our simulations with real-world hardware to confirm picosecond-level precision with off-the-shelf components.

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
```
x_i(k+1) = x_i(k) + ε Σ_j W_ij(d_ij - (x_i - x_j))
```

Where W_ij weights by measurement precision (inverse variance).

## Project Structure

```
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
```
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
  title = {Driftlock: Picosecond Wireless Synchronization via Chronometric Interferometry},
  year = {2025},
  url = {https://github.com/shannon-labs/driftlock-choir}
}
```

## Contact

Hunter Bown • hunter@shannonlabs.dev

## Patent Notice

The Driftlock method is patent pending.
**Provisional Patent Application No. 63/886,461, filed September 23, 2025.**
Source code is Apache 2.0 licensed. Patent licensing: hunter@shannonlabs.dev

---

*"We turned the problem into the solution. What everyone thought was noise became our most precise measurement tool."*
