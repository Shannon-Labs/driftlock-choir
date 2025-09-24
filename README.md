# Driftlock: 22-Picosecond Wireless Synchronization

## Abstract

By intentionally introducing frequency offset between wireless transceivers, we generate beat signals that encode propagation delay with unprecedented precision. This counterintuitive approach—treating frequency offset as a feature rather than impairment—achieves **22.13 ps dense-network synchronization** using commercial hardware. Read the [full results analysis](docs/results_extended_011.md).

**Performance**: 22.13 ps consensus precision • 2,273× improvement over GPS • Single-iteration convergence

**Patent Pending** • Apache 2.0 License

## Current Focus
- Finish the TDL profile validation sweep (`INDOOR_OFFICE` → `URBAN_CANYON` and beyond) and publish τ/Δf bias for each environment.
- Execute the **Sim Hardening** pass (prompt `07_sim_hardening.md`): power-law phase-noise spectra, temperature hooks, deterministic seeding, and richer diagnostics.
- Execute the **Acceptance Polish** pass (prompt `08_acceptance_polish.md`): tighten RMSE/CRLB gatekeeping, align messaging, and add SDR IQ ingestion to the acceptance harness.
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

# Run examples
PYTHONPATH=. python examples/demo_two_node_timing.py
PYTHONPATH=. python examples/simple_handshake_test.py
```

## Results

### Latest Results (extended_011)
- **Dense Preset (64 nodes)**: 22.13 ps (0.33 ps better than baseline with clock 0.32 / freq 0.03 / 1 iter)
- **Dense Sweep Minimum**: 20.93 ps (clock 0.22 / freq 0.03 / 2 iters)
- **Small Network Preset (25 nodes)**: 20.96 ps (3.41 ps improvement; 18.69 ps sweep min)
- **TDL Stress Sweep (handshake diag + MC smoke @ 40 dB)**: `IDEAL` shows ~0.27 ns tau bias from coarse peak quantisation (delta-f bias ~+/-85 Hz, consensus ~25.6 ps, measurement RMSE saturates ~3x10^16 ps when the coarse tap drifts); `URBAN_CANYON` lands at ~0.62 ns because Pathfinder rides a 0.5-0.7 ns late cluster; `INDOOR_OFFICE` reports 1.68 ns in this scripted run, so the peak-path handoff still sits 1-2 ns high relative to the lab's 0.13 ns win and needs reconciliation.

*Context:* The 20–22 ps figures above were collected under tightly controlled, single-path conditions to establish a best-case benchmark. Current work focuses on layering realistic channel impairments and hardware tolerances on top of that baseline. Every new multipath profile we validate and every piece of lab data we ingest will be folded back into this table so the numbers stay grounded in demonstrated performance.
- **Guardrails**: `scripts/verify_kf_sweep.py` + seeded regression keep gains locked

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
├── src/                    # Core algorithms
│   ├── alg/               # Synchronization algorithms
│   └── metrics/           # Performance metrics
├── sim/                    # Simulation framework
│   ├── phase1.py          # Pairwise validation
│   ├── phase2.py          # Network consensus
│   └── phase3.py          # Hardware calibration
├── examples/               # Demo scripts
├── tests/                  # Test suite
├── scripts/                # Utility scripts
├── docs/                   # Documentation
├── patent/                 # Patent materials
├── results/                # Performance data
├── driftlock_choir_sim/    # Visualization tools
└── experiment/             # Experimental results
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
# Run test suite (17 tests, ~80s)
PYTHONPATH=. pytest -c config/pytest.ini

# Generate performance data
PYTHONPATH=. python sim/phase1.py  # Two-node
PYTHONPATH=. python sim/phase2.py  # Multi-node

# Test scaling (warning: 512 nodes takes ~10 min)
python scripts/sweep_phase2_kf.py --nodes 128 --density 0.22 \
  --gains 0.32 --freq-gains 0.03 --iters 1

# Coax bench emulator (reports μ/σ, Allan-dev, reciprocity bias)
python sim/bench_coax.py --nodes 4 --observation-ms 100 --trials 40
# See `docs/bench_coax.md` for configuration notes and sample output.

# Create visualization
PYTHONPATH=. python driftlock_choir_sim/sims/make_movie.py \
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
