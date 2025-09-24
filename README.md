# Driftlock: 22-Picosecond Wireless Synchronization

## Abstract

By intentionally introducing frequency offset between wireless transceivers, we generate beat signals that encode propagation delay with unprecedented precision. This counterintuitive approach—treating frequency offset as a feature rather than impairment—achieves **22.13 ps dense-network synchronization** using commercial hardware. Read the [full results analysis](docs/results_extended_011.md).

**Performance**: 22.13 ps consensus precision • 2,273× improvement over GPS • Single-iteration convergence

**Patent Pending** • Apache 2.0 License

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
- **Guardrails**: `scripts/verify_kf_sweep.py` + seeded regression keep gains locked

### Scaling Performance
- **128 nodes**: 22.97 ps RMSE (51s runtime)
- **256 nodes**: 21.64 ps RMSE (better precision at scale!)
- **512 nodes**: 20.09 ps RMSE (10.5 min runtime)

The algorithm gets *more* precise with larger networks - a counterintuitive result enabled by variance-weighted consensus.

For reproduction commands, regression guardrails, and repo hygiene notes see [docs/scaling_results.md](docs/scaling_results.md).

### Next Experiments: Handoff Notes for Driftlock Choir Simulation Debugging

**Objective:**
The primary goal is to run a 50-seed Monte Carlo simulation using the new `INDOOR_OFFICE` channel profile and recently added hardware impairments to generate statistically sound performance benchmarks for the Driftlock Choir platform.

**Current Status: Blocked - Catastrophic Divergence**
The simulation is currently failing. The consensus algorithm is not just converging slowly; it's catastrophically diverging, with the timing error exploding to physically impossible values (on the order of hours). This happens consistently, even in short smoke tests.

**Diagnosis:**
The divergence is a classic feedback loop failure. The new, high-fidelity `INDOOR_OFFICE` channel model is introducing a level of noise and multipath error that the consensus algorithm, in its current tuning, cannot handle. The algorithm overcorrects for these large errors, which amplifies the error in the next state, leading to an uncontrolled spiral.

**Key Debugging Steps & Fixes Implemented:**
I've made significant progress in debugging and improving the simulation script. Here are the key changes I've implemented:

1.  **Parallelization:** The `scripts/run_monte_carlo.py` script now supports parallel execution using Python's `multiprocessing` library. You can control the number of workers with the `--num-workers` flag.
2.  **Smoke Test:** A `--smoke-test` flag has been added to the script. This will run a quick, 2-seed, 16-node simulation to validate the entire pipeline before committing to a long run.
3.  **Profiling:** A `--profile` flag has been added to enable `cProfile` for a single run, which will help identify any performance bottlenecks.
4.  **Shock Therapy (Attempt 1):** I've implemented a two-pronged attack to try to stabilize the system:
    *   **Extreme Gain Reduction:** The Kalman filter gains (`local_kf_clock_gain` and `local_kf_freq_gain`) have been reduced by a factor of 1,000 from their original values.
    *   **Outlier Rejection Clamp:** An outlier clamp has been added to the `_run_local_kf` method in `sim/phase2.py` to discard any measurement with an error greater than 1 microsecond.
5.  **Bug Fixes:** I've fixed several bugs in the simulation code, including:
    *   An incorrect call order in the main simulation loop.
    *   A bug due to an immutable dataclass.
    *   An `AttributeError` due to a renamed attribute.

**Recommendations for Next Steps:**

1.  **Verify the Divergence:** Run the smoke test to observe the divergence firsthand: `python scripts/run_monte_carlo.py --smoke-test --channel-profile INDOOR_OFFICE`
2.  **Investigate the Consensus Algorithm:** The "shock therapy" I implemented was not enough to stabilize the system. The next step should be a deeper dive into the consensus algorithm itself (`src/alg/consensus.py`). It's possible that there are other parameters that can be tuned to dampen the system, or that a more fundamental change to the algorithm is required to handle this level of noise.
3.  **Profile the Code:** Use the `--profile` flag to identify any bottlenecks in the code. This will help to focus optimization efforts.
4.  **Examine the Channel Model:** The `INDOOR_OFFICE` channel profile is the root cause of the divergence. It would be wise to examine the `src/chan/tdl.py` file to see how this profile is implemented. It's possible that there is a bug in this file that is causing the extreme noise.
5.  **Isolate the Problem:** If all else fails, it might be necessary to temporarily disable the `INDOOR_OFFICE` channel profile and see if the simulation converges with a simpler channel model. This would help to isolate the problem and confirm that the consensus algorithm is still fundamentally sound.

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

Where B_rms is RMS bandwidth, T is observation time. We achieve 0.83× this theoretical limit.

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
