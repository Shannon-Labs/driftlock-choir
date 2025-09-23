# Driftlock: 22-Picosecond Wireless Synchronization

![Demo](docs/images/demo.gif)

## Abstract

By intentionally introducing frequency offset between wireless transceivers, we generate beat signals that encode propagation delay with unprecedented precision. This counterintuitive approach—treating frequency offset as a feature rather than impairment—achieves **22 picosecond** synchronization using commercial hardware.

**Performance**: 22.13 ps consensus precision • 2,273× improvement over GPS • Single-iteration convergence

**Patent Pending** • Apache 2.0 License

## The Core Insight

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
```

![Network Convergence](docs/images/convergence.png)

## Results

### Experimental Validation
- **Synchronization**: 22.13 ps (Kalman-filtered consensus)
- **Raw Measurement**: 45.0 ps (0.83× Cramér-Rao bound)
- **Bias Reduction**: 4,500× via reciprocity calibration
- **Network Size**: Tested up to 64 nodes
- **Convergence**: Single iteration

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
PYTHONPATH=. pytest

# Generate performance data
PYTHONPATH=. python sim/phase1.py  # Two-node
PYTHONPATH=. python sim/phase2.py  # Multi-node

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

The Driftlock method is patent pending. Source code is Apache 2.0 licensed. Patent licensing: hunter@shannonlabs.dev

---

*"We turned the problem into the solution. What everyone thought was noise became our most precise measurement tool."*