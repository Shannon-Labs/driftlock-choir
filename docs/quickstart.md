# Driftlock Quick Start Guide

## What You'll Learn

In 10 minutes, you'll:
- Run your first Driftlock simulation
- Achieve **22 picosecond** network synchronization
- Understand the core innovation
- Run Monte Carlo validation suite

## Prerequisites

- Python 3.10 or higher
- Basic understanding of wireless communications (helpful but not required)

## Installation

```bash
# Clone the repository
git clone https://github.com/shannon-labs/driftlock
cd driftlock

# Create virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Your First Synchronization

### 1. Simple Two-Node Demo

Run the basic demonstration:

```bash
python demo_two_node_timing.py
```

Expected output:
```
Driftlock Two-Node Synchronization Demo
========================================
Distance: 100.0 m
SNR: 20.0 dB
Frequency offset: 1.00 MHz

Running synchronization...
✓ Timing accuracy achieved: 2.081 ns RMS
✓ Frequency accuracy: 93.2 Hz
✓ Better than GPS by factor of: 4.8×
```

### 2. Understanding What Just Happened

The demo:
1. Created two wireless nodes 100m apart
2. Intentionally offset their frequencies by 1 MHz
3. Generated beat signals encoding timing information
4. Extracted propagation delay with 2ns accuracy
5. No GPS, no master clock, no external reference!

## Exploring the Core Concept

### Create Your Own Simulation

```python
import numpy as np
from src.alg.chronometric_handshake import (
    ChronometricNode,
    ChronometricNodeConfig,
    ChronometricHandshakeConfig,
    simulate_handshake_pair
)

# Create two nodes with intentional frequency offset
node_a = ChronometricNode(ChronometricNodeConfig(
    node_id=1,
    carrier_freq_hz=2.4e9,  # 2.4 GHz carrier
    phase_offset_rad=0,
    clock_bias_s=100e-9,  # 100ns clock bias
    freq_error_ppm=2.0
))

node_b = ChronometricNode(ChronometricNodeConfig(
    node_id=2,
    carrier_freq_hz=2.4e9 + 1e6,  # 1 MHz offset!
    phase_offset_rad=np.pi/4,
    clock_bias_s=-50e-9,  # Different clock bias
    freq_error_ppm=2.0
))

# Configure the handshake
config = ChronometricHandshakeConfig(
    beat_duration_s=20e-6,  # 20 microseconds
    coarse_enabled=True,
    coarse_bandwidth_hz=20e6
)

# Run synchronization
rng = np.random.default_rng(42)
result, traces = simulate_handshake_pair(
    node_a, node_b,
    distance_m=100,
    snr_db=20,
    rng=rng,
    config=config,
    capture_trace=True
)

print(f"Achieved synchronization: {result.tof_est_s*1e9:.3f} ns")
print(f"True propagation time: {result.tof_true_s*1e9:.3f} ns")
print(f"Error: {(result.tof_est_s - result.tof_true_s)*1e12:.1f} ps")
```

## Running Full Simulations

### Phase 1: Two-Node Monte Carlo

Validate the accuracy across different conditions:

```bash
python -m sim.phase1 --snr-db "20,10,0" --retune-offsets-hz "1e6,5e6" --coarse-bw-hz "20e6,40e6" --num-trials 150 --rng-seed 2024 --make-plots
```

Need to sanity-check a configuration before committing? Use the lightweight preview mode:

```bash
python -m sim.phase1 --snr-db "20" --retune-offsets-hz "1e6" --coarse-bw-hz "40e6" --dry-run --echo-config
```

`--dry-run` prints the resolved JSON config (without saving artifacts) while `--echo-config` keeps the description even on real runs. Outputs land under `results/phase1/`, including alias-map manifests that now capture reciprocity bias summaries and MAC guard times.

### Phase 2: Network Consensus

Simulate a 50-node network and exercise the Kalman pre-filter:

```bash
python -m sim.phase2 --nodes 50 --density 0.15 --local-kf on --weighting metropolis_var --target-rmse-ps 100 --rng-seed 4242 --echo-config
```

Toggle the front-end filter with `--local-kf {auto,on,off,baseline}` to compare convergence. Results (JSON + CSV + plots) are written to `results/phase2/` with Kalman telemetry embedded in the manifests.

### Batch Monte Carlo sweeps

For larger studies, drive both phases from the refreshed Monte Carlo harness:

```bash
python scripts/run_mc.py all -c sim/configs/mc_extended.yaml -o results/mc_runs -r extended_010
```

The YAML file lists Phase 1 alias-map jobs (with hardware delays + calibration modes) and Phase 2 consensus presets. Each job writes its own subdirectory and the tool stitches together `final_results.json` plus a human-readable `simulation_report.txt` summarizing bias reductions and consensus convergence.

With the new shrinkage-based local pre-filter, the dense 64-node preset now
lands at **22.13 ps** (vs. 22.45 ps without the pre-filter) and the 25-node
topology drops to **20.96 ps**. Full telemetry lives under
`results/mc_runs/<run_id>/phase2/*/phase2_results.json`.

Want to probe optimal gains? Run the sweep helper, then validate the JSON via
`scripts/verify_kf_sweep.py` to confirm the best-mean combo sticks to 0.32 / 0.03 / 1 iteration.

Regression guardrails now live in `tests/test_consensus.py::test_dense_kf_vs_baseline`; run it standalone to make sure the tuned preset keeps ≥1 ps over the baseline (or execute `scripts/run_verification_checks.sh` to run both the regression and sweep verifier in one command):

```bash
pytest tests/test_consensus.py::test_dense_kf_vs_baseline -q
# or
scripts/run_verification_checks.sh
```

Then iterate with the sweep helper:

```bash
python scripts/sweep_phase2_kf.py --nodes 64 --density 0.22 \
  --gains 0.18,0.22,0.25,0.28,0.32 --freq-gains 0.03,0.05,0.07 \
  --iters 1,2 --seeds 4040,4141,4242 --epsilon 0.02 --timestep-ms 0.5 \
  --write-json --output-dir results/kf_sweeps --run-id dense_combo_scan --baseline
```

This emits `kf_sweep_summary.json` files summarising per-gain RMSE and baseline
comparisons so you can keep pushing the front-end toward the physics limit.

Want a quick smoke test instead? Use the lightweight wrapper:

```bash
python scripts/run_presets.py phase1-alias --num-trials 120 --calib-mode loopback
python scripts/run_presets.py phase2-consensus --nodes 40 --local-kf off
```

This helper pins sensible defaults while letting you override the essentials via CLI flags.

## Visualizing Results

### Generate Performance Plots

```python
from sim.phase1 import Phase1Config, Phase1Simulator

config = Phase1Config(
    snr_values_db=[0, 10, 20, 30],
    n_monte_carlo=100,
    plot_results=True,  # Enable plotting
    save_results=True
)

simulator = Phase1Simulator(config)
results = simulator.run_full_simulation()
```

### View Beat Signal Processing

The simulation can capture internal signals:

```python
# With capture_trace=True, you get:
# - Raw beat signal
# - Filtered beat signal
# - Phase evolution
# - Parameter extraction

import matplotlib.pyplot as plt

if traces:
    trace = traces['forward']

    fig, axes = plt.subplots(2, 1, figsize=(10, 8))

    # Beat signal
    axes[0].plot(trace.time_us, np.real(trace.beat_filtered))
    axes[0].set_xlabel('Time (μs)')
    axes[0].set_ylabel('Beat Signal')
    axes[0].set_title('Filtered Beat Signal (1 MHz offset)')

    # Phase evolution
    axes[1].plot(trace.adc_time_us, trace.unwrapped_phase, 'o-', label='Measured')
    axes[1].plot(trace.adc_time_us, trace.phase_fit, 'r-', label='Fitted')
    axes[1].set_xlabel('Time (μs)')
    axes[1].set_ylabel('Phase (radians)')
    axes[1].set_title('Phase Extraction for Timing')
    axes[1].legend()

    plt.tight_layout()
    plt.show()
```

## Key Parameters to Experiment With

### Frequency Offset (Δf)
```python
# Try different offsets (default: 1 MHz)
config.retune_offsets_hz = (0.5e6,)  # 500 kHz
config.retune_offsets_hz = (1e6, 5e6)  # Multi-carrier for ambiguity resolution
```

### Signal Quality
```python
# Vary SNR to see impact on accuracy
snr_db = 30  # Excellent conditions → <1 ps error
snr_db = 10  # Moderate conditions → ~5 ps error
snr_db = 0   # Poor conditions → ~20 ps error
```

### Network Size
```python
# Phase 2 configuration
config = Phase2Config(
    n_nodes=50,  # Try 10, 20, 50, 100 nodes
    comm_range_m=150.0
)
```

## Understanding the Innovation

### Why Intentional Offset?

Traditional systems fight frequency offset. We embrace it:

```python
# Traditional: Δf ≈ 0 (trying to match frequencies)
# Problem: No beat signal, limited timing information

# Driftlock: Δf = 1 MHz (intentional offset)
# Solution: Strong beat signal encoding timing with ps precision!
```

### The Beat Signal Magic

The beat phase evolution directly encodes timing:

```
Phase slope → Frequency offset (Δf)
Phase intercept → Propagation delay (τ)
```

More precision than GPS, using only commercial hardware!

## Next Steps

### 1. Explore Advanced Features

- **Multi-carrier unwrapping**: Resolve phase ambiguities
- **Distributed consensus**: Scale to many nodes
- **Hardware calibration**: Compensate for delays

### 2. Run Comprehensive Tests

```bash
# Full test suite
pytest

# Specific test categories
pytest tests/test_chronometric.py  # Core algorithm tests
pytest tests/test_consensus.py      # Network consensus tests
```

### 3. Contribute

- Report issues on GitHub
- Share your results
- Propose new applications
- Improve the simulations

## Troubleshooting

### Common Issues

**Import errors**: Ensure you're in the driftlock directory and virtual environment is activated

**Numerical warnings**: Some extreme test cases may generate warnings - this is expected

**Memory usage**: Large Monte Carlo runs (>1000 trials) may require significant RAM

### Getting Help

- GitHub Issues: https://github.com/shannon-labs/driftlock/issues
- Documentation: See `/docs` folder
- Contact: hello@shannonlabs.dev

## Academic Citation

If you use Driftlock in research:

```bibtex
@software{driftlock2025,
  title = {Driftlock: Sub-Nanosecond Wireless Synchronization},
  author = {Bown, Hunter},
  year = {2025},
  organization = {Shannon Labs, Inc.},
  note = {Patent Pending}
}
```

## Commercial Use

For commercial applications, contact: licensing@shannonlabs.com

---

**Ready to revolutionize wireless synchronization? You just did!** 🚀
