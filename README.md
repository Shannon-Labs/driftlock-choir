# Driftlock: Picosecond-Scale Wireless Synchronization

[![Patent Pending](https://img.shields.io/badge/Patent-Pending-orange)](patent/PROVISIONAL_PATENT_APPLICATION.md)
[![License: Academic](https://img.shields.io/badge/License-Academic_Free-green)](LICENSE-ACADEMIC.md)
[![License: Commercial](https://img.shields.io/badge/License-Commercial_Contact-red)](LICENSE-COMMERCIAL.md)
[![Test Suite](https://img.shields.io/badge/Tests-17_passed,_1_skipped-brightgreen)](tests)
[![Shannon Labs](https://img.shields.io/badge/by-Shannon_Labs-blue)](https://shannonlabs.dev)
[![Website](https://img.shields.io/badge/Website-driftlock-choir.net-cyan)](https://driftlock-choir.net)

## Abstract

Driftlock is a novel wireless synchronization technique that achieves picosecond-scale accuracy using commodity hardware. The core innovation, **Chronometric Interferometry**, intentionally introduces small frequency offsets between wireless nodes to create low-frequency beat signals. These beat signals encode high-precision timing information, allowing for robust time-of-flight estimation. Through two-way measurements to cancel clock bias and a variance-weighted consensus algorithm for network-wide agreement, this method has achieved **22 picosecond** synchronization precision in simulations, outperforming traditional GPS/PTP-based systems by several orders of magnitude. This repository contains the open-source simulation framework for the Driftlock technology, enabling full validation and reproducibility of our results.

---

### The Two-Pillar System: Driftlock™ and Drittlock™ API

This repository concerns **Driftlock Choir**, the RF/time synchronization technology. It is one part of a two-pillar security stack developed by Shannon Labs.

1.  **Driftlock™ Choir (This Repository):** The underlying RF and time-layer technology that provides picosecond-scale synchronization.
2.  **Drittlock™ API:** A software layer for zero-training anomaly detection using compression-based analysis (CbAD). It uses the pristine timing signals from Driftlock to achieve high-performance security monitoring.

---

## Key Performance Results

### Synchronization Accuracy
- **22.13 ps** network consensus precision (Kalman filter tuned).
- **2,273× better than GPS** (50 ns → 22 ps).
- **4,500× bias reduction** via reciprocity calibration (12 ns → 2.65 ps).
- **Single-iteration convergence** for 25-64 node networks.

### Simulation Validation
- **Δf Aperture Spike:** **58.1 dB** (387% over 15 dB requirement).
- **Coherent RMSE:** **45.0 ps** (0.83× CRLB, 62% better than required).
- **Computational Efficiency:** Full suite completes in **3.7 s** on an Apple M2 Max.
- [View Latest Validation Summary](docs/results_extended_011.md)
- [View Full Monte Carlo Summary](results/mc_runs/extended_011/SUMMARY.md)

### Video Demonstrations
- [**Full Technical Demo (22ps Sync)**](driftlock_choir_sim/outputs/movies/demo_choir_sim.mp4)
- [**Quick Teaser Video**](driftlock_choir_sim/outputs/movies/demo_teaser_choir_sim.mp4)
- [**Comparison vs. GPS/PTP Baseline**](driftlock_choir_sim/outputs/movies/baseline_choir_sim.mp4)

*(Note: The links above point to video files within this repository. To view them, you may need to download the files or clone the repository.)*

## Getting Started

### Installation
```bash
git clone https://github.com/shannon-labs/driftlock-choir
cd driftlock-choir
pip install -r requirements.txt
```

### Reproducing Our Results

1.  **Run the Test Suite:** Verify the environment and baseline algorithms.
    ```bash
    PYTHONPATH=. pytest
    ```
    *(Expected result: 17 passed, 1 skipped)*

2.  **Run a Two-Node Handshake Simulation:** This is the fundamental building block (Phase 1).
    ```bash
    PYTHONPATH=. python sim/phase1.py
    ```
    *(This saves results and plots to `results/phase1/`)*

3.  **Run a Multi-Node Consensus Simulation:** This demonstrates network-wide synchronization (Phase 2).
    ```bash
    PYTHONPATH=. python sim/phase2.py
    ```
    *(This saves results and plots to `results/phase2/`)*

4.  **Generate Demo Videos:** Recreate the videos linked above.
    ```bash
    # Generate the full technical demo video
    PYTHONPATH=. python driftlock_choir_sim/sims/make_movie.py \
      --config driftlock_choir_sim/configs/demo_movie.yaml

    # Generate the baseline (GPS/PTP) comparison video
    PYTHONPATH=. python driftlock_choir_sim/sims/make_movie.py \
      --config driftlock_choir_sim/configs/baseline_movie.yaml
    ```
    *(Outputs are saved to `driftlock_choir_sim/outputs/movies/`)*


## Core Innovation: Chronometric Interferometry

Driftlock uses beat signals from intentionally offset carriers to encode timing information:
`φ_beat(t) = 2π Δf (t - τ) + phase_terms`

**Key Technical Steps:**
1.  **Intentional Frequency Offset:** Creates a measurable low-frequency beat signal.
2.  **Phase Extraction:** The phase of the beat signal reveals the propagation delay `τ`.
3.  **Two-Way Measurement:** A forward and reverse handshake allows for cancellation of clock bias.
4.  **Distributed Consensus:** A variance-weighted consensus algorithm achieves network-wide agreement.

[Read the full theory →](docs/theory.md) | [Explore the Choir Simulation Lab →](driftlock_choir_sim/README.md)

## Project Structure
```
driftlock-choir/
├── sim/                    # Simulation framework (open source)
│   ├── phase1.py           # Two-node synchronization
│   └── phase2.py           # Multi-node consensus
├── src/                    # Core algorithms (patent pending)
│   ├── alg/                # Consensus, estimators, Kalman tools
│   ├── phy/                # Oscillators, noise, signal models
│   └── metrics/            # CRLB + bias/variance analysis
├── tests/                  # Pytest suite for all modules
├── docs/                   # Technical documentation and results
└── patent/                 # Patent materials (provisional, claims, figures)
```

## Citation
If you use Driftlock in your research, please cite our work:
```bibtex
@software{driftlock_choir_2025,
  title = {Driftlock: Picosecond-Scale Wireless Synchronization via Chronometric Interferometry},
  author = {Bown, Hunter},
  year = {2025},
  organization = {Shannon Labs, Inc.},
  url = {https://github.com/shannon-labs/driftlock-choir}
}
```

## Licensing and Patent
- **Patent Pending:** The core Driftlock method is patent pending.
- **Academic Use:** This source code is freely available for academic and research purposes under the [Academic License](LICENSE-ACADEMIC.md).
- **Commercial Use:** Commercial applications require a license. Please contact [licensing@shannonlabs.com](mailto:licensing@shannonlabs.com).

## How to Get Involved
- **Star this repo** to follow our progress.
- **Read our papers** in the [docs](docs/) folder.
- **Report bugs** or suggest improvements in the [Issues](https://github.com/shannon-labs/driftlock-choir/issues) tab.
- **Contribute improvements** via pull requests. Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.
