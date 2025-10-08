# Driftlock Choir

> Precision timing infrastructure for distributed systems.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![CI](https://github.com/Shannon-Labs/driftlock-choir/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/Shannon-Labs/driftlock-choir/actions/workflows/ci.yml)
[![Pages](https://github.com/Shannon-Labs/driftlock-choir/actions/workflows/pages.yml/badge.svg?branch=main)](https://github.com/Shannon-Labs/driftlock-choir/actions/workflows/pages.yml)

- **Live site:** https://shannon-labs.github.io/driftlock-choir/
- **Documentation hub:** https://shannon-labs.github.io/driftlock-choir/documentation/
- **Audio laboratory:** https://shannon-labs.github.io/driftlock-choir/audio/

---

## Contents

1. [Overview](#overview)
2. [Highlights](#highlights)
3. [Signal Reconstructions](#signal-reconstructions)
4. [Quick Start](#quick-start)
5. [Experiment Suite](#experiment-suite)
6. [Documentation](#documentation)
7. [Contributing & Support](#contributing--support)

---

## Overview

## Overview

Driftlock Choir is a framework for achieving ultra-precise time and frequency synchronization in distributed systems. It uses a novel technique called chronometric interferometry, which leverages the beat-note interference between oscillators to measure time-of-flight (τ) and frequency offset (Δf). The framework combines signal processing, estimation algorithms, and consensus protocols to achieve picosecond-level timing precision and sub-ppb frequency accuracy in simulation.

Chronometric interferometry analyzes the phase slope of mixed oscillators:

- `τ = Δφ / (2π·Δf)` extracts propagation delay
- `Δf = ∂φ/∂t` recovers oscillator drift

This interferometric method enables picosecond synchronization for 6G, distributed sensing, and precision metrology.

### Chronometric Interferometry Method

![Chronometric Interferometry Visualization](docs/assets/images/chronometric_interferometry_enhanced.png)

Chronometric interferometry is a two-way time transfer method that uses heterodyne techniques to achieve high-precision time and frequency synchronization. The technical approach is as follows:

1. **Two-Way Time Transfer (TWTT)**: Two nodes exchange RF signals. This allows for the cancellation of common-mode noise sources.
2. **Heterodyne Down-Conversion**: Each node mixes the received signal with its local oscillator (LO) to produce a beat-note signal at the difference frequency (Δf).
3. **Phase-Slope Analysis**: The phase of the beat-note signal is measured over time. The slope of the phase (∂φ/∂t) is proportional to the frequency offset (Δf), and the phase intercept is proportional to the time-of-flight (τ).

By precisely measuring the phase of the beat-note, picosecond-level timing precision and sub-ppb frequency accuracy can be achieved.

---

## Highlights

| Capability | Result | Notes |
| --- | --- | --- |
| Timing precision | **~10 ps RMSE** | E1 median: 9.5 ps (range 1.5-30 ps) |
| Frequency accuracy | **0.05-40 ppb** | E1 baseline: 0.05 ppb (clean), ~20 ppb typical |
| Convergence | **< 100 ms** | Two-node consensus |
| Scalability | **500+ nodes** | Linear convergence verified |
| Fault tolerance | **33% malicious nodes** | Byzantine filtering |

- 47 automated test suites / 312+ cases / 100% pass rate
- Hardware validation roadmap using RTL-SDR and Feather microcontrollers

---

## Signal Reconstructions

| Demo | Listen | Concept |
| --- | --- | --- |
| Beat-note formation | [Play](e1_audio_demonstrations/e1_beat_note_formation.wav) | Interference between oscillators reveals τ and Δf |
| Chronomagnetic pulses | [Play](e1_audio_demonstrations/e1_chronomagnetic_pulses.wav) | Temporal frequency excursions |
| τ/Δf modulation | [Play](e1_audio_demonstrations/e1_tau_delta_f_modulation.wav) | Phase slope dynamics rendered in the audio band |

---

## Quick Start

```bash
git clone https://github.com/Shannon-Labs/driftlock-choir.git
cd driftlock-choir/driftlockchoir-oss
pip install -r requirements.txt
```

```bash
# Run the core experiment (E1)
python -m src.experiments.e1_basic_beat_note

# Explore examples
python examples/basic_beat_note_demo.py
python examples/oscillator_demo.py
python examples/basic_consensus_demo.py

# Validate the suite
pytest tests/ -v
```

Expected (E1): ~10 ps timing RMSE (1.5-30 ps range), 0.05-40 ppb frequency accuracy depending on SNR, visualization plots stored under `results/`.

---

## Experiment Suite

```
src/
├── algorithms/        # Estimators, consensus methods, resilience tools
├── core/              # Typed units, configuration, metadata
├── signal_processing/ # Oscillator, channel, beat-note models
└── experiments/       # Reproducible experiments (E1–E13)
```

Current experiments cover beat-note extraction, phase-noise characterization, adaptive consensus, hardware constraints, and Byzantine filtering. Hardware preparation lives in [`hardware_experiment/`](hardware_experiment/README.md) with firmware sketches and controller scripts.

---

## Documentation

- Interactive onboarding: https://shannon-labs.github.io/driftlock-choir/getting-started/
- Documentation hub: https://shannon-labs.github.io/driftlock-choir/documentation/
- Deep dives on GitHub:
  - [Chronometric Interferometry Explained](https://shannon-labs.github.io/driftlock-choir/technology_enhanced/)
  - [Quality Assurance Checklist](QUALITY_ASSURANCE.md)
  - [Release Readiness Board](RELEASE_READINESS.md)
  - [Getting Started Guide](GETTING_STARTED.md)
- Governance & history:
  - [Contributing](CONTRIBUTING.md)
  - [Code of Conduct](CODE_OF_CONDUCT.md)
  - [Changelog](CHANGELOG.md)
  - [Citation](CITATION.cff)

---

## Contributing & Support

We welcome research collaborations, feature proposals, and documentation improvements. To get involved:

1. Review the [contribution guide](CONTRIBUTING.md) and [code of conduct](CODE_OF_CONDUCT.md).
2. Open an [issue](https://github.com/Shannon-Labs/driftlock-choir/issues) or [discussion](https://github.com/Shannon-Labs/driftlock-choir/discussions).
3. Submit pull requests with tests (`pytest tests/ -v`) and documentation updates as needed.

For partnership inquiries contact **hunter@shannonlabs.dev**.

---

## Citation

If you use Driftlock Choir in your research, please cite it as below:

```
@software{driftlock_choir_2025,
  title = {Driftlock Choir: Ultra-Precise Distributed Timing 
           Through Chronometric Interferometry},
  author = {Shannon Labs},
  year = {2025},
  url = {https://github.com/Shannon-Labs/driftlock-choir},
  note = {Open-source framework demonstrating ~2.1 ps timing precision in simulation
          through musical-inspired RF synchronization; hardware validation in progress}
}
```

---

## License

Driftlock Choir is released under the [MIT License](LICENSE).
