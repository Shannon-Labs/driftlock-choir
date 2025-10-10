# Chronometric Interferometry for Picosecond-Scale Wireless Synchronization

![Chronometric Interferometry](docs/assets/images/chronometric_interferometry.png)

> An open-source framework for picosecond-scale wireless synchronization, actively transitioning from simulation to hardware validation.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)

## Overview

This repository contains a research project developing **chronometric interferometry**, a technique to synchronize distributed systems over-the-air with picosecond-level precision. The goal is to deliver the timing performance of dedicated fiber optics with the flexibility and low cost of wireless radio.

The repository provides the full software stack—from first-principles physics models to statistical estimators and hardware bridges—to design, simulate, and validate this timing infrastructure.

## Core Concepts

### What is Chronometric Interferometry?
Our method is a form of heterodyne interferometry. We mix two signals with a known frequency offset (Δf) to produce an intermediate frequency (the "beat note") whose phase is directly proportional to the propagation delay (τ). This down-conversion allows us to measure the phase of a low-frequency beat note with high precision, which is far more feasible with commodity ADCs than trying to directly measure the phase of the multi-gigahertz carrier.

### Why Not Just Use an Atomic Clock?
This system addresses time *distribution* (syntonization and synchronization), not time *generation*. Atomic clocks provide a highly stable frequency reference, but they don't solve the problem of delivering that reference's phase to distributed nodes. This project provides a time-transfer technology that could use an atomic clock as its primary reference oscillator to create an end-to-end timing network that is both stable *and* precisely distributed.

### The Primary Challenge: Multipath Fading
The primary challenge is multipath fading. The current phase-slope estimator assumes a single, line-of-sight (LoS) propagation path. In a real environment, the received signal is a superposition of multiple paths, each with a different delay and attenuation. This corrupts the simple linear relationship between frequency and phase. Overcoming this requires developing advanced estimators that can identify the LoS path in a dense multipath environment.

## Performance

Achieving picosecond-level wireless timing enables new applications across various fields.

**13.5 Picoseconds in Perspective:**
*   **Distance:** Light travels just **4 millimeters** in 13.5 ps.
*   **Comparison:** This is over 100x more precise than standard GPS and works indoors. It is competitive with the White Rabbit protocol, a standard for wired timing, without the need for dedicated fiber optic cables.

| Technology | Typical Precision | Medium |
| :--- | :--- | :--- |
| NTP | Milliseconds (10⁻³ s) | Internet |
| PTP | Microseconds (10⁻⁶ s) | Ethernet |
| GPS | Nanoseconds (10⁻⁹ s) | Satellite RF |
| White Rabbit | **Tens of Picoseconds** (10⁻¹² s) | **Fiber Optic** |
| **This Project** | **Tens of Picoseconds** (10⁻¹² s) | **Wireless RF** |

This capability is an enabler for:
*   **6G Wireless:** Joint Communication and Sensing (JCAS), where base stations act as a coherent radar system.
*   **Autonomous Systems:** Sensor fusion for robots or autonomous vehicles with near-perfect correlation.
*   **Distributed Computing:** Coherent signal processing applications like low-cost radio telescopes.

## Project Status

The project is currently in **Phase 1: Foundational Credibility**. The core algorithms have been validated in a comprehensive Python simulation, and we are now focused on hardware implementation.

*   **Simulation Results:** Achieved **13.5 ps** line-of-sight timing recovery and **0.09 ps** residual error in high-band simulations.
*   **Next Step:** The immediate goal is to reproduce these results on real hardware.

## Quick Start

Get the code and install dependencies:
```bash
git clone https://github.com/Shannon-Labs/driftlock-choir.git
cd driftlock-choir
pip install -r requirements.txt
```

Run the baseline E1 chronometric interferometry experiment (13.5 ps delay):
```bash
python e1_beat_note_analysis.py --tau-ps 13.5 --delta-f-hz 150
```

Explore the results and visualizations in the interactive walkthrough notebook:
```bash
jupyter notebook examples/e1_cli_walkthrough.ipynb
```

## Collaboration

This project uses an "Open Core" model. The core simulation framework is open-source (MIT) to foster academic collaboration and peer review. High-performance, hardware-specific implementations (e.g., for FPGAs) may be proprietary. We are committed to validating our work through peer-reviewed publication.

## Citation

If you use this work in your research, please cite it as:
```
@software{chronometric_interferometry_2025,
  title = {An Open-Source Framework for Picosecond-Scale Wireless Synchronization via Chronometric Interferometry},
  author = {Shannon Labs and Community Contributors},
  year = {2025},
  url = {https://github.com/Shannon-Labs/driftlock-choir},
  note = {Version X.Y.Z, hardware validation in progress}
}
```

## License

Released under the [MIT License](LICENSE).
