# Driftlock Choir: Shannon Labs Two-Pillar Security Stack

[![Patent Pending](https://img.shields.io/badge/Patent-Pending-orange)](patent/PROVISIONAL_PATENT_APPLICATION.md)
[![License: Academic](https://img.shields.io/badge/License-Academic_Free-green)](LICENSE-ACADEMIC.md)
[![License: Commercial](https://img.shields.io/badge/License-Commercial_Contact-red)](LICENSE-COMMERCIAL.md)
[![Shannon Labs](https://img.shields.io/badge/by-Shannon_Labs-blue)](https://shannonlabs.dev)
[![Entruptor](https://img.shields.io/badge/Entruptor-CbAD_Security-purple)](https://www.entruptor.com)
[![Website](https://img.shields.io/badge/Website-driftlock.net-cyan)](https://driftlock.net)

**Driftlock Choir** is the RF/time synchronization pillar of Shannon Labs' two-pillar security stack. **Entruptor** (CbAD software layer) + **Driftlock** (22ps synchronization) deliver secure, synchronized systems where information is both the signal and the security.

> 🚀 **Two-Pillar Security Stack**: Entruptor (CbAD anomaly detection) + Driftlock (22ps synchronization) = secure, synchronized systems. [Choir Simulation Lab →](driftlock_sim/README.md)

## 🚀 Two-Pillar Security Architecture

**Shannon Labs presents the two-pillar security stack:**

1. **Entruptor** (CbAD Software Layer) — Zero-training anomaly detection using compression-based analysis
2. **Driftlock** (RF/Time Layer) — 22ps synchronization feeding pristine signals to Entruptor

**Information is both the signal and the security.** Together they deliver secure, synchronized systems for critical infrastructure where intentional frequency offsets create beat patterns that encode both timing precision and security monitoring.

## ⚡ Two-Pillar Performance Targets

### Driftlock Choir (RF/Time Synchronization)
- **22.13 picoseconds** synchronization precision (Kalman filter tuned)
- **2,273× better than GPS** (50ns → 22ps)
- **4,500× bias reduction** via reciprocity calibration (12ns → 2.65ps)
- **Single iteration convergence** for 25-64 node networks
- **100% alias resolution** success rate at SNR ≥ 0dB

### Choir Simulation Lab Acceptance Criteria
- **Δf spike ≥ 15 dB** (aperture detection threshold)
- **RMSE ≤ 120 ps** (synchronization accuracy)
- **BER < 1e-3 at 20 dB** (bit error rate performance)
- **Runtime < 60 s** (computational efficiency)

**Latest run (see `driftlock_sim/outputs/csv/acceptance_summary.json`):**
- Δf aperture spike **58.1 dB** with 2Δf at **58.1 dB**
- Coherent RMSE **45.0 ps** with RMSE/CRLB **0.83** across 14 trials
- Payload coexistence adds **+8.6%** RMSE with **0 observed BER** @ 20 dB
- Full suite completes in **3.7 s** on Apple M2 Max (Python 3.12, NumPy 1.26)

Run `PYTHONPATH=. python driftlock_sim/sims/run_acceptance.py` to reproduce; artifacts land under `driftlock_sim/outputs/{csv,figs}`.

### Integration with Entruptor (CbAD)
- **Zero training required** for anomaly detection
- **Pristine signal feed** from 22ps synchronization
- **Information-theoretic security** through compression analysis

## 🎯 Two-Pillar Integration Demo

```python
import driftlock
from entruptor import CbADDetector

# Create two nodes with intentional frequency offset (Driftlock)
node_a = driftlock.Node(frequency=2.4e9)  # 2.4 GHz
node_b = driftlock.Node(frequency=2.4e9 + 1e6)  # 1 MHz offset

# Run synchronization (feeds pristine signals to Entruptor)
sync_result = driftlock.synchronize(node_a, node_b)
print(f"Timing accuracy: {sync_result.accuracy_ns} ns")

# Initialize Entruptor CbAD detector with synchronized signals
detector = CbADDetector(sync_result.signals)
anomaly_score = detector.detect_anomalies()
print(f"Anomaly detection confidence: {anomaly_score}")
```

## 📦 Installation

### Academic Users (Free)
```bash
git clone https://github.com/shannon-labs/driftlock
cd driftlock
pip install -r requirements.txt
python sim/phase1.py  # Run validation
```

### Commercial Users
Commercial use requires a license from Shannon Labs, Inc.
- **Contact**: licensing@shannonlabs.com
- **Website**: [driftlock.net](https://driftlock.net)

## 🔬 Two-Pillar Science

### Driftlock Choir (RF/Time Synchronization)
Driftlock uses **Chronometric Interferometry** - beat signals from intentionally offset carriers encode timing information:

```
φ_beat(t) = 2π Δf (t - τ) + phase_terms
```

**Key innovations:**
1. **Intentional frequency offset** creates measurable beat signals
2. **Phase extraction** from beats reveals propagation delay τ
3. **Two-way measurements** cancel clock bias
4. **Distributed consensus** with variance weighting achieves 22ps sync

### Entruptor Integration (CbAD Security)
**Compression-based anomaly detection** analyzes signal compressibility:

```
anomaly_score = 1 - (compressed_size / original_size)
```

**Key innovations:**
1. **Zero training** - works on any signal type
2. **Information-theoretic** - measures statistical anomalies
3. **Real-time processing** - sub-millisecond detection
4. **Pristine signal feed** from 22ps synchronization

[Read the full theory →](docs/theory.md) | [Choir Lab →](driftlock_sim/README.md)

## 📊 Latest Validation Results ([Extended Run 011](docs/results_extended_011.md))

| Metric | Value | Significance |
|--------|-------|-------------|
| **Network Consensus** | **22.13 ps** (dense w/ KF @ 0.32/0.03/1) | Tuned shrinkage beats baseline by 0.33 ps |
| | 20.96 ps (small w/ KF) | 14% improvement over no-filter |
| **Reciprocity Calibration** | **2.65 ps** bias | **4,500× improvement** from 12ns hardware bias |
| **Convergence** | **1 iteration** | Instant lock for production deployment |
| **Monte Carlo Validation** | **600+ simulations** | Full statistical confidence |
| **Guardrails** | Sweep verifier + seeded regression | scripts/verify_kf_sweep.py & pytest dense preset

### 🔬 Monte Carlo & Regression Framework
New automated validation pipeline:
```bash
# Run full Monte Carlo suite with presets
python scripts/run_mc.py all -c sim/configs/mc_extended.yaml -o results/mc_runs -r extended_011

# Verify dense sweep artifacts
scripts/verify_kf_sweep.py results/kf_sweeps/dense_combo_scan/kf_sweep_summary.json \
  --expected-min 20.9337 --expected-best-mean 21.89 \
  --expected-clock 0.32 --expected-freq 0.03 --expected-iterations 1

# Quick preset testing
python scripts/run_presets.py phase2-consensus --job dense_network_no_kf

# Regression guardrail
pytest tests/test_consensus.py::test_dense_kf_vs_baseline -q

# One-shot verification (regression + sweep)
scripts/run_verification_checks.sh
```

[View RMSE comparison chart →](results/mc_runs/extended_011/phase2_rmse_bar.png)

## 🏗️ Project Structure

```
driftlock/
├── sim/                    # Simulation framework (open source)
│   ├── phase1.py          # Two-node synchronization
│   ├── phase2.py          # Multi-node consensus
│   └── phase3.py          # Hardware calibration
├── src/                    # Core algorithms (patent pending)
│   ├── alg/               # Synchronization algorithms
│   ├── phy/               # Physical layer models
│   ├── mac/               # MAC layer scheduling
│   └── chan/              # Channel models
├── docs/                   # Documentation
│   ├── theory.md          # Mathematical foundation
│   ├── quickstart.md      # Getting started guide
│   └── results.md         # Performance validation
├── patent/                 # Patent materials (provisional, claims, figures, prior art)
└── tests/                  # Test suite
```

---

## 🎓 Citation

If you use Driftlock in your research, please cite:

```bibtex
@software{driftlock2025,
  title = {Driftlock: Sub-Nanosecond Wireless Synchronization via Chronometric Interferometry},
  author = {Bown, Hunter},
  year = {2025},
  organization = {Shannon Labs, Inc.},
  note = {Patent Pending - Driftlock Method}
}
```

## 🛡️ Patent Notice

**Patent Pending**: Driftlock Method
Provisional Patent Application Filed: September 19, 2025

This repository contains:
- ✅ **Simulation code**: Open source for validation and research
- ✅ **Academic license**: Free for research and education
- ⚠️ **Core innovation**: Protected by pending patent
- 📧 **Commercial use**: Requires license from Shannon Labs

## 🚀 Two-Pillar Applications

The Entruptor + Driftlock stack enables revolutionary secure, synchronized systems:

### Critical Infrastructure Security
- **5G/6G Networks**: Secure coordinated beamforming with anomaly detection
- **Financial Systems**: Synchronized trading with fraud detection
- **Industrial IoT**: Precision manufacturing with intrusion detection
- **Smart Grids**: Synchronized power distribution with anomaly monitoring

### Scientific & Research
- **Quantum Networks**: Synchronized quantum state measurements with security validation
- **Radio Astronomy**: Ultra-precise timing arrays with signal integrity verification
- **Gravitational Wave Detection**: Coordinated sensor networks with anomaly filtering

### Autonomous Systems
- **Swarm Robotics**: Synchronized coordination with security monitoring
- **Autonomous Vehicles**: Precise timing with environmental anomaly detection
- **Distributed Sensing**: Coordinated sensor networks with threat detection

## 🤝 Get Involved

### Academics & Researchers
- 📚 Read the [theory paper](docs/theory.md)
- 🧪 Run the [simulations](docs/quickstart.md)
- 💡 Contribute improvements via pull requests
- 📝 Cite us in your research

### Industry & Commercial
- 📊 Evaluate the [performance data](docs/results.md)
- 💼 Contact for licensing: licensing@shannonlabs.com
- 🔗 Visit [driftlock.net](https://driftlock.net)
- 🤝 Partner with Shannon Labs

### Everyone
- ⭐ **Star this repo** to follow our progress
- 👀 Watch for hardware validation results
- 💬 Join the discussion in Issues
- 🐛 Report bugs and suggest improvements

---

## 📈 Status

- ✅ **2.081 nanosecond** synchronization achieved in simulation
- ✅ **500 Monte Carlo trials** validated
- ✅ **Patent Pending** (September 2025)
- 🚧 Hardware validation in progress
- 📧 Commercial inquiries: licensing@shannonlabs.com

## 🏢 About Shannon Labs

[Shannon Labs](https://shannonlabs.dev) is advancing the frontiers of information theory and wireless communications. Our two-pillar security stack combines:

- **Entruptor**: Compression-based anomaly detection (CbAD) for zero-training security
- **Driftlock**: 22 picosecond wireless synchronization for pristine signal delivery

Together they deliver secure, synchronized systems where information is both the signal and the security.

**Contact**: hello@shannonlabs.dev
**Website**: [shannonlabs.dev](https://shannonlabs.dev)
**Entruptor**: [entruptor.com](https://www.entruptor.com)
**Driftlock**: [driftlock.net](https://driftlock.net)

---

<div align="center">

**🌟 Help us reach 1000+ stars and establish Driftlock as the community standard for wireless synchronization!**

[![Star Driftlock](https://img.shields.io/github/stars/shannon-labs/driftlock?style=social)](https://github.com/shannon-labs/driftlock)

</div>
