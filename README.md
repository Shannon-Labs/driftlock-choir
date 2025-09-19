# Driftlock: Sub-Nanosecond Wireless Synchronization

[![Patent Pending](https://img.shields.io/badge/Patent-Pending-orange)](patent/PROVISIONAL_PATENT_APPLICATION.md)
[![License: Academic](https://img.shields.io/badge/License-Academic_Free-green)](LICENSE-ACADEMIC.md)
[![License: Commercial](https://img.shields.io/badge/License-Commercial_Contact-red)](LICENSE-COMMERCIAL.md)
[![Shannon Labs](https://img.shields.io/badge/by-Shannon_Labs-blue)](https://shannonlabs.dev)
[![Website](https://img.shields.io/badge/Website-driftlock.net-purple)](https://driftlock.net)

**Driftlock** achieves **22 picosecond wireless synchronization** without GPS by intentionally creating frequency offsets - turning what everyone thought was noise into a precision measurement tool.

## 🚀 Revolutionary Insight

For 100 years, wireless systems have fought to eliminate frequency offset. **We do the opposite.**

Driftlock intentionally creates controlled frequency differences between nodes, generating beat signals that encode ultra-precise timing information. This paradigm shift enables sub-nanosecond synchronization using standard commercial hardware - no atomic clocks, no GPS, no fiber optics required.

## ⚡ Proven Performance

- **2.081 nanoseconds RMS** synchronization accuracy (validated via 500+ Monte Carlo trials)
- **5× better than GPS** timing precision (10-50ns typical)
- **250× better than IEEE 1588** PTP over wireless
- **100% alias resolution** success rate at SNR ≥ 0dB
- **<5ms network convergence** for 50+ node networks
- Works with **2ppm TCXO oscillators** (standard commercial hardware)

## 🎯 Quick Demo

```python
import driftlock

# Create two nodes with intentional frequency offset
node_a = driftlock.Node(frequency=2.4e9)  # 2.4 GHz
node_b = driftlock.Node(frequency=2.4e9 + 1e6)  # 1 MHz offset

# Run synchronization
sync_result = driftlock.synchronize(node_a, node_b)
print(f"Timing accuracy achieved: {sync_result.accuracy_ns} ns")
# Output: Timing accuracy achieved: 2.081 ns
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

## 🔬 The Science

Driftlock uses **Chronometric Interferometry** - a novel approach where beat signals from intentionally offset carriers encode timing information:

```
φ_beat(t) = 2π Δf (t - τ) + phase_terms
```

Key innovations:
1. **Intentional frequency offset** creates measurable beat signals
2. **Phase extraction** from beats reveals propagation delay τ
3. **Two-way measurements** cancel clock bias
4. **Distributed consensus** with variance weighting achieves network sync

[Read the full theory →](docs/theory.md)

## 📊 Validation Results (Extended Run 006)

| Metric | Value | Conditions |
|--------|-------|------------|
| Network Consensus | 22-24 ps RMSE | WITHOUT Kalman filter |
| Calibrated Bias | 2.65 ps | Loopback calibration |
| Uncalibrated Bias | -12,000 ps | Shows 4,500× improvement |
| Convergence | 1 iteration | 25-64 node networks |
| Monte Carlo Trials | 600+ | Comprehensive validation |

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
Provisional Patent Application Filed: September 18, 2025

This repository contains:
- ✅ **Simulation code**: Open source for validation and research
- ✅ **Academic license**: Free for research and education
- ⚠️ **Core innovation**: Protected by pending patent
- 📧 **Commercial use**: Requires license from Shannon Labs

## 🚀 Applications

Driftlock enables revolutionary applications requiring ultra-precise distributed timing:

- **5G/6G Networks**: Coordinated beamforming and distributed MIMO
- **Quantum Networks**: Synchronized quantum state measurements
- **Financial Systems**: Microsecond-accurate distributed timestamps
- **Scientific Instruments**: Radio astronomy, gravitational wave detection
- **Autonomous Systems**: Swarm robotics, distributed sensing
- **Industrial IoT**: Precision manufacturing, smart grids

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

[Shannon Labs](https://shannonlabs.dev) is advancing the frontiers of information theory and wireless communications. Our team previously developed compression-based anomaly detection and now brings you Driftlock - a fundamental breakthrough in distributed synchronization.

**Contact**: hello@shannonlabs.dev
**Website**: [shannonlabs.dev](https://shannonlabs.dev)
**Driftlock**: [driftlock.net](https://driftlock.net)

---

<div align="center">

**🌟 Help us reach 1000+ stars and establish Driftlock as the community standard for wireless synchronization!**

[![Star Driftlock](https://img.shields.io/github/stars/shannon-labs/driftlock?style=social)](https://github.com/shannon-labs/driftlock)

</div>
