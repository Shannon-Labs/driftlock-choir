# <div align="center">Driftlock Choir</div>

## <div align="center">Ultra-Precise Distributed Clock Synchronization</div>

<div align="center">

[![MIT License](https://img.shields.io/badge/license-MIT-blue)](https://github.com/Shannon-Labs/driftlock-choir/blob/main/LICENSE)
[![Python Version](https://img.shields.io/badge/python-3.8+-blue)](https://www.python.org/)

**Picosecond-level clock synchronization through chronometric interferometry**

</div>

---

## 🚀 What We've Achieved

<div align="center">

| Metric | Demonstrated Performance | Industry Standard |
|--------|---------------------|------------------|
| **Timing RMSE** | **~100 ps** (clean conditions) | ~100 ns (NTP) |
| **Frequency RMSE** | **~10 ppb** (clean conditions) | ~1 ppm (PTP) |
| **Beat Note Analysis** | **Sub-Hz precision** | Standard approaches |
| **Consensus Demo** | **Basic 2-node sync** | Manual coordination |

</div>

### Visual Results

<div align="center">

![E1 Experiment Results](e1_experiment_result.png)

*Experiment E1: Beat-note chronometric interferometry demonstrating sub-100ps timing extraction from simple oscillator interference patterns*

</div>

---

## 🎯 What This Demonstrates

### 🔬 **Chronometric Interferometry Foundation**
- Clean beat-note generation from dual oscillators
- Phase slope analysis for τ/Δf extraction
- Validated against analytical solutions
- Demonstrates the fundamental physics principles

### ⚡ **Signal Processing Pipeline**
- TCXO and ideal oscillator models
- AWGN channel simulation
- Real-time beat-note processing
- Instantaneous frequency analysis

### 🧮 **Basic Consensus Framework**
- Two-node synchronization demonstration
- Simple Metropolis consensus algorithm
- Convergence validation
- Foundation for distributed timing

### 📊 **Experiment E1: Beat-Note Validation**
Complete implementation of our foundational experiment proving that chronometric interferometry can extract precise timing and frequency information from simple beat patterns.

---

## 🏗️ Architecture Overview

```
driftlockchoir-oss/
├── src/
│   ├── core/              # Fundamental types and constants
│   ├── signal_processing/ # Oscillators, channels, beat-notes
│   ├── algorithms/        # Basic estimators and consensus
│   └── experiments/       # Experiment E1 implementation
├── examples/              # Usage demonstrations
├── tests/                 # Test suite
└── docs/                  # Documentation
```

---

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/Shannon-Labs/driftlock-choir.git
cd driftlock-choir

# Install dependencies
pip install -r requirements.txt

# Run the basic validation
python test_e1.py
```

### Minimal Working Example

```python
from src.experiments.e1_basic_beat_note import ExperimentE1
from src.experiments.runner import ExperimentContext

# Create experiment
experiment = ExperimentE1()
config = experiment.create_default_config()

# Create context
context = ExperimentContext(
    config=config,
    output_dir="results",
    random_seed=42,
    verbose=True
)

# Run experiment
result = experiment.run_experiment(context, config.parameters)

print(f"Success: {result.success}")
print(f"Timing RMSE: {result.metrics.rmse_timing:.1f} ps")
print(f"Frequency RMSE: {result.metrics.rmse_frequency:.1f} ppb")
```

### Running Experiment E1

```bash
# Run the core chronometric interferometry demonstration
python -m src.experiments.e1_basic_beat_note

# This will generate beat-note waveforms, analyze phase slopes,
# and demonstrate sub-100ps timing extraction
```

---

## 📊 Experiment E1: Core Demonstration

### What E1 Proves

**Hypothesis**: Two-way chronometric interferometry can extract τ (time-of-flight) and Δf (frequency offset) from simple beat patterns with known ground truth.

**Key Results**:
- ✅ Clean beat-note generation from dual oscillators
- ✅ Phase slope analysis achieving ~100 ps timing resolution
- ✅ Frequency estimation with ~10 ppb precision
- ✅ Validation against analytical solutions
- ✅ Noise floor characterization

### Visual Results

The experiment generates four key plots:
1. **Beat Note Waveform** - Clean interferometric signal
2. **Beat Note Spectrum** - Frequency domain analysis
3. **Instantaneous Phase** - Phase evolution over time
4. **Instantaneous Frequency** - Real-time frequency tracking

### Technical Validation

```python
# Expected vs Measured Results
true_tau_ps = 1000.0        # 1 ns time-of-flight
estimated_tau_ps = ~1000 ± 50  # Within 50 ps typically

true_delta_f_hz = 50.0      # 50 Hz frequency offset  
estimated_delta_f_hz = ~50 ± 1  # Within 1 Hz typically
```

---

## 🧪 Key Components

### Signal Processing

- **`Oscillator`**: TCXO and ideal oscillator models with configurable phase noise
- **`BeatNoteProcessor`**: Real-time beat-note generation and analysis
- **`ChannelSimulator`**: AWGN channel with thermal noise modeling

### Algorithms

- **`EstimatorFactory`**: Phase slope and advanced estimation methods
- **`ConsensusAlgorithm`**: Basic Metropolis consensus for distributed sync
- **`ExperimentRunner`**: Framework for reproducible experimental validation

### Experiments

- **`ExperimentE1`**: Complete beat-note formation and analysis demonstration
- **Parameter sweeps** across SNR and frequency offset ranges
- **Statistical validation** with uncertainty quantification

---

## 📈 Performance Characteristics

### Noise Performance

| SNR (dB) | Timing RMSE (ps) | Frequency RMSE (ppb) |
|----------|------------------|----------------------|
| 20       | ~500             | ~50                  |
| 30       | ~200             | ~20                  |
| 40       | ~100             | ~10                  |
| 50       | ~50              | ~5                   |

### Frequency Range

| Offset (Hz) | Performance | Notes |
|-------------|-------------|--------|
| 10-100      | Excellent   | Optimal beat-note range |
| 100-1000    | Good        | Wider bandwidth needed |
| 1000+       | Limited     | Sampling constraints |

---

## 🌟 What This Enables

### For Researchers
- **Physics Validation**: Verify chronometric interferometry principles
- **Algorithm Development**: Test new estimation and consensus methods
- **Educational Tool**: Learn distributed timing concepts

### For Developers
- **Foundation Framework**: Build upon proven signal processing pipeline
- **Consensus Playground**: Experiment with distributed synchronization
- **Performance Baseline**: Compare against demonstrated capabilities

### For Industry
- **Proof of Concept**: Demonstrate feasibility for specific applications
- **Technology Evaluation**: Assess fit for timing-critical systems
- **Integration Planning**: Understand requirements and interfaces

---

## 🛣️ Roadmap

- [x] **Core Experiment E1** - Beat-note validation and analysis
- [x] **Signal Processing Pipeline** - Oscillators, channels, noise models
- [x] **Basic Consensus** - Two-node synchronization demonstration
- [ ] **Enhanced Visualization** - Interactive plots and real-time displays
- [ ] **Hardware Interface** - Connect to real oscillators and networks
- [ ] **Multi-node Demos** - Scale beyond two-node synchronization
- [ ] **Performance Benchmarks** - Standardized test suites
- [ ] **Integration Examples** - Real-world application demonstrations

---

## 📄 License & Contact

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

**Interested in the full technology?** Contact Hunter at **hunter@shannonlabs.dev**

---

## 🤝 Contributing

We welcome contributions! Areas where help is especially valuable:

- **Documentation**: Improve explanations and add tutorials
- **Testing**: Expand test coverage and edge case validation
- **Visualization**: Enhance plots and add interactive displays
- **Examples**: Create new demonstration scripts
- **Hardware**: Interface with real timing hardware
- **Algorithms**: Implement alternative estimation methods

### Development Setup

```bash
git clone https://github.com/Shannon-Labs/driftlock-choir.git
cd driftlock-choir
pip install -r requirements.txt
python -m pytest tests/ -v
```

---



## 🔬 Scientific Foundation

This work builds on foundational research in:

- **Chronometric interferometry** and time-frequency metrology
- **Distributed consensus algorithms** and graph theory  
- **Wireless channel modeling** and signal processing
- **Clock synchronization** and timing systems

### Key Publications

*Publications and citations to be added as research is published*

---

## 📞 Contact

- **Questions & Collaboration**: hunter@shannonlabs.dev
- **Bug Reports**: [GitHub Issues](https://github.com/Shannon-Labs/driftlock-choir/issues)

---

<div align="center">

**⭐ Star this repository if you find it useful!**

**Building the future of distributed timing, one picosecond at a time**

Made with ❤️ by the Driftlock Choir Team

</div>