# DriftLock

DriftLock is a comprehensive simulation framework for physical-layer synchronization in wireless networks, incorporating hardware imperfections, realistic channel models, and distributed consensus algorithms.

## Overview

DriftLock provides a complete simulation environment for analyzing synchronization performance in wireless networks under realistic conditions. The framework includes:

- **Physical-layer models**: Allan deviation noise, channel effects, and various noise sources
- **Hardware imperfections**: Oscillator drift, ADC effects, IQ imbalance, and transceiver models
- **Advanced algorithms**: Closed-form estimators, consensus algorithms, and Kalman filtering
- **Network protocols**: MAC layer simulation, topology generation, and asynchronous behavior
- **Comprehensive metrics**: CRLB analysis, bias-variance decomposition, and conditioning analysis

## Architecture

```
DriftLock/
├── src/
│   ├── phy/               # Physical-layer truth
│   │   ├── osc.py         # Allan-deviation noise generators
│   │   ├── chan.py        # LOS + Doppler + multipath (K-factor)
│   │   └── noise.py       # AWGN, phase-noise, jitter
│   ├── hw/                # Hardware imperfection layer
│   │   ├── trx.py         # TransceiverNode class
│   │   ├── lo.py          # Unlocked LO drift model
│   │   ├── adc.py         # Sub-sampling ADC: aperture jitter, ENOB
│   │   └── iq.py          # IQ imbalance & DC offset
│   ├── alg/               # Algorithms
│   │   ├── ci.py          # Closed-form τ, Δf extractor
│   │   ├── consensus.py   # Vanilla + Chebyshev accelerated
│   │   └── kalman.py      # Optional EKF for temporal fusion
│   ├── net/               # Network & protocol
│   │   ├── topo.py        # Random-geometric graph builder
│   │   ├── mac.py         # BEACON / RESPONSE packet sim
│   │   └── async.py       # Random tx offset & drop model
│   └── metrics/
│       ├── crlb.py        # Joint CRLB for τ, Δf
│       ├── biasvar.py     # Bias-variance decomposition
│       └── cond.py        # Jacobian condition-number scan
├── sim/
│   ├── phase1.py          # Two-node physical-limit sweep
│   ├── phase2.py          # 50-node convergence
│   ├── phase3.py          # Mobility, scale, oscillator sweep
│   └── configs/           # YAML parameter files
├── scripts/
│   └── run_mc.py          # Fire-and-forget Monte-Carlo runner
└── README.md              # Usage & replication instructions
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/shannonlabs/driftlock.git
cd driftlock
```

2. Install dependencies:
```bash
pip install numpy scipy matplotlib pyyaml dataclasses
```

## Usage

### Quick Start

Run a basic two-node synchronization analysis:

```python
from sim.phase1 import Phase1Simulator, Phase1Config

# Configure simulation
config = Phase1Config(
    snr_range_db=list(range(0, 31, 5)),
    bandwidth_range=[1e5, 5e5, 1e6, 5e6, 1e7],
    duration_range=[1e-4, 5e-4, 1e-3, 5e-3, 1e-2],
    n_monte_carlo=100
)

# Run simulation
simulator = Phase1Simulator(config)
results = simulator.run_full_simulation()
```

### Command-Line Interface

Use the Monte Carlo runner for comprehensive simulations:

```bash
# Run Phase 1 simulation with default configuration
python scripts/run_mc.py phase1

# Run all phases with custom configuration and parallel processing
python scripts/run_mc.py all -c configs/custom.yaml -w 4 -o results/experiment1

# Run Phase 2 with specific run ID
python scripts/run_mc.py phase2 --run-id convergence_analysis
```

### Configuration

Simulations are configured using YAML files. See `sim/configs/default.yaml` for a complete example:

```yaml
# Network Parameters
network:
  n_nodes: 50
  area_size: 1000.0
  comm_range: 200.0

# Physical Layer Parameters
physical:
  carrier_freq: 2.4e9
  sample_rate: 1.0e6
  snr_db: 15.0

# Simulation Parameters
simulation:
  duration: 10.0
  n_monte_carlo: 100
  save_results: true
```

## Simulation Phases

### Phase 1: Two-Node Physical Limits

Explores fundamental synchronization limits between two nodes:
- SNR vs. estimation performance
- Bandwidth effects on delay/frequency estimation
- Observation time optimization
- CRLB comparison with practical estimators

```python
from sim.phase1 import Phase1Simulator, Phase1Config

config = Phase1Config(
    snr_range_db=list(range(-10, 31, 2)),
    bandwidth_range=[1e5, 2e5, 5e5, 1e6, 2e6, 5e6, 1e7],
    duration_range=[1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2]
)

simulator = Phase1Simulator(config)
results = simulator.run_full_simulation()
```

### Phase 2: 50-Node Convergence

Analyzes distributed synchronization in multi-node networks:
- Network topology effects
- Consensus algorithm comparison (vanilla vs. accelerated)
- Scalability analysis
- Robustness to node failures and measurement errors

```python
from sim.phase2 import Phase2Simulator, Phase2Config

config = Phase2Config(
    n_nodes=50,
    max_iterations=1000,
    tolerance=1e-6,
    n_monte_carlo=100
)

simulator = Phase2Simulator(config)
results = simulator.run_full_simulation()
```

### Phase 3: Advanced Scenarios

Explores complex scenarios with mobility and hardware variations:
- Node mobility effects (random walk, linear motion)
- Large-scale network analysis
- Oscillator characteristic variations
- Combined effect analysis

```python
from sim.phase3 import Phase3Simulator, Phase3Config

config = Phase3Config(
    mobility_speeds=[0.0, 1.0, 5.0, 10.0, 20.0],
    network_sizes=[20, 30, 50, 75, 100],
    allan_dev_values=[1e-11, 1e-10, 1e-9, 1e-8, 1e-7],
    n_monte_carlo=50
)

simulator = Phase3Simulator(config)
results = simulator.run_full_simulation()
```

## Key Features

### Physical Layer Models

- **Oscillator Models**: Allan deviation-based noise generation with realistic frequency drift
- **Channel Models**: Multipath fading with Rician K-factor, Doppler effects, path loss
- **Noise Models**: AWGN, phase noise, timing jitter with configurable parameters

### Hardware Imperfections

- **Transceiver Models**: Complete RF front-end simulation with realistic impairments
- **Local Oscillator**: Unlocked LO with temperature drift, aging, and phase noise
- **ADC Effects**: Quantization noise, aperture jitter, ENOB degradation
- **IQ Imbalance**: Amplitude/phase imbalance, DC offset, image rejection analysis

### Advanced Algorithms

- **Parameter Estimation**: ML, LS, and ESPRIT-based delay/frequency estimators
- **Consensus Algorithms**: Vanilla and Chebyshev-accelerated distributed consensus
- **Kalman Filtering**: Extended Kalman filter for temporal parameter tracking

### Comprehensive Metrics

- **CRLB Analysis**: Joint Cramér-Rao lower bounds for delay and frequency estimation
- **Bias-Variance**: Complete bias-variance decomposition with confidence intervals
- **Conditioning**: Jacobian analysis for parameter estimation robustness

## Results and Analysis

Simulation results include:

1. **Performance Curves**: Estimation error vs. SNR, bandwidth, duration
2. **Convergence Analysis**: Consensus convergence rates and success probabilities
3. **Scalability Metrics**: Performance vs. network size and complexity
4. **Robustness Analysis**: Effects of node failures and measurement errors
5. **Hardware Impact**: Performance degradation due to realistic imperfections

Results are automatically saved in JSON format with accompanying visualization plots.

## Extending the Framework

### Adding New Models

1. **Physical Models**: Extend classes in `src/phy/` for new channel or noise models
2. **Hardware Models**: Add new imperfection models in `src/hw/`
3. **Algorithms**: Implement new estimators or consensus algorithms in `src/alg/`
4. **Metrics**: Add custom performance metrics in `src/metrics/`

### Custom Simulations

Create custom simulation scripts following the pattern in `sim/`:

```python
from src.phy.osc import OscillatorParams
from src.alg.ci import ClosedFormEstimator
from src.metrics.crlb import JointCRLBCalculator

# Define custom simulation class
class CustomSimulator:
    def __init__(self, config):
        self.config = config
        
    def run_simulation(self):
        # Implement custom simulation logic
        pass
```

## Performance Optimization

- **Parallel Processing**: Use the `--workers` option for multi-core execution
- **Reduced Trials**: Adjust `n_monte_carlo` for faster prototyping
- **Selective Phases**: Run individual phases instead of full simulation suite
- **Configuration Tuning**: Optimize parameters in YAML configuration files

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-model`)
3. Commit changes (`git commit -am 'Add new channel model'`)
4. Push to branch (`git push origin feature/new-model`)
5. Create Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use DriftLock in your research, please cite:

```bibtex
@software{driftlock2024,
  title={DriftLock: Physical-Layer Synchronization Framework},
  author={Shannon Labs},
  year={2024},
  url={https://github.com/shannonlabs/driftlock}
}
```

## Contact

For questions and support, please open an issue on GitHub or contact the Shannon Labs team.
