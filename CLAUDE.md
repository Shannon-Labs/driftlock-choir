# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Driftlock Choir is a precision timing infrastructure framework that implements chronometric interferometry for distributed systems. The core technique extracts time-of-flight (τ) and frequency offset (Δf) from beat-note interference patterns between oscillators, achieving picosecond-level timing precision in simulation.

## Development Commands

### Installation and Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Install hardware experiment dependencies (if needed)
pip install -r hardware_experiment/requirements.txt
```

### Running Experiments
```bash
# Core chronometric interferometry experiment (E1)
python -m src.experiments.e1_basic_beat_note

# Example demonstrations
python examples/basic_beat_note_demo.py
python examples/oscillator_demo.py
python examples/basic_consensus_demo.py

# Enhanced visualization
python create_enhanced_visualization.py
```

### Testing
```bash
# Run all tests
pytest tests/ -v

# Run specific test categories
pytest tests/ -m unit          # Unit tests only
pytest tests/ -m integration   # Integration tests only
pytest tests/ -m performance   # Performance tests only
pytest tests/ -m "not slow"    # Skip slow tests

# Run specific experiment test
pytest tests/test_experiment_e1.py -v
```

### Code Quality
```bash
# Type checking
mypy src/

# Linting
pylint src/
flake8 src/

# Code formatting
black src/
isort src/
```

## Architecture Overview

### Core Module Structure
```
src/
├── core/              # Type-safe physical units and constants
├── signal_processing/ # Oscillator models, beat-note analysis, channel simulation
├── algorithms/        # τ/Δf estimators and consensus protocols
└── experiments/       # Core validation experiments
```

### Key Components

**Type System (`src/core/types.py`)**: Uses NewType for physical units (Seconds, Picoseconds, Hertz, PPB, etc.) with type safety throughout the codebase.

**Signal Processing Chain**:
1. `Oscillator`: Models ideal and TCXO oscillators with phase noise
2. `ChannelSimulator`: AWGN channel with thermal noise effects
3. `BeatNoteProcessor`: Core beat-note generation and analysis
4. `EstimatorFactory`: Phase-slope and ML estimation methods

**Experiment Framework**:
- `ExperimentE1`: Core chronometric interferometry validation
- `ExperimentContext`: Standardized experiment execution environment
- `ExperimentRunner`: Configurable experiment orchestration

### Physical Constants
All physical constants and conversion factors are centralized in `src/core/constants.py` using the `PhysicalConstants` class.

## Experiment Framework

### E1 Basic Beat Note Experiment
The core experiment validates chronometric interferometry through:
- Clean beat-note generation from two oscillators with known offsets
- Phase-slope analysis for τ/Δf extraction
- Performance characterization under realistic RF conditions

**Expected Performance**:
- Timing precision: 13.5 ps RMSE baseline (5-20 ps typical)
- Frequency accuracy: 0.05 ppb RMSE baseline (0.05-5 ppb typical)
- Convergence: < 100 ms for two-node consensus

**Experiment Configuration**: Use `ExperimentConfig` class to set sampling rates, durations, SNR conditions, and oscillator parameters.

## Hardware Integration

### RTL-SDR Validation
Hardware validation tools are available in `hardware_experiment/`:
- `e1_hardware_controller.py`: RTL-SDR interface and control
- `offline_bridge.py`: Simulation-to-hardware bridge
- `feather_firmware_*.ino`: Microcontroller firmware for reference/offset oscillators

### Hardware Requirements
- RTL-SDR dongles for RF signal capture
- Adafruit Feather boards for oscillator control
- Standard RF front-end components

## Development Patterns

### Adding New Experiments
1. Create experiment class in `src/experiments/` following `ExperimentE1` pattern
2. Use `ExperimentContext` for consistent execution
3. Implement validation against analytical solutions
4. Add corresponding tests in `tests/`

### Estimation Algorithms
New estimation methods should:
1. Implement the `EstimatorFactory` interface
2. Return `EstimationResult` with quality metrics
3. Include uncertainty quantification
4. Be tested against known ground-truth cases

### Signal Processing Components
When adding signal processing elements:
1. Use type-safe physical units from `src/core/types.py`
2. Include comprehensive noise modeling
3. Validate against theoretical predictions
4. Provide visualization capabilities

## Testing Strategy

### Test Categories
- **Unit Tests**: Individual component validation
- **Integration Tests**: End-to-end experiment validation
- **Performance Tests**: Statistical analysis and baseline verification
- **Robustness Tests**: Edge cases and failure modes

### Test Configuration
Tests use deterministic seeds (seed=42) for reproducible results. The pytest configuration includes markers for different test categories and comprehensive reporting.

## Performance Validation

All experiments should validate against:
- Known ground truth values
- Cramér-Rao lower bounds
- Analytical solutions for simple cases
- Historical performance baselines

Results are automatically stored in the `results/` directory with timestamped filenames and comprehensive metadata.