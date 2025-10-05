# Getting Started with Driftlock Choir

Driftlock Choir demonstrates ultra-precise clock synchronization through chronometric interferometry.

## Quick Start

### 1. Installation
```bash
# Clone or copy this folder
cd driftlockchoir-oss

# Install dependencies
pip install -r requirements.txt
```

### 2. Verify Installation
```bash
# Quick validation test
python quick_test.py

# Basic functionality test
python test_e1.py
```

### 3. Run Examples

#### Basic Beat Note Demo
```bash
python examples/basic_beat_note_demo.py
```
This demonstrates chronometric interferometry fundamentals by generating beat notes from two oscillators and extracting timing information.

#### Oscillator Comparison Demo  
```bash
python examples/oscillator_demo.py
```
Shows different oscillator models (ideal vs TCXO) and channel effects.

#### Basic Consensus Demo
```bash
python examples/basic_consensus_demo.py
```
Demonstrates two-node consensus synchronization.

### 4. Run Experiment E1
```bash
# Run the core experiment
python -m src.experiments.e1_basic_beat_note
```

This validates the fundamental physics of chronometric interferometry with clean beat-note generation and sub-100ps timing extraction.

## What You'll See

### Expected Performance
- **Timing accuracy**: ~100 ps RMSE in clean conditions
- **Frequency accuracy**: ~10 ppb in clean conditions  
- **Beat note analysis**: Sub-Hz frequency precision
- **Consensus convergence**: 5-10 iterations for 2-node networks

### Generated Outputs
- Beat note waveforms and analysis plots
- Convergence history visualizations
- Performance metrics and statistics
- Validation against theoretical predictions

## Key Components

### Core Signal Processing
- `Oscillator`: Ideal and TCXO models with phase noise
- `BeatNoteProcessor`: Real-time beat note generation and analysis
- `ChannelSimulator`: AWGN channel with thermal noise

### Algorithms
- `EstimatorFactory`: Phase slope and ML estimation methods
- `MetropolisConsensus`: Basic distributed consensus algorithm
- Parameter sweep and statistical validation tools

### Experiments
- `ExperimentE1`: Complete beat-note validation experiment
- Reproducible experimental framework with detailed logging

## Understanding the Results

### Beat Note Analysis
The beat note plots show:
1. **Waveform**: Time-domain interferometric signal
2. **Spectrum**: Frequency content revealing beat frequencies
3. **Instantaneous Phase**: Phase evolution over time
4. **Instantaneous Frequency**: Real-time frequency tracking

### Timing Extraction
The phase slope method extracts:
- **τ (tau)**: Time-of-flight delay in picoseconds
- **Δf (delta-f)**: Frequency offset in Hz
- **Uncertainties**: Statistical estimation bounds

### Performance Validation
Results are validated against:
- Known ground truth values
- Cramér-Rao lower bounds
- Analytical solutions for simple cases

## Next Steps

### For Researchers
- Modify oscillator parameters and noise models
- Implement new estimation algorithms
- Explore multi-node consensus scenarios

### For Developers  
- Integrate with real hardware interfaces
- Add new visualization and analysis tools
- Extend to larger network topologies

### For Advanced Applications
Interested in the full technology? Contact hunter@shannonlabs.dev

## Troubleshooting

### Common Issues
1. **Import errors**: Ensure Python path includes src directory
2. **Matplotlib warnings**: Install latest matplotlib for plotting
3. **Numerical precision**: Results may vary slightly across platforms
4. **Random seed**: Set seed=42 for reproducible results

### Performance Notes
- Longer signal durations improve estimation accuracy
- Higher SNR conditions give better performance
- Complex scenarios may require parameter tuning

## Support

- **Documentation**: See README.md for full details
- **Issues**: Report bugs and feature requests on GitHub
- **Questions & Collaboration**: hunter@shannonlabs.dev

---

**Welcome to the future of distributed timing!** 🚀