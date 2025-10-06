# E1 Hardware Experiment
## Real-World Implementation of RF Chronometric Interferometry

This directory contains a complete hardware implementation of the E1 chronometric interferometry experiment using **Adafruit Feather M4 Express** boards and **RTL-SDR**.

## Quick Start

1. **Flash the Feathers**: Load firmware onto two Feather M4 boards
2. **Install Python dependencies**: `pip install -r requirements.txt`
3. **Run the experiment**: `python3 e1_hardware_controller.py --ref-port /dev/ttyACM0 --offset-port /dev/ttyACM1`

## Files Overview

### Firmware
- `feather_firmware_reference.ino` - Reference signal generator (1.000000 MHz)
- `feather_firmware_offset.ino` - Offset signal generator (1.000100 MHz)

### Control Software
- `e1_hardware_controller.py` - Main experiment controller
- `requirements.txt` - Python dependencies

### Documentation
- `EXPERIMENT_INSTRUCTIONS.md` - Complete step-by-step guide
- `README.md` - This file

## Hardware Requirements

- 2x Adafruit Feather M4 Express boards
- 1x RTL-SDR v3 dongle
- Simple wire antennas (~17cm each)
- USB cables

## Expected Results

The experiment demonstrates:
- **Beat frequency measurement**: 100 Hz ± 5 Hz
- **Timing precision**: ~100 ps
- **SNR**: >20 dB with proper setup

## What This Demonstrates

This hardware implementation validates the core principles of **RF Chronometric Interferometry**:

1. **Signal Generation**: Two precisely offset RF signals
2. **Beat Note Creation**: Natural interference between signals  
3. **Phase Analysis**: Extraction of timing from beat patterns
4. **Frequency Estimation**: Precise measurement of frequency offsets

These are the same principles used in advanced wireless timing systems for 5G networks, precision positioning, and scientific instrumentation.

## Experiment Flow

```
Feather A (433.0 MHz) ─┐
                       ├── RTL-SDR ──> Beat Analysis ──> Results
Feather B (433.1 MHz) ─┘
```

The 100 Hz beat frequency contains both timing and frequency information that can be extracted using the same algorithms demonstrated in the software E1 experiment.

## Scientific Significance

This experiment bridges the gap between theoretical chronometric interferometry and real-world implementation, demonstrating:

- **Practical feasibility** of picosecond timing with commodity hardware
- **Scalability** to larger networks and higher frequencies
- **Robustness** of the underlying physics principles

## Support

See `EXPERIMENT_INSTRUCTIONS.md` for detailed setup and troubleshooting.

For questions: hunter@shannonlabs.dev