# Hardware Experiment Instructions
## RF Chronometric Interferometry with Adafruit Feathers and RTL-SDR

This document provides complete step-by-step instructions for performing the hardware experiment demonstrating chronometric interferometry using real hardware.

## Overview

This experiment demonstrates the core principles of **RF Chronometric Interferometry** by:
- Generating two RF signals with a precise 100 Hz frequency offset
- Capturing both signals simultaneously using RTL-SDR
- Analyzing beat notes to extract timing and frequency information
- Validating the chronometric interferometry concept with real hardware

**Expected Results:**
- Beat frequency: 100 Hz ± 5 Hz
- Timing precision: ~100 ps
- SNR: >20 dB (good signal quality)

## Required Hardware

### Essential Components
1. **Two Adafruit Feather M4 Express boards** - [Product Link](https://www.adafruit.com/product/3857)
   - ARM Cortex-M4 @ 120 MHz
   - Built-in 12-bit DAC
   - USB connectivity

2. **RTL-SDR v3 Dongle** - Generic RTL2832U + R820T2 
   - Frequency range: 24-1766 MHz
   - USB 2.0 connectivity
   - SMA antenna connector

3. **USB Cables**
   - 2x USB-A to Micro-USB (for Feathers)
   - 1x USB-A extension (for RTL-SDR)

4. **Simple Wire Antennas**
   - 2x ~17cm wire pieces (¼ wavelength at 433 MHz)
   - Alternatively: commercial 433 MHz antennas

### Optional Components
- **Breadboards** (for antenna connections)
- **SMA to breadboard adapters**
- **Oscilloscope** (for signal verification)
- **Spectrum analyzer** (for frequency verification)

## Software Requirements

### Arduino IDE Setup
1. **Install Arduino IDE** (1.8.19 or newer)
   ```
   Download from: https://www.arduino.cc/en/software
   ```

2. **Add Adafruit Board Support**
   - Open Arduino IDE → File → Preferences
   - Add to "Additional Board Manager URLs":
     ```
     https://adafruit.github.io/arduino-board-index/package_adafruit_index.json
     ```
   - Go to Tools → Board → Boards Manager
   - Search for "Adafruit SAMD" and install latest version

3. **Install Required Libraries**
   - Go to Sketch → Include Library → Manage Libraries
   - Install: "Adafruit Zero DMA Library"

### Python Environment Setup
1. **Install Python 3.8+**
   ```bash
   # Check version
   python3 --version
   ```

2. **Install Required Packages**
   ```bash
   pip install numpy matplotlib scipy pyserial pyrtlsdr
   ```

3. **RTL-SDR Drivers** (OS-specific)

   **Linux (Ubuntu/Debian):**
   ```bash
   sudo apt update
   sudo apt install rtl-sdr librtlsdr-dev
   # Blacklist kernel driver
   echo 'blacklist dvb_usb_rtl28xxu' | sudo tee -a /etc/modprobe.d/blacklist-rtl.conf
   ```

   **macOS (with Homebrew):**
   ```bash
   brew install rtl-sdr
   ```

   **Windows:**
   - Download and install RTL-SDR drivers from: https://www.rtl-sdr.com/rtl-sdr-quick-start-guide/
   - Use Zadig to install WinUSB drivers

## Hardware Setup

### Step 1: Prepare the Feather Boards

1. **Flash Reference Feather (Board A)**
   - Connect Feather A to computer via USB
   - Open Arduino IDE
   - Load `feather_firmware_reference.ino`
   - Select Board: "Adafruit Feather M4 Express (SAMD51)"
   - Select correct COM port
   - Upload firmware
   - Open Serial Monitor (115200 baud) to verify operation

2. **Flash Offset Feather (Board B)**
   - Connect Feather B to computer via USB
   - Load `feather_firmware_offset.ino`
   - Upload firmware (same process as above)
   - Verify operation in Serial Monitor

### Step 2: Antenna Setup

1. **Prepare Wire Antennas**
   - Cut 2 pieces of wire to 17.3 cm length (¼λ at 433 MHz)
   - Strip 1 cm of insulation from one end of each wire

2. **Connect Antennas to Feathers**
   - **Feather A (Reference)**: Connect antenna wire to pin A0
   - **Feather B (Offset)**: Connect antenna wire to pin A0
   - Ensure good electrical connection

3. **Position Antennas**
   - Separate antennas by at least 30 cm to avoid coupling
   - Orient antennas vertically for best radiation pattern
   - Keep away from metal objects that could cause reflections

### Step 3: RTL-SDR Setup

1. **Connect RTL-SDR**
   - Plug RTL-SDR into USB port
   - If using external antenna, connect to SMA connector
   - For this experiment, the internal antenna stub may work for short distances

2. **Test RTL-SDR Connection**
   ```bash
   # Linux/macOS
   rtl_test
   
   # Should show: "Found 1 device(s):"
   ```

3. **Position RTL-SDR**
   - Place RTL-SDR within 1-2 meters of both Feather antennas
   - Avoid blocking line-of-sight between antennas and RTL-SDR

## Running the Experiment

### Step 1: Identify Serial Ports

1. **Find Feather Serial Ports**
   
   **Linux:**
   ```bash
   ls /dev/ttyACM*
   # Should show /dev/ttyACM0, /dev/ttyACM1, etc.
   ```
   
   **macOS:**
   ```bash
   ls /dev/tty.usbmodem*
   # Should show /dev/tty.usbmodem..., etc.
   ```
   
   **Windows:**
   - Open Device Manager → Ports (COM & LPT)
   - Look for "USB Serial Device" entries (COM3, COM4, etc.)

2. **Identify Which Board is Which**
   - Connect to each port using serial terminal
   - Reference board shows: "Reference Signal Generator"
   - Offset board shows: "Offset Signal Generator"

### Step 2: Run the Experiment

1. **Navigate to Experiment Directory**
   ```bash
   cd driftlockchoir-oss/hardware_experiment
   ```

2. **Execute the Experiment**
   ```bash
   # Linux/macOS example:
   python3 e1_hardware_controller.py --ref-port /dev/ttyACM0 --offset-port /dev/ttyACM1 --duration 10
   
   # Windows example:
   python e1_hardware_controller.py --ref-port COM3 --offset-port COM4 --duration 10
   ```

3. **Follow the Experiment Progress**
   The script will:
   - Connect to both Feathers and RTL-SDR
   - Display hardware status
   - Start signal generation on both boards
   - Capture 10 seconds of RF data
   - Stop signal generation
   - Analyze captured data
   - Generate analysis plots
   - Display results

### Step 3: Analyze Results

The experiment generates a comprehensive analysis plot with four panels:

1. **RF Spectrum**: Shows the two generated signals at 433.0 and 433.1 MHz
2. **Beat Signal Time Domain**: Shows the time-domain beat pattern
3. **Beat Frequency Spectrum**: Shows the extracted 100 Hz beat frequency
4. **Results Summary**: Quantitative analysis results

**Success Criteria:**
- Beat frequency within 10 Hz of expected 100 Hz
- SNR > 20 dB
- Timing precision < 500 ps

## Expected Results and Interpretation

### Typical Good Results
```
Beat Frequency: 99.8 ± 2.1 Hz
Frequency Error: 0.2 Hz  
Timing Offset: 85.3 ± 12.7 ps
SNR: 28.5 dB
```

### What the Results Mean

1. **Beat Frequency**: Should be close to 100 Hz
   - This validates the frequency offset generation
   - Small errors are normal due to oscillator drift

2. **Timing Offset**: The measured propagation delay
   - Not calibrated to absolute zero (depends on hardware delays)
   - Stability and precision are more important than absolute accuracy

3. **SNR**: Signal quality indicator
   - >20 dB: Good signal quality
   - 10-20 dB: Marginal but usable
   - <10 dB: Poor signal, check setup

## Troubleshooting

### Connection Issues

**Problem**: Cannot connect to Feather boards
- **Check**: USB cables and connections
- **Try**: Different USB ports
- **Verify**: Boards appear in device manager/lsusb
- **Reset**: Press reset button on Feathers

**Problem**: RTL-SDR not found
- **Check**: RTL-SDR drivers installed correctly
- **Try**: Different USB port, avoid USB hubs
- **Test**: `rtl_test` command
- **Windows**: Verify WinUSB driver with Zadig

### Signal Issues

**Problem**: No RF signals detected
- **Check**: Antenna connections to pin A0
- **Verify**: Feathers are generating signals (LED should blink fast when active)
- **Test**: Use `STATUS` command in serial monitor
- **Position**: Move RTL-SDR closer to antennas

**Problem**: Low SNR (<10 dB)
- **Improve**: Antenna positioning and orientation
- **Reduce**: Distance between Feathers and RTL-SDR
- **Check**: Antenna connections
- **Avoid**: RF interference sources (WiFi, Bluetooth, microwaves)

**Problem**: Beat frequency far from 100 Hz
- **Verify**: Both Feathers running correct firmware
- **Check**: No compile-time errors in firmware
- **Reset**: Both boards and restart experiment
- **Calibrate**: Use `FREQ` command to fine-tune frequencies

### Analysis Issues

**Problem**: Python script crashes
- **Check**: All required packages installed
- **Verify**: Serial port permissions (Linux: add user to dialout group)
- **Update**: Matplotlib and numpy to latest versions
- **Try**: Shorter capture duration (--duration 5)

**Problem**: No beat signal extracted
- **Increase**: Capture duration to 20+ seconds
- **Check**: Both signals present in RF spectrum plot
- **Adjust**: RTL-SDR gain (script uses auto-gain)
- **Verify**: Antenna separation sufficient

## Advanced Experiments

### Experiment Variations

1. **Distance Measurement**
   - Move one Feather to different distances
   - Measure timing offset changes
   - Calculate distance: d = c × Δt / 2

2. **Frequency Sweep**
   - Use `FREQ` command to adjust base frequencies
   - Test different frequency offsets
   - Characterize system performance vs. frequency

3. **Multi-Point Measurements**
   - Take multiple measurements over time
   - Calculate Allan variance for stability analysis
   - Study temperature effects

### Hardware Improvements

1. **Better Antennas**
   - Use proper 433 MHz antennas
   - Add antenna ground planes
   - Consider directional antennas

2. **RF Amplification**
   - Add small RF amplifiers to increase signal strength
   - Use RF attenuators to control signal levels
   - Implement AGC (Automatic Gain Control)

3. **Frequency Stability**
   - Use external crystal oscillators
   - Add temperature compensation
   - Implement GPS disciplining

## Safety and Regulations

### RF Safety
- **Power Levels**: Feather DAC outputs are very low power (<1 mW)
- **Frequency**: 433 MHz is ISM band in most countries
- **Duration**: Short experiment duration minimizes any interference

### Regulatory Compliance
- **ISM Band**: 433 MHz experiments generally allowed
- **Low Power**: Signal levels well below regulatory limits
- **Check**: Local regulations for your specific location

## Scientific Significance

This experiment demonstrates:

1. **Chronometric Interferometry**: Core concept of using beat notes for timing measurement
2. **RF Phase Analysis**: Extraction of timing information from RF signals
3. **Distributed Timing**: Foundation for wireless clock synchronization
4. **Signal Processing**: Practical implementation of advanced DSP techniques

The principles demonstrated here scale to:
- **5G/6G Networks**: Ultra-precise timing for advanced features
- **Indoor Positioning**: Sub-meter accuracy systems
- **Scientific Instruments**: Distributed sensor synchronization
- **GNSS Augmentation**: Enhanced positioning accuracy

## References and Further Reading

1. **Chronometric Interferometry Theory**
   - Original experiment documentation in this repository
   - IEEE papers on wireless timing synchronization

2. **RTL-SDR Resources**
   - RTL-SDR.com blog and tutorials
   - GNU Radio integration possibilities

3. **Adafruit Documentation**
   - Feather M4 Express technical documentation
   - SAMD51 microcontroller reference manual

## Support and Community

- **Issues**: Report problems on GitHub repository
- **Questions**: Contact hunter@shannonlabs.dev
- **Improvements**: Submit pull requests with enhancements
- **Collaboration**: Join the distributed timing research community

---

This experiment demonstrates advanced concepts in a simplified form. Real-world chronometric interferometry systems involve much more sophisticated hardware and algorithms, but the fundamental principles are the same.