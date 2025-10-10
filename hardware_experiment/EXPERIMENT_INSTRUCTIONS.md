# Hardware Experiment Instructions (Raspberry Pi Pico)

This document provides complete step-by-step instructions for performing the hardware experiment using two Raspberry Pi Picos and an RTL-SDR.

## Overview

This experiment demonstrates the core principles of RF Chronometric Interferometry by:
- Generating two RF signals with a precise 100 Hz frequency offset using the Pico's Programmable I/O (PIO).
- Capturing both signals simultaneously using an RTL-SDR.
- Analyzing the resulting beat note to extract timing and frequency information.

## Required Hardware

1.  **Two Raspberry Pi Picos**
2.  **One RTL-SDR v3 Dongle** (or equivalent)
3.  **Two Micro-USB cables** (for the Picos)
4.  **Simple wire antennas** (two ~17cm pieces of solid-core wire for the 433MHz band)

## Software Setup

This process involves two main parts: setting up the Picos with MicroPython and preparing your computer to run the experiment.

### Part A: Install MicroPython on the Picos (One-Time Setup)

You will need to do this for **both** of your Raspberry Pi Picos.

1.  **Download MicroPython:** Go to the official Raspberry Pi Pico downloads page and download the latest "MicroPython UF2" file.
    *   [https://www.raspberrypi.com/documentation/microcontrollers/micropython.html](https://www.raspberrypi.com/documentation/microcontrollers/micropython.html)

2.  **Enter Bootloader Mode:** Unplug your Pico. Press and hold the **BOOTSEL** button on the Pico, and while holding it, plug the Pico into your computer's USB port.

3.  **Copy the UF2 File:** The Pico will appear as a new USB drive on your computer named `RPI-RP2`. Drag and drop the downloaded MicroPython UF2 file onto this drive. The Pico will automatically reboot, and MicroPython will be installed.

4.  **Repeat for the second Pico.**

### Part B: Install the Firmware on the Picos

The easiest way to copy the Python firmware scripts to the Picos is by using the **Thonny IDE**.

1.  **Install Thonny:** Download and install Thonny from [https://thonny.org/](https://thonny.org/). It's a beginner-friendly Python IDE with built-in support for MicroPython devices.

2.  **Configure Thonny for Pico:**
    *   Open Thonny.
    *   Go to `Tools -> Options...` (or `Thonny -> Preferences...` on macOS).
    *   Click on the **Interpreter** tab.
    *   Select **MicroPython (Raspberry Pi Pico)** from the interpreter dropdown list.
    *   Select the serial port for your Pico under the **Port** dropdown. It will usually be auto-detected.
    *   Click **OK**.

3.  **Upload Firmware to Pico #1 (Reference):**
    *   Connect the first Pico to your computer.
    *   In Thonny, open the `pico_firmware_reference.py` file from the `hardware_experiment` directory.
    *   Go to `File -> Save copy...`.
    *   When prompted where to save, select **Raspberry Pi Pico**.
    *   Save the file with the name `main.py`. This ensures the script runs automatically when the Pico is powered on.

4.  **Upload Firmware to Pico #2 (Offset):**
    *   Disconnect the first Pico.
    *   Connect the second Pico.
    *   In Thonny, open the `pico_firmware_offset.py` file.
    *   Go to `File -> Save copy...` and select **Raspberry Pi Pico**.
    *   Save this file as `main.py` on the second Pico.

Your Picos are now programmed.

### Part C: Prepare Your Computer

1.  **Install Python Dependencies:**
    ```bash
    # Navigate to the hardware_experiment directory
    pip install -r requirements.txt
    ```

2.  **Install RTL-SDR Drivers:** Follow the instructions in the original `EXPERIMENT_INSTRUCTIONS.md` for your operating system (Linux, macOS, or Windows) if you haven't already.

## Hardware Setup

### Wiring Diagram

The wiring for this experiment is minimal. For each of the two Picos, you only need to connect a single wire antenna to GPIO Pin 15.

```
+--------------------------+
|                          |
|      Raspberry Pi        |
|          Pico            |
|                          |
|       +---------+        |
|       |   USB   |        |
|       +---------+        |
|                          |
|             ...          |
| [ ] GND                  |
| [ ] GP14                 |
| [X] GP15 <---- ~17cm Wire Antenna
| [ ] RUN                  |
|             ...          |
|                          |
+--------------------------+
```

1.  **Connect Antennas:** For each Pico, connect a ~17cm wire antenna to **GPIO Pin 15**.
2.  **Power the Picos:** Connect both Picos to your computer or a USB power source. As soon as they are powered, they will begin listening for serial commands.
3.  **Connect RTL-SDR:** Plug your RTL-SDR dongle into a USB port on your computer.

## Running the Experiment

1.  **Find the Pico Serial Ports:** Identify the serial ports for your two Picos. They will appear as `/dev/ttyACM0`, `/dev/ttyACM1` (Linux), `/dev/tty.usbmodem...` (macOS), or `COM3`, `COM4` (Windows).

2.  **Run the Controller Script:** Execute the `e1_hardware_controller.py` script, providing the two serial ports you found. It does not matter which port is for the reference or offset.
    ```bash
    # Example on Linux/macOS
    python e1_hardware_controller.py --ref-port /dev/ttyACM0 --offset-port /dev/ttyACM1

    # Example on Windows
    python e1_hardware_controller.py --ref-port COM3 --offset-port COM4
    ```

The script will connect to the Picos, command them to start transmitting, capture the RF data with the RTL-SDR, and then perform the analysis, producing a plot of the results.

## Troubleshooting

*   **Cannot connect to Pico:** Ensure you have selected the correct MicroPython interpreter and port in Thonny. Make sure no other program (like a serial monitor) is connected to the Pico's port.
*   **RTL-SDR not found:** Verify drivers are installed correctly using the `rtl_test` command.
*   **Low SNR:** Check your antenna connections to GPIO 15. Try moving the Picos and the RTL-SDR closer together.
