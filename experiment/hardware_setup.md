# Hardware Setup

This guide walks you through wiring and firmware setup for two Adafruit Feather LoRa boards and an RTL‑SDR.

Bill of materials

- Adafruit Feather M0 with RFM95 (915 MHz)
- Adafruit Feather 32u4 with RFM95 (915 MHz)
- Two antennas suitable for 915 MHz (or coax & attenuators for cabled bench testing)
- USB cables for both boards
- RTL‑SDR (RTL2832U + R820T/2 or equivalent)

1) Install CircuitPython

- Follow Adafruit’s guide to load CircuitPython on each board:  
  https://learn.adafruit.com/welcome-to-circuitpython/installing-circuitpython
- After flashing, the board mounts as a USB drive `CIRCUITPY`.

2) Install the RFM95 library on each board

- Download the Adafruit CircuitPython Bundle (matching your CircuitPython version):  
  https://circuitpython.org/libraries
- From the bundle’s `lib/`, copy `adafruit_rfm9x.mpy` to `CIRCUITPY/lib/`.

3) Wiring/CS/RST pins

- Many Feather LoRa boards define convenient pin aliases in CircuitPython: `board.RFM9X_CS` and `board.RFM9X_RST`.
- If you receive a runtime error referencing CS/RST, replace those with explicit pins based on your board’s schematic. Common mappings:
  - Feather M0 RFM9x: CS = `board.D8`, RST = `board.D4`
  - Feather 32u4 RFM9x: CS = `board.D8`, RST = `board.D4`
  Example code change:
  ```python
  cs = digitalio.DigitalInOut(board.D8)
  reset = digitalio.DigitalInOut(board.D4)
  ```

4) Load the transmitter scripts

- On Feather M0: copy `experiment/node_a_transmit.py` to the board as `code.py`
- On Feather 32u4: copy `experiment/node_b_transmit.py` to the board as `code.py`
- Eject/soft‑reset; each board starts transmitting automatically.

5) Antennas and safety

- Attach antennas before enabling TX to avoid damaging the PA.
- Keep TX power low (e.g., `tx_power = 5`) and test in a shielded environment or with attenuators for compliance.

6) Host setup (RTL‑SDR)

- Install librtlsdr drivers per your platform.  
  https://pyrtlsdr.readthedocs.io/en/latest/install.html
- On macOS: `brew install librtlsdr` (if using Homebrew).  
  On Linux: use your package manager (`rtl-sdr`, `librtlsdr`).

7) Run the capture and analyzer

- In a Python venv, install host deps: `pip install -r experiment/requirements.txt`
- Run `python experiment/beat_recorder.py` then `python experiment/driftlock_analyzer.py`.

Notes and options

- Pure CW: The SX127x supports continuous wave test modes via registers. The high‑level library sends LoRa waveforms; for true CW, set TX CW test mode (datasheet) or use SDR/generators.
- Beat spacing: If local interference around 1 kHz is strong, retune Node B to Δf = 2–5 kHz and adjust analyzer bandpass accordingly.
