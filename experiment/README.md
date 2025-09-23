# Driftlock Hardware Validation Experiment

This experiment demonstrates the Bown Driftlock Method on real hardware using two Adafruit Feather boards with integrated RFM95 (LoRa) radios and a low‑cost RTL‑SDR. The setup intentionally creates a small frequency offset between two transmitters; the resulting beat signal’s phase encodes timing information. Moving one node changes the delay, which appears as a beat‑phase shift.

Summary

- Node A: continuous transmission at 915.000000 MHz
- Node B: continuous transmission at 915.001000 MHz (Δf = 1 kHz)
- RTL‑SDR records the combined RF; baseband contains a 1 kHz beat
- Analyzer extracts the beat phase and demonstrates timing sensitivity

What this proves

- Frequency offset is intentionally created and used as a measurement signal
- The beat phase evolves linearly and changes proportionally with path delay
- Changing physical distance produces a measurable phase/time change

Quick start

1) Prepare microcontrollers (CircuitPython)
- Install CircuitPython on both Feathers (M0 and 32u4 RFM95).  
  Guide: https://learn.adafruit.com/welcome-to-circuitpython/installing-circuitpython
- Copy the RFM9x library onto CIRCUITPY/lib (from the Adafruit CircuitPython bundle):  
  adafruit-circuitpython-bundle/lib/adafruit_rfm9x.mpy
- On Feather M0: copy `node_a_transmit.py` to the board as `code.py`
- On Feather 32u4: copy `node_b_transmit.py` to the board as `code.py`

2) Prepare the host (Python 3.10+)
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r experiment/requirements.txt
```
- Install RTL‑SDR drivers (librtlsdr). Platform‑specific instructions:  
  https://pyrtlsdr.readthedocs.io/en/latest/install.html

3) Run capture + analysis
```bash
cd experiment
python beat_recorder.py       # records ~1s to results/beat_capture_<ts>.npy
python driftlock_analyzer.py  # extracts beat phase and timing indication
```
- `beat_recorder.py` accepts overrides such as `--duration 5`, `--gain 12`, `--center-freq 915.002`, `--sample-rate 2.4`, and `--output-dir results_hw/`. Each run now emits a JSON metadata file alongside the `.npy` capture documenting the RF settings.
- `driftlock_analyzer.py` auto-loads metadata, supports `--capture <path>` and `--reference <path>` for explicit comparisons, and honors `--sample-rate` / `--band-low` / `--band-high` tweaks when retuning the filter.
- Want a guided walkthrough? Open `experiment/quickstart.html` in your browser for a follow-along version of these steps.

One-command option

```bash
cd experiment
python run_simple_demo.py     # captures + analyzes with sensible defaults
```
- Designed as an “experiments for dummies” workflow: records ~1.5 s, auto-analyzes, and compares against the previous capture if one exists.
- Flags remain optional: try `--duration 3`, `--gain 10`, or `--no-plot` to tailor behaviour without leaving the simplified flow.

Deeper experimentation (batch runs)

```bash
cd experiment
python run_series_demo.py --runs 6 --sleep 5
```
- Captures a labeled series, logs timing deltas vs. baseline/previous runs, and produces CSV/JSON plus a Δτ plot for post-analysis.
- Use this when stepping the node through known distances or testing environmental perturbations.

Expected results

- Beat frequency near 1000 Hz (± a few Hz depending on tolerance)
- Analyzer prints a “timing offset” derived from beat phase; absolute value includes unknown initial phases, so focus on deltas
- Move a node by 30 cm → delay change ≈ 1 ns → measurable phase shift; analyzer reports a corresponding ps‑scale change between captures

Important notes

- The RFM95 (SX127x) is a LoRa modem; its high‑level drivers transmit LoRa waveforms, not true CW. Many radios include a continuous wave (test) mode, but it may require low‑level register access. The provided example “continuous transmit” loops keep the RF active and are adequate for a benchtop beat demonstration; spectral content will reflect the modem’s behavior. If you prefer pure CW, enable the chip’s TX continuous wave test mode (datasheet) or use a signal generator/SDR for carriers.
- Keep TX power low (e.g., 5 dBm) and use shielding/attenuators/faraday bag or coax coupling for indoor tests. Follow local regulations for ISM band emissions.
- Absolute timing extraction requires two‑way exchange to cancel clock bias. This single‑ended demo is designed to show a beat‑phase shift with distance (relative timing).

File map

- `hardware_setup.md` — wiring and CircuitPython library setup
- `node_a_transmit.py` — Feather M0 @ 915.000000 MHz
- `node_b_transmit.py` — Feather 32u4 @ 915.001000 MHz
- `node_a_tx_cw.py` — Feather M0 attempts SX127x TX continuous‑wave test mode @ 915.000000 MHz
- `node_b_tx_cw.py` — Feather 32u4 attempts SX127x TX continuous‑wave test mode @ 915.001000 MHz
- `beat_recorder.py` — RTL‑SDR capture and quick visualization
- `driftlock_analyzer.py` — beat‑phase extraction and timing estimation
- `run_simple_demo.py` — one-touch capture + analysis wrapper (quickstart)
- `run_series_demo.py` — batch capture harness for structured experiments
- `quickstart.html` — beginner-friendly follow-along web page
- `driftlock_phase_delta.py` — compute phase/timing/distance delta between two captures
- `requirements.txt` — host dependencies
- `results/` — saved captures and plots

Troubleshooting

- If no beat is visible, reduce gain or separate antennas (avoid saturation); verify both transmitters are on and near 915 MHz.
- Try Δf = 2–5 kHz by editing the Node B frequency if 1 kHz is too close to mains/alias frequencies in your environment.
- For cleaner CW, consider enabling SX127x continuous wave test mode (datasheet) or using two SDRs/signal generators.

Optional: TX Continuous‑Wave test mode

- The provided `node_*_tx_cw.py` scripts use low‑level register writes to put the SX127x into LoRa TX continuous mode so the RF stays on indefinitely, closer to a CW. This is intended for benchtop verification only and may not be identical to a pure unmodulated carrier. Use with low power and shielding. If unsupported on your board/firmware, fall back to `node_*_transmit.py`.
