# Driftlock Feather Demo (Low-Cost Variant)

This demonstrates the core two‑way timing handshake and stability trends using two Adafruit Feather boards and a simple ISM‑band radio link. It’s designed for fast iteration and investor demos without SDRs.

## Goals
- Show two‑way (reciprocal) timing handshake over RF.
- Log round‑trip time (RTT) samples with deterministic scheduling.
- Convert samples to a stability trend (RMS around mean) in picoseconds.
- Target stability: sub‑microsecond in cabled/short‑range setups.

## Hardware Options
- 2× Adafruit Feather boards (32u4 LoRa, M0 LoRa, M4, RP2040, ESP32‑S3)
- Radio options:
  - RFM95 (LoRa): Feather 32u4 LoRa or Feather M0 LoRa (recommended, dummy‑proof)
  - RFM69 FeatherWing (non‑LoRa)
  - Built‑in BLE (nRF52/ESP32) using GATT (advanced)
  - nRF24L01+ breakout (advanced)
- Optional: RF cables + attenuators for cabled demo
- Host laptop with Python 3.11+

## Firmware Protocol (overview)
- Node A (initiator) sends `PING(seq)` every T ms.
- Node B replies immediately with `PONG(seq, t_rx_ping_us)`.
- Node A stamps send/receive times (`t_tx_ping_us`, `t_rx_pong_us`) using `micros()`.
- Compute RTT_us = `t_rx_pong_us - t_tx_ping_us`.
- Print one line per exchange over USB serial as JSON:
  ```json
  {"seq":123,"t_us":173742.5,"rtt_us":213.0}
  ```

Prefer these one‑file sketches that need no edits:

- LoRa (RFM95): `firmware/feather_rpr_rfm95/feather_rpr_rfm95.ino`
- RFM69: `firmware/feather_rpr_rfm69/feather_rpr_rfm69.ino`

Both use a single jumper on A0 to choose the role (floating=initiator; GND=responder).

## Host Capture and Plot

1) Install deps:
```
pip install -r requirements.txt  # includes pyserial
```

2) Read serial, compute stability, and export JSONL:
```
python scripts/feather_log_parser.py \
  --port /dev/tty.usbmodem1101 \
  --baud 115200 \
  --output rmse_stream.jsonl \
  --window 50
```
- Computes rolling RMS (around the mean) of `rtt_us` and outputs JSONL lines with `t` (s) and `rmse_ps`.

3) Produce demo artifacts (summary + PNG):
```
python scripts/hw_demo_logger.py \
  --output-dir results/hw_demo_001 \
  --run-id feather_loopback \
  --format jsonl \
  --input rmse_stream.jsonl \
  --notes "Feather demo; cabled; BLE or LoRa"
```

## Acceptance Criteria (Feather Variant)
- Cabled or short‑range LOS: RMS stability < 500,000 ps (0.5 µs) over 60 s
- Decreasing trend with averaging/consensus enabled in host script
- Deterministic seeds/schedule; reproducible logs and PNG

## Notes
- This demo proves reciprocal timing stability with cheap dev boards; SDR demos can push toward sub‑ns and phase‑based estimation.
- If your radio supports channelized Δf or continuous‑wave test modes, you can extend this to beat‑phase extraction later.
