# Feather Demo — 5‑Minute Quickstart (Dummy‑Proof)

This is the simplest path to a working Driftlock demo using two Adafruit Feathers. No SDRs, no configs, one jumper decides the role.

## What to buy
- 2× Adafruit Feather boards (M0, M4, ESP32‑S3, or RP2040 all work)
- 2× Adafruit RFM69HCW FeatherWing (match frequency to your region: 433/868/915 MHz)
- 2× Short jumper wires (to set roles)
- USB cables for both boards

## Assemble (2 minutes)
1. Plug each RFM69 FeatherWing onto a Feather.
2. Put a short jumper wire into A0 and GND on ONE board only.
   - That board = Responder (A0 to GND)
   - The other board = Initiator (A0 floating)
3. Attach the antennas that came with the FeatherWings.

## Flash the SAME firmware to both (2 minutes)
1. Open Arduino IDE → Sketch → Add File →
   - For LoRa (RFM95, e.g., Feather 32u4 LoRa / Feather M0 LoRa): `firmware/feather_rpr_rfm95/feather_rpr_rfm95.ino`
   - For RFM69 FeatherWing (non‑LoRa): `firmware/feather_rpr_rfm69/feather_rpr_rfm69.ino`
2. Tools → Board → select your Feather; Tools → Port → pick the correct port.
3. Sketch → Upload. Repeat for the second board.
   - No code edits. The A0 jumper decides the role.

## See it working (1 minute)
Option A — Arduino Serial Monitor:
- Tools → Serial Monitor at 115200 baud on the initiator board.
- You’ll see lines like: `{ "seq": 12, "t_us": 123456.0, "rtt_us": 210.0 }`

Option B — One‑command artifacts:
```
pip install -r requirements.txt
python scripts/feather_demo_run.py --duration 60
```
Outputs into `results/hw_demo_quick/feather_YYYYMMDDThhmmssZ/`:
- `run_manifest.json` — simple metadata
- `rmse_stream.jsonl` — time series
- `summary.json` — stats
- `rmse_trend.png` — pretty plot for your deck

## Tips
- Start with the boards next to each other (line‑of‑sight). For cabled demos, use attenuators.
- If nothing prints, swap which USB port you’re watching (initiator prints JSON).
- If you bought the 433/868 MHz wing, set `RADIO_FREQ_MHZ` at the top of the sketch accordingly.

You now have a working, recordable demo with two Feathers and one jumper.
