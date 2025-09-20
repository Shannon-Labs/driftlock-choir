# Driftlock Hardware Live Demo Plan (SDR)

This plan documents the 2–4 week path from simulation to a repeatable SDR demo with JSON artifacts and plots suitable for pilots and investor diligence.

## Objectives

- Demonstrate sub‑nanosecond timing convergence using two SDRs with intentional Δf.
- Record JSON/PNG artifacts and verify with the same guardrails used in sim.
- Establish acceptance criteria and a repeatable script for partner pilots.

## Bill of Materials (example)

- 2× SDR: LimeSDR Mini 2.0 or USRP B200‑class
- 2× Antennas or RF cables + 30–60 dB attenuators (for cabled demo)
- 1× Linux laptop (Python 3.11+), powered USB hub, RF cables
- Optional: GPSDO (to test disciplining off), spectrum analyzer

## Measurement Topologies

1. Cabled Loopback (baseline): TX↔RX via attenuators for SNR ≥ 20 dB
2. Short‑range OTA: 1–3 m indoor, line‑of‑sight, limited reflections

## Procedure

1. Configure two radios with Δf = {1, 3, 5} MHz; preamble BW 20–40 MHz
2. Run two‑way chronometric handshake (coarse+narrowband)
3. Log per‑slot RMSE estimates and alias resolution outcomes
4. Run variance‑weighted consensus vs baseline preset
5. Export JSON summary and RMSE‑over‑time plot

## Acceptance Criteria

- Stability (loopback): timing RMSE < 500 ps over 60 s at SNR ≥ 0 dB
- Trend: RMSE decreasing toward 100–200 ps in cabled setup
- Alias resolution: ≥ 95% success across Δf at 0/10/20 dB
- Reproducibility: deterministic seeds; JSON/PNG artifacts; one‑command replay

## Artifacts

- `results/hw_demo_XXX/`
  - `run_manifest.json` (device info, seeds, config)
  - `rmse_stream.jsonl` (timestamped entries)
  - `summary.json` (metrics: mean/min/max/last/percentiles, alias stats)
  - `rmse_trend.png`

## Commands

```
# Example: record a run from a CSV/JSONL stream
python scripts/hw_demo_logger.py \
  --output-dir results/hw_demo_001 \
  --run-id loopback_smoke \
  --format jsonl \
  --input rmse_stream.jsonl \
  --notes "Loopback; Δf=1 MHz; SNR~20 dB"
```

## Notes

- Start with cabled loopback to minimize channel variability; add OTA once stable.
- Use fixed RNG seeds and persist a small device inventory in the manifest.
- Reuse sim guardrails for verification (seeded regression philosophy).

---

## Feather Variant (Low‑Cost)

If you prefer to avoid SDRs for the first demo, use two Adafruit Feathers with BLE/LoRa/nRF24 for a reciprocal ping ranging (RPR) handshake.

- Target: sub‑µs stability trends over 60 s (cabled/short‑range)
- Artifacts: produced via `scripts/feather_log_parser.py` → `scripts/hw_demo_logger.py`
- Details: see `docs/feather_demo.md`
