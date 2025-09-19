Subject: Pilot request: Software-only time telemetry overlay for existing radios

Hi <Name>,

We’ve built a software-only time telemetry overlay that turns tiny, intentional frequency offsets (Δf) into precise timing telemetry on top of existing radios. No new spectrum or hardware — we reuse the radio’s CFO/phase estimators and carve microsecond-scale pilots.

What we’d like to pilot (45 minutes to scope, then a small on-site demo):
- In-band pilots or a low-duty sideband (LoRa) beacons for a small topology (3–5 nodes)
- Our SDK extracts beat-phase, runs variance-weighted consensus, and emits `get_clock_bias`/`get_quality`
- A simple dashboard shows “clock quality” and bias stability

Why it matters:
- Works indoors and without GPS; masterless synchronization across devices
- Software-only overlay on Wi‑Fi/5G/private networks; minimal throughput impact
- Enables coordinated sensing/actuation and robust timing in edge/industrial settings

We can demo live in ~10 minutes (Feather + RTL‑SDR bench). If you’re open, I’d love to set up a quick call to align on a scoped pilot (1–2 weeks, low risk).

Best,
<Your Name>
Shannon Labs — Driftlock
<email> | <phone>
