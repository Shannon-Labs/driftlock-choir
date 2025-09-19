# Driftlock — a16z Speedrun Deck Outline (10 slides)

1) Title & One‑Liner
- Wireless Time Telemetry Overlay for Existing Radios
- We exploit tiny, intentional Δf to synchronize distributed devices — software only.

2) Why Now
- Edge AI + private 5G growth; GPS fragility; need for masterless sync.
- Radios already estimate CFO/phase — we reuse that machinery.

3) Product
- SDK + control plane; orchestrates Δf/Δt micro‑beacons; extracts beat‑phase; variance‑weighted consensus; APIs: get_clock_bias, get_quality.

4) How It Integrates
- Pilots/training → beat‑phase extraction → consensus → time telemetry API.
- Modes: in‑band pilots, sideband LoRa/BLE, SDR reference.

5) Demo
- 60s bench video: beat at ~1 kHz; move 30 cm → Δτ ~ 1 ns; SDK prints bias + quality; Grafana panel.

6) Evidence (Simulation)
- Alias calibration: off → ~−12 ns bias; loopback → ~+2.65 ps.
- 64‑node & 25‑node: converge in 1 iteration to ~22–24 ps RMSE (no KF).
- Conservative language: simulation results; pilots incoming.

7) IP & Moat
- Paradigm shift (teaching‑away: everyone eliminates CFO).
- Provisional condensed to 25 claims; US + PCT drafts ready.

8) Markets & Wedge
- Private 5G/industrial, robotics/swarm, distributed sensing.
- Pricing: SDK + device SaaS; pilots with clear deliverables.

9) 90‑Day Plan
- Weeks 0–2: Demo + SDK alpha + telemetry sink; reproducible repo.
- Weeks 2–6: Private 5G + robotics pilots; dashboards.
- Weeks 6–12: SDK beta + control plane; US/PCT filed; 2 LOIs.

10) Team & Asks
- Multidisciplinary team; advisors (radio/standards).
- Asks: pilot intros (private 5G, robotics), infra GTM mentor.
