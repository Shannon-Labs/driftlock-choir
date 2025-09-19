# Driftlock — a16z Speedrun One‑Pager

- One‑liner: Software‑only time telemetry overlay for existing radios. We exploit tiny, intentional Δf in pilots/beacons to synchronize distributed devices — no new spectrum or hardware.

- Why now: Edge AI + private 5G demand precise, masterless timing; GPS is fragile indoors/contested; radios already estimate CFO/phase — we reuse that machinery.

- Product: SDK + control plane. Orchestrates Δf/Δt micro‑beacons, extracts beat‑phase, runs variance‑weighted consensus, and exposes `get_clock_bias`/`get_quality` APIs + dashboards.

- Demo: Two Feathers + RTL‑SDR generate ~1 kHz beat; moving one node 30 cm produces Δτ ~ 1 ns. Analyzer + SDK show bias and quality. Repo includes reproducible scripts and sample artifacts.

- Evidence (simulation):
  - Alias calibration: off → ~−12 ns bias; loopback → ~+2.65 ps (extended_006).
  - Consensus: 64‑node/25‑node converge in 1 iteration to ~22–24 ps RMSE (no KF).
  - Conservative language: simulation results; hardware pilots in progress.

- Moat/IP: Paradigm shift (offset as feature, teaching‑away). Provisional condensed to 25 claims; US + PCT drafts ready.

- Wedge markets: Private 5G/industrial, robotics/swarm, distributed sensing. Pricing: SDK + device SaaS; pilot‑first GTM.

- 90‑day plan: (1) SDK alpha + telemetry sink + demo; (2) two pilots (private 5G + robotics), dashboards; (3) SDK beta + control plane; US/PCT filed; 2 LOIs.

- Asks: pilot intros (private 5G integrators, robotics labs), infra GTM advisor.
