# Slide 7 — Architecture Snapshot

Title: Production API meets research-grade modelling

Bullets
- Customer-facing layer: Next.js 14 + Supabase handles auth, billing, streaming ingestion, and anomaly alerts.
- Modelling spine (`src/`): oscillators/noise (`phy/`), RF front-end (`hw/`), estimation/control (`alg/`), MAC/topology (`net/`).
- Simulation harnesses (`sim/phase1-3.py`) and Monte Carlo runner (`scripts/run_mc.py`) validate algorithms that graduate into the API.
- TelemetryExporter + Supabase pipelines keep product metrics and research telemetry in lockstep for audits and SLAs.

Notes for presenter
- Show how the production repo (“driftlock”) and research repo (“driftlock-choir”) share schemas and validators.
- Highlight that this alignment shortens the path from research insight to customer feature.
- Offer to open the repos during diligence to demonstrate transparency.
