# Slide 9 — Evidence: DriftLock API Benchmarks

Title: Production-grade anomaly detection performance

Bullets
- Latency: ~80 ms end-to-end detection on CPU-only infrastructure (no GPUs, no warm-up).
- Throughput: 159k requests/sec sustained in load tests with <50 MB memory footprint.
- Accuracy: 71.5% F1 on CICIDS2017 without training; compression heuristics generalize out of the box.
- Deployment: Next.js 14 + Supabase + Vercel stack running live design-partner traffic today.

Notes for presenter
- Show benchmark graphs or tables (appendix) to substantiate each metric.
- Mention observability hooks (Supabase, telemetry dashboards) that prove readiness for enterprise SLAs.
- Use this slide to segue into the Choir evidence—API customers will inherit those upgrades.
