# DriftLock — a16z Deck Outline

1. Title
   - DriftLock by Shannon Labs, Inc. (formerly Entruptor/CBAD)
   - Founder credit: Hunter Bown
   - Tagline: Take agency back from the agents

2. Vision & Company Snapshot
   - Mission: build infrastructure that restores human agency in data-heavy environments
   - Ethos: evidence-first, operator-centric tools
   - Product lineage: DriftLock API live today; Driftlock Choir expands the control plane

3. Product Today — DriftLock API
   - Instant anomaly detection API (Next.js + Supabase) powering current revenue conversations
   - Zero-training, compression-first detection (~80 ms latency, 159k RPS, CPU-only)
   - Customers drop in via REST/SDK; integrates with existing observability stacks

4. Deep Tech Platform — Driftlock Choir
   - Chronometric synchronization engine that underpins next-gen control modules
   - Shared telemetry & modelling stack informs both the API and future closed-loop offerings
   - Roadmap ties Choir breakthroughs back into API experiences (detections → actions)

5. Problem & Why Now
   - Distributed systems drift, costing reliability, dollars, and trust
   - Teams drown in reactive tooling; lack predictive coordination
   - Data growth + autonomy mandates synchronization that scales today

6. DriftLock Solution Loop
   - Today: API delivers “sense” and “flag” via fast anomaly detection
   - Emerging: Choir extends into “estimate → consensus → action” for proactive stabilization
   - Unified delivery: SaaS tiers & integrations reuse shared telemetry/validation assets

7. Architecture Snapshot
   - Production layer: Next.js + Supabase API, telemetry ingestion, customer dashboards
   - Modelling spine (`src/`): oscillators/noise (`phy/`), RF front-end (`hw/`), estimation/control (`alg/`), MAC/topology (`net/`)
   - Simulation harnesses (`sim/phase*`) and TelemetryExporter feed both anomaly API analytics and Choir R&D
   - CI-ready Monte Carlo scripts (`scripts/run_mc.py`) reproduce plots stored in `results/phase*/`

8. Technical Moat
   - Compression-driven anomaly detection yields GPU-free performance edge today
   - Chronometric interferometry + decentralized consensus proven to 100 ps timing RMS in single-digit ms
   - Statistical Validator framework enforces CRLB compliance before we ship new detection/consensus features
   - Gemini roadmap (HITL, dynamic channels, hybrid estimators) extends defensibility

9. Evidence — API Performance
   - Benchmarks: ~80 ms detection, 159k RPS, 47 MB memory, F1 71.5% (CICIDS2017)
   - Dark/fast user experience: zero training, first-request insights
   - Production telemetry funnels into Supabase-backed dashboards (show request analytics)

10. Evidence — Phase 1 Handshakes
    - Plot: `driftlock_choir_sim/outputs/phase1_enhancements/figures/wls_performance_improvement.png`
    - Alias resolution >95%, CRLB-tracking RMSE across SNR sweep
    - Config reproducibility: `sim/configs/default.yaml` with deterministic seeds

11. Evidence — Phase 2 Consensus
    - Plot: `results/phase2/phase2_convergence.png`
    - Predicted vs measured iterations align within ±1; KF pre-filter improves RMS by 35%
    - Telemetry streamed via `TelemetryExporter` for diligence-ready audits

12. Market Opportunity
    - Beachheads: mission-critical networking, industrial robotics, edge AI analytics
    - Drift-driven downtime averages $300k/hour; observability + reliability spend >$20B/year
    - Autonomy + regulation demand auditable synchronization and anomaly response

13. Business Model
    - Current: API subscriptions (usage tiers) with add-ons for dedicated throughput/SLOs
    - Near-term: Stabilize/Orchestrate tiers layer closed-loop controls from Choir research
    - Expansion via controls marketplace and simulation-backed assessments

14. Go-To-Market Motion
    - Design partner-led diagnostics leveraging API onboarding + simulation insights
    - 30-day ROI reports using existing Monte Carlo + Supabase telemetry
    - Community content: anomaly benchmarks, synchronization studies, acceptance digests

15. Traction & Pipeline
    - API pilots in edge analytics / security; anomaly detections in production workloads
    - Choir POCs with industrial automation and mission networks showing >10× timing gains
    - Advisor bench + hiring funnel aligned with customer commitments

16. Roadmap & R&D Horizons
    - 0–6 mo: DriftLock GA, integration SDKs, SLA dashboards, on-call tooling for design partners
    - 6–18 mo: Execute Gemini backlog — HITL loop, dynamic channels, hybrid coherent/aperture fusion
    - 18+ mo: MAC/consensus co-design, autonomy bundles, regulated-industry certification

17. Team
    - Solo founder story: Hunter Bown, law student-turned-builder, supported by domain advisors and contractors
    - Hiring plan focused on controls, platform, and HITL specialists as capital deploys

18. Competition & Differentiation
    - Incumbents: legacy PTP/GNSS, observability dashboards, vendor-specific appliances
    - DriftLock edge: instant anomaly detection today; evidence-bound control plane tomorrow
    - Capability matrix in appendix (robustness, telemetry, autonomy readiness)

19. The Ask & Use of Funds
    - Raise $X seed to execute 18-month roadmap
    - 40% product acceleration, 35% go-to-market, 25% research moat (HITL, dynamic channels, hybrid estimators)

20. Appendix
    - Company & product notes (`docs/pitch/company_notes.md`)
    - API benchmark sheet, repository pointers, research backlog (Gemini roadmap)
    - Extra plots, configs, Monte Carlo summaries
