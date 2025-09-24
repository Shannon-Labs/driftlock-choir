### Prompt: "Acceptance Polish" – CRLB Gatekeeping & Message Discipline

**[START PROMPT]**

Persona: You are Jules, acting as Verification Steward. The acceptance harness already enforces runtime, alias sanity, payload coexistence, and RMSE thresholds. After the latest audit, you need to tighten the guardrails and harmonize the messaging so published numbers reflect the real test envelope.

#### Goals
1. Ensure RMSE/CRLB ratios make physical sense (no sub-bound “passes” caused by conservative CRLB inputs).
2. Align README/acceptance reports so outsiders see the statistically bounded multipath numbers first, with the 22 ps single-path result preserved as historical best-case.
3. Automate hardware-in-the-loop staging by piping SDR IQ captures into the same acceptance framework.

#### Tasks
1. **CRLB gate refinement**
   - Update acceptance thresholds so test suites expect `1.0 <= RMSE/CRLB <= 1.5` (configurable). Treat `< 0.9` as “bound too low” and recompute using the residual-based CRLB already in `metrics/crlb.py` before failing.
   - Extend unit/integration tests to cover the new behaviour (include a fixture where the spectrum-derived bound is intentionally conservative, verifying the warning path).

2. **Reporting discipline**
   - Reorder README and `docs/results_*.md` so acceptance-harness numbers (e.g., 45 ps RMSE @ RMSE/CRLB ≈1.2) lead, with clean-room 20–22 ps figures clearly labeled as idealized benchmarks.
   - Add a short “interpretation guide” explaining the relationship between CRLB, consensus RMSE, and multipath bias.

3. **Hardware ingestion hook**
   - Create `scripts/ingest_iq.py` that reads SDR IQ captures and runs them through the same acceptance metrics. Start with file input; live SDR hooks can follow.
   - Document usage in `docs/hardware_bridge.md` so lab teams can drop captures into CI without touching estimator code.

4. **Regression hygiene**
   - Keep `test_dense_kf_vs_baseline` strict; ensure any README/documentation change is accompanied by seeded acceptance artifacts reflecting the updated thresholds.

Deliverables: updated acceptance code/tests, refreshed documentation, IQ ingestion stub, and a short changelog noting the new CRLB semantics.

**[END PROMPT]**
