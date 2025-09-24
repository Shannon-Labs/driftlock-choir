### Prompt: "Simulation Hardening" – Phase-Noise & Reproducibility Pass

**[START PROMPT]**

**Persona**: You are Jules, acting as the Modeling Reliability Lead. The multipath sweep is complete; we now need to eliminate the most glaring physics gaps flagged in the recent audit before layering on new modulation or Harmony work.

#### Scope
Focus on the immediate 1–2 week items from the audit:
- Power-law phase-noise spectrum (h\_-2 … h\_2) for oscillators.
- Temperature-dependent LO parameters (non-linear coefficients + optional thermal time constant stubs).
- Deterministic seeding across all Monte Carlo entry points.
- Hooks to make per-edge measurement diagnostics easy to export.

#### Tasks
1. **Oscillator model**
   - Extend the existing oscillator/phase-noise generator to accept `{ -2: h_-2, …, 2: h_2 }` parameters.
   - Implement spectrum synthesis (e.g., sum of filtered noise processes) with unit tests showing regression vs known spectra.

2. **Temperature hooks**
   - Add optional non-linear temperature coefficients and a simple thermal RC to the LO model.
   - Provide a stub API for injecting temperature traces (actual data integration can come later).

3. **Reproducibility**
   - Audit all stochastic entry points (`sim/phase*.py`, Monte Carlo scripts) to ensure `np.random.default_rng(seed)` is used and seeds are surfaced in manifests.

4. **Diagnostics**
   - Add a lightweight export helper so `_populate_measurements` (Phase 2) can dump per-edge τ/Δf errors when requested (CLI flag / config option).

5. **Verification**
   - Extend the test suite (`tests/`) with coverage for the new phase-noise generator and seeding behaviour.
   - Run `pytest` and document new baseline artefacts under `results/` if any.

Deliver a short changelog (`SIM_HARDENING_NOTES.md`) summarizing the new knobs and default values so downstream projects (Swing, Harmony) can consume them easily.

**[END PROMPT]**
