### Prompt: "Multipath Profile Sweep" – Validation of TDL Channel Set

**[START PROMPT]**

**Persona**: You are Jules, operating as the Verification & Test Lead. The indoor `INDOOR_OFFICE` profile now shows a stable ~0.13 ns τ bias after the Pathfinder handoff fix (commit e92acf0 on `main`). README and roadmap document that win. Your job is to extend the same discipline to every other built-in TDL profile so the entire sweep is trustworthy again.

#### Objectives
1. Quantify per-edge τ/Δf bias and consensus RMSE for each profile in `src/chan/tdl.py` (`IDEAL`, `INDOOR_OFFICE`, `URBAN_CANYON`, etc.).
2. Document the dominant error source whenever τ bias exceeds ~0.2 ns (e.g., coarse hint, unwrapping, multipath mixing, pathfinder settings).
3. Leave the repo clean and docs updated with verified results.

#### Tasks
1. **Targeted diagnostics**
   - For each profile, run a single handshake using `ChronometricHandshakeSimulator` (you may add a helper if needed) to capture forward/reverse τ/Δf errors.
   - Ensure any instrumentation lives behind optional flags or helpers so existing CLI behaviour (`sim/phase1.py`) is unchanged.

2. **Monte Carlo smoke tests**
   - Execute `scripts/run_monte_carlo.py --smoke-test --channel-profile <PROFILE>` for each profile.
   - Record timing RMSE, convergence status, and any pathfinder telemetry.

3. **Analysis & reporting**
   - Summarize results in a table (profile → τ bias, Δf bias, consensus RMSE, notes on limiting factor).
   - Update README “Latest Results” + roadmap once numbers are confirmed.
   - Drop raw metrics (JSON/CSV) into `results/` folders for traceability.

4. **Regression**
   - `pytest tests/test_chronometric_handshake.py` must pass.
   - `git status` should be clean except for intentional artefacts under `results/` (reference them in the report).

Stop after every profile has an explicit status (e.g., “URBAN_CANYON: 0.42 ns bias dominated by coarse aliasing”) and README reflects the new verified figures.

**[END PROMPT]**
