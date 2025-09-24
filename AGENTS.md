# Repository Guidelines

## Project Structure & Module Organization
- Core modelling lives in `src/`: `alg/` hosts estimation and control helpers, `phy/` captures oscillators and noise, `hw/` mirrors RF front-end components, `net/` handles MAC/topology utilities, and `metrics/` provides CRLB and bias analysis.
- Simulator entry points sit in `sim/phase1.py`, `sim/phase2.py`, and `sim/phase3.py`; reusable experiment presets live under `sim/configs/`.
- Batch orchestration (Monte Carlo, sweeps) is provided by `scripts/run_mc.py`; outputs land in `results/phase*/` and should remain committed for reproducibility snapshots.
- Tests mirror the module layout in `tests/`, while `docs/` and `examples/` host explainer material and minimal scenarios.

## Build, Test, and Development Commands
- `python -m venv .venv && source .venv/bin/activate`: create and enter a local virtual environment.
- `pip install -r requirements.txt`: install core numerical, network, and plotting dependencies.
- `python sim/phase1.py` / `python sim/phase2.py`: run handshake and consensus studies; inspect saved plots in `results/phase1/` and `results/phase2/`.
- `python scripts/run_mc.py --simulation-type all --config sim/configs/default.yaml`: execute the Monte Carlo bundle; pass `--n-workers` to parallelize.

## Coding Style & Naming Conventions
- Follow PEP 8, 4-space indentation, and descriptive `snake_case` for functions; reserve `UpperCamelCase` for dataclasses, simulators, and protocol descriptors.
- Gate scripts with `if __name__ == "__main__":` and prefer pure module imports so simulations compose cleanly.
- Annotate public APIs with type hints and docstrings; add concise comments only for non-obvious math or control flow.

## Testing Guidelines
- Use `pytest` with deterministic seeds (`np.random.default_rng`) for stochastic components.
- Name tests after the feature under test (e.g., `test_consensus_variance.py`) and colocate fixtures beside the module-under-test in `tests/`.
- Run `pytest` locally before submitting patches; add scenario-derived regression cases when fixing numeric drift.

## Commit & Pull Request Guidelines
- Adopt Conventional Commits (`feat:`, `fix:`, `refactor:`) with imperative subjects under 72 characters.
- PRs should explain scope, list validation commands, link tracking issues, and attach key artifacts from `results/` when behaviour changes.
- Ensure configuration changes are versioned under `sim/configs/` with expected output paths noted in the PR.

## Current Multipath Validation Status
- `INDOOR_OFFICE` profile now holds ~0.13 ns bias after the Pathfinder peak-path fix; README reflects this.
- Consensus Monte Carlo smoke tests run clean but still report high timing RMSE because edge τ bias remains ~0.13 ns.
- No other channel profiles have been re-validated since the fix; commit `8b2c85d` is the known-good baseline.

## Next Agent Checklist
- fix(align): Implement fractional coarse alignment to mitigate quantization bias.
- fix(pathfinder): Refine late-path pruning to pass URBAN_CANYON guardrails.
- test(indoor): Investigate and reconcile INDOOR_OFFICE simulation/lab discrepancy.
- **Profile sweep:** For each TDL profile in `src/chan/tdl.py` (e.g., `URBAN_CANYON`, `IDEAL`, others), run single-handshake diagnostics and `scripts/run_monte_carlo.py --smoke-test --channel-profile <PROFILE>`. Capture τ/Δf bias and consensus RMSE.
- **Bias forensics:** Whenever τ bias exceeds ~0.2 ns, note whether coarse hints, phase unwrapping, or pathfinder behaviour is responsible. Do not chase the legacy `<3 ps` target—focus on realistic multipath behaviour.

### 2025-09-24 Update (agent)
- **CODE RESTORED TO `8b2c85d`:** After a series of failed experiments with a new pathfinder and a circular convolution bug, the codebase has been reverted to this known-good commit.
- **`fftconvolve` bug FIXED:** The `_coarse_delay_estimate` function now correctly uses `signal.convolve`, which has resolved the catastrophic `first_path_error_ns` and vastly improved the `rmse_over_crlb` ratio.
- **New Performance Baseline:**
  - **IDEAL:** -0.13 ns bias, 1.05 RMSE/CRLB. **PASSING GUARDRAILS.**
  - **URBAN_CANYON:** 0.45 ns bias, 9.88 RMSE/CRLB. Failing CRLB, but bias is good.
  - **INDOOR_OFFICE:** 1.76 ns bias, 21.5 RMSE/CRLB. Failing bias and CRLB, as expected.
- **Next Steps:** With the codebase stable and the primary bug fixed, the next agent is cleared to begin **Project Swing**. The remaining multipath issues in `URBAN_CANYON` and `INDOOR_OFFICE` will be addressed after the Project Swing integration.
- **Minimal edits:** If Phase1/Phase2 helpers need tweaks to expose metrics, make the smallest change possible and keep the existing CLI flags working. Always re-run `pytest tests/test_chronometric_handshake.py`.
- **Document results:** Update README “Latest Results” and roadmap only after you have verified metrics for a profile. Summaries should read like `URBAN_CANYON: 0.45 ns bias (dominated by …)`.
- **Repo hygiene:** Leave the tree clean (`git status` empty), remove scratch files, and note new artifacts under `results/` when relevant.
