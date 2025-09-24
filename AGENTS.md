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
- **Profile sweep:** For each TDL profile in `src/chan/tdl.py` (e.g., `URBAN_CANYON`, `IDEAL`, others), run single-handshake diagnostics and `scripts/run_monte_carlo.py --smoke-test --channel-profile <PROFILE>`. Capture τ/Δf bias and consensus RMSE.
- **Bias forensics:** Whenever τ bias exceeds ~0.2 ns, note whether coarse hints, phase unwrapping, or pathfinder behaviour is responsible. Do not chase the legacy `<3 ps` target—focus on realistic multipath behaviour.
- **Minimal edits:** If Phase1/Phase2 helpers need tweaks to expose metrics, make the smallest change possible and keep the existing CLI flags working. Always re-run `pytest tests/test_chronometric_handshake.py`.
- **Document results:** Update README “Latest Results” and roadmap only after you have verified metrics for a profile. Summaries should read like `URBAN_CANYON: 0.45 ns bias (dominated by …)`.
- **Repo hygiene:** Leave the tree clean (`git status` empty), remove scratch files, and note new artifacts under `results/` when relevant.
