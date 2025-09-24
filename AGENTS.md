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
