# Repository Guidelines

## Project Structure & Module Organization
The modelling stack lives in `src/`, split by layer: `alg/` contains estimation and control algorithms, `phy/` models oscillators and noise sources, `hw/` captures RF front-end components, `net/` defines MAC and topology utilities, and `metrics/` holds analysis helpers such as CRLB and bias/variance calculators. Simulation entry points sit in `sim/phase1.py`, `sim/phase2.py`, and `sim/phase3.py`; shared experiment options live in `sim/configs/`. Batched Monte Carlo execution is provided by `scripts/run_mc.py`. Place new reusable primitives alongside peers in `src/` and keep scenario-specific glue in `sim/`.

## Build, Test, and Development Commands
- `python -m venv .venv && source .venv/bin/activate`: create an isolated environment.
- `pip install numpy scipy matplotlib pyyaml pytest`: install core numerical, plotting, YAML, and testing dependencies used across modules.
- `python sim/phase1.py`: run the two-node physical limit study with interactive plots and optional result dumps.
- `python scripts/run_mc.py --simulation-type all --config sim/configs/default.yaml`: execute a full Monte Carlo sweep; override `--n-workers` to parallelize.

## Coding Style & Naming Conventions
Follow PEP 8 with 4-space indentation, descriptive snake_case for functions, and UpperCamelCase for dataclasses or simulators. Leverage type hints and dataclass annotations as seen across `sim/phase*.py`. Keep modules pure where feasible so they can be imported without side effects; gate CLI execution behind `if __name__ == "__main__":`.

## Testing Guidelines
Use `pytest` for unit coverage and stash tests in a `tests/` directory that mirrors the `src/` layout. Target deterministic pieces (e.g., CRLB calculators, topology utilities) with assertions on numeric tolerances, and isolate stochastic behaviour by seeding RNGs. Before large runs, smoke test scenarios with reduced Monte Carlo counts (`--n-workers 1 --save-intermediate`) to validate configuration handling.

## Commit & Pull Request Guidelines
Adopt a Conventional Commits style (`feat:`, `fix:`, `refactor:`, etc.) with an imperative subject capped at 72 characters. Provide context in the body for parameter changes or new experiment presets. PRs should include: purpose and scope, links to tracking issues, sample commands used for validation, and (when applicable) artifacts or plots saved under `results/`. Request review once CI or local pytest runs pass and configuration files load without schema errors.

## Configuration & Reproducibility
Version any new YAML presets under `sim/configs/` and document expected output directories. Prefer relative paths so runs remain portable; long experiments should emit metadata via `MonteCarloRunner._save_run_config` for traceability.
