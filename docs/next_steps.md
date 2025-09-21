# Follow-up Actions

## 1. Wire Smoke Tests into CI
- Promote the quick configs used in the recent smoke runs into a dedicated YAML (e.g., `sim/configs/smoke.yaml`).
- Add a `make smoke`-style helper (or a short script) that runs `python sim/phase1.py --atomic-mode --num-trials 5` and `PYTHONPATH=. python scripts/run_mc.py …` against the smoke config.
- Extend the GitHub Actions workflow to invoke the smoke helper on every PR, failing on non-zero exit and uploading the small telemetry bundle for debugging.

## 2. Rebaseline Monte Carlo Analytics
- Execute `python scripts/run_mc.py all -c sim/configs/mc_extended.yaml -o results/mc_regression -r $(date +%Y%m%d)`.
- Compare `SUMMARY.md` statistics against the latest published results; flag any deviations in RMSE, CRLB ratios, or convergence characteristics.
- Archive the new output under `results/` and refresh documentation tables/figures that cite these metrics.

## 3. Harden Fidelity Validation
- Update `scripts/validate_fidelity.py` to accept a telemetry path emitted by Phase 2 runs so it exercises real JSONL data rather than synthetic placeholders.
- Add a targeted pytest (e.g., `tests/metrics/test_fidelity.py`) that generates a tiny telemetry sample and asserts the CRLB/consensus checks pass.
- Document the validation workflow in `docs/automation.md`, including commands and expected artifacts, so the pipeline is reproducible by future contributors.
