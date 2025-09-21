# Fidelity Validation Workflow

## Overview
The fidelity validation workflow uses `scripts/validate_fidelity.py` to check simulation telemetry against analytical bounds (CRLB for Phase 1, consensus convergence for Phase 2). It supports both synthetic simulations (default) and real JSONL telemetry from Phase 2 runs (e.g., `phase2_runs.jsonl`). The script computes RMSE, CRLB ratios, and convergence metrics, generating reports for pass/fail assessment. Integration with pytest ensures automated testing.

## Prerequisites
- Install dependencies: `pip install -r requirements.txt`
- Set `PYTHONPATH=.` for imports from `src/`
- For real telemetry: Run Phase 2 simulation first (e.g., `python sim/phase2.py` or `python scripts/run_mc.py phase2 ...`) to generate JSONL files under `results/phase2/.../phase2_runs.jsonl`

## Usage

### Synthetic Validation (Default)
Run with config for simulated data:
```
PYTHONPATH=. python scripts/validate_fidelity.py sim/configs/hw_emulation.yaml -o results/fidelity_run -n 100 --seed 42 --log-level INFO
```
- Skips `--telemetry-path`; generates synthetic Phase 1/2 runs.
- Use `--no-phase1` to skip Phase 1, `--phase2` to include consensus.
- Add `--skip-crlb-validation` if you only need consensus checks (no Phase 1) and want to avoid CRLB cross-checks.

### Real Telemetry Validation
Load existing JSONL from Phase 2:
```
PYTHONPATH=. python scripts/validate_fidelity.py sim/configs/default.yaml \
  --telemetry-path results/phase2/dense_network_kf/phase2_runs.jsonl \
  --output results/fidelity_real --log-level INFO
```
- Extracts metrics like `rmse_tau_ps`, `converged`, `snr_db` from JSONL lines.
- Computes CRLB from telemetry params (SNR, BW, duration); validates consensus RMSE and convergence.
- Falls back to config values if missing in telemetry.
- Use `--skip-hw-validation` to bypass the hardware emulation cross-check when only consensus telemetry is available.
- Combine with `--skip-crlb-validation` for telemetry sources that lack CRLB support (e.g., aggregated JSONL samples).

### Options
- `--config <path>`: YAML config (default: `sim/configs/hw_emulation.yaml`)
- `--telemetry-path <path>`: JSONL file; skips simulation if provided
- `-o, --output <dir>`: Output directory (default: `results/validate_fidelity`)
- `-r, --run-id <id>`: Unique run ID (auto-generated if omitted)
- `-n, --n-trials <int>`: Trials for synthetic runs (default: 100)
- `--seed <int>`: RNG seed for reproducibility (default: 42)
- `--log-level <level>`: DEBUG/INFO/WARNING (default: INFO)
- `--skip-hw-validation`: omit the hardware emulation check (useful for consensus-only telemetry)
- `--skip-crlb-validation`: omit CRLB cross-validation (useful for quick smoke tests or telemetry without variance data)

## Expected Artifacts
Validation outputs to `<output>/<run_id>/`:
- `fidelity_report.json`: Detailed results (overall_pass, per-validation pass/fail, discrepancy_summary)
- `validation_summary.txt`: Human-readable summary (e.g., "Passed: 3/3, Overall: PASS")
- `loaded_telemetry.json` (real mode): Parsed telemetry entries
- `validation_telemetry.json` (synthetic mode): Combined Phase 1/2 data
- Logs in console or file if `--log-level DEBUG`

Example summary.txt:
```
Fidelity Validation Report
Run ID: fidelity_1726940575
Config: sim/configs/default.yaml
Passed: 3/3
Overall: PASS
```

Exit code: 0 (PASS), 1 (FAIL on discrepancies > threshold).

## Integration with Pytest
The test suite includes `tests/metrics/test_fidelity.py` for automated validation:
```
pytest tests/metrics/test_fidelity.py -v
```
- Generates synthetic 5-trial JSONL, runs `validate_fidelity.py --telemetry-path <temp.jsonl>`
- Asserts exit 0 and `overall_pass: true` in report.json
- Uses tolerances for numeric checks (e.g., RMSE ~30 ps, converged=True)
- Run with CI: Included in `.github/workflows/test.yml` via `pytest tests/`

For custom tests, extend with bad telemetry (e.g., high RMSE) to assert FAIL.

## Troubleshooting
- **ModuleNotFoundError 'src'**: Ensure `PYTHONPATH=.`
- **No telemetry loaded**: Check JSONL structure (keys: rmse_tau_ps, converged, snr_db)
- **CRLB mismatch**: Verify config/telemetry params (SNR, BW)
- **Report not generated**: Check output dir permissions; use `--output .`

This workflow ensures reproducible fidelity checks, integrable into CI/CD for regression testing.