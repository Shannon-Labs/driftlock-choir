# Changelog

All notable changes to Driftlock will be documented in this file.

## [0.10.2] - TBD

### Planned
- Investigate frequency-gain parity (0.03 vs 0.05) and multi-iteration dense presets
- Extend seeded sweeps (5001/5003/5005) into automated CI regression
- Add optional small-network gain retune (0.25/0.05 baseline) with parity checker
- Publish deployment variant landing pages (telecom, defense) sourced from investor strategy

---

## [0.10.1] - October 2, 2025

### Added
- **Dense preset retune**: Monte Carlo `extended_011` locks clock gain 0.32 / freq gain 0.03 / 1 iteration
  - Dense networks: 22.13ps (with KF) vs 22.45ps (baseline) → +0.33ps edge
  - Regression guardrail: `tests/test_consensus.py::test_dense_kf_vs_baseline` enforces ≥1ps improvement (seed 5001)
- `scripts/verify_kf_sweep.py` sweep checker to validate JSON artifacts (min 20.93ps, best mean 21.89ps)
- `docs/results_extended_011.md` tech brief summarizing extended_011 artifacts and verification flow

### Changed
- README, quickstart, and website now highlight 22.13ps dense preset and verification workflow
- Investor collateral (Speedrun packet, deployment summary, CTA messaging) updated to point at extended_011 artifacts and 15-test pytest run

### Files Updated
- `sim/configs/mc_extended.yaml`, `results/mc_runs/extended_011/*` (regenerated artifacts)
- `tests/test_consensus.py` (new regression)
- `scripts/verify_kf_sweep.py` (new helper)
- `README.md`, `docs/quickstart.md`, `docs/results_extended_011.md`, `index.html`
- `a16z-speedrun/` collateral & deployment summaries refreshed for 22.13ps narrative

## [0.10.0] - September 19, 2025

### Added
- **Major Breakthrough**: Variance-weighted Kalman pre-filter now BEATS baseline
  - Dense networks: 22.08ps (with KF) vs 22.45ps (baseline)
  - Small networks: 20.96ps (with KF) vs 24.38ps (baseline)
- Shrinkage-based local smoother in `_run_local_kf()`
- Extended Run 010 validation results
- Strategic investor outreach strategy document
- Bell Labs moment positioning throughout materials

### Changed
- Kalman filter now enhances rather than degrades performance
- Updated all documentation to reflect September 2025 dates
- Website now highlights 22.08ps achievement with KF
- README updated with latest validation results

### Technical Improvements
- Variance-weighted shrinkage for per-edge measurements
- Numerical stability through intelligent clamping
- Maintains single-iteration convergence

### Files Updated
- `docs/results_extended_010.md` - New validation results
- `docs/quickstart.md` - Updated with new performance metrics
- `index.html` - Website reflects 22.08ps breakthrough
- `README.md` - Latest results and Monte Carlo documentation
- `a16z-speedrun/` - Multiple files with Bell Labs positioning

## [0.9.0] - September 18, 2025

### Added
- Initial Monte Carlo validation framework
- Extended Run 009 with 600+ simulations
- Loopback calibration (2.65ps bias achievement)
- a16z Speedrun application materials

### Changed
- Baseline consensus: 22.45ps (dense), 24.38ps (small)
- Patent application filed (September 2025)

---

*For full commit history, see [GitHub](https://github.com/shannon-labs/driftlock choir)*