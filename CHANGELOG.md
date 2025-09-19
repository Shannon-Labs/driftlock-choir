# Changelog

All notable changes to Driftlock will be documented in this file.

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

*For full commit history, see [GitHub](https://github.com/shannon-labs/driftlock)*