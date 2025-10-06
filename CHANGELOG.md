# Changelog

All notable changes to the Driftlock Choir project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- GitHub Pages documentation hub with governance and technical primers.
- Governance documents (Code of Conduct, Quality Assurance, Release Readiness, etc.) aligned with the open-source launch.

### Changed
- Refreshed README and Getting Started guide with accurate commands and live-site links.
- Standardized examples, tests, and experiment imports to use `pathlib`-based project resolution.
- Updated GitHub Actions workflow dependencies to the latest maintained versions.

### Fixed
- Restored backward compatibility for `OscillatorModel.frequency` and beat-note frequency expectations used in the existing test suite.
- Hardened Experiment E1 parameter validation to gracefully handle invalid configurations.

## [0.1.0] - 2024-10-06

### Initial Open-Source Release
- Core chronometric interferometry primitives (oscillators, channel simulator, beat-note processor).
- Experiment E1 for basic beat-note formation and τ/Δf estimation.
- Example scripts, pytest suite, and audio demonstrations for the Driftlock Choir OSS drop.
- Initial documentation set including theoretical overview and hardware experiment roadmap.
