# Phase 1 Chronometric Handshake Redesign

## Objective
Rebuild the two-node simulation so it explicitly models the Chronometric Interferometry handshake described in the provisional patent. The simulator must produce realistic beat signals, execute the closed-form τ/Δf estimator, and emit diagnostic plots required for Deliverable 1.

## Key Model Elements
- **Nodes**: Sampled via a `ChronometricNode` wrapper storing actual carrier `f_i`, intentional offset `Δf`, random phase `θ_i`, and TCXO-like clock bias/frequency error (ppm).
- **Channel**: Deterministic LOS propagation with geometric delay `τ_geo = d / c` plus relative clock bias; noise comes from the existing `NoiseGenerator` (AWGN + optional phase jitter hooks).
- **Mixing Pipeline**:
  1. Generate the analytic transmit tone at node i for 10–20 μs at a baseband rate tied to `Δf`.
  2. Apply propagation delay and LO offsets to build the complex beat expression.
  3. Inject AWGN, then apply a 2nd-order Butterworth band-pass centred on the beat.
  4. Sub-sample with an ADC operating at ≈ `2·Δf` to emulate the low-rate capture path.
- **Measurement Extraction**:
  - Unwrap beat phase, fit a line, and recover `Δf` from the slope.
  - Solve Equation (2) for `τ` using the known `θ_i`, `θ_j`, `f_i`, resolving the cycle ambiguity around the expected geometric delay.
  - Track residual phase RMS as a quality metric.
- **Two-Way Handshake**:
  - `run_two_way()` returns directional estimates `(τ̂_ij, Δf̂_ij)` and `(τ̂_ji, Δf̂_ji)`.
  - Combine them to recover clock bias `δt = 0.5(τ̂_ij - τ̂_ji)` and geometric ToF `τ̂_geo = 0.5(τ̂_ij + τ̂_ji)`.

## API Changes in `sim/phase1.py`
- Extended `Phase1Config` with handshake-specific knobs (`distance_m`, `delta_f_hz`, beat duration, rate factors, ppm/bias statistics, RNG seed, optional trace capture SNR).
- Added `ChronometricHandshakeSimulator`, `ChronometricNode`, and rich result structures (`DirectionalMeasurement`, `TwoWayHandshakeResult`, `HandshakeTrace`).
- Replaced the legacy CRLB sweeps with an SNR Monte Carlo around the patent estimator, capturing RMSE/Bias for τ, Δf, and clock offset.
- Added JSON serialization helpers and plotting routines for beat waveforms and error curves.

## Testing Hooks
- Deterministic seeding via `rng_seed` for reproducibility.
- Optional trace capture (first SNR sweep or user-selected SNR) feeding waveform/phase plots.

## Next Steps
1. Expose the handshake simulator to Phase 2 so pairwise measurements can be reused across consensus experiments.
2. Add unit tests that seed the RNG and assert τ/Δf RMSE against analytic expectations for high-SNR cases.
3. Extend the channel helper with Doppler knobs in preparation for Phase 3 mobility studies.
