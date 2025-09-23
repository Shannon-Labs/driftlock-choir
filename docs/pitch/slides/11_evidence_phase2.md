# Slide 11 — Evidence: Phase 2 Consensus

Title: Network-wide convergence with predictive guarantees

Bullets
- Plot: `results/phase2/phase2_convergence.png` — timing RMSE drops below 100 ps in <4 ms (ε auto-tuned via spectral margin).
- Telemetry: predicted vs. measured iterations align within ±1 step using `alg/spectral_predictor.py`.
- Residuals stay zero-mean with tight histograms (`phase2_residuals.png`); CRLB ratios hover at 1.1×.
- Local KF pre-filter yields 35% clock RMS improvement pre-consensus; JSONL/CSV streamed via `TelemetryExporter` for audits.

Notes for presenter
- Highlight the predictive element: investors love deterministic-looking automation.
- Call out baseline comparison mode to show 10–100× uplift vs GNSS/PTP when asked.
- Explain how convergence guarantees translate into future API SLAs for automated responses.
