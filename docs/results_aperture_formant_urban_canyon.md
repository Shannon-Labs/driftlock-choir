# Project Aperture × Formant — URBAN_CANYON sweep (2025-09-25)

This note captures the joint Aperture/Formant experiments executed on 2025-09-25. All artifacts live under `results/project_aperture_formant/URBAN_CANYON/`, with the most relevant runs:

- `20250925T193300Z_urban_guard`: initial blend/guard combinations (no first-path accuracy metrics).
- `20250925T195000Z_urban_guard_fpmetrics`: same grid with the new first-path accuracy counters.
- `20250925T205000Z_urban_guard_scan2`: threshold/noise-guard/first-path-blend variations.
- `20250925T210000Z_guard_focus`: targeted reruns of the most promising guard settings.
- `20250925T210600Z_noise2_focus`: extreme threshold experiment (`noise2_thresh18_pre15_guard30`).
- `20250925T211000Z_baseline_metrics`: baseline formant sweep with updated metrics.
- `20250925T211200Z_blend025_metrics`: widened guard/aperture (`blend025_pre30_guard60`).
- `20250925T211400Z_include_disable`: literal-fundamental test (`include_disable_pre20_guard40`).

## Key Metrics

| scenario | pre_guard_ns | guard_ns | aperture_ns | first_path_blend | noise_guard | rel_threshold_db | missing_f0 | coarse_lock_fwd | coarse_lock_rev | first_path≤10ns_fwd | first_path≤10ns_rev | first_path_neg_fwd | first_path_neg_rev | two_way_tau_bias_ns |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline_guarded_formant25k (`20250925T211000Z`) | 0.0 | 30.0 | 100.0 | 0.05 | 6.0 | -12.0 | enabled | 0.531 | 0.583 | 0.281 | 0.240 | 0.250 | 0.323 | 19.76 |
| blend025_pre30_guard60 (`20250925T211200Z`) | 30.0 | 60.0 | 160.0 | 0.25 | 6.0 | -12.0 | enabled | 0.781 | 0.792 | 0.167 | 0.198 | 0.250 | 0.281 | 22.13 |
| noise3_thresh18_pre15_guard35 (`20250925T210000Z`) | 15.0 | 35.0 | 110.0 | 0.30 | 3.0 | -18.0 | disabled | 0.583 | 0.646 | 0.271 | 0.229 | 0.260 | 0.333 | 20.18 |
| pre5_guard25_blend05 (`20250925T210000Z`) | 5.0 | 25.0 | 90.0 | 0.05 | 6.0 | -12.0 | enabled | 0.458 | 0.531 | 0.281 | 0.240 | 0.250 | 0.323 | 19.75 |
| noise2_thresh18_pre15_guard30 (`20250925T210600Z`) | 15.0 | 30.0 | 100.0 | 0.30 | 2.0 | -18.0 | disabled | 0.573 | 0.604 | 0.042 | 0.083 | 0.833 | 0.833 | 17.57 |
| include_disable_pre20_guard40 (`20250925T211400Z`) | 20.0 | 40.0 | 120.0 | 0.20 | 6.0 | -12.0 | disabled + include_f0 | 0.604 | 0.677 | 0.188 | 0.198 | 0.250 | 0.292 | 22.71 |

*Notes:* `missing_f0` indicates whether the missing-fundamental decoder was engaged. `include_f0` applies only in the last row (literal fundamental injected into the comb).

## Observations

- **Baseline vs widened guard:** increasing pre-/guard/aperture from (0,30,100) to (30,60,160) lifts coarse lock to 0.78/0.79 but cuts “within 10 ns” hits to ≈0.17/0.20. Mean τ bias remains >22 ns and first-path negatives persist (~25%).
- **Aggressive thresholds (`noise3`):** relaxing both the relative threshold (−18 dB) and noise guard (3×) while keeping a modest (15 ns, 35 ns) guard boosts within‑10ns captures to ≈0.27/0.23, but one quarter to one third of trials still claim a negative first-path (early-noise) hit.
- **Shrinking guard to 5/25 ns:** coarse lock falls to ~0.46/0.53 but the within‑10ns and negative rates match the baseline almost exactly; the detector continues to oscillate between the direct path and the late cluster.
- **Extreme guard relax (`noise2`):** pushing the noise guard multiplier to 2× drives 83% of trials into negative first-path errors (hundreds of ns early). Two-way bias drops slightly (17.6 ns) only because the forward and reverse errors swing in opposite directions; the result is unusable.
- **Disabling the missing-fundamental decoder:** with the literal fundamental present (`include_disable_pre20_guard40`) the aperture still reports ~25% negative hits and τ bias >22 ns, indicating the bias is dominated by guard window choice, not the decoder itself.
- Across experiments, forward alias success stays ≥0.99 and vowel labels remain stable when the decoder is active (100% on the chosen vowel). When disabled, formant fields drop to `null` as expected.

## Files of Interest

- Summaries (Markdown + JSON): `results/project_aperture_formant/URBAN_CANYON/20250925T210000Z_guard_focus/summary.md`, `.../20250925T211000Z_baseline_metrics/summary.md`, etc.
- Per-scenario JSON includes the new metrics: `first_path_within_{5,10}ns_rate` and `first_path_negative_rate` for both directions.

These sweeps indicate that simply expanding the guard or relaxing thresholds cannot eliminate the 20–30 ns bias—each change trades off “early noise” vs. “late cluster” selection rather than centering on the true first path. Further progress likely requires smarter aperture scoring (e.g., frequency-consistent scoring or multipath-aware pruning) rather than scalar guard tuning alone.

## First-Path Accuracy Snapshot (2025-09-25)

| scenario | guard_ns | blend | missing_f0 | f_within10 | r_within10 | f_negative | r_negative | tau_bias_ns |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| pre5_guard25_blend05 | 25.0 | 0.05 | on | 0.281 | 0.240 | 0.250 | 0.323 | 19.75 |
| pre5_guard25_blend10 | 25.0 | 0.10 | on | 0.281 | 0.240 | 0.250 | 0.323 | 19.75 |
| baseline_guarded_formant25k | 30.0 | 0.05 | on | 0.281 | 0.240 | 0.250 | 0.323 | 19.76 |
| noise2_thresh18_pre15_guard30 | 30.0 | 0.30 | off | 0.042 | 0.083 | 0.833 | 0.833 | 17.57 |
| noise3_thresh18_pre15_guard35 | 35.0 | 0.30 | off | 0.271 | 0.229 | 0.260 | 0.333 | 20.18 |
| include_disable_pre20_guard40 | 40.0 | 0.20 | off | 0.188 | 0.198 | 0.250 | 0.292 | 22.71 |
| blend025_pre30_guard60 | 60.0 | 0.25 | on | 0.167 | 0.198 | 0.250 | 0.281 | 22.13 |

*Interpretation:* the baseline guard (0/30/100 ns) already sees ~25–32% negative picks; aggressive thresholding without missing-fundamental (`noise2`) floods the detector with early noise, while widening guard to 60 ns improves coarse lock but drops the ≤10 ns hit rate to ~0.17/0.20. Run `python scripts/guard_clutter_report.py results/project_aperture_formant/URBAN_CANYON` to regenerate this table from the JSON artifacts.
