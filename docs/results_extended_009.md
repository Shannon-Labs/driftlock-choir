# Monte Carlo Summary (extended_009)

This run used the refreshed `scripts/run_mc.py` harness with the configuration in
`sim/configs/mc_extended.yaml` and writes artifacts under
`results/mc_runs/extended_009/`.

## Phase 1 – alias-map calibration sweep

| Calibration mode | Mean reciprocity bias (ps) | Bias vs retune offsets (ps) | Bias vs SNR (ps) |
| --- | ---:| --- | --- |
| off | −1.20e4 | [−1.20e4, −1.20e4] | [−1.20e4, −1.20e4, −1.20e4] |
| loopback | 2.65 | [2.65, 2.65] | [2.65, 2.65, 2.65] |

Loopback calibration collapses the hardware-delay induced bias from ~12 ns down to about
3 ps across both retune offsets and SNR slices. Manifests live under:
- `results/mc_runs/extended_009/phase1/alias_calibration_sweep/calib_off/alias_map/`
- `results/mc_runs/extended_009/phase1/alias_calibration_sweep/calib_loopback/alias_map/`

## Phase 2 – consensus presets

![Phase 2 RMSE comparison](../results/mc_runs/extended_009/phase2_rmse_bar.png)

| Job | Weighting | Local KF | Converged | Final RMSE (ps) | Notes |
| --- | --- | --- | :--: | ---: | --- |
| dense_network_kf | metropolis_var | on | ✅ | 33.11 | KF bounded to ±30 ps / 2 kHz to stabilise start-up; hits target with small residual bias |
| dense_network_no_kf | metropolis | off | ✅ | 22.45 | Baseline consensus (metropolis) |
| small_network_kf_on | inverse_variance | on | ✅ | 40.04 | Local KF blended, meets 120 ps target |
| small_network_kf_off | inverse_variance | off | ✅ | 24.38 | Baseline for small network |

All consensus runs now converge reliably (previous KF presets diverged). Frequency RMS drops
slightly with the Kalman pre-filter (≈0.6% for the dense case) while clock residuals are capped by
the new magnitude clamps (`local_kf_max_abs_ps`, `local_kf_max_abs_freq_hz`).

These metrics feed directly into the generated `simulation_report.txt` for downstream packaging
(e.g., Speedrun application decks).
