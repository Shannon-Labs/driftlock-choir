# Monte Carlo Summary (extended_011)

## Dense Preset Tuning Update

We reran the extended Monte Carlo after promoting the **clock gain 0.32 / freq gain 0.03 / 1 iteration** combo from the dense KF sweep. The Metropolis-variance weighting with the updated shrinkage now delivers a reproducible improvement over the no-KF baseline:

- **Dense network (64 nodes)**: **22.13 ps** vs **22.45 ps** baseline (Δ ≈ **0.33 ps**)
- **Small network (25 nodes)**: unchanged at **20.96 ps** with KF vs **24.38 ps** baseline

The gain change is confined to the dense KF preset in `sim/configs/mc_extended.yaml`; small network parameters remain at 0.25 / 0.05.

## Phase 1 – Alias-Map Calibration Sweep

Cal results mirror the prior run.

| Calibration Mode | Mean Reciprocity Bias (ps) |
| --- | ---:|
| off | −1.20e4 |
| loopback | 2.65 |

Loopback calibration continues to collapse the hardware-delay induced bias from ~12 ns down to ~2.65 ps across all conditions.

## Phase 2 – Consensus Presets

| Job | Weighting | Local KF | Converged | Final RMSE (ps) | Baseline Δ |
| --- | --- | --- | :--: | ---: | ---: |
| dense_network_kf | metropolis_var | **on** | ✅ | **22.13** | **−0.33** |
| dense_network_no_kf | metropolis | off | ✅ | 22.45 | reference |
| small_network_kf_on | inverse_variance | **on** | ✅ | **20.96** | **−3.41** |
| small_network_kf_off | inverse_variance | off | ✅ | 24.38 | reference |

### Sweep Verification

Use `scripts/verify_kf_sweep.py` to sanity-check dense sweep artifacts before documentation updates:

```bash
scripts/verify_kf_sweep.py \
  results/kf_sweeps/dense_combo_scan/kf_sweep_summary.json \
  --expected-min 20.9337 --expected-best-mean 21.89 \
  --expected-clock 0.32 --expected-freq 0.03 --expected-iterations 1
```

The checker confirms the dense scan still bottoms out at **20.93 ps** (clock 0.22 / freq 0.03 / 2 iter) and that the best mean sits at **21.89 ps** for the adopted **0.32 / 0.03 / 1 iter** combination, with a +1.76 ps mean gain against the baseline preset.

For alternate seeds (5001/5003/5005) the combo continues to hold: see `results/kf_sweeps/dense_combo_scan_alt/kf_sweep_summary.json` (min **20.63 ps**, mean **21.81 ps**) and validate with the same CLI.

### Regression Guardrail

`pytest tests/test_consensus.py::test_dense_kf_vs_baseline -q` now exercises **both** dense presets using `Phase2Simulation` seeded at 5001. The run asserts that the tuned KF stays at least 1 ps better than the baseline, flagging any regressions in network sampling, measurement conditioning, or shrinkage tuning.

## Artifacts

- Results root: `results/mc_runs/extended_011`
- Dense KF manifest: `results/mc_runs/extended_011/phase2/dense_network_kf/phase2_results.json`
- Verification script: `scripts/verify_kf_sweep.py`

---

*Generated: September 2025*
*Driftlock v0.10.1 – Dense KF combo locked to 0.32 ⧸ 0.03*
