# Investor-Ready Artifacts Summary
*October 2, 2025 — Optimized for a16z Speedrun Review*

## 🎯 Key Refresh for Extended_011

### 1. ✅ Lead With Live Artifacts
**Hero Section Updates:**
- Subheading now highlights **"22.13 ps dense-network synchronization"** with direct link to `docs/results_extended_011.md`
- Primary button: `results/mc_runs/extended_011/SUMMARY.md`
- Secondary button: `results/kf_sweeps/dense_combo_scan/kf_sweep_summary.json`
- Badge copy: **"Tuned 0.32 / 0.03 / 1 combo"** with sweep link for verification

### 2. ✅ Verification Strip = Proof, Not Hype
- **`pytest -q`** → 15 tests in 9.65s (includes dense KF vs baseline regression)
- **Monte Carlo command** → `extended_011` results JSON
- **Sweep checker** → `scripts/verify_kf_sweep.py` confirming 20.93 ps min & 21.89 ps best mean
- All cards link straight to artifacts; messaging mentions guardrails

### 3. ✅ Concrete Metrics Front And Center
- **Dense preset (64 nodes)**: 22.13 ps (0.33 ps better than baseline with clock 0.32 / freq 0.03 / 1 iter)
- **Dense sweep minimum**: 20.93 ps (clock 0.22 / freq 0.03 / 2 iters)
- **Small network preset (25 nodes)**: 20.96 ps (3.41 ps improvement; 18.69 ps sweep min)
- **Guardrails**: scripts/verify_kf_sweep.py + seeded regression keep gains locked

### 4. ✅ Reproduction Flow Updated
```bash
# 1. Full Monte Carlo (600 trials, dense + small presets)
python scripts/run_mc.py all -c sim/configs/mc_extended.yaml -o results/mc_runs -r extended_011

# 2. Dense Network Sweep (~2.5 min, same parameter grid)
python scripts/sweep_phase2_kf.py --nodes 64 --density 0.22 \
  --gains 0.18,0.22,0.25,0.28,0.32 --freq-gains 0.03,0.05,0.07 \
  --iters 1,2 --seeds 4040,4141,4242 --write-json \
  --output-dir results/kf_sweeps --run-id dense_combo_scan --baseline

# 3. Validate sweep JSON against expectations
scripts/verify_kf_sweep.py results/kf_sweeps/dense_combo_scan/kf_sweep_summary.json \
  --expected-min 20.9337 --expected-best-mean 21.89 \
  --expected-clock 0.32 --expected-freq 0.03 --expected-iterations 1

# 4. Regression guardrail (seeded comparison)
pytest tests/test_consensus.py::test_dense_kf_vs_baseline -q
```
- Lists exact artifacts emitted and expected runtimes (~2.5 min sweep, ~3 min MC)
- Reinforces that documentation is gated by automated checks

### 5. ✅ Speedrun CTA Mirrors New Results
- Tagline: "Validated in Monte Carlo run extended_011"
- Metric tiles: 22.13 ps dense preset • 0.33 ps edge • 15 passing tests
- Buttons remain "Request Speedrun Demo" and "Download Tech Brief" (now points to `docs/results_extended_011.md`)
- Artifact links swapped to `extended_011`

### 6. ✅ Data Source References Updated
- All narrative references moved from extended_010 → extended_011
- Sweep discussion still cites 20.93 ps minimum & 21.89 ± 0.75 ps mean from dense scan
- Notes that the preset promotion is anchored by the sweep checker and regression

## 📊 Artifacts Ready for Review

### Primary Data Files
1. **Full Results**: `results/mc_runs/extended_011/final_results.json`
2. **KF Sweep**: `results/kf_sweeps/dense_combo_scan/kf_sweep_summary.json`
3. **Run Summary**: `results/mc_runs/extended_011/SUMMARY.md`
4. **Tech Brief**: `docs/results_extended_011.md`

### Verification Commands
```bash
# Quick validation (pytest with regression)
pytest -q

# Confirm dense preset beats baseline at seed 5001
pytest tests/test_consensus.py::test_dense_kf_vs_baseline -q

# Inspect sweep JSON for 0.32 / 0.03 / 1 combo
scripts/verify_kf_sweep.py results/kf_sweeps/dense_combo_scan/kf_sweep_summary.json \
  --expected-best-mean 21.89 --expected-clock 0.32 --expected-freq 0.03
```

## 🚀 Why This Resonates With Technical Investors

1. **Evidence-first**: Every claim backed by artifacts and run IDs
2. **Reproducibility**: Deterministic seeds, regression test, sweep checker
3. **Speed**: <5 minutes end-to-end for full validation
4. **Transparency**: Gains/iterations spelled out, JSON accessible
5. **Guardrails**: Automated checks prevent stale documentation
6. **Scalability Story**: Dense vs small network metrics presented together

## 📧 For Speedrun Reviewers

Reviewers will immediately see:
- 22.13 ps dense preset advantage, quantified vs baseline
- Direct links to extended_011 data + sweep JSON
- Exact commands (with expected runtimes) to reproduce results
- Regression + verification scripts that protect against drift

## 🎯 The Killer Stats (Extended_011)

```
Dense Preset:     22.13 ps (clock 0.32, freq 0.03, 1 iter)
Baseline Delta:    0.33 ps advantage (metropolis_var vs metropolis)
Dense Sweep Min:  20.93 ps (clock 0.22, freq 0.03, 2 iters)
Small Preset:     20.96 ps (clock 0.25, freq 0.05) → 3.41 ps edge
Calibration:       2.65 ps residual bias (loopback)
Pytest:            15 tests (seeded dense regression included)
```

Perfect alignment for the Speedrun diligence checklist: rigorous, reproducible, and fast to verify. 🔬