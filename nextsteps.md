# Driftlock – Next Steps for Choir Acceptance (handoff)

This brief captures the post-breakthrough state of the Choir acceptance harness so the next engineer (AI or human) can keep momentum and polish the deliverables for investors, partners, and the website.

## Context (what we’re shipping)
- Driftlock Choir simulates a multi-carrier "choir/organ" transmitter with two receiver paths:
  - **Coherent path**: per-tone phasors → phase unwrap vs. frequency → weighted least squares → τ̂ with confidence intervals.
  - **Aperture path**: envelope/cepstrum detects the missing fundamental Δf (and harmonic peaks) even when that tone is absent.
- The acceptance harness lives in `driftlock_choir_sim/sims/run_acceptance.py` and now produces JSON + PNG + PDF artifacts in **3.7 seconds**.

## Current status (2025-10-XX)
- ✅ All acceptance checks pass with large margins: 45.0 ps RMSE, RMSE/CRLB 0.83, Δf spike 58 dB, payload BER 0.
- ✅ Reciprocity loopback collapses bias to **2.65 ps**; Monte Carlo extended run 011 locks dense consensus at **22.13 ps**.
- ✅ Executive summary PDF (`driftlock_choir_sim/outputs/figs/executive_summary.pdf`) renders the latest metrics on every run.
- ✅ Seeded regression (`pytest tests/test_consensus.py::test_dense_kf_vs_baseline -q`) guards the dense preset and sweep artifacts.
- ✅ Acceptance movie (`driftlock_choir_sim/sims/make_movie.py`) now renders annotated overlays with deterministic jitter via `driftlock_choir_sim/configs/demo_movie.yaml`.
- ✅ Investor reel minted at `driftlock_choir_sim/outputs/movies/demo_choir_sim.mp4` (60 s annotated seed 2025; overlays cover Δf SNR + CRLB guidance).
- ✅ Baseline GNSS/PTP comparison clip minted at `driftlock_choir_sim/outputs/movies/baseline_choir_sim.mp4` using `driftlock_choir_sim/configs/baseline_movie.yaml`.

## Immediate priorities
1. **Finalize executive summary layout**
   - Polish typography in `executive_summary_pdf` (spacing, section headers) if we want investor-ready printouts.
   - Add space for aperture/coherent figures if we decide to inline thumbnails.

2. **Acceptance + baseline movies**
   - Use `driftlock_choir_sim/sims/make_movie.py --config driftlock_choir_sim/configs/demo_movie.yaml` for the investor reel and `--config driftlock_choir_sim/configs/baseline_movie.yaml` for the GNSS/PTP comparison (both default to 60 s @ 24 fps under `driftlock_choir_sim/outputs/movies/`).
   - Expect ~20–25 minutes for the acceptance cut, ~35 minutes for the baseline reel on M2 Max; lower `movie.seconds` when iterating.
   - Tweak `movie.header_lines` / `info_box` fields in YAML if we want alternative talking points in the overlays.

3. **Website + README refresh**
   - Mirror the latest metrics (22.13 ps consensus, 45 ps acceptance, 3.7 s runtime) in `index.html` and `README.md`.
   - Link directly to `driftlock_choir_sim/outputs/csv/acceptance_summary.json`, the PDF, and `results/mc_runs/extended_011/` artifacts.

4. **Artifact packaging**
   - Add `results/mc_runs/extended_011/SUMMARY.md` and `acceptance_summary.json` to investor-facing bundles.
   - Consider a `docs/acceptance_changelog.md` summarizing key deltas per extended run.

## Stretch goals (after the above)
- Consensus curve visual: plug `src/alg/consensus.py` outputs into a small plotting script for dense and small networks.
- Choir health ROC: mute or detune a tone, sweep SNR, and tabulate detection probability for both receiver paths.
- Compressibility probe: gzip/LZMA short baseband windows and correlate compressed size with injected anomalies (Entruptor tie-in).
- Hardware roadmap: keep `docs/hw_demo_plan.md` aligned with any new acceptance thresholds (runtime, SNR margins).

## Verification checklist
- `PYTHONPATH=. python driftlock_choir_sim/sims/run_acceptance.py`
  - Aperture Δf spike ≥ 15 dB; 2Δf logged.
  - Coherent RMSE ≤ 120 ps with ratio ≤ 1.5; 100% CI coverage and unwrap sanity.
  - Robustness τ̂ finite at 0 dB SNR.
  - Payload RMSE delta ≤ 25%; observed BER < 1e-3.
  - Total runtime < 60 s; artifacts saved under `driftlock_choir_sim/outputs/`.
- `scripts/verify_kf_sweep.py` guardrail before updating docs.
- `pytest -q` to ensure seeded regression remains green.

## File map
- Acceptance harness: `driftlock_choir_sim/sims/run_acceptance.py`
- Movie generation: `driftlock_choir_sim/sims/make_movie.py`
- Consensus + KF sweeps: `scripts/` + `results/kf_sweeps/`
- Website assets: `index.html`, `docs/`, `README.md`, `driftlock_choir_sim/outputs/`

With the harness now green and fast, the focus shifts to storytelling: polish artifacts, sync the narrative across docs and the site, and package everything for investor demos and PR.
