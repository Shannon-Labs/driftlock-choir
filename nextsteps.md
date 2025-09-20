# Driftlock – Next Steps for Choir Acceptance (handoff)

This brief captures the post-breakthrough state of the Choir acceptance harness so the next engineer (AI or human) can keep momentum and polish the deliverables for investors, partners, and the website.

## Context (what we’re shipping)
- Driftlock Choir simulates a multi-carrier "choir/organ" transmitter with two receiver paths:
  - **Coherent path**: per-tone phasors → phase unwrap vs. frequency → weighted least squares → τ̂ with confidence intervals.
  - **Aperture path**: envelope/cepstrum detects the missing fundamental Δf (and harmonic peaks) even when that tone is absent.
- The acceptance harness lives in `driftlock_sim/sims/run_acceptance.py` and now produces JSON + PNG + PDF artifacts in **3.7 seconds**.

## Current status (2025-10-XX)
- ✅ All acceptance checks pass with large margins: 45.0 ps RMSE, RMSE/CRLB 0.83, Δf spike 58 dB, payload BER 0.
- ✅ Reciprocity loopback collapses bias to **2.65 ps**; Monte Carlo extended run 011 locks dense consensus at **22.13 ps**.
- ✅ Executive summary PDF (`driftlock_sim/outputs/figs/executive_summary.pdf`) renders the latest metrics on every run.
- ✅ Seeded regression (`pytest tests/test_consensus.py::test_dense_kf_vs_baseline -q`) guards the dense preset and sweep artifacts.
- ⏳ Short acceptance movie (`driftlock_sim/sims/make_movie.py`) needs updated overlays + deterministic config for the website.
- ⏳ Website and README copy still reference older nanosecond-era results; syncing content is in progress.

## Immediate priorities
1. **Finalize executive summary layout**
   - Polish typography in `executive_summary_pdf` (spacing, section headers) if we want investor-ready printouts.
   - Add space for aperture/coherent figures if we decide to inline thumbnails.

2. **Deterministic acceptance movie**
   - Update `driftlock_sim/sims/make_movie.py` + config to use the new acceptance signals (missing fundamental + payload).
   - Export 60–90 s MP4 with overlays: left RF comb, right-top envelope spectrum (Δf marker), right-bottom τ̂ strip chart.
   - Drop output to `driftlock_sim/outputs/movies/driftlock_choir_sim.mp4` for the website.

3. **Website + README refresh**
   - Mirror the latest metrics (22.13 ps consensus, 45 ps acceptance, 3.7 s runtime) in `index.html` and `README.md`.
   - Link directly to `driftlock_sim/outputs/csv/acceptance_summary.json`, the PDF, and `results/mc_runs/extended_011/` artifacts.

4. **Artifact packaging**
   - Add `results/mc_runs/extended_011/SUMMARY.md` and `acceptance_summary.json` to investor-facing bundles.
   - Consider a `docs/acceptance_changelog.md` summarizing key deltas per extended run.

## Stretch goals (after the above)
- Consensus curve visual: plug `src/alg/consensus.py` outputs into a small plotting script for dense and small networks.
- Choir health ROC: mute or detune a tone, sweep SNR, and tabulate detection probability for both receiver paths.
- Compressibility probe: gzip/LZMA short baseband windows and correlate compressed size with injected anomalies (Entruptor tie-in).
- Hardware roadmap: keep `docs/hw_demo_plan.md` aligned with any new acceptance thresholds (runtime, SNR margins).

## Verification checklist
- `PYTHONPATH=. python driftlock_sim/sims/run_acceptance.py`
  - Aperture Δf spike ≥ 15 dB; 2Δf logged.
  - Coherent RMSE ≤ 120 ps with ratio ≤ 1.5; 100% CI coverage and unwrap sanity.
  - Robustness τ̂ finite at 0 dB SNR.
  - Payload RMSE delta ≤ 25%; observed BER < 1e-3.
  - Total runtime < 60 s; artifacts saved under `driftlock_sim/outputs/`.
- `scripts/verify_kf_sweep.py` guardrail before updating docs.
- `pytest -q` to ensure seeded regression remains green.

## File map
- Acceptance harness: `driftlock_sim/sims/run_acceptance.py`
- Movie generation: `driftlock_sim/sims/make_movie.py`
- Consensus + KF sweeps: `scripts/` + `results/kf_sweeps/`
- Website assets: `index.html`, `docs/`, `README.md`, `driftlock_sim/outputs/`

With the harness now green and fast, the focus shifts to storytelling: polish artifacts, sync the narrative across docs and the site, and package everything for investor demos and PR.
