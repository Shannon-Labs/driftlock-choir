# Driftlock – Next Steps for Choir Acceptance (handoff)

This document briefs the next engineer (AI or human) to finish the Driftlock Choir acceptance and keep everything deterministic, fast, and reproducible.

## Context (what we’re building)
- Driftlock Choir simulates a multi‑carrier “choir/organ” transmitter and two receiver paths:
  - Coherent: per‑tone phasors → phase unwrap vs frequency → WLS slope → τ̂ with CI.
  - Aperture: envelope/cepstrum shows the “missing fundamental” at Δf (and pairwise differences) even when no explicit tone exists at Δf.
- Acceptance criteria live in `driftlock_sim/sims/run_acceptance.py` and target precision, robustness, payload coexistence, and report/movie generation.

## Current status (observed locally)
- Aperture acceptance passes (Δf spike is strong).
- Coherent precision still fails (RMSE ≫ CRLB) and acceptance runtime exceeds 60s in a clean checkout.
- Payload coexistence fails (timing RMSE degrades >25% and BER is not observed on payload tones).
- A small bug in `executive_summary_pdf` used `res[...]` instead of `data[...]` (variable rename).

## Priority fixes (in order)
1) Executive summary variable bug
- File: `driftlock_sim/sims/run_acceptance.py`
- In `executive_summary_pdf(data)`, replace any `res['...']` with `data['...']`.

2) Keep acceptance < 60 seconds
- Precision block: fs=20e6, Δf=1e6, M=21 (±1..±10, missing‑fundamental), duration ≈ 0.008–0.010 s, trials=12–16.
- Payload block: duration ≈ 0.008–0.010 s, trials=8–12.
- Avoid per‑trial plotting; generate figures once per check at the end.
- Ensure coherent path is vectorized (it is) and avoid large N FFTs in loops.

3) Inject truth delay at the signal level (before noise)
- File: `driftlock_sim/sims/run_acceptance.py`
- Use `impose_fractional_delay_fft(x, fs, truth_tau)` from `driftlock_sim/dsp/time_delay.py` immediately after generating the comb and before applying channel/impairments.
- Use complex FFT (already implemented) with Hann and de‑weight.

4) Weighted LS with per‑tone SNR
- File: `driftlock_sim/dsp/rx_coherent.py`
- Implement a simple per‑tone SNR estimator:
  - Option A: ring/noise estimate around each tone; SNR_k ≈ |Y_k|^2 / σ_noise^2.
  - Option B (OK for acceptance): robust magnitude stats (median/mad) per tone neighborhood.
- Use `w_k = max(SNR_k, 1e-6)` in `wls_delay` for an efficient estimator.

5) CRLB with RMS bandwidth
- Files: `driftlock_sim/dsp/metrics.py`, `driftlock_sim/dsp/crlb.py`
- Compute Brms = sqrt(Σ w_k (f_k − f̄)^2 / Σ w_k) with w_k ≈ a_k^2 (or SNR_k if available).
- Feed Brms into the CRLB function; report CRLB in ps and the ratio RMSE/CRLB in acceptance output.

6) Observed BER and payload coexistence
- Files: `driftlock_sim/dsp/tx_comb.py`, `driftlock_sim/sims/run_acceptance.py`
- Reserve 1/8 carriers for QPSK payload; keep pilots fixed.
- Expose payload bits per payload tone from TX; add a demod (hard decision) on those tones to compute observed BER.
- Tune duration/trials so that at SNR≥20 dB, BER < 1e‑3 and timing RMSE degrades < 25% vs. no‑payload baseline.

## Verification (must meet)
- Run: `PYTHONPATH=. python driftlock_sim/sims/run_acceptance.py`
  - Aperture: Δf spike ≥ 15 dB (log 2Δf too).
  - Coherent: RMSE(τ̂) ≤ 120 ps; RMSE/CRLB ≤ 1.5 (target ≤ 1.3) at SNR=30 dB.
  - Robustness: Δf detectable at 0 dB; τ̂ finite.
  - Payload: ΔRMSE ≤ 25%; observed BER < 1e‑3 at 20 dB.
  - Total runtime ≤ 60 s.
- Outputs present:
  - `driftlock_sim/outputs/csv/acceptance_summary.json`
  - `driftlock_sim/outputs/figs/executive_summary.pdf`
  - (When run) `driftlock_sim/outputs/movies/driftlock_choir_sim.mp4`

## Stretch goals (after green checks)
- Two‑way bias cancellation and N‑node variance‑weighted consensus curve (reuse `src/alg/consensus.py`).
- Choir health ROC: mute/detune one tone across SNR/M and sweep a “health index” threshold.
- Compressibility probe: gzip/lzma short baseband windows and correlate size with anomalies (bridge to Entruptor narrative).
- Movie polish: 60–90 s MP4 with overlays: left RF comb, right‑top envelope Δf, right‑bottom τ̂ strip chart vs truth.

## File map (where to edit)
- Coherent path: `driftlock_sim/dsp/rx_coherent.py` (SNR_k, WLS)
- CRLB & Brms: `driftlock_sim/dsp/metrics.py`, `driftlock_sim/dsp/crlb.py`
- Truth delay: `driftlock_sim/dsp/time_delay.py` (already present), used in `run_acceptance.py`
- Acceptance harness: `driftlock_sim/sims/run_acceptance.py`
- TX comb/payload: `driftlock_sim/dsp/tx_comb.py`
- Movie: `driftlock_sim/sims/make_movie.py`

## Common pitfalls
- Aliasing: keep all tones within Nyquist; use symmetric indices in missing‑fundamental mode.
- Leakage: ensure estimator evaluates exactly at fk; otherwise window or correlate.
- Weighting: unweighted LS increases RMSE; use SNR_k.
- Time budget: plotting inside trial loops will blow the 60 s target.

## Quick commands
- Single demo: `PYTHONPATH=. python driftlock_sim/sims/run_single.py --config driftlock_sim/configs/demo_movie.yaml`
- Sweep: `PYTHONPATH=. python driftlock_sim/sims/run_sweep.py --config driftlock_sim/configs/sweep_small.yaml`
- Acceptance: `PYTHONPATH=. python driftlock_sim/sims/run_acceptance.py`
- Movie: `PYTHONPATH=. python driftlock_sim/sims/make_movie.py --config driftlock_sim/configs/demo_movie.yaml`

## Definition of Done
- Acceptance green, runtime ≤ 60 s, artifacts written.
- Executive summary PDF reports Δf SNR, τ̂ RMSE, CRLB (Brms), RMSE/CRLB, payload BER, and pass/fail for each check.
- Code remains deterministic (seeded RNG) and vectorized; no changes to existing `src/` code or root tests.

Good luck — once green, we’ll wire the MP4 and figures into the website’s “Choir Simulation Lab” section and reference results in the Speedrun deck.
