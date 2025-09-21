# Driftlock Choir — Full‑Spectrum RF Simulation

A reproducible simulation lab for a multi‑carrier "choir" transmitter and
an "aperture" receiver that reconstructs the missing fundamental and
estimates time‑of‑flight via phase‑slope group delay. Includes coherent and
aperture paths, impairments, sweeps, figures, and an optional MP4.

Quick start:

```
# Install local deps (isolated venv recommended)
pip install -r driftlock_choir_sim/requirements.txt

# Run a single demo (writes CSV/PNGs under driftlock_choir_sim/outputs)
python driftlock_choir_sim/sims/run_single.py --config driftlock_choir_sim/configs/demo_movie.yaml

# Run a small sweep
python driftlock_choir_sim/sims/run_sweep.py --config driftlock_choir_sim/configs/sweep_small.yaml

# Make a short MP4 (requires ffmpeg)
python driftlock_choir_sim/sims/make_movie.py --config driftlock_choir_sim/configs/demo_movie.yaml
```

Outputs land under `driftlock_choir_sim/outputs/{csv,figs,movies,logs}`.

## Acceptance quick run

```bash
PYTHONPATH=. python driftlock_choir_sim/sims/run_acceptance.py
```

Writes:
- `driftlock_choir_sim/outputs/csv/acceptance_summary.json`
- `driftlock_choir_sim/outputs/figs/executive_summary.pdf`

Latest deterministic pass (M2 Max) reports: Δf SNR **58 dB**, coherent RMSE **45 ps** (RMSE/CRLB **0.83**), payload RMSE delta **+8.6%** with **0 BER**, runtime **3.7 s**.

See `THEORY.md` for the model overview and `configs/*.yaml` for parameters.

