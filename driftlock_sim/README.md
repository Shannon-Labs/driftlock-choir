# Driftlock Choir — Full‑Spectrum RF Simulation

A reproducible simulation lab for a multi‑carrier "choir" transmitter and
an "aperture" receiver that reconstructs the missing fundamental and
estimates time‑of‑flight via phase‑slope group delay. Includes coherent and
aperture paths, impairments, sweeps, figures, and an optional MP4.

Quick start:

```
# Install local deps (isolated venv recommended)
pip install -r driftlock_sim/requirements.txt

# Run a single demo (writes CSV/PNGs under driftlock_sim/outputs)
python driftlock_sim/sims/run_single.py --config driftlock_sim/configs/demo_movie.yaml

# Run a small sweep
python driftlock_sim/sims/run_sweep.py --config driftlock_sim/configs/sweep_small.yaml

# Make a short MP4 (requires ffmpeg)
python driftlock_sim/sims/make_movie.py --config driftlock_sim/configs/demo_movie.yaml
```

Outputs land under `driftlock_sim/outputs/{csv,figs,movies,logs}`.

See `THEORY.md` for the model overview and `configs/*.yaml` for parameters.

