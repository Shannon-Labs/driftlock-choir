# Coax Bench Emulation

`sim/bench_coax.py` provides a quick stand-in for the planned coax validation: it evaluates 2–7 node complete graphs with intentional Δf sweeps, reports per-link mean/σ, Allan deviation, and reciprocity bias, and optionally dumps the stats to JSON for later analysis.

## Usage

```bash
python sim/bench_coax.py \
  --nodes 4 \
  --delta-f-min 5e5 --delta-f-max 2e6 \
  --observation-ms 100 \
  --trials 40 \
  --snr-db 35 \
  --output results/bench/coax_4node.json
```

Example console output:

```
Nodes: 3 | Trials per link: 10
Δf sweep: 0.50–2.00 MHz | Observation window: 50.0 ms
pair       μ_ps    σ_ps    allan_ps  bias_ps
(0, 1)       0.01   0.02     0.02    0.00
(0, 2)       0.00   0.02     0.02    0.00
(1, 2)       0.01   0.01     0.01    0.00
```

- **μ_ps / σ_ps**: sample mean and standard deviation (ps) across the Monte Carlo trials.
- **allan_ps**: two-sample Allan deviation of the residual sequence, useful for spotting drift.
- **bias_ps**: mean reciprocity residual when swapping the A/B chains (should stay near zero on coax).

### Notes

- The handshake profile caps the beat duration at 1 ms for runtime; use the `--observation-ms` flag to scale Monte Carlo trials instead.
- JSON summaries live under `results/` by default, which is ignored—move curated artifacts into `docs/` if you want to check them in.
- For hardware bring-up, start with 50 ms windows at 0.5–2 MHz offsets; bump SNR down once the lab path loss data is available.
