# Scaling Benchmarks & Repo Hygiene

_Last updated: September 2025_

## Large-Network Sweep Summary

We reran the variance-weighted consensus sweep with the tuned local-KF preset (clock gain 0.32, freq gain 0.03, single iteration) across progressively larger random geometric graphs. Each run used `--density 0.22` and a single seed so the runtime stayed manageable while showcasing scaling behaviour.

| Nodes | Runtime | Timing RMSE (ps) | Baseline Δ (ps) | Command |
| ---: | ---: | ---: | ---: | --- |
| 64 | 38 s | 22.13 | −0.33 | `python scripts/sweep_phase2_kf.py --nodes 64 --density 0.22 --gains 0.32 --freq-gains 0.03 --iters 1 --seeds 4040 --write-json --output-dir results/kf_sweeps/dense_combo_scan --run-id quick64` |
| 128 | 51 s | 22.97 | −4.50 | `python scripts/sweep_phase2_kf.py --nodes 128 --density 0.22 --gains 0.32 --freq-gains 0.03 --iters 1 --seeds 4040 --write-json --output-dir results/kf_sweeps/scaling_tests --run-id n128` |
| 256 | 2.7 min | 21.64 | −4.54 | `python scripts/sweep_phase2_kf.py --nodes 256 --density 0.22 --gains 0.32 --freq-gains 0.03 --iters 1 --seeds 4040 --write-json --output-dir results/kf_sweeps/scaling_tests --run-id n256` |
| 512 | 10.5 min | 20.09 | −5.29 | `python scripts/sweep_phase2_kf.py --nodes 512 --density 0.22 --gains 0.32 --freq-gains 0.03 --iters 1 --seeds 4040 --write-json --output-dir results/kf_sweeps/scaling_tests --run-id n512` |

The RMSE steadily drops as the network grows; the 512-node case lands at 20.09 ps despite the consensus still converging in a single iteration.

## Regression Guardrails

- `tests/test_consensus.py::test_dense_kf_vs_baseline` keeps the canonical 64-node dense scenario pinned ≥1 ps better than the no-KF baseline.
- `tests/test_consensus.py::test_large_network_kf_vs_baseline` now exercises a 128-node graph with a fast-envelope handshake profile, ensuring the tuned preset still wins by ≥0.5 ps without blowing up CI runtime (~11 s locally).
- `Phase2Config` exposes optional handshake overrides (`handshake_beat_duration_s`, `handshake_baseband_rate_factor`, etc.) so future tests can downscale waveform generation without touching production defaults.

## Repository Hygiene

- `.gitignore` blocks private/support artifacts (`.claude/`, `a16z*/`, `investor*/`, `*_STRATEGY.*`, and the entire `results/` tree) to prevent accidental disclosure of prompts, decks, or generated plots.
- Confirmed that historical prompt and patent-strategy drafts (`CLAUDE.md`, `GEMINI.md`, `PATENT_ATTORNEY_SUMMARY.md`, etc.) have been removed from `main`; only the provisionals and supporting public docs remain under `patent/`.
- Scaling JSON/PNG outputs stay under `results/` and are ignored by default. Use `--write-json` when you need reproducibility, then move curated summaries into `docs/` before committing.

> Tip: run `git status -sb` before committing—any appearance of `private/`, `.claude/`, or `results/` means something slipped past the hygiene rules.
