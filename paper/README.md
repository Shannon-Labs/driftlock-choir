Driftlock Choir — arXiv Paper Draft
===================================

This folder contains an arXiv-ready LaTeX manuscript describing Chronometric Interferometry and the Driftlock Choir consensus system, distilled from the patent documents under `patent/` and the simulation stack in `src/` + `sim/`.

Quick start

- Compile: `pdflatex main && bibtex main && pdflatex main && pdflatex main`
- Or with latexmk: `latexmk -pdf main.tex`

Figure sources

- Reuses existing repo artifacts to keep the paper self-consistent:
  - `patent/figures/fig3_beat_phase_extraction.png`
  - `patent/figures/fig4_convergence.png`
  - `results/phase2/phase2_topology.png`
  - `results/phase2/phase2_residuals.png`
  - `results/phase1/alias_map/tau_rmse_ps_retune_1000000p0.png`

Reproducibility

- See Section "Reproducibility" in the paper for exact commands, matching `AGENTS.md` guidance.

Notes

- Bibliography entries are included as stable placeholders. Replace `TBD` fields with DOIs if desired prior to submission.

