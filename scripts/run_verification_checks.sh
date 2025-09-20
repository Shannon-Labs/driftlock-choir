#!/usr/bin/env bash
set -euo pipefail

# Run the dense preset regression test and sweep verifier to ensure the
# promoted gains remain valid before committing or in CI.

pytest tests/test_consensus.py::test_dense_kf_vs_baseline -q

scripts/verify_kf_sweep.py \
  results/kf_sweeps/dense_combo_scan/kf_sweep_summary.json \
  --expected-min 20.9337 \
  --expected-best-mean 21.89 \
  --expected-clock 0.32 \
  --expected-freq 0.03 \
  --expected-iterations 1
