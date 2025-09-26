# Spectrum Beacons with Vowel-Coded Preambles

This note sketches a spectrum-coordination application for the Aperture × Formant stack. Instead of targeting nanosecond timing, nodes advertise spectrum availability by transmitting short, vowel-coded beacons. Pathfinder’s missing-fundamental analysis reconstructs the label even when the literal fundamental carrier is suppressed, letting receivers confirm "ownership" of a spectrum lane without lighting up the exact beat frequency. Multipath resilience becomes a feature: if a neighbour can recover the same vowel and missing fundamental through heavy reflections, both parties infer a consistent channel view.

## Beacon Architecture

1. **Beacon Synthesis** – Use `synthesize_formant_preamble` with the fundamental omitted (missing-fundamental mode). Each vowel maps to a specific occupancy message (e.g., `A = anchor slot`, `E = short-term lease`, `I = telemetry burst`).
2. **Channel Probe** – Receivers capture short segments (1–2k samples) and feed them to `analyze_missing_fundamental`.
3. **Consistency Checks** – Beyond the decoded label, receivers inspect the reconstructed missing fundamental and dominant harmonic. Stable triplets across neighbours indicate a truthful beacon; mismatch or high scores suggest either interference or malicious use.
4. **Guard Metrics** – The same guard/pre-guard sweep tooling logs first-path bias and negative hits, which now translate into "clutter scores" for the spectrum lane. Nodes that see 25–30% negative hits can flag the channel as echo-heavy, steering wideband traffic elsewhere.

## Simulation Tooling (`scripts/run_spectrum_beacon_sim.py`)

The new simulator exercises this workflow without running the full handshake. It draws random SNRs, multipath families, and optionally "empty" slots (no beacon) to measure:

- **Label recovery** – fraction of trials where the decoded vowel matches the transmitted profile.
- **Missing fundamental accuracy** – statistics of the reconstructed fundamental for successful detections (and for false positives).
- **Detection heuristics** – per-trial scores and an optional `--score-threshold` so teams can tune for high precision vs. high recall.

Example command:

```bash
python scripts/run_spectrum_beacon_sim.py \
  --profiles A E I O U \
  --num-trials 512 \
  --snr-db 15 35 \
  --max-extra-paths 4 \
  --max-delay-ns 120 \
  --empty-prob 0.3 \
  --phase-jitter 0.25 \
  --score-threshold 3.5e12 \
  --output results/project_aperture_formant/URBAN_CANYON/20250925T211700Z_spectrum_beacon.json
```

## Initial Findings (2025-09-25)

Artifact (`URBAN_CANYON`, tolerance checks enabled): `results/project_aperture_formant/URBAN_CANYON/20250925T212300Z_spectrum_beacon_tol.json`

- With SNR ∈ [15, 35] dB, 4-tap random multipath, and 30% empty trials, the enhanced detector (score + missing-f₀ + dominant checks) recovers ≈67% of beacons with zero false positives.
- Empty slots still cluster near the canonical 3.98×10¹² score but fail the harmonic checks, so thresholds now act purely on recall.
- Guard sweeps continue to show ~25–33% negative first-path hits, reinforcing the need for cooperative voting or history windows.

### Score Threshold Sweep

Per-trial dumps (`*.trials.jsonl`) enable ROC-style analysis. Thresholds below ~3.2×10¹² eliminate false positives; once the cutoff crosses the empty-slot score (≈3.98×10¹²) the false-positive rate jumps to 1.0.

| threshold | detection | label_accuracy | false_positive | precision | recall |
| --- | ---: | ---: | ---: | ---: | ---: |
| none | 0.670 | 0.670 | 0.000 | 1.000 | 0.670 |
| 1.00e12 | 0.009 | 0.009 | 0.000 | 1.000 | 0.009 |
| 2.00e12 | 0.014 | 0.014 | 0.000 | 1.000 | 0.014 |
| 3.00e12 | 0.094 | 0.094 | 0.000 | 1.000 | 0.094 |
| 3.20e12 | 0.153 | 0.153 | 0.000 | 1.000 | 0.153 |
| 3.40e12 | 0.179 | 0.179 | 0.000 | 1.000 | 0.179 |
| 3.60e12 | 0.270 | 0.270 | 0.000 | 1.000 | 0.270 |
| 3.80e12 | 0.335 | 0.335 | 0.000 | 1.000 | 0.335 |
| 4.00e12 | 0.352 | 0.352 | 0.000 | 1.000 | 0.352 |

### Multi-Profile Envelope

- **IDEAL** (`results/project_aperture_formant/IDEAL/20250925T212400Z_spectrum_beacon_tol.json`): 82% detection with zero false positives; thresholds ≥3.4×10¹² still recover 21–42% of beacons cleanly.
- **INDOOR_OFFICE** (`results/project_aperture_formant/INDOOR_OFFICE/20250925T212500Z_spectrum_beacon_tol.json`): 61% detection with zero false positives despite six-path multipath; a 3.8×10¹² cutoff preserves ~32% recall with perfect precision.

### Multi-Receiver Voting

Use `scripts/aggregate_beacon_votes.py` to combine `--dump-trials` outputs across receivers. In URBAN_CANYON, fusing two independent seeds yields:

- `--vote-threshold 1` (any receiver can claim): 77% detection, 0 false positives.
- `--vote-threshold 2` (both must agree on the label): 6% detection, 0 false positives—precision stays perfect but recall collapses, indicating receivers often disagree in heavy multipath.

Artifacts: `results/project_aperture_formant/URBAN_CANYON/20250925T212700Z_spectrum_beacon_vote{1,2}.json`.

## Next Directions

1. **Consensus on Beacons** – Fuse multiple receivers’ scores to downweight solitary detections, preventing a single noisy node from reserving the band.
2. **Metadata Packing** – Encode extra bits via vowel pairs (e.g., `AI` within a superframe) or by modulating the missing-fundamental offset (±500 Hz) to flag slot durations.
3. **Guard-Aware Scheduling** – Feed the guard sweep statistics into a spectrum map: high negative-hit ratios flag reflective environments, prompting radios to choose narrower or more redundant signalling.

This line of work reframes the missing-fundamental path not just as a coarse timing aide, but as a robust metadata tag for cooperative spectrum etiquette.
