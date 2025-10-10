# `analysis_records` Schema

The CLI and experiment runners export `analysis_records` as part of each structured JSON artifact. Each record captures a single beat-note analysis result with consistent units.

## Top-level fields

| Field | Type | Units | Description |
| --- | --- | --- | --- |
| `timestamp.time` | float | seconds | Experiment epoch for the beat-note sample. |
| `timestamp.uncertainty` | float | picoseconds | Timing uncertainty associated with the timestamp. |
| `timestamp.quality` | enum | — | MeasurementQuality enum (`excellent`, `good`, `fair`, `poor`, `invalid`). |
| `tx_frequency_hz` | float | hertz | Nominal transmit oscillator frequency. |
| `rx_frequency_hz` | float | hertz | Nominal receive oscillator frequency after mixing. |
| `beat_frequency_hz` | float | hertz | Measured beat-note frequency magnitude. |
| `tau_estimate_ps` | float | picoseconds | Estimated propagation delay τ. |
| `tau_uncertainty_ps` | float | picoseconds | One-sigma uncertainty for τ. |
| `delta_f_estimate_hz` | float | hertz | Estimated residual frequency offset Δf. |
| `delta_f_uncertainty_hz` | float | hertz | One-sigma uncertainty for Δf. |
| `snr_db` | float | dB | Estimated signal-to-noise ratio of the beat note. |
| `estimation_method` | string | — | Estimator identifier (e.g., `phase_slope_single`). |
| `quality` | enum | — | Quality tag propagated from MeasurementQuality.

## Sample record

```json
{
  "timestamp": {
    "time": 0.0,
    "uncertainty": 0.0,
    "quality": "excellent"
  },
  "tx_frequency_hz": 2400000000.0,
  "rx_frequency_hz": 2400000150.0,
  "beat_frequency_hz": 149.9998914732555,
  "tau_estimate_ps": 11.93062029925049,
  "tau_uncertainty_ps": 6.6440252724574265,
  "delta_f_estimate_hz": 0.0001085969553855648,
  "delta_f_uncertainty_hz": 0.000001726184583789089,
  "snr_db": 35.0,
  "estimation_method": "phase_slope_single",
  "quality": "good"
}
```

For a complete export, see [`artifacts/e1_cli_clean.json`](artifacts/e1_cli_clean.json).
