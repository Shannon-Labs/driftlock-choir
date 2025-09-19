#!/usr/bin/env python3
"""
Minimal orchestrator example driving the DriftlockClient.

Simulates a neighbor measurement every 100 ms and logs telemetry.
Replace the synthetic generator with real beat-phase stats from your radio.
"""
import time
import math
from driftlock_sdk.client import DriftlockClient, SyncConfig, Measurement


def synthetic_measurements(k: int):
    # Simulate a slow bias drift and a beat frequency near config.delta_f_hz
    phase_intercept = 0.05 * math.sin(2 * math.pi * (k % 100) / 100.0)
    beat_freq = 1000.0 + 0.5 * math.sin(2 * math.pi * (k % 200) / 200.0)
    variance = 0.01
    return beat_freq, phase_intercept, variance


def main():
    cfg = SyncConfig(node_id=1, neighbors=(2,), delta_f_hz=1000.0, schedule_us=(0.0,))
    client = DriftlockClient(cfg)
    client.start_sync()
    for k in range(100):
        bf, phi0, var = synthetic_measurements(k)
        meas = Measurement(neighbor_id=2, beat_freq_hz=bf, phase_intercept=phi0, variance_phase=var, timestamp_s=time.time())
        client.ingest_measurement(meas)
        client.run_consensus_step()
        time.sleep(0.1)
    print(f"Bias estimate (s): {client.get_clock_bias():.3e}")
    print(f"Quality: {client.get_quality()}")


if __name__ == "__main__":
    main()

