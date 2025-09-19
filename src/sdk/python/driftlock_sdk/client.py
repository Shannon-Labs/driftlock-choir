from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import json
import time
import os


@dataclass
class SyncConfig:
    node_id: int
    neighbors: Tuple[int, ...]
    delta_f_hz: float = 1_000.0
    schedule_us: Tuple[float, ...] = (0.0,)


@dataclass
class Measurement:
    neighbor_id: int
    beat_freq_hz: float
    phase_intercept: float
    variance_phase: float
    timestamp_s: float


@dataclass
class Quality:
    rms_resid_rad: float
    snr_db: Optional[float] = None


class DriftlockClient:
    def __init__(self, config: SyncConfig):
        self.config = config
        self.state_bias_s: float = 0.0
        self.state_df_hz: float = 0.0
        self._last_quality = Quality(rms_resid_rad=0.0)
        self._telemetry_dir = os.environ.get("DRIFTLOCK_TEL_DIR", "results/time_telemetry")
        os.makedirs(self._telemetry_dir, exist_ok=True)

    def start_sync(self) -> None:
        # Placeholder for scheduler initialization
        self._log({"event": "start_sync", "config": self.config.__dict__})

    def ingest_measurement(self, m: Measurement) -> None:
        # Extremely simple placeholder: use phase variance to weight
        w = 1.0 / max(m.variance_phase, 1e-9)
        # Small update toward neighbor differential (toy)
        self.state_bias_s += 1e-12 * w * (m.phase_intercept / (2.0 * 3.141592653589793 * 915e6))
        self.state_df_hz += 1e-3 * w * (m.beat_freq_hz - self.config.delta_f_hz)
        self._last_quality = Quality(rms_resid_rad=max(0.0, (1.0 / (w ** 0.5))))
        self._log({
            "event": "ingest",
            "neighbor": m.neighbor_id,
            "beat": m.beat_freq_hz,
            "phase_intercept": m.phase_intercept,
            "variance_phase": m.variance_phase,
        })

    def run_consensus_step(self) -> None:
        # Placeholder: in real use, apply variance-weighted Laplacian update
        self._log({"event": "consensus_step", "bias_s": self.state_bias_s, "df_hz": self.state_df_hz})

    def get_clock_bias(self) -> float:
        return self.state_bias_s

    def get_quality(self) -> Quality:
        return self._last_quality

    def _log(self, payload: Dict) -> None:
        payload["ts"] = time.time()
        path = os.path.join(self._telemetry_dir, f"telemetry_{int(payload['ts'])}.json")
        try:
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(payload) + "\n")
        except Exception:
            pass

