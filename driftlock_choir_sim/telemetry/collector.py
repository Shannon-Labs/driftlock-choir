from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any

@dataclass
class TelemetryFrame:
    """Frame-level telemetry for acceptance runs."""
    frame_idx: int
    tau_hat_ps: float
    ci_ps: float
    crlb_ps: float
    df_snr_db: float
    rmse_ps: float
    truth_tau_ps: float
    timestamp: float

class TelemetryCollector:
    """Collector for frame-level stats during simulation."""
    def __init__(self, output_path: Path):
        self.output_path = output_path
        self.frames: list[TelemetryFrame] = []
        self.start_time = Path(output_path).parent.stat().st_mtime if output_path.exists() else 0.0

    def add_frame(self, frame_idx: int, tau_hat_ps: float, ci_ps: float, crlb_ps: float, df_snr_db: float, rmse_ps: float, truth_tau_ps: float):
        """Add a frame's telemetry."""
        self.frames.append(TelemetryFrame(
            frame_idx=frame_idx,
            tau_hat_ps=tau_hat_ps,
            ci_ps=ci_ps,
            crlb_ps=crlb_ps,
            df_snr_db=df_snr_db,
            rmse_ps=rmse_ps,
            truth_tau_ps=truth_tau_ps,
            timestamp=Path(self.output_path).stat().st_mtime if Path(self.output_path).exists() else self.start_time
        ))

    def save(self):
        """Save telemetry to JSONL."""
        with open(self.output_path, 'w') as f:
            for frame in self.frames:
                f.write(json.dumps({
                    'frame_idx': frame.frame_idx,
                    'tau_hat_ps': frame.tau_hat_ps,
                    'ci_ps': frame.ci_ps,
                    'crlb_ps': frame.crlb_ps,
                    'df_snr_db': frame.df_snr_db,
                    'rmse_ps': frame.rmse_ps,
                    'truth_tau_ps': frame.truth_tau_ps,
                    'timestamp': frame.timestamp
                }) + '\n')

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self.output_path.exists():
            self.save()