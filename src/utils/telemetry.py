from __future__ import annotations
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple, Union
from dataclasses import dataclass, asdict, field
import numpy as np
from typing_extensions import Self

@dataclass
class TelemetryMetadata:
    """Metadata for telemetry records."""
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    seed: Optional[int] = None
    config: Optional[Dict[str, Any]] = None
    run_id: Optional[str] = None
    variant: Optional[str] = None  # e.g., 'baseline', 'driftlock choir'

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class TelemetryRecord:
    """Base telemetry record with metadata and data."""
    metadata: TelemetryMetadata = field(default_factory=TelemetryMetadata)
    data: Dict[str, Any] = field(default_factory=dict)
    baseline_data: Optional[Dict[str, Any]] = None  # For comparative fields

    def to_dict(self) -> Dict[str, Any]:
        result = {"metadata": self.metadata.to_dict(), "data": self.data}
        if self.baseline_data:
            result["baseline_data"] = self.baseline_data
            # Compute comparative deltas
            result["comparative"] = {}
            for key in self.data:
                if key in self.baseline_data:
                    delta = self.data[key] - self.baseline_data[key]
                    result["comparative"][f"{key}_delta"] = delta
        return result

    def add_baseline(self, baseline: Dict[str, Any]) -> Self:
        self.baseline_data = baseline
        return self

class TelemetryExporter:
    """Unified exporter for telemetry data supporting JSONL, CSV, batch aggregation, and streaming."""

    def __init__(self, output_dir: Union[str, Path], variant: str = "driftlock choir"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.variant = variant
        self.records: List[TelemetryRecord] = []
        self.jsonl_path: Optional[Path] = None
        self.csv_path: Optional[Path] = None
        self._csv_writer = None
        self._fieldnames: List[str] = []

    def add_record(self, data: Dict[str, Any], metadata: Optional[Dict[str, Any]] = None,
                   baseline_data: Optional[Dict[str, Any]] = None, seed: Optional[int] = None) -> None:
        """Add a single telemetry record."""
        meta_dict = metadata or {}
        if seed is not None:
            meta_dict["seed"] = seed
        meta_dict["variant"] = self.variant
        record = TelemetryRecord(
            metadata=TelemetryMetadata(**meta_dict),
            data=data,
            baseline_data=baseline_data
        )
        self.records.append(record)
        self._stream_record(record)

    def _stream_record(self, record: TelemetryRecord) -> None:
        """Stream a single record to JSONL and CSV in real-time."""
        record_dict = record.to_dict()
        # JSONL streaming
        if self.jsonl_path is None:
            self.jsonl_path = self.output_dir / f"telemetry_{self.variant}.jsonl"
        with open(self.jsonl_path, "a") as f:
            f.write(json.dumps(record_dict) + "\n")

        flat_row = self._flatten_record(record_dict)

        # CSV streaming (flatten for CSV)
        if self.csv_path is None:
            self.csv_path = self.output_dir / f"telemetry_{self.variant}.csv"
            self._initialize_csv(flat_row)
        else:
            self._write_csv_row(flat_row)

    def _initialize_csv(self, flat_row: Dict[str, Any]) -> None:
        """Initialize CSV with fieldnames and first row."""
        self._fieldnames = list(flat_row.keys())
        with open(self.csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self._fieldnames)
            writer.writeheader()
            writer.writerow(flat_row)

    def _flatten_record(self, data: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
        """Flatten nested dictionaries for CSV output."""
        flat: Dict[str, Any] = {}
        for key, value in data.items():
            name = f"{prefix}{key}" if not prefix else f"{prefix}_{key}"
            if isinstance(value, dict):
                flat.update(self._flatten_record(value, name))
            elif isinstance(value, list):
                flat[name] = json.dumps(value)
            else:
                flat[name] = value
        return flat

    def _write_csv_row(self, flat_row: Dict[str, Any]) -> None:
        """Write flattened row to CSV, expanding headers as needed."""
        new_keys = [k for k in flat_row.keys() if k not in self._fieldnames]
        if new_keys:
            self._fieldnames.extend(new_keys)
            self._rewrite_csv()
            return
        with open(self.csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self._fieldnames)
            writer.writerow({key: flat_row.get(key, "") for key in self._fieldnames})

    def _rewrite_csv(self) -> None:
        if not self.csv_path:
            return
        with open(self.csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self._fieldnames)
            writer.writeheader()
            for record in self.records:
                flat = self._flatten_record(record.to_dict())
                writer.writerow({key: flat.get(key, "") for key in self._fieldnames})

    def batch_aggregate(self, keys: List[str]) -> Dict[str, Any]:
        """Aggregate batch statistics (mean, std, min, max) for specified keys."""
        if not self.records:
            return {}
        agg_data = {}
        values = {key: [] for key in keys}
        for record in self.records:
            for key in keys:
                if key in record.data:
                    values[key].append(record.data[key])
        for key in keys:
            vals = np.array(values[key])
            if len(vals) > 0:
                agg_data[f"{key}_mean"] = float(np.mean(vals))
                agg_data[f"{key}_std"] = float(np.std(vals))
                agg_data[f"{key}_min"] = float(np.min(vals))
                agg_data[f"{key}_max"] = float(np.max(vals))
        # Add aggregated record
        agg_record = TelemetryRecord(data=agg_data, metadata=self.records[0].metadata)
        self.add_record(agg_record.data, metadata=agg_record.metadata.to_dict())
        return agg_data

    def export_batch(self, filename: Optional[str] = None) -> Tuple[Path, Path]:
        """Export all collected records to JSONL and CSV (non-streaming)."""
        if not self.records:
            raise ValueError("No records to export.")
        if filename:
            jsonl_name = f"{filename}.jsonl"
            csv_name = f"{filename}.csv"
        else:
            jsonl_name = f"batch_telemetry_{self.variant}.jsonl"
            csv_name = f"batch_telemetry_{self.variant}.csv"
        jsonl_path = self.output_dir / jsonl_name
        csv_path = self.output_dir / csv_name

        # JSONL
        with open(jsonl_path, "w") as f:
            for record in self.records:
                f.write(json.dumps(record.to_dict()) + "\n")

        # CSV
        fieldnames: List[str] = []
        flat_records: List[Dict[str, Any]] = []
        for record in self.records:
            flat = self._flatten_record(record.to_dict())
            flat_records.append(flat)
            for key in flat.keys():
                if key not in fieldnames:
                    fieldnames.append(key)
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for flat in flat_records:
                writer.writerow({key: flat.get(key, "") for key in fieldnames})

        return jsonl_path, csv_path

    def stream_generator(self) -> Generator[Dict[str, Any], None, None]:
        """Generator for streaming records (for real-time processing)."""
        for record in self.records:
            yield record.to_dict()

    def clear(self) -> None:
        """Clear collected records."""
        self.records.clear()
        self.jsonl_path = None
        self.csv_path = None
        self._csv_writer = None
        self._fieldnames = []

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self.records:
            self.export_batch()