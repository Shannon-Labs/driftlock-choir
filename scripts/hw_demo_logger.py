#!/usr/bin/env python3
"""
Hardware demo logger for Driftlock SDR runs.

Reads a time series of RMSE entries (CSV or JSONL) and produces:
- summary.json: aggregate metrics
- rmse_trend.png: RMSE vs time plot
- run_manifest.json: minimal metadata (notes, run id)

JSONL schema per line:
  {"t": <unix_seconds>, "rmse_ps": <float>, "alias": true/false}

CSV columns (header required):
  t,rmse_ps,alias

Usage:
  python scripts/hw_demo_logger.py \
    --output-dir results/hw_demo_001 \
    --run-id loopback_smoke \
    --format jsonl \
    --input rmse_stream.jsonl \
    --notes "Loopback; Δf=1 MHz; SNR~20 dB"
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt


@dataclass
class Entry:
    t: float
    rmse_ps: float
    alias: Optional[bool] = None


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_jsonl(p: Path) -> List[Entry]:
    out: List[Entry] = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            out.append(Entry(float(rec["t"]), float(rec["rmse_ps"]), bool(rec.get("alias")) if "alias" in rec else None))
    return out


def read_csv(p: Path) -> List[Entry]:
    out: List[Entry] = []
    with p.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            alias_val = row.get("alias")
            alias = None if alias_val is None or alias_val == "" else (str(alias_val).strip().lower() in ("1", "true", "yes"))
            out.append(Entry(float(row["t"]), float(row["rmse_ps"]), alias))
    return out


def summarize(entries: List[Entry]) -> dict:
    if not entries:
        return {"count": 0}
    xs = [e.rmse_ps for e in entries]
    ts = [e.t for e in entries]
    n = len(xs)
    xs_sorted = sorted(xs)
    p95 = xs_sorted[int(0.95 * (n - 1))]
    alias_vals = [e.alias for e in entries if e.alias is not None]
    alias_rate = None
    if alias_vals:
        alias_rate = sum(1 for a in alias_vals if a) / len(alias_vals)
    return {
        "count": n,
        "first_ts": ts[0],
        "last_ts": ts[-1],
        "duration_s": ts[-1] - ts[0],
        "rmse_ps": {
            "last": xs[-1],
            "mean": sum(xs) / n,
            "min": min(xs),
            "max": max(xs),
            "p95": p95,
        },
        "alias_success_rate": alias_rate,
    }


def plot_trend(entries: List[Entry], out_path: Path) -> None:
    if not entries:
        return
    ts = [e.t - entries[0].t for e in entries]
    xs = [e.rmse_ps for e in entries]
    plt.figure(figsize=(8, 4))
    plt.plot(ts, xs, color="#00D9FF", linewidth=2)
    plt.xlabel("Time (s)")
    plt.ylabel("Timing RMSE (ps)")
    plt.title("Driftlock Hardware Demo — RMSE Trend")
    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--format", choices=["csv", "jsonl"], required=True)
    ap.add_argument("--input", required=True)
    ap.add_argument("--notes", default="")
    args = ap.parse_args()

    out_dir = _ensure_dir(Path(args.output_dir) / args.run_id)
    inp = Path(args.input)

    if args.format == "csv":
        entries = read_csv(inp)
    else:
        entries = read_jsonl(inp)

    # Persist manifest
    manifest = {
        "run_id": args.run_id,
        "notes": args.notes,
        "input": os.fspath(inp),
        "schema": "t (s), rmse_ps (float), alias (bool)",
    }
    (out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    # Summary
    summary = summarize(entries)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # Plot
    plot_trend(entries, out_dir / "rmse_trend.png")

    print(json.dumps({"output_dir": os.fspath(out_dir), **summary}, indent=2))


if __name__ == "__main__":
    main()

