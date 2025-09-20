#!/usr/bin/env python3
"""
Feather Demo Runner — the simplest end-to-end for the low-cost RPR demo.

What it does:
  - Auto-detect a serial port printing JSON lines {seq, t_us, rtt_us}
  - Read for a fixed duration (default 60s)
  - Write rmse_stream.jsonl + summary.json + rmse_trend.png into an output dir

Usage:
  python scripts/feather_demo_run.py --duration 60 --out results/hw_demo_quick
  # or specify port
  python scripts/feather_demo_run.py --port /dev/tty.usbmodem1101
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Deque, List, Optional

import matplotlib.pyplot as plt

try:
    import serial  # type: ignore
    from serial.tools import list_ports  # type: ignore
except Exception:
    serial = None
    list_ports = None


def rolling_rms(xs: List[float]) -> float:
    if not xs:
        return float("nan")
    m = mean(xs)
    return math.sqrt(mean([(x - m) ** 2 for x in xs]))


def pick_port(explicit: Optional[str]) -> str:
    if explicit:
        return explicit
    if list_ports is None:
        raise SystemExit("pyserial not installed. pip install pyserial")
    ports = list(list_ports.comports())
    if not ports:
        raise SystemExit("No serial ports found. Plug in the Feather and try again.")
    # Prefer Feather-ish names on macOS/Linux
    for p in ports:
        name = (p.device or "") + " " + (p.description or "")
        if "usbmodem" in name or "ttyACM" in name or "Feather" in name or "Adafruit" in name:
            return p.device
    # Fallback to first
    return ports[0].device


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", help="Serial port (auto-detect if omitted)")
    ap.add_argument("--duration", type=int, default=60, help="Seconds to record")
    ap.add_argument("--out", default="results/hw_demo_quick", help="Output root directory")
    args = ap.parse_args()

    if serial is None:
        raise SystemExit("pyserial not installed. pip install pyserial")

    port = pick_port(args.port)
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    out_dir = ensure_dir(Path(args.out) / f"feather_{ts}")

    print(f"Using serial port: {port}")
    print(f"Recording for {args.duration}s …")

    q: Deque[float] = deque(maxlen=50)
    entries = []
    t0 = time.time()
    with serial.Serial(port, 115200, timeout=0.5) as ser, open(out_dir / "rmse_stream.jsonl", "w", encoding="utf-8") as stream:
        while True:
            if time.time() - t0 > args.duration:
                break
            line = ser.readline().decode("utf-8", errors="ignore").strip()
            if not line or not (line.startswith("{") and line.endswith("}")):
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            if "t_us" not in rec or "rtt_us" not in rec:
                continue
            t_s = float(rec["t_us"]) * 1e-6
            rmse_ps = float(rec["rtt_us"]) * 1e6  # treat RTT as stability proxy; convert us->ps
            q.append(rmse_ps)
            val = rolling_rms(list(q))
            payload = {"t": t_s, "rmse_ps": val}
            stream.write(json.dumps(payload) + "\n")
            entries.append(payload)
            # Console feedback
            print(f"t={t_s:6.2f}s  rmse≈{val:10.0f} ps", end="\r")

    print("\nFinished. Summarizing …")
    # Summary
    if entries:
        xs = [e["rmse_ps"] for e in entries]
        p95 = sorted(xs)[int(0.95 * (len(xs) - 1))]
        summary = {
            "count": len(xs),
            "duration_s": entries[-1]["t"] - entries[0]["t"],
            "rmse_ps": {
                "last": xs[-1],
                "mean": sum(xs) / len(xs),
                "min": min(xs),
                "max": max(xs),
                "p95": p95,
            },
        }
    else:
        summary = {"count": 0}

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # Plot
    if entries:
        t0s = entries[0]["t"]
        ts_vals = [e["t"] - t0s for e in entries]
        xs = [e["rmse_ps"] for e in entries]
        plt.figure(figsize=(8, 4))
        plt.plot(ts_vals, xs, color="#00D9FF", linewidth=2)
        plt.xlabel("Time (s)")
        plt.ylabel("RMSE proxy (ps)")
        plt.title("Driftlock Feather Demo — Stability Trend")
        plt.grid(True, alpha=0.2)
        plt.tight_layout()
        plt.savefig(out_dir / "rmse_trend.png", dpi=160)
        plt.close()

    # Manifest
    manifest = {
        "port": port,
        "duration_s": args.duration,
        "timestamp": ts,
        "notes": "Feather RPR quick demo",
    }
    (out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Artifacts → {os.fspath(out_dir)}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

