#!/usr/bin/env python3
"""
Read serial JSON lines from the Feather demo and produce a rmse_stream.jsonl
with fields {"t": <seconds>, "rmse_ps": <float>} for plotting/logging.

Input line schema (JSON per line): {"seq": int, "t_us": float, "rtt_us": float}

Usage:
  python scripts/feather_log_parser.py \
    --port /dev/tty.usbmodem1101 --baud 115200 \
    --output rmse_stream.jsonl --window 50
"""

from __future__ import annotations

import argparse
import json
import math
from collections import deque
from statistics import mean
from typing import Deque, List

try:
    import serial  # type: ignore
except Exception as e:  # pragma: no cover
    serial = None


def rolling_rms(xs: List[float]) -> float:
    if not xs:
        return float("nan")
    m = mean(xs)
    return math.sqrt(mean([(x - m) ** 2 for x in xs]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", required=True)
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--output", required=True)
    ap.add_argument("--window", type=int, default=50)
    args = ap.parse_args()

    if serial is None:
        raise SystemExit("pyserial not installed. pip install pyserial")

    q: Deque[float] = deque(maxlen=args.window)
    with serial.Serial(args.port, args.baud, timeout=1) as ser, open(args.output, "w", encoding="utf-8") as out:
        while True:
            line = ser.readline().decode("utf-8", errors="ignore").strip()
            if not line:
                continue
            if not (line.startswith("{") and line.endswith("}")):
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            if "t_us" not in rec or "rtt_us" not in rec:
                continue
            t_s = float(rec["t_us"]) * 1e-6
            rtt_ps = float(rec["rtt_us"]) * 1e6  # us -> ps
            q.append(rtt_ps)
            rmse_ps = rolling_rms(list(q))
            out.write(json.dumps({"t": t_s, "rmse_ps": rmse_ps}) + "\n")
            out.flush()
            # Minimal console echo for operator
            print(f"t={t_s:.3f}s rmse={rmse_ps:,.0f} ps")


if __name__ == "__main__":
    main()

