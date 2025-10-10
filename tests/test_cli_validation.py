"""CLI validation suite covering clean, stressed, and hardware-bridge scenarios."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUN_EXPERIMENT = PROJECT_ROOT / "run_experiment.py"


def _run_cli(export_path: Path, *args: str, expect_success: bool) -> dict:
    command = [
        sys.executable,
        str(RUN_EXPERIMENT),
        "--export",
        str(export_path),
        *args,
    ]
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=False,
    )

    if expect_success:
        assert result.returncode == 0, result.stderr or result.stdout
    else:
        assert result.returncode != 0, "Expected CLI to raise for stressed scenario"

    data = json.loads(export_path.read_text(encoding="utf-8"))
    return data


def test_clean_24ghz_cli(tmp_path: Path) -> None:
    export_path = tmp_path / "clean.json"
    data = _run_cli(
        export_path,
        "--band",
        "2.4GHz",
        "--channel-profile",
        "line_of_sight",
        "--duration-ms",
        "2.0",
        "--sampling-rate-msps",
        "40",
        "--tau-ps",
        "13.5",
        "--delta-f-hz",
        "150",
        "--no-phase-noise",
        "--no-additive-noise",
        expect_success=True,
    )

    summary = data["summary"]
    assert summary["success"] is True
    assert summary["validation"]["timing_error_ps"] <= 5.0
    assert summary["analysis"]["quality"] in {"excellent", "good"}


def test_high_band_58ghz_cli(tmp_path: Path) -> None:
    export_path = tmp_path / "high_band.json"
    data = _run_cli(
        export_path,
        "--band",
        "5.8GHz",
        "--channel-profile",
        "line_of_sight",
        "--duration-ms",
        "2.0",
        "--sampling-rate-msps",
        "40",
        "--tau-ps",
        "9.5",
        "--delta-f-hz",
        "240",
        "--no-phase-noise",
        "--no-additive-noise",
        expect_success=True,
    )

    summary = data["summary"]
    assert summary["success"] is True
    assert summary["validation"]["timing_error_ps"] <= 1.0
    assert summary["analysis"]["delta_f_estimate_hz"] == pytest.approx(0.0, abs=0.01)


def test_noisy_multipath_cli_failure(tmp_path: Path) -> None:
    export_path = tmp_path / "multipath.json"
    data = _run_cli(
        export_path,
        "--band",
        "2.4GHz",
        "--channel-profile",
        "rf_multipath",
        "--duration-ms",
        "4.0",
        "--sampling-rate-msps",
        "40",
        "--tau-ps",
        "13.5",
        "--delta-f-hz",
        "150",
        "--snr-db",
        "15",
        expect_success=False,
    )

    summary = data["summary"]
    assert summary["success"] is False
    assert summary["validation"]["meets_frequency"] is False
    assert summary["analysis"]["tau_estimate_ps"] > 1e5


def test_hardware_bridge_dry_run() -> None:
    from hardware_experiment.offline_bridge import estimate_from_capture

    sampling_rate_hz = 1.0e6
    duration_seconds = 0.002
    samples = int(sampling_rate_hz * duration_seconds)
    time = np.arange(samples) / sampling_rate_hz

    # Baseband captures representing a 150 Hz beat note
    tx_capture = np.exp(1j * 2 * np.pi * 150.0 * time)
    rx_capture = np.exp(1j * 2 * np.pi * 150.0 * time)

    result, metadata = estimate_from_capture(
        tx_capture=tx_capture,
        rx_capture=rx_capture,
        sampling_rate_hz=sampling_rate_hz,
        tx_frequency_hz=150.0,
        rx_frequency_hz=150.0,
        snr_db=35.0,
    )

    assert metadata["samples"] == samples
    assert result.tau_uncertainty > 0
    assert result.quality.value in {"excellent", "good", "fair"}
