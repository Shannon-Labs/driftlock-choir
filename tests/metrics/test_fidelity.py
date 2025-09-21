"""Pytest for fidelity validation using synthetic telemetry."""

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest
import numpy as np

TELEMETRY_SAMPLES = 5
EXPECTED_RMSE_PS = 30.0
EXPECTED_CONVERGED = True


@pytest.fixture
def synthetic_telemetry() -> Path:
    """Generate tiny synthetic telemetry JSONL for testing."""
    data = []
    for _ in range(TELEMETRY_SAMPLES):
        rmse_tau = EXPECTED_RMSE_PS + np.random.normal(0, 0.5)
        rmse_df = 1e3 + np.random.normal(0, 50)
        entry = {
            "config": {
                "snr_db": 25.0,
                "coarse_bandwidth_hz": 20e6,
                "coarse_duration_s": 0.01,
                "min_baseband_rate_hz": 20e6,
                "max_iterations": 50,
            },
            "consensus": {
                "timing_rms_ps": [rmse_tau, rmse_tau],
                "frequency_rms_hz": [rmse_df, rmse_df],
                "converged": EXPECTED_CONVERGED,
                "convergence_iteration": 1,
                "target_rmse_ps": 100.0,
            },
            "statistics": {
                "rmse_tau": {"point_estimate": rmse_tau},
                "rmse_df": {"point_estimate": rmse_df},
            },
            "edge_diagnostics": {
                "predicted_tau_std_ps": 5.0,
                "predicted_df_std_hz": 100.0,
                "measurement_rmse_tau_ps": rmse_tau,
                "measurement_rmse_df_hz": rmse_df,
            },
        }
        data.append(entry)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        for entry in data:
            f.write(json.dumps(entry) + "\n")
        temp_path = Path(f.name)
    yield temp_path
    temp_path.unlink()


def test_fidelity_validation(synthetic_telemetry: Path) -> None:
    """Test that validate_fidelity.py passes on synthetic good telemetry."""
    repo_root = Path(__file__).resolve().parents[2]
    run_id = "pytest_smoke"
    output_dir = repo_root / "results" / "validate_fidelity_test"
    cmd = [
        sys.executable,
        "scripts/validate_fidelity.py",
        "sim/configs/default.yaml",
        "--telemetry-path",
        str(synthetic_telemetry),
        "--n-trials",
        "5",
        "--log-level",
        "WARNING",
        "--run-id",
        run_id,
        "--output",
        str(output_dir),
        "--skip-hw-validation",
        "--skip-crlb-validation",
    ]

    env = os.environ.copy()
    env.setdefault("PYTHONPATH", str(repo_root))

    result = subprocess.run(cmd, capture_output=True, text=True, cwd=repo_root, env=env)
    assert result.returncode == 0, f"Validation failed: {result.stderr}"

    report_path = output_dir / run_id / "fidelity_report.json"
    assert report_path.exists(), "Report not generated; check output dir"
    with open(report_path, "r", encoding="utf-8") as f:
        report = json.load(f)
    assert report.get("overall_pass") is True, "Report indicates failure"
