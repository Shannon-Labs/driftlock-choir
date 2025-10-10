"""Generate reproducible metrics snapshots for chronometric interferometry demonstration."""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List

from ..core.types import ExperimentConfig, ExperimentResult
from .e1_basic_beat_note import ExperimentE1
from .runner import ExperimentContext


METRICS_DIR = Path("results/metrics")


def _clone_config(base: ExperimentConfig, overrides: Dict[str, float]) -> ExperimentConfig:
    return ExperimentConfig(
        experiment_id=base.experiment_id,
        description=base.description,
        parameters={**base.parameters, **overrides},
        seed=base.seed,
        start_time=base.start_time,
        expected_duration=base.expected_duration,
    )


def _context_for(config: ExperimentConfig, seed_offset: int = 0) -> ExperimentContext:
    seed = config.seed if config.seed is not None else 42
    return ExperimentContext(
        config=config,
        output_dir="results/e1_basic_beat_note",
        random_seed=seed + seed_offset,
        verbose=False,
    )


def _result_summary(result: ExperimentResult) -> Dict[str, float]:
    metrics = result.metrics
    return {
        "rmse_timing_ps": float(metrics.rmse_timing),
        "rmse_frequency_ppb": float(metrics.rmse_frequency),
        "success": 1.0 if result.success else 0.0,
    }


def _run_baseline(experiment: ExperimentE1, base_config: ExperimentConfig) -> Dict[str, float]:
    params = {**base_config.parameters, "plot_results": False, "save_plots": False}
    config = _clone_config(base_config, params)
    context = _context_for(config)
    result = experiment.run_experiment(context, config.parameters)
    return _result_summary(result)


def _run_sweeps(
    experiment: ExperimentE1,
    base_config: ExperimentConfig,
    snr_values: Iterable[float],
    delta_f_values: Iterable[float],
) -> List[Dict[str, float]]:
    records: List[Dict[str, float]] = []
    seed_offset = 1

    for snr in snr_values:
        for delta_f in delta_f_values:
            overrides = {
                "plot_results": False,
                "save_plots": False,
                "snr_db": snr,
                "true_delta_f_hz": delta_f,
            }
            config = _clone_config(base_config, overrides)
            context = _context_for(config, seed_offset=seed_offset)
            seed_offset += 1
            result = experiment.run_experiment(context, config.parameters)

            summary = _result_summary(result)
            summary.update(
                {
                    "snr_db": snr,
                    "delta_f_hz": delta_f,
                }
            )
            records.append(summary)

    return records


def main() -> None:
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    experiment = ExperimentE1()
    base_config = experiment.create_default_config()

    timestamp = datetime.now(timezone.utc).isoformat()

    baseline_metrics = _run_baseline(experiment, base_config)
    baseline_payload = {
        "timestamp": timestamp,
        "tx_frequency_hz": float(base_config.parameters["tx_frequency_hz"]),
        "rx_frequency_hz": float(base_config.parameters["rx_frequency_hz"]),
        "sampling_rate_hz": float(base_config.parameters["sampling_rate_hz"]),
        "duration_seconds": float(base_config.parameters["duration_seconds"]),
        "true_tau_ps": float(base_config.parameters["true_tau_ps"]),
        "true_delta_f_hz": float(base_config.parameters["true_delta_f_hz"]),
        "snr_db": float(base_config.parameters["snr_db"]),
        "metrics": baseline_metrics,
    }

    with (METRICS_DIR / "e1_baseline.json").open("w", encoding="utf-8") as fp:
        json.dump(baseline_payload, fp, indent=2)

    snr_values = [20.0, 25.0, 30.0, 35.0, 40.0]
    delta_f_values = [10.0, 25.0, 50.0, 75.0, 100.0]
    sweep_records = _run_sweeps(experiment, base_config, snr_values, delta_f_values)

    fieldnames = [
        "snr_db",
        "delta_f_hz",
        "rmse_timing_ps",
        "rmse_frequency_ppb",
        "success",
    ]

    with (METRICS_DIR / "e1_sweep_metrics.csv").open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for record in sweep_records:
            writer.writerow({key: record.get(key, "") for key in fieldnames})


if __name__ == "__main__":
    main()
