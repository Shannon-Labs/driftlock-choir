#!/usr/bin/env python3
"""Run the core chronometric interferometry experiment with clean outputs.

This script executes the E1 chronometric interferometry experiment, focusing on
scientific data quality. It disables automatic file generation, exposes standard
RF bands (2.4 GHz / 5.8 GHz), and reports timing/frequency estimates that can be
consumed directly by the visualization and analysis tooling.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from src.core.types import BeatNoteAnalysisRecord
from src.experiments.e1_basic_beat_note import ExperimentE1
from src.experiments.runner import ExperimentContext

BAND_FREQUENCIES = {
    "2.4GHz": 2.4e9,
    "5.8GHz": 5.8e9,
}

PRECISION_GATE_PS = 5.0  # Absolute tolerance for 13.5 ps validation


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Clean chronometric interferometry experiment runner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--band",
        choices=sorted(BAND_FREQUENCIES.keys()),
        default="2.4GHz",
        help="RF band to simulate (sets nominal TX frequency unless overridden)",
    )
    parser.add_argument(
        "--tx-frequency-hz",
        type=float,
        help="Override transmit frequency (Hz). Defaults to the selected band.",
    )
    parser.add_argument(
        "--rx-frequency-hz",
        type=float,
        help="Override receive frequency (Hz). Defaults to TX + delta-f.",
    )
    parser.add_argument(
        "--delta-f-hz",
        type=float,
        default=150.0,
        help="Target beat-note frequency difference Δf (Hz). Ignored if RX frequency is provided.",
    )
    parser.add_argument(
        "--residual-delta-f-hz",
        type=float,
        default=0.0,
        help="Residual frequency offset after mixing (Hz).",
    )
    parser.add_argument(
        "--tau-ps",
        type=float,
        default=13.5,
        help="True time-of-flight delay τ in picoseconds.",
    )
    parser.add_argument(
        "--snr-db",
        type=float,
        default=35.0,
        help="Signal-to-noise ratio in dB.",
    )
    parser.add_argument(
        "--duration-ms",
        type=float,
        default=1.0,
        help="Signal duration in milliseconds.",
    )
    parser.add_argument(
        "--sampling-rate-msps",
        type=float,
        default=20.0,
        help="Sampling rate in mega-samples per second.",
    )
    parser.add_argument(
        "--channel-profile",
        choices=["rf_multipath", "line_of_sight"],
        default="line_of_sight",
        help="Channel impairment profile to apply.",
    )
    parser.add_argument(
        "--no-phase-noise",
        action="store_true",
        help="Disable oscillator phase-noise modeling.",
    )
    parser.add_argument(
        "--no-additive-noise",
        action="store_true",
        help="Disable additive thermal/channel noise.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--export",
        type=Path,
        help="Optional path to export structured experiment data (JSON).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose experiment logging for debugging.",
    )
    return parser.parse_args()


def configure_experiment(
    experiment: ExperimentE1, args: argparse.Namespace
) -> tuple[ExperimentContext, Dict[str, float]]:
    """Configure the experiment context and return metadata."""
    config = experiment.create_default_config()

    tx_frequency_hz = args.tx_frequency_hz or BAND_FREQUENCIES[args.band]
    rx_frequency_hz = args.rx_frequency_hz or (tx_frequency_hz + args.delta_f_hz)
    beat_frequency_hz = rx_frequency_hz - tx_frequency_hz
    residual_delta_f_hz = args.residual_delta_f_hz

    sampling_rate_hz = args.sampling_rate_msps * 1e6
    duration_seconds = args.duration_ms / 1e3

    config.parameters.update(
        {
            "tx_frequency_hz": tx_frequency_hz,
            "rx_frequency_hz": rx_frequency_hz,
            "sampling_rate_hz": sampling_rate_hz,
            "duration_seconds": duration_seconds,
            "true_tau_ps": args.tau_ps,
            "true_delta_f_hz": residual_delta_f_hz,
            "snr_db": args.snr_db,
            "add_noise": not args.no_additive_noise,
            "add_phase_noise": not args.no_phase_noise,
            "plot_results": False,
            "save_plots": False,
            "channel_profile": args.channel_profile,
            "oscillator_type": config.parameters.get("oscillator_type", "tcxo"),
            "estimation_method": config.parameters.get("estimation_method", "phase_slope"),
        }
    )

    context = ExperimentContext(
        config=config,
        random_seed=args.seed,
        verbose=args.verbose,
    )

    metadata = {
        "tx_frequency_hz": tx_frequency_hz,
        "rx_frequency_hz": rx_frequency_hz,
        "beat_frequency_hz": beat_frequency_hz,
        "residual_delta_f_hz": residual_delta_f_hz,
        "sampling_rate_hz": sampling_rate_hz,
        "duration_seconds": duration_seconds,
        "snr_db": args.snr_db,
        "phase_noise": not args.no_phase_noise,
        "additive_noise": not args.no_additive_noise,
        "channel_profile": args.channel_profile,
    }

    return context, metadata


def print_header(metadata: Dict[str, float], args: argparse.Namespace) -> None:
    """Print experiment header information."""
    print("🔬 Chronometric Interferometry — Experiment E1")
    print("=" * 72)
    print(
        "RF band: {band} | TX {tx:.6f} GHz | RX {rx:.6f} GHz".format(
            band=args.band,
            tx=metadata["tx_frequency_hz"] / 1e9,
            rx=metadata["rx_frequency_hz"] / 1e9,
        )
    )
    print(
        "Target τ: {tau:.3f} ps | Residual Δf target: {delta:.3f} Hz".format(
            tau=args.tau_ps, delta=metadata["residual_delta_f_hz"]
        )
    )
    print(
        "Sampling: {fs:.2f} MS/s for {dur:.2f} ms | SNR: {snr:.1f} dB".format(
            fs=metadata["sampling_rate_hz"] / 1e6,
            dur=metadata["duration_seconds"] * 1e3,
            snr=metadata["snr_db"],
        )
    )
    print(
        "Nominal beat note: {beat:.3f} Hz".format(
            beat=metadata["beat_frequency_hz"]
        )
    )
    print(
        "Phase noise: {phase} | Additive noise: {additive} | Channel: {channel}".format(
            phase="enabled" if metadata["phase_noise"] else "disabled",
            additive="enabled" if metadata["additive_noise"] else "disabled",
            channel=metadata["channel_profile"],
        )
    )
    print()


def _extract_analysis_record(result) -> Optional[BeatNoteAnalysisRecord]:
    """Return the first analysis record, building one from telemetry if necessary."""
    if result.analysis_records:
        return result.analysis_records[0]

    if result.telemetry:
        candidate = result.telemetry[0]
        if isinstance(candidate, dict):
            try:
                return BeatNoteAnalysisRecord.from_dict(candidate)
            except Exception:  # pragma: no cover - best-effort fallback
                return None
    return None


def summarize_result(
    result, args: argparse.Namespace, metadata: Dict[str, float]
) -> Dict[str, Any]:
    """Print and return a summary dictionary for downstream use."""
    analysis_record = _extract_analysis_record(result)

    metrics = result.metrics
    rmse_timing_ps = float(metrics.rmse_timing)
    rmse_frequency_ppb = float(metrics.rmse_frequency)

    summary: Dict[str, Any] = {
        "metrics": {
            "rmse_timing_ps": rmse_timing_ps,
            "rmse_frequency_ppb": rmse_frequency_ppb,
            "convergence_time_s": float(metrics.convergence_time),
            "iterations_to_convergence": metrics.iterations_to_convergence,
        },
        "success": result.success,
        "error_message": result.error_message,
    }

    expected_delta_f = metadata["residual_delta_f_hz"]
    expected_beat_frequency = metadata["beat_frequency_hz"]

    print("📈 Primary Estimates")
    if analysis_record:
        tau_estimate_ps = float(analysis_record.tau_estimate)
        tau_unc_ps = float(analysis_record.tau_uncertainty)
        delta_f_estimate_hz = float(analysis_record.delta_f_estimate)
        delta_f_unc_hz = float(analysis_record.delta_f_uncertainty)
        beat_frequency_hz = float(analysis_record.beat_frequency)
        quality = analysis_record.quality.value

        timing_error_ps = abs(tau_estimate_ps - args.tau_ps)
        frequency_error_hz = abs(delta_f_estimate_hz - expected_delta_f)
        beat_frequency_error_hz = abs(beat_frequency_hz - expected_beat_frequency)

        precision_gate = max(PRECISION_GATE_PS, 3.0 * tau_unc_ps)
        frequency_gate = max(1.0, 3.0 * delta_f_unc_hz)
        beat_gate = max(1.0, 3.0 * delta_f_unc_hz)

        summary.update(
            {
                "analysis": {
                    "tau_estimate_ps": tau_estimate_ps,
                    "tau_uncertainty_ps": tau_unc_ps,
                    "delta_f_estimate_hz": delta_f_estimate_hz,
                    "delta_f_uncertainty_hz": delta_f_unc_hz,
                    "beat_frequency_hz": beat_frequency_hz,
                    "expected_beat_frequency_hz": expected_beat_frequency,
                    "expected_residual_delta_f_hz": expected_delta_f,
                    "snr_db": float(analysis_record.snr_db),
                    "quality": quality,
                    "method": analysis_record.estimation_method,
                },
                "validation": {
                    "timing_error_ps": timing_error_ps,
                    "frequency_error_hz": frequency_error_hz,
                    "beat_frequency_error_hz": beat_frequency_error_hz,
                    "precision_gate_ps": precision_gate,
                    "frequency_gate_hz": frequency_gate,
                    "beat_frequency_gate_hz": beat_gate,
                    "meets_precision": timing_error_ps <= precision_gate,
                    "meets_frequency": frequency_error_hz <= frequency_gate,
                    "meets_beat_frequency": beat_frequency_error_hz <= beat_gate,
                },
            }
        )

        print(
            f"  • τ estimate: {tau_estimate_ps:.3f} ps ± {tau_unc_ps:.3f} ps"
        )
        print(
            f"  • Δf estimate: {delta_f_estimate_hz:.3f} Hz ± {delta_f_unc_hz:.3f} Hz"
        )
        print(
            f"  • Beat frequency: {beat_frequency_hz:.3f} Hz (expected {expected_beat_frequency:.3f} Hz) | Quality: {quality}"
        )
        print()

        print("🎯 Validation Checks")
        precision_flag = "✅" if timing_error_ps <= precision_gate else "❌"
        frequency_flag = "✅" if frequency_error_hz <= frequency_gate else "❌"
        beat_flag = "✅" if beat_frequency_error_hz <= beat_gate else "❌"
        print(
            f"  • Timing accuracy: {precision_flag} |Δτ| = {timing_error_ps:.3f} ps (gate ≤ {precision_gate:.3f} ps)"
        )
        print(
            f"  • Frequency bias: {frequency_flag} |Δf| = {frequency_error_hz:.3f} Hz (gate ≤ {frequency_gate:.3f} Hz)"
        )
        print(
            f"  • Beat note match: {beat_flag} |Δf_beat| = {beat_frequency_error_hz:.3f} Hz (gate ≤ {beat_gate:.3f} Hz)"
        )
    else:
        print("  • No beat-note analysis record attached to the result.")

    print()
    print("📊 Performance Metrics")
    print(f"  • RMSE τ: {rmse_timing_ps:.3f} ps")
    print(f"  • RMSE Δf: {rmse_frequency_ppb:.3f} ppb")
    print(
        f"  • Success flag: {'✅' if result.success else '❌'}"
        + (f" — {result.error_message}" if result.error_message else "")
    )
    print()

    return summary


def export_data(
    path: Path,
    context: ExperimentContext,
    result,
    summary: Dict[str, Any],
) -> None:
    """Export experiment data to a JSON file."""
    payload: Dict[str, Any] = {
        "experiment_id": context.config.experiment_id,
        "description": context.config.description,
        "parameters": context.config.parameters,
        "summary": summary,
        "analysis_records": [record.to_dict() for record in result.analysis_records],
        "telemetry": result.telemetry,
        "metrics": {
            "rmse_timing_ps": float(result.metrics.rmse_timing),
            "rmse_frequency_ppb": float(result.metrics.rmse_frequency),
            "convergence_time_s": float(result.metrics.convergence_time),
            "iterations_to_convergence": result.metrics.iterations_to_convergence,
            "communication_overhead_bytes": result.metrics.communication_overhead,
            "computation_time_s": float(result.metrics.computation_time),
        },
        "success": result.success,
        "error_message": result.error_message,
        "completion_time": result.completion_time.to_datetime().isoformat(),
    }

    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    print(f"📝 Exported structured data to {path}")


def main() -> int:
    args = parse_args()

    experiment = ExperimentE1()
    context, metadata = configure_experiment(experiment, args)

    print_header(metadata, args)

    result = experiment.run_experiment(context, context.config.parameters)

    summary = summarize_result(result, args, metadata)

    if args.export:
        export_data(args.export, context, result, summary)

    return 0 if result.success else 1


if __name__ == "__main__":
    sys.exit(main())
