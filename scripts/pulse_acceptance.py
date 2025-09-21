#!/usr/bin/env python3
"""
Pulse Acceptance Pipeline - End-to-End Validation and Artifact Generation.

Orchestrates acceptance tests, movie generation, telemetry collection, composite creation, and markdown summary.
"""

import os
import subprocess
import yaml
import json
from pathlib import Path
from typing import Dict, Any

from src.io.telemetry import TelemetryExporter
from src.metrics.stats import StatisticalValidator, StatsParams
import argparse
import time
import shutil
import csv
from scipy import stats

def run_command(cmd: list[str], cwd: Path = Path(".")) -> Dict[str, Any]:
    """Run a subprocess command and return output."""
    env = os.environ.copy()
    env["PYTHONPATH"] = "."
    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, env=env)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed: {result.stderr}")
    return {"stdout": result.stdout, "stderr": result.stderr}

def load_historical_metrics(historical_path: str) -> Dict[str, Any]:
    """Load historical metrics from JSON."""
    with open(historical_path, "r") as f:
        return json.load(f)

def generate_markdown_digest(fresh_metrics: Dict[str, Any], historical: Dict[str, Any], run_type: str = 'acceptance') -> str:
    """Generate markdown summary comparing fresh vs historical."""
    md = f"# {run_type.upper()} Research Digest\n\n"
    md += "## Fresh Metrics\n"
    for key, value in fresh_metrics.items():
        if isinstance(value, dict):
            md += f"- **{key}**:\n"
            for subkey, subvalue in value.items():
                md += f"  - {subkey}: {subvalue}\n"
        else:
            md += f"- **{key}**: {value}\n"
    if historical:
        md += "\n## Historical Comparison\n"
        for key in fresh_metrics:
            if isinstance(fresh_metrics[key], (int, float)) and key in historical and isinstance(historical[key], (int, float)):
                delta = fresh_metrics[key] - historical[key]
                md += f"- **{key}**: Fresh {fresh_metrics[key]} vs Historical {historical[key]} (Δ {delta:+.2f})\n"
    return md

def main():
    parser = argparse.ArgumentParser(description="Pulse Acceptance Pipeline - End-to-End Validation and Artifact Generation.")
    parser.add_argument('--run-type', choices=['acceptance', 'baseline', 'ablation'], default='acceptance')
    args = parser.parse_args()

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"driftlock_choir_sim/outputs/research/{args.run_type}_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)

    exporter = TelemetryExporter(output_dir / "telemetry", variant=args.run_type)

    fresh_metrics = {}
    stats = {}

    if args.run_type == 'acceptance':
        # Step 1: Run acceptance
        print("Running acceptance test...")
        cmd = ["python", "driftlock_choir_sim/sims/run_acceptance.py"]
        result = run_command(cmd)
        # Parse stdout for metrics (assume JSON at end)
        lines = result["stdout"].strip().split("\n")
        fresh_metrics = json.loads(lines[-1])  # Last line is JSON
        acceptance_json = output_dir / "acceptance_summary.json"
        with open(acceptance_json, "w") as f:
            json.dump(fresh_metrics, f, indent=2)

        # Move acceptance artifacts
        acceptance_artifacts = Path("driftlock_choir_sim/outputs")
        if acceptance_artifacts.exists():
            shutil.copytree(acceptance_artifacts, output_dir / "acceptance_artifacts", dirs_exist_ok=True)
            # Clean up original if needed
            # shutil.rmtree(acceptance_artifacts / "csv")
            # etc., but skip clean for now

        exporter.add_record(data={"acceptance_metrics": fresh_metrics})

        # Integrate stats for acceptance
        validator = StatisticalValidator(StatsParams())
        if 'coherent' in fresh_metrics:
            coh = fresh_metrics['coherent']
            n = coh.get('n_trials', 14)
            mean_rmse = coh['rmse_ps']
            crlb = coh['crlb_ps']
            sem = crlb / np.sqrt(n)
            ci_level = 0.95
            df = n - 1
            t_crit = stats.t.ppf((1 + ci_level) / 2, df)
            margin = t_crit * sem
            ci = {
                'point_estimate': mean_rmse,
                'ci_lower': mean_rmse - margin,
                'ci_upper': mean_rmse + margin,
                'method': 'parametric_t',
                'n_samples': n
            }
            stats['coherent_rmse'] = ci
        exporter.add_record(data={'acceptance_stats': stats})

        # Step 2: Run make_movie for demo with telemetry
        print("Generating demo movie...")
        demo_yaml = output_dir / "demo_movie.yaml"
        demo_config = {
            "movie": {"telemetry": True}
        }  # Add telemetry flag
        with open(demo_yaml, "w") as f:
            yaml.dump(demo_config, f)
        cmd = ["python", "driftlock_choir_sim/sims/make_movie.py", "--config", str(demo_yaml)]
        run_command(cmd, cwd=output_dir)
        demo_telemetry = output_dir / "demo_telemetry.jsonl"
        # Assume telemetry saved

        if demo_telemetry.exists():
            with open(demo_telemetry, "r") as f:
                for line in f:
                    if line.strip():
                        try:
                            telemetry_data = json.loads(line.strip())
                            exporter.add_record(data=telemetry_data, variant="demo")
                        except json.JSONDecodeError:
                            pass

        # Step 3: Run make_movie for baseline with telemetry
        print("Generating baseline movie...")
        baseline_yaml = output_dir / "baseline_movie.yaml"
        baseline_config = {
            "movie": {"telemetry": True}
        }
        with open(baseline_yaml, "w") as f:
            yaml.dump(baseline_config, f)
        cmd = ["python", "driftlock_choir_sim/sims/make_movie.py", "--config", str(baseline_yaml)]
        run_command(cmd, cwd=output_dir)
        baseline_telemetry = output_dir / "baseline_telemetry.jsonl"

        if baseline_telemetry.exists():
            with open(baseline_telemetry, "r") as f:
                for line in f:
                    if line.strip():
                        try:
                            telemetry_data = json.loads(line.strip())
                            exporter.add_record(data=telemetry_data, variant="baseline")
                        except json.JSONDecodeError:
                            pass

        # Step 4: Generate teasers (copy configs with seconds=15)
        demo_teaser_yaml = output_dir / "demo_teaser.yaml"
        teaser_config = {
            "movie": {"seconds": 15, "telemetry": True}
        }
        with open(demo_teaser_yaml, "w") as f:
            yaml.dump(teaser_config, f)
        demo_teaser_telemetry = output_dir / "demo_teaser_telemetry.jsonl"
        cmd = ["python", "driftlock_choir_sim/sims/make_movie.py", "--config", str(demo_teaser_yaml)]
        run_command(cmd, cwd=output_dir)

        if demo_teaser_telemetry.exists():
            with open(demo_teaser_telemetry, "r") as f:
                for line in f:
                    if line.strip():
                        try:
                            telemetry_data = json.loads(line.strip())
                            exporter.add_record(data=telemetry_data, variant="demo_teaser")
                        except json.JSONDecodeError:
                            pass

        baseline_teaser_yaml = output_dir / "baseline_teaser.yaml"
        with open(baseline_teaser_yaml, "w") as f:
            yaml.dump(teaser_config, f)
        baseline_teaser_telemetry = output_dir / "baseline_teaser_telemetry.jsonl"
        cmd = ["python", "driftlock_choir_sim/sims/make_movie.py", "--config", str(baseline_teaser_yaml)]
        run_command(cmd, cwd=output_dir)

        if baseline_teaser_telemetry.exists():
            with open(baseline_teaser_telemetry, "r") as f:
                for line in f:
                    if line.strip():
                        try:
                            telemetry_data = json.loads(line.strip())
                            exporter.add_record(data=telemetry_data, variant="baseline_teaser")
                        except json.JSONDecodeError:
                            pass

        # Step 5: Create composite
        print("Creating composite reel...")
        cmd = [
            "python", "driftlock_choir_sim/viz/compositor.py",
            "--left-config", str(baseline_yaml),
            "--right-config", str(demo_yaml),
            "--layout", "side-by-side",
            "--burn-in-metrics",
            "--num-frames", "360",
            "--seed", "2025",
            "--output-name", "pulse_comparison"
        ]
        run_command(cmd, cwd=output_dir)

    elif args.run_type == 'baseline':
        print("Running baseline simulations...")
        baseline_subdir = output_dir / "runs"
        baseline_subdir.mkdir()
        baseline_rmses = []
        n_mc = 20
        for i in range(n_mc):
            seed = 2025 + i
            run_dir = baseline_subdir / f"run_{i}"
            run_dir.mkdir()
            cmd = [
                "python", "sim/phase2.py",
                "--baseline-mode",
                "--rng-seed", str(seed),
                "--results-dir", str(run_dir),
                "--no-plots",
                "--save-results"
            ]
            try:
                run_command(cmd)
                jsonl_path = run_dir / "phase2_runs.jsonl"
                if jsonl_path.exists():
                    with open(jsonl_path, 'r') as f:
                        lines = [line.strip() for line in f if line.strip()]
                        if lines:
                            last = json.loads(lines[-1])
                            rmse_list = last.get('consensus', {}).get('timing_rms_ps', [])
                            if rmse_list:
                                rmse = rmse_list[-1]
                                baseline_rmses.append(rmse)
            except Exception as e:
                print(f"Error in baseline run {i}: {e}")
        if baseline_rmses:
            validator = StatisticalValidator(StatsParams())
            ci = validator.confidence_intervals_for_rmse(np.array(baseline_rmses))
            stats['baseline_rmse'] = ci
            fresh_metrics = {
                'mean_rmse_ps': float(np.mean(baseline_rmses)),
                'std_rmse_ps': float(np.std(baseline_rmses)),
                'n_runs': len(baseline_rmses)
            }
            exporter.add_record(data={'baseline_metrics': fresh_metrics, 'baseline_stats': stats})

    elif args.run_type == 'ablation':
        print("Running ablation sweeps...")
        ablation_subdir = output_dir / "sweeps"
        ablation_subdir.mkdir()
        cmd = [
            "python", "scripts/ablation_sweeps.py",
            "--config", "sim/configs/ablations.yaml",
            "--output", str(ablation_subdir)
        ]
        run_command(cmd)
        summary_csv = ablation_subdir / "ablation_summary.csv"
        ablation_rmses = []
        if summary_csv.exists():
            with open(summary_csv, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    rmse_str = row.get('mean_final_rmse_ps', '')
                    if rmse_str:
                        try:
                            ablation_rmses.append(float(rmse_str))
                        except ValueError:
                            pass
        if ablation_rmses:
            validator = StatisticalValidator(StatsParams())
            ci = validator.confidence_intervals_for_rmse(np.array(ablation_rmses))
            stats['ablation_rmse'] = ci
            fresh_metrics = {
                'mean_rmse_ps': float(np.mean(ablation_rmses)),
                'std_rmse_ps': float(np.std(ablation_rmses)),
                'n_combos': len(ablation_rmses)
            }
            exporter.add_record(data={'ablation_metrics': fresh_metrics, 'ablation_stats': stats})

    # Common: Load historical and generate digest
    historical_path = "results/mc_runs/extended_011/final_results.json"
    historical = {}
    if os.path.exists(historical_path):
        historical = load_historical_metrics(historical_path)
    exporter.add_record(data={"historical_metrics": historical}, variant="historical")

    digest = generate_markdown_digest(fresh_metrics, historical, args.run_type)
    # Enhance with stats
    if stats:
        digest += "\n## Statistical Findings\n"
        for key, s in stats.items():
            if isinstance(s, dict) and 'ci_lower' in s:
                digest += f"- **{key}**: point {s['point_estimate']:.2f}, 95% CI [{s['ci_lower']:.2f}, {s['ci_upper']:.2f}]\n"
    with open(output_dir / "digest.md", "w") as f:
        f.write(digest)

    exporter.add_record(data={"comparison": fresh_metrics}, baseline_data=historical)

    exporter.export_batch(f"{args.run_type}_summary")

    print(f"All artifacts generated in {output_dir}")

if __name__ == "__main__":
    main()