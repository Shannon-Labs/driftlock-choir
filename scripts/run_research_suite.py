#!/usr/bin/env python3
"""
Research Suite Runner - Comprehensive Scientific Validation Protocol

This script orchestrates the complete research workflow for Driftlock Choir scientific validation,
regenerating all evidence: acceptance vs baseline comparisons, ablation studies, statistical tests,
model fidelity validation, and comprehensive documentation.

Usage:
    python scripts/run_research_suite.py --all                    # Run complete suite
    python scripts/run_research_suite.py --acceptance            # Acceptance tests only
    python scripts/run_research_suite.py --baseline              # Baseline comparisons only
    python scripts/run_research_suite.py --ablations             # Ablation studies only
    python scripts/run_research_suite.py --validation            # Statistical validation only
    python scripts/run_research_suite.py --fidelity              # Model fidelity checks only
"""

import argparse
import subprocess
import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
import sys
import os

@dataclass
class ResearchConfig:
    """Configuration for research suite execution."""
    base_seed: int = 2025
    output_base: str = "driftlock_choir_sim/outputs/research"
    results_base: str = "results"
    n_monte_carlo: int = 20
    n_workers: int = 4
    run_timestamp: str = ""

    # Component flags
    run_acceptance: bool = False
    run_baseline: bool = False
    run_ablations: bool = False
    run_validation: bool = False
    run_fidelity: bool = False
    run_historical: bool = False

    # Quality gates
    require_significance: bool = True
    significance_threshold: float = 0.05
    require_fidelity: bool = True
    fidelity_threshold: float = 2.0  # RMSE within 2x CRLB

    export_site_data: bool = True
def setup_environment() -> None:
    """Setup Python environment and paths."""
    # Ensure we're in the right directory
    script_dir = Path(__file__).parent
    os.chdir(script_dir.parent)

    # Set PYTHONPATH
    env = os.environ.copy()
    env["PYTHONPATH"] = "."

    # Check if virtual environment exists
    venv_path = Path(".venv")
    if not venv_path.exists():
        print("Warning: Virtual environment not found. Creating one...")
        subprocess.run([sys.executable, "-m", "venv", ".venv"], check=True)

    # Install dependencies if needed
    try:
        import numpy
        import scipy
        import matplotlib
        import networkx
        print("✓ All dependencies available")
    except ImportError as e:
        print(f"Missing dependency: {e}")
        print("Installing requirements...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], check=True)

def run_command(cmd: List[str], description: str, cwd: Optional[Path] = None) -> bool:
    """Run a command with proper error handling and logging."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}")

    try:
        env = os.environ.copy()
        env["PYTHONPATH"] = "."

        result = subprocess.run(
            cmd,
            cwd=cwd or Path("."),
            capture_output=True,
            text=True,
            env=env,
            check=True
        )

        print(f"✓ {description} completed successfully")
        if result.stdout:
            print("Output:", result.stdout[-500:])  # Last 500 chars
        return True

    except subprocess.CalledProcessError as e:
        print(f"✗ {description} failed")
        print(f"Error: {e.stderr}")
        return False
    except Exception as e:
        print(f"✗ {description} failed with exception: {e}")
        return False

def run_acceptance_tests(config: ResearchConfig) -> bool:
    """Run comprehensive acceptance tests with enhanced telemetry."""
    print(f"\n{'='*80}")
    print("ACCEPTANCE TESTS - Enhanced Scientific Validation")
    print(f"{'='*80}")

    timestamp = config.run_timestamp
    output_dir = f"{config.output_base}/acceptance_{timestamp}"

    success = True

    # Run acceptance test
    cmd = [
        sys.executable, "scripts/pulse_acceptance.py",
        "--run-type", "acceptance",
        "--output-dir", output_dir
    ]

    if not run_command(cmd, "Enhanced Acceptance Tests"):
        success = False

    # Run with statistical validation
    cmd = [
        sys.executable, "scripts/pulse_acceptance.py",
        "--run-type", "acceptance",
        "--output-dir", f"{output_dir}_stats"
    ]

    if not run_command(cmd, "Acceptance Tests with Statistical Validation"):
        success = False

    return success

def run_baseline_comparisons(config: ResearchConfig) -> bool:
    """Run baseline comparisons against legacy GNSS/PTP systems."""
    print(f"\n{'='*80}")
    print("BASELINE COMPARISONS - Legacy System Validation")
    print(f"{'='*80}")

    timestamp = config.run_timestamp
    output_dir = f"{config.output_base}/baseline_{timestamp}"

    success = True

    # Run baseline emulation
    cmd = [
        sys.executable, "scripts/pulse_acceptance.py",
        "--run-type", "baseline",
        "--output-dir", output_dir
    ]

    if not run_command(cmd, "Baseline System Comparison"):
        success = False

    # Run comparative Monte Carlo analysis
    cmd = [
        sys.executable, "scripts/run_mc.py",
        "--simulation-type", "all",
        "--config", "sim/configs/gnss_ptp.yaml",
        "--comparative-runs",
        "--n-workers", str(config.n_workers),
        "--output-dir", f"{config.results_base}/comparative_mc_{timestamp}"
    ]

    if not run_command(cmd, "Comparative Monte Carlo Analysis"):
        success = False

    return success

def run_ablation_studies(config: ResearchConfig) -> bool:
    """Run comprehensive ablation parameter studies."""
    print(f"\n{'='*80}")
    print("ABLATION STUDIES - Parameter Sensitivity Analysis")
    print(f"{'='*80}")

    timestamp = config.run_timestamp
    output_dir = f"{config.results_base}/ablations_{timestamp}"

    success = True

    # Run ablation sweeps
    cmd = [
        sys.executable, "scripts/ablation_sweeps.py",
        "--config", "sim/configs/ablations.yaml",
        "--output-dir", output_dir,
        "--n-monte-carlo", str(config.n_monte_carlo),
        "--base-seed", str(config.base_seed)
    ]

    if not run_command(cmd, "Parameter Ablation Sweeps"):
        success = False

    # Run enhanced Monte Carlo with ablation configs
    cmd = [
        sys.executable, "scripts/run_mc.py",
        "--simulation-type", "all",
        "--config", "sim/configs/ablations.yaml",
        "--ablation-config", "sim/configs/ablations.yaml",
        "--n-workers", str(config.n_workers),
        "--output-dir", f"{config.results_base}/mc_ablations_{timestamp}"
    ]

    if not run_command(cmd, "Monte Carlo Ablation Analysis"):
        success = False

    return success

def run_statistical_validation(config: ResearchConfig) -> bool:
    """Run comprehensive statistical validation and analysis."""
    print(f"\n{'='*80}")
    print("STATISTICAL VALIDATION - Confidence Intervals and Hypothesis Testing")
    print(f"{'='*80}")

    timestamp = config.run_timestamp
    output_dir = f"{config.output_base}/statistical_validation_{timestamp}"

    success = True

    # Run enhanced acceptance with stats
    cmd = [
        sys.executable, "scripts/pulse_acceptance.py",
        "--run-type", "acceptance",
        "--output-dir", f"{output_dir}/acceptance"
    ]

    if not run_command(cmd, "Statistical Validation - Acceptance"):
        success = False

    # Run baseline with stats
    cmd = [
        sys.executable, "scripts/pulse_acceptance.py",
        "--run-type", "baseline",
        "--output-dir", f"{output_dir}/baseline"
    ]

    if not run_command(cmd, "Statistical Validation - Baseline"):
        success = False

    # Run ablation with stats
    cmd = [
        sys.executable, "scripts/pulse_acceptance.py",
        "--run-type", "ablation",
        "--output-dir", f"{output_dir}/ablation"
    ]

    if not run_command(cmd, "Statistical Validation - Ablation"):
        success = False

    return success

def run_model_fidelity_checks(config: ResearchConfig) -> bool:
    """Run model fidelity validation against analytical predictions."""
    print(f"\n{'='*80}")
    print("MODEL FIDELITY VALIDATION - CRLB and Theoretical Validation")
    print(f"{'='*80}")

    timestamp = config.run_timestamp
    output_dir = f"{config.output_base}/fidelity_validation_{timestamp}"

    success = True

    # Run fidelity validation
    cmd = [
        sys.executable, "scripts/validate_fidelity.py",
        "--config", "sim/configs/hw_emulation.yaml",
        "--output-dir", output_dir
    ]

    if not run_command(cmd, "Model Fidelity Validation"):
        success = False

    # Run CRLB cross-validation
    cmd = [
        sys.executable, "scripts/validate_fidelity.py",
        "--config", "sim/configs/default.yaml",
        "--output-dir", f"{output_dir}/crlb_validation",
        "--crlb-only"
    ]

    if not run_command(cmd, "CRLB Cross-Validation"):
        success = False

    return success

def run_historical_alignment(config: ResearchConfig) -> bool:
    """Run historical data alignment and coupling analysis."""
    print(f"\n{'='*80}")
    print("HISTORICAL ALIGNMENT - Monte Carlo Coupling Analysis")
    print(f"{'='*80}")

    timestamp = config.run_timestamp
    output_dir = f"{config.output_base}/historical_alignment_{timestamp}"

    success = True

    # Find latest research outputs
    research_dirs = list(Path(config.output_base).glob("acceptance_*"))
    if not research_dirs:
        print("No recent acceptance test data found for historical alignment")
        return False

    latest_research = max(research_dirs, key=lambda x: x.stat().st_mtime)

    # Run historical alignment
    cmd = [
        sys.executable, "scripts/compare_historical.py",
        "--realtime-dir", str(latest_research),
        "--historical-dir", "results/mc_runs/extended_011",
        "--output-dir", output_dir
    ]

    if not run_command(cmd, "Historical Data Alignment Analysis"):
        success = False

    return success

def validate_results(config: ResearchConfig) -> Dict[str, Any]:
    """Validate research results against quality gates."""
    print(f"\n{'='*80}")
    print("VALIDATION - Quality Gate Assessment")
    print(f"{'='*80}")

    validation_results = {
        "quality_gates": {},
        "recommendations": [],
        "overall_pass": True
    }

    # Check for required output files
    timestamp = config.run_timestamp
    required_files = [
        f"{config.output_base}/acceptance_{timestamp}/digest.md",
        f"{config.output_base}/baseline_{timestamp}/digest.md",
        f"{config.results_base}/ablations_{timestamp}/ablation_summary.csv",
        f"{config.output_base}/fidelity_validation_{timestamp}/report.json",
        f"{config.output_base}/historical_alignment_{timestamp}/historical_alignment_report.md"
    ]

    files_present = 0
    for file_path in required_files:
        if Path(file_path).exists():
            files_present += 1
        else:
            validation_results["recommendations"].append(f"Missing required file: {file_path}")

    validation_results["quality_gates"]["required_files"] = {
        "present": files_present,
        "total": len(required_files),
        "pass": files_present == len(required_files)
    }

    # Load and validate statistical results
    try:
        stats_file = f"{config.output_base}/statistical_validation_{timestamp}/acceptance/digest.md"
        if Path(stats_file).exists():
            # Basic validation - check for statistical significance
            with open(stats_file, 'r') as f:
                content = f.read()
                if "p < 0.001" in content or "p < 0.05" in content:
                    validation_results["quality_gates"]["statistical_significance"] = True
                else:
                    validation_results["quality_gates"]["statistical_significance"] = False
                    validation_results["recommendations"].append("Statistical significance not clearly demonstrated")
    except Exception as e:
        validation_results["quality_gates"]["statistical_significance"] = False
        validation_results["recommendations"].append(f"Could not validate statistical results: {e}")

    # Load and validate fidelity results
    try:
        fidelity_file = f"{config.output_base}/fidelity_validation_{timestamp}/report.json"
        if Path(fidelity_file).exists():
            with open(fidelity_file, 'r') as f:
                fidelity_data = json.load(f)
                # Check CRLB ratios
                crlb_ratios = fidelity_data.get("crlb_ratios", {})
                if crlb_ratios:
                    max_ratio = max(crlb_ratios.values())
                    validation_results["quality_gates"]["model_fidelity"] = max_ratio <= config.fidelity_threshold
                    if max_ratio > config.fidelity_threshold:
                        validation_results["recommendations"].append(
                            f"Model fidelity exceeds threshold: {max_ratio:.2f}x CRLB (max allowed: {config.fidelity_threshold}x)"
                        )
    except Exception as e:
        validation_results["quality_gates"]["model_fidelity"] = False
        validation_results["recommendations"].append(f"Could not validate model fidelity: {e}")

    # Overall assessment
    all_gates_pass = all([
        validation_results["quality_gates"].get("required_files", {}).get("pass", False),
        validation_results["quality_gates"].get("statistical_significance", False),
        validation_results["quality_gates"].get("model_fidelity", False)
    ])

    validation_results["overall_pass"] = all_gates_pass

    if not all_gates_pass:
        validation_results["recommendations"].append("Not all quality gates passed - review recommendations above")

    return validation_results

def generate_final_report(config: ResearchConfig, validation_results: Dict[str, Any]) -> str:
    """Generate comprehensive final research report."""
    print(f"\n{'='*80}")
    print("GENERATING FINAL RESEARCH REPORT")
    print(f"{'='*80}")

    timestamp = config.run_timestamp
    report_dir = Path(f"{config.output_base}/research_suite_{timestamp}")
    report_dir.mkdir(parents=True, exist_ok=True)

    report_path = report_dir / "research_suite_report.md"

    with open(report_path, 'w') as f:
        f.write("# Driftlock Choir Research Suite Report\n\n")
        f.write(f"Generated: {datetime.now(timezone.utc).isoformat()}\n")
        f.write(f"Base Seed: {config.base_seed}\n")
        f.write(f"Monte Carlo Runs: {config.n_monte_carlo}\n")
        f.write(f"Workers: {config.n_workers}\n\n")

        f.write("## Executive Summary\n\n")

        if validation_results["overall_pass"]:
            f.write("✅ **All quality gates passed** - Research suite completed successfully.\n\n")
        else:
            f.write("⚠️ **Some quality gates failed** - Review recommendations below.\n\n")

        f.write("## Quality Gates Status\n\n")
        f.write("| Gate | Status | Details |\n")
        f.write("|------|--------|---------|\n")

        for gate_name, gate_data in validation_results["quality_gates"].items():
            if isinstance(gate_data, bool):
                status = "✅ Pass" if gate_data else "❌ Fail"
                details = ""
            elif isinstance(gate_data, dict):
                status = "✅ Pass" if gate_data.get("pass", False) else "❌ Fail"
                details = f"{gate_data.get('present', 0)}/{gate_data.get('total', 0)}"
            else:
                status = "❓ Unknown"
                details = str(gate_data)

            f.write(f"| {gate_name} | {status} | {details} |\n")

        f.write("\n## Component Results\n\n")

        components = [
            ("Acceptance Tests", "acceptance", f"{config.output_base}/acceptance_{timestamp}/digest.md"),
            ("Baseline Comparisons", "baseline", f"{config.output_base}/baseline_{timestamp}/digest.md"),
            ("Ablation Studies", "ablations", f"{config.results_base}/ablations_{timestamp}/ablation_summary.csv"),
            ("Statistical Validation", "validation", f"{config.output_base}/statistical_validation_{timestamp}/acceptance/digest.md"),
            ("Model Fidelity", "fidelity", f"{config.output_base}/fidelity_validation_{timestamp}/report.json"),
            ("Historical Alignment", "historical", f"{config.output_base}/historical_alignment_{timestamp}/historical_alignment_report.md")
        ]

        for component_name, component_key, path in components:
            if getattr(config, f"run_{component_key}", False):
                if Path(path).exists():
                    f.write(f"### {component_name}\n")
                    f.write(f"- **Status**: ✅ Completed\n")
                    f.write(f"- **Output**: `{path}`\n")
                    if component_key == "ablations":
                        f.write(f"- **Summary**: Available at `{path}`\n")
                    f.write("\n")
                else:
                    f.write(f"### {component_name}\n")
                    f.write(f"- **Status**: ❌ Failed or incomplete\n")
                    f.write(f"- **Expected**: `{path}`\n\n")

        f.write("## Recommendations\n\n")
        if validation_results["recommendations"]:
            for rec in validation_results["recommendations"]:
                f.write(f"- {rec}\n")
        else:
            f.write("No specific recommendations - all quality gates passed.\n")

        f.write("\n## Reproducibility\n\n")
        f.write("To reproduce this research suite:\n\n")
        f.write("```bash\n")
        f.write(f"python scripts/run_research_suite.py --all --timestamp {timestamp}\n")
        f.write("```\n\n")

        f.write("All experiments use deterministic seeding with base seed 2025.\n")
        f.write("See `docs/appendix_experiments.md` for detailed methodology.\n\n")

        f.write("## File Manifest\n\n")
        f.write("Key outputs from this research suite:\n\n")

        manifest_items = [
            f"{config.output_base}/acceptance_{timestamp}/digest.md",
            f"{config.output_base}/baseline_{timestamp}/digest.md",
            f"{config.results_base}/ablations_{timestamp}/ablation_summary.csv",
            f"{config.output_base}/statistical_validation_{timestamp}/",
            f"{config.output_base}/fidelity_validation_{timestamp}/report.json",
            f"{config.output_base}/historical_alignment_{timestamp}/historical_alignment_report.md",
            "docs/appendix_experiments.md",
            "docs/results_overview.md"
        ]

        for item in manifest_items:
            f.write(f"- `{item}`\n")

    print(f"Final report generated: {report_path}")
    return str(report_path)

def main():
    parser = argparse.ArgumentParser(
        description="Run comprehensive Driftlock Choir research suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/run_research_suite.py --all                    # Complete suite
  python scripts/run_research_suite.py --acceptance --baseline  # Core validation
  python scripts/run_research_suite.py --ablations             # Parameter studies
  python scripts/run_research_suite.py --validation --fidelity # Statistical checks
        """
    )

    parser.add_argument("--all", action="store_true",
                       help="Run complete research suite")
    parser.add_argument("--acceptance", action="store_true",
                       help="Run acceptance tests")
    parser.add_argument("--baseline", action="store_true",
                       help="Run baseline comparisons")
    parser.add_argument("--ablations", action="store_true",
                       help="Run ablation studies")
    parser.add_argument("--validation", action="store_true",
                       help="Run statistical validation")
    parser.add_argument("--fidelity", action="store_true",
                       help="Run model fidelity checks")
    parser.add_argument("--historical", action="store_true",
                       help="Run historical alignment analysis")

    parser.add_argument("--timestamp", type=str, default="",
                       help="Timestamp suffix for outputs (default: auto-generated)")
    parser.add_argument("--base-seed", type=int, default=2025,
                       help="Base random seed for reproducibility")
    parser.add_argument("--n-monte-carlo", type=int, default=20,
                       help="Number of Monte Carlo runs per configuration")
    parser.add_argument("--n-workers", type=int, default=4,
                       help="Number of parallel workers")
    parser.add_argument("--output-base", type=str, default="driftlock_choir_sim/outputs/research",
                       help="Base output directory")
    parser.add_argument("--no-validate", action="store_true",
                       help="Skip quality gate validation")
    parser.add_argument("--export-site-data", nargs="?", default="true", const="true", choices=["true", "false"],
                        help="Enable site data export after successful runs (default: true)")

    args = parser.parse_args()

    # Setup configuration
    timestamp = args.timestamp or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    config = ResearchConfig(
        base_seed=args.base_seed,
        output_base=args.output_base,
        n_monte_carlo=args.n_monte_carlo,
        n_workers=args.n_workers,
        run_timestamp=timestamp,
        export_site_data=args.export_site_data == "true"
    )

    # Set component flags
    if args.all:
        config.run_acceptance = True
        config.run_baseline = True
        config.run_ablations = True
        config.run_validation = True
        config.run_fidelity = True
        config.run_historical = True
    else:
        config.run_acceptance = args.acceptance
        config.run_baseline = args.baseline
        config.run_ablations = args.ablations
        config.run_validation = args.validation
        config.run_fidelity = args.fidelity
        config.run_historical = args.historical

    # Validate that at least one component is selected
    if not any([config.run_acceptance, config.run_baseline, config.run_ablations,
                config.run_validation, config.run_fidelity, config.run_historical]):
        print("Error: No components selected. Use --all or specify individual components.")
        return 1

    print("="*80)
    print("DRIFTLOCK CHOIR RESEARCH SUITE")
    print("="*80)
    print(f"Timestamp: {timestamp}")
    print(f"Base Seed: {config.base_seed}")
    print(f"Monte Carlo Runs: {config.n_monte_carlo}")
    print(f"Workers: {config.n_workers}")
    print("="*80)

    # Setup environment
    setup_environment()

    # Track results
    results = {
        "acceptance": False,
        "baseline": False,
        "ablations": False,
        "validation": False,
        "fidelity": False,
        "historical": False
    }

    # Run selected components
    if config.run_acceptance:
        results["acceptance"] = run_acceptance_tests(config)

    if config.run_baseline:
        results["baseline"] = run_baseline_comparisons(config)

    if config.run_ablations:
        results["ablations"] = run_ablation_studies(config)

    if config.run_validation:
        results["validation"] = run_statistical_validation(config)

    if config.run_fidelity:
        results["fidelity"] = run_model_fidelity_checks(config)

    if config.run_historical:
        results["historical"] = run_historical_alignment(config)

    # Validation
    if not args.no_validate:
        validation_results = validate_results(config)
    else:
        validation_results = {"overall_pass": True, "quality_gates": {}, "recommendations": []}
    # Site data export integration
    if config.export_site_data and validation_results["overall_pass"]:
        export_success = export_site_data(config)
        validation_results["quality_gates"]["site_data_export"] = {
            "status": "success" if export_success else "failed",
            "pass": export_success
        }
    else:
        reason = "disabled" if not config.export_site_data else "quality gates failed"
        validation_results["quality_gates"]["site_data_export"] = {
            "status": "skipped",
            "reason": reason,
            "pass": True
        }

    # Generate final report
    report_path = generate_final_report(config, validation_results)

    # Summary
    print(f"\n{'='*80}")
    print("RESEARCH SUITE EXECUTION COMPLETE")
    print(f"{'='*80}")

    completed = sum(results.values())
    total = len(results)
    print(f"Components completed: {completed}/{total}")

    for component, success in results.items():
        status = "✅" if success else "❌"
        print(f"  {status} {component}")

    if validation_results["overall_pass"]:
        print("✅ All quality gates passed")
    else:
        print("⚠️ Some quality gates failed - check recommendations")

    print(f"\n📊 Final report: {report_path}")
    print(f"📈 All outputs: {config.output_base}/research_suite_{timestamp}/")

    # Exit code
    if validation_results["overall_pass"]:
        print("🎉 Research suite completed successfully!")
        return 0
    else:
        print("⚠️ Research suite completed with warnings.")
        return 1

if __name__ == "__main__":
    exit(main())