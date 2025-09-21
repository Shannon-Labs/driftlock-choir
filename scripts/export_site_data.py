"""Export research results to JSON files for Next.js site consumption.

This script parses outputs from the research suite (e.g., digest.md, JSON reports, CSV summaries)
in a specified or latest run directory under results/, extracts structured data for key metrics,
and generates JSON files in docs/site_data/. It supports CLI options for specific timestamps or latest run.

Designed to be called post-run_research_suite.py execution.
"""

import argparse
import csv
import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml  # Assuming YAML frontmatter in digest.md; install if needed via requirements.txt


def find_latest_run(results_dir: str = "results") -> Optional[str]:
    """Find the most recent run directory based on timestamp in folder name.

    Assumes run directories are named like '2025-09-21_07-05-04' under results/.

    Args:
        results_dir (str): Base directory for runs.

    Returns:
        Optional[str]: Full path to latest run dir, or None if none found.
    """
    run_dirs = [
        d for d in os.listdir(results_dir)
        if os.path.isdir(os.path.join(results_dir, d)) and re.match(r"\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}", d)
    ]
    if not run_dirs:
        return None
    # Parse timestamps and find max
    timestamps = [
        datetime.strptime(d, "%Y-%m-%d_%H-%M-%S") for d in run_dirs
    ]
    latest_idx = timestamps.index(max(timestamps))
    return os.path.join(results_dir, run_dirs[latest_idx])


def parse_digest_md(digest_path: str) -> Dict[str, Any]:
    """Parse digest.md for key summaries like acceptance rates, baselines.

    Assumes Markdown with YAML frontmatter for structured data, or ## sections with key: value lines.

    Args:
        digest_path (str): Path to digest.md.

    Returns:
        Dict[str, Any]: Extracted data, e.g., {'acceptance': 0.95, 'baseline_rmse': 0.1}.

    Raises:
        ValueError: If parsing fails or file not found.
    """
    if not os.path.exists(digest_path):
        raise ValueError(f"digest.md not found at {digest_path}")

    with open(digest_path, "r", encoding="utf-8") as f:
        content = f.read()

    data: Dict[str, Any] = {}
    # Try YAML frontmatter first
    yaml_match = re.match(r"---\n(.*?)\n---", content, re.DOTALL)
    if yaml_match:
        try:
            data.update(yaml.safe_load(yaml_match.group(1)))
        except yaml.YAMLError as e:
            raise ValueError(f"YAML parsing error in digest.md: {e}")

    # Fallback: Parse ## sections for key: value
    sections = re.split(r"\n#{2,}\s+([^#\n]+)", content)
    for i in range(1, len(sections), 2):
        section_title = sections[i].strip().lower().replace(" ", "_")
        section_content = sections[i + 1].strip()
        # Look for lines like "Key: value"
        for line in section_content.split("\n"):
            kv_match = re.match(r"(\w+):\s*(.+)", line.strip())
            if kv_match:
                data[f"{section_title}_{kv_match.group(1).lower()}"] = kv_match.group(2).strip()

    # Validate required keys (customize as needed)
    required = ["acceptance", "baseline"]
    missing = [k for k in required if k not in data]
    if missing:
        raise ValueError(f"Missing keys in digest.md: {missing}")

    return data


def extract_from_json(json_path: str, keys: List[str]) -> Dict[str, Any]:
    """Safely extract specific keys from a JSON file.

    Args:
        json_path (str): Path to JSON file.
        keys (List[str]): Keys to extract.

    Returns:
        Dict[str, Any]: Extracted values, with missing as None.

    Raises:
        ValueError: If file not found or invalid JSON.
    """
    if not os.path.exists(json_path):
        raise ValueError(f"JSON file not found at {json_path}")

    try:
        with open(json_path, "r") as f:
            data = json.load(f)
        return {k: data.get(k) for k in keys}
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON at {json_path}: {e}")


def extract_from_csv(csv_path: str, metrics: List[str]) -> Dict[str, List[float]]:
    """Extract mean/aggregate metrics from CSV summary (assumes columns like 'rmse', rows as trials).

    Args:
        csv_path (str): Path to CSV.
        metrics (List[str]): Column names to aggregate (mean).

    Returns:
        Dict[str, List[float]]: Aggregated values per metric.

    Raises:
        ValueError: If file issues.
    """
    if not os.path.exists(csv_path):
        raise ValueError(f"CSV not found at {csv_path}")

    data: Dict[str, List[float]] = {m: [] for m in metrics}
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for m in metrics:
                if m in row and row[m]:
                    try:
                        data[m].append(float(row[m]))
                    except ValueError:
                        pass  # Skip invalid

    # Compute means if lists non-empty
    result = {}
    for m, vals in data.items():
        if vals:
            result[m] = {"mean": sum(vals) / len(vals), "ci": 1.96 * (sum(v**2 for v in vals)/len(vals) - (sum(vals)/len(vals))**2)**0.5 / len(vals)**0.5}  # Simple 95% CI approx
        else:
            result[m] = {"mean": None, "ci": None}

    return result


def extract_performance_metrics(run_dir: str) -> Dict[str, Any]:
    """Extract performance metrics (RMSE, CRLB ratios, CIs, gains) from phase JSONs/CSVs.

    Assumes structure: run_dir/phase1/metrics.json, phase2/summary.csv, etc.

    Args:
        run_dir (str): Path to run directory.

    Returns:
        Dict[str, Any]: Compiled metrics.
    """
    perf: Dict[str, Any] = {}

    # Phase 1: Chronometric handshake
    try:
        phase1_json = os.path.join(run_dir, "phase1", "metrics.json")
        perf["phase1"] = extract_from_json(phase1_json, ["rmse", "crlb_ratio", "confidence_interval"])
        # Gains vs baseline (assume baseline in config or separate file)
        baseline_rmse = 0.15  # Placeholder; parse from baseline_comparison.yaml or digest
        if "rmse" in perf["phase1"]:
            perf["phase1"]["gain_vs_baseline"] = (baseline_rmse - perf["phase1"]["rmse"]) / baseline_rmse
    except ValueError:
        perf["phase1"] = {"error": "Metrics not available"}

    # Phase 2: Consensus
    try:
        phase2_csv = os.path.join(run_dir, "phase2", "summary.csv")
        perf["phase2"] = extract_from_csv(phase2_csv, ["consensus_time", "variance_reduction"])
    except ValueError:
        perf["phase2"] = {"error": "Metrics not available"}

    # Phase 3: Full sim
    try:
        phase3_json = os.path.join(run_dir, "phase3", "performance.json")
        perf["phase3"] = extract_from_json(phase3_json, ["overall_rmse", "crlb_achieved"])
    except ValueError:
        perf["phase3"] = {"error": "Metrics not available"}

    return perf


def extract_statistical_validation(run_dir: str) -> Dict[str, Any]:
    """Extract p-values, effect sizes, significance from stats files.

    Assumes run_dir/stats/validation.json with tests.

    Args:
        run_dir (str): Path to run directory.

    Returns:
        Dict[str, Any]: Validation data.
    """
    try:
        stats_path = os.path.join(run_dir, "stats", "validation.json")
        keys = ["p_value_acceptance", "effect_size_baseline", "significance_tests"]
        return extract_from_json(stats_path, keys)
    except ValueError:
        return {"error": "Statistical data not available"}


def extract_ablation_studies(run_dir: str) -> Dict[str, Any]:
    """Extract ablation results: sensitivity, best configs from sweeps.

    Assumes run_dir/ablations/results.json.

    Args:
        run_dir (str): Path to run directory.

    Returns:
        Dict[str, Any]: Ablation data.
    """
    try:
        ablations_path = os.path.join(run_dir, "ablations", "results.json")
        keys = ["parameter_sensitivity", "best_config", "sweep_summary"]
        return extract_from_json(ablations_path, keys)
    except ValueError:
        return {"error": "Ablation data not available"}


def extract_model_fidelity(run_dir: str) -> Dict[str, Any]:
    """Extract fidelity: CRLB validation, discrepancies from fidelity checks.

    Assumes run_dir/fidelity/validation.json.

    Args:
        run_dir (str): Path to run directory.

    Returns:
        Dict[str, Any]: Fidelity data.
    """
    try:
        fidelity_path = os.path.join(run_dir, "fidelity", "validation.json")
        keys = ["crlb_validation", "discrepancies", "status"]
        return extract_from_json(fidelity_path, keys)
    except ValueError:
        return {"error": "Fidelity data not available"}


def extract_historical_alignment(run_dir: str) -> Dict[str, Any]:
    """Extract historical alignment: coupling metrics, quality from comparisons.

    Assumes run_dir/historical/alignment.json.

    Args:
        run_dir (str): Path to run directory.

    Returns:
        Dict[str, Any]: Alignment data.
    """
    try:
        hist_path = os.path.join(run_dir, "historical", "alignment.json")
        keys = ["coupling_metrics", "alignment_quality", "drift_history"]
        return extract_from_json(hist_path, keys)
    except ValueError:
        return {"error": "Historical data not available"}


def generate_artifacts_manifest(run_dir: str) -> Dict[str, List[str]]:
    """Generate manifest of artifacts: plots, reports, data files.

    Uses glob for PNG, PDF, JSON, CSV, MD in run_dir.

    Args:
        run_dir (str): Path to run directory.

    Returns:
        Dict[str, List[str]]: Relative paths grouped by type.
    """
    manifest: Dict[str, List[str]] = {"plots": [], "reports": [], "data": []}
    base = Path(run_dir)

    # Plots: PNG, SVG, PDF
    for ext in ["*.png", "*.svg", "*.pdf"]:
        manifest["plots"].extend([
            str(p.relative_to(base)) for p in base.glob(f"**/{ext}")
        ])

    # Reports: MD, JSON summaries
    for ext in ["*.md", "*.json"]:
        if ext == "*.json" and "metrics" in ext:  # Avoid data JSons
            continue
        manifest["reports"].extend([
            str(p.relative_to(base)) for p in base.glob(f"**/{ext}")
        ])

    # Data: CSV
    manifest["data"].extend([
        str(p.relative_to(base)) for p in base.glob("**/ *.csv")
    ])

    return manifest


def extract_research_metadata(run_dir: str, digest_data: Dict[str, Any]) -> Dict[str, Any]:
    """Extract metadata: timestamps, seeds, config, quality gates.

    Combines dir timestamp, digest info, config.yaml.

    Args:
        run_dir (str): Path to run directory.
        digest_data (Dict[str, Any]): From digest.md.

    Returns:
        Dict[str, Any]: Metadata.
    """
    # Timestamp from dir name
    dir_name = os.path.basename(run_dir)
    timestamp = datetime.strptime(dir_name, "%Y-%m-%d_%H-%M-%S").isoformat() if re.match(r"\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}", dir_name) else None

    # Config
    config_path = os.path.join(run_dir, "config.yaml")
    config = {}
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

    # Seeds, quality from digest or config
    metadata = {
        "run_timestamp": timestamp,
        "random_seeds": digest_data.get("seeds", config.get("seeds", [])),
        "configuration": config,
        "quality_gates": digest_data.get("quality_gates", {"passed": True}),
        "run_id": dir_name
    }

    # Validate
    if not timestamp:
        raise ValueError("Invalid run directory timestamp")

    return metadata


def export_to_site_data(run_dir: str, site_data_dir: Path) -> None:
    """Main extraction and export logic.

    Parses all sources and writes JSONs.

    Args:
        run_dir (str): Run directory.
        site_data_dir (Path): Output directory.

    Raises:
        ValueError: On extraction failures.
    """
    site_data_dir.mkdir(parents=True, exist_ok=True)

    # Parse digest for common data
    digest_path = os.path.join(run_dir, "digest.md")
    digest_data = parse_digest_md(digest_path)

    # Extract sections
    performance = extract_performance_metrics(run_dir)
    statistical = extract_statistical_validation(run_dir)
    ablations = extract_ablation_studies(run_dir)
    fidelity = extract_model_fidelity(run_dir)
    historical = extract_historical_alignment(run_dir)
    manifest = generate_artifacts_manifest(run_dir)
    metadata = extract_research_metadata(run_dir, digest_data)

    # Write JSONs
    json_files = {
        "performance_metrics.json": performance,
        "statistical_validation.json": statistical,
        "ablation_studies.json": ablations,
        "model_fidelity.json": fidelity,
        "historical_alignment.json": historical,
        "artifacts_manifest.json": manifest,
        "research_metadata.json": metadata
    }

    for fname, data in json_files.items():
        filepath = site_data_dir / fname
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2, default=str)  # default=str for datetimes

    # Validate outputs (basic: keys present, no None in critical)
    for fname, data in json_files.items():
        if "error" in str(data).lower():
            raise ValueError(f"Incomplete data in {fname}")


def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(
        description="Export research results to site data JSONs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument("--timestamp", type=str, help="Specific run timestamp (YYYY-MM-DD_HH-MM-SS)")
    group.add_argument("--latest", action="store_true", default=True, help="Use latest run (default)")

    args = parser.parse_args()

    results_dir = "results"
    if not os.path.exists(results_dir):
        raise ValueError(f"Results directory '{results_dir}' not found. Run research suite first.")

    if args.timestamp:
        run_dir = os.path.join(results_dir, args.timestamp)
    else:
        run_dir = find_latest_run(results_dir)

    if not run_dir or not os.path.exists(run_dir):
        raise ValueError(f"Run directory not found: {run_dir}")

    site_data_dir = Path("docs/site_data")
    try:
        export_to_site_data(run_dir, site_data_dir)
        print(f"Successfully exported data from {run_dir} to {site_data_dir}")
    except ValueError as e:
        print(f"Export failed: {e}", file=os.sys.stderr)
        raise


if __name__ == "__main__":
    main()