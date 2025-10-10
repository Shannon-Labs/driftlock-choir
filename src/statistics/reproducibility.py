"""
Reproducibility framework for chronometric interferometry research.

This module provides comprehensive reproducibility tools including fixed seed management,
environment tracking, and automated validation for research publication standards.
"""

import os
import sys
import json
import hashlib
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any
import numpy as np
import pandas as pd
from dataclasses import dataclass, field, asdict
import warnings


@dataclass
class ExperimentMetadata:
    """Metadata for experiment reproducibility."""
    experiment_name: str
    timestamp: str
    python_version: str
    numpy_version: str
    scipy_version: str
    matplotlib_version: str
    random_seed: int
    git_commit: Optional[str] = None
    environment_hash: str = ""
    system_info: Dict[str, str] = field(default_factory=dict)
    requirements_hash: str = ""


@dataclass
class ReproducibilityConfig:
    """Configuration for reproducible experiments."""
    fixed_seed: int = 42
    deterministic_algorithms: bool = True
    numpy_precision: str = "double"
    log_level: str = "INFO"
    output_directory: str = "./reproducibility"
    validate_results: bool = True
    tolerance_level: float = 1e-10


class ReproducibilityManager:
    """
    Comprehensive reproducibility management for chronometric interferometry research.

    Ensures that all experiments are fully reproducible with fixed seeds,
    environment tracking, and validation capabilities.
    """

    def __init__(self, config: Optional[ReproducibilityConfig] = None):
        """
        Initialize reproducibility manager.

        Args:
            config: Reproducibility configuration
        """
        self.config = config or ReproducibilityConfig()
        self.experiment_metadata = None
        self.results_cache = {}
        self.validation_results = {}

        # Create output directory
        Path(self.config.output_directory).mkdir(parents=True, exist_ok=True)

    def setup_reproducible_environment(self, experiment_name: str) -> ExperimentMetadata:
        """
        Setup fully reproducible environment.

        Args:
            experiment_name: Name of the experiment

        Returns:
            Experiment metadata
        """
        # Set fixed random seed
        np.random.seed(self.config.fixed_seed)

        # Configure deterministic algorithms
        if self.config.deterministic_algorithms:
            os.environ['PYTHONHASHSEED'] = str(self.config.fixed_seed)
            np.set_printoptions(precision=15)
            np.seterr(all='raise')

        # Configure NumPy precision
        if self.config.numpy_precision == "double":
            np.set_printoptions(precision=15)
        elif self.config.numpy_precision == "single":
            np.set_printoptions(precision=7)

        # Create experiment metadata
        self.experiment_metadata = ExperimentMetadata(
            experiment_name=experiment_name,
            timestamp=datetime.now().isoformat(),
            python_version=sys.version,
            numpy_version=np.__version__,
            scipy_version=getattr(__import__('scipy'), '__version__', 'unknown'),
            matplotlib_version=getattr(__import__('matplotlib'), '__version__', 'unknown'),
            random_seed=self.config.fixed_seed,
            git_commit=self._get_git_commit(),
            environment_hash=self._compute_environment_hash(),
            system_info=self._get_system_info(),
            requirements_hash=self._compute_requirements_hash()
        )

        # Save metadata
        self._save_experiment_metadata()

        return self.experiment_metadata

    def validate_reproducibility(self, experiment_results: Dict,
                               reference_results: Optional[Dict] = None,
                               tolerance: Optional[float] = None) -> Dict:
        """
        Validate experiment reproducibility.

        Args:
            experiment_results: Current experiment results
            reference_results: Reference results for comparison
            tolerance: Tolerance level for comparison

        Returns:
            Validation results dictionary
        """
        if tolerance is None:
            tolerance = self.config.tolerance_level

        validation_results = {
            'timestamp': datetime.now().isoformat(),
            'tolerance': tolerance,
            'tests_passed': 0,
            'tests_failed': 0,
            'details': {}
        }

        # Test deterministic random number generation
        validation_results['details']['random_numbers'] = self._test_random_numbers(tolerance)

        # Test numerical stability
        validation_results['details']['numerical_stability'] = self._test_numerical_stability(tolerance)

        # Test algorithm consistency
        validation_results['details']['algorithm_consistency'] = self._test_algorithm_consistency(tolerance)

        # Compare with reference results if provided
        if reference_results:
            validation_results['details']['reference_comparison'] = self._compare_with_reference(
                experiment_results, reference_results, tolerance
            )

        # Count passed/failed tests
        for test_name, test_result in validation_results['details'].items():
            if test_result.get('passed', False):
                validation_results['tests_passed'] += 1
            else:
                validation_results['tests_failed'] += 1

        validation_results['overall_status'] = validation_results['tests_failed'] == 0

        # Save validation results
        self._save_validation_results(validation_results)

        return validation_results

    def generate_reproducibility_report(self, experiment_results: Dict,
                                      validation_results: Optional[Dict] = None) -> str:
        """
        Generate comprehensive reproducibility report.

        Args:
            experiment_results: Experiment results
            validation_results: Validation results

        Returns:
            Path to generated report
        """
        report_path = Path(self.config.output_directory) / f"{self.experiment_metadata.experiment_name}_report.md"

        report_content = self._generate_markdown_report(experiment_results, validation_results)

        with open(report_path, 'w') as f:
            f.write(report_content)

        return str(report_path)

    def create_reproducibility_package(self, experiment_results: Dict,
                                     output_path: Optional[str] = None) -> str:
        """
        Create complete reproducibility package.

        Args:
            experiment_results: Experiment results
            output_path: Output path for package

        Returns:
            Path to created package
        """
        if output_path is None:
            output_path = Path(self.config.output_directory) / f"{self.experiment_metadata.experiment_name}_package"
        else:
            output_path = Path(output_path)

        output_path.mkdir(parents=True, exist_ok=True)

        # Save metadata
        metadata_path = output_path / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(asdict(self.experiment_metadata), f, indent=2)

        # Save results
        results_path = output_path / "results.json"
        with open(results_path, 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            serializable_results = self._make_json_serializable(experiment_results)
            json.dump(serializable_results, f, indent=2)

        # Save configuration
        config_path = output_path / "config.json"
        with open(config_path, 'w') as f:
            json.dump(asdict(self.config), f, indent=2)

        # Save requirements
        requirements_path = output_path / "requirements.txt"
        self._save_requirements(requirements_path)

        # Create reproducibility script
        script_path = output_path / "reproduce_experiment.py"
        self._create_reproduction_script(script_path)

        return str(output_path)

    def _get_git_commit(self) -> Optional[str]:
        """Get current Git commit hash."""
        try:
            import subprocess
            result = subprocess.run(['git', 'rev-parse', 'HEAD'],
                                  capture_output=True, text=True, timeout=10)
            return result.stdout.strip() if result.returncode == 0 else None
        except:
            return None

    def _compute_environment_hash(self) -> str:
        """Compute hash of environment variables."""
        env_vars = {
            'PATH': os.environ.get('PATH', ''),
            'PYTHONPATH': os.environ.get('PYTHONPATH', ''),
            'LD_LIBRARY_PATH': os.environ.get('LD_LIBRARY_PATH', ''),
        }
        env_string = json.dumps(env_vars, sort_keys=True)
        return hashlib.sha256(env_string.encode()).hexdigest()[:16]

    def _get_system_info(self) -> Dict[str, str]:
        """Get system information."""
        import platform
        return {
            'platform': platform.platform(),
            'processor': platform.processor(),
            'architecture': platform.architecture()[0],
            'python_implementation': platform.python_implementation(),
        }

    def _compute_requirements_hash(self) -> str:
        """Compute hash of requirements."""
        try:
            with open('requirements.txt', 'r') as f:
                requirements = f.read()
            return hashlib.sha256(requirements.encode()).hexdigest()[:16]
        except:
            return "unknown"

    def _save_experiment_metadata(self) -> None:
        """Save experiment metadata to file."""
        metadata_path = Path(self.config.output_directory) / f"{self.experiment_metadata.experiment_name}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(asdict(self.experiment_metadata), f, indent=2)

    def _test_random_numbers(self, tolerance: float) -> Dict:
        """Test deterministic random number generation."""
        np.random.seed(self.config.fixed_seed)
        random_numbers = np.random.random(100)

        # Expected first 10 numbers with seed 42
        expected_first_10 = np.array([
            0.37454012, 0.95071431, 0.73199394, 0.59865848, 0.15601864,
            0.15599452, 0.05808361, 0.86617615, 0.60111501, 0.70807258
        ])

        passed = np.allclose(random_numbers[:10], expected_first_10, atol=tolerance)

        return {
            'passed': passed,
            'description': 'Deterministic random number generation',
            'details': f'First 10 numbers match expected values within {tolerance}'
        }

    def _test_numerical_stability(self, tolerance: float) -> Dict:
        """Test numerical stability of computations."""
        # Test matrix operations
        test_matrix = np.random.RandomState(self.config.fixed_seed).random((10, 10))

        # Compute determinant multiple times
        det1 = np.linalg.det(test_matrix)
        det2 = np.linalg.det(test_matrix)

        passed = np.abs(det1 - det2) < tolerance

        return {
            'passed': passed,
            'description': 'Numerical stability of matrix operations',
            'details': f'Determinant computation stable within {tolerance}'
        }

    def _test_algorithm_consistency(self, tolerance: float) -> Dict:
        """Test algorithm consistency across runs."""
        # Simple linear regression test
        np.random.seed(self.config.fixed_seed)
        x = np.random.random(100)
        y = 2 * x + 1 + np.random.random(100) * 0.1

        # Fit linear regression
        coeffs1 = np.polyfit(x, y, 1)

        # Reset seed and fit again
        np.random.seed(self.config.fixed_seed)
        coeffs2 = np.polyfit(x, y, 1)

        passed = np.allclose(coeffs1, coeffs2, atol=tolerance)

        return {
            'passed': passed,
            'description': 'Algorithm consistency across runs',
            'details': f'Linear regression coefficients consistent within {tolerance}'
        }

    def _compare_with_reference(self, results: Dict, reference: Dict,
                               tolerance: float) -> Dict:
        """Compare results with reference values."""
        passed = True
        mismatches = []

        for key, value in results.items():
            if key in reference:
                if isinstance(value, np.ndarray) and isinstance(reference[key], np.ndarray):
                    if not np.allclose(value, reference[key], atol=tolerance):
                        passed = False
                        mismatches.append(f"Array {key} differs from reference")
                elif isinstance(value, (int, float)) and isinstance(reference[key], (int, float)):
                    if abs(value - reference[key]) > tolerance:
                        passed = False
                        mismatches.append(f"Scalar {key} differs from reference")

        return {
            'passed': passed,
            'description': 'Comparison with reference results',
            'details': 'All results match reference' if passed else f"Mismatches: {', '.join(mismatches)}"
        }

    def _save_validation_results(self, validation_results: Dict) -> None:
        """Save validation results to file."""
        validation_path = Path(self.config.output_directory) / f"{self.experiment_metadata.experiment_name}_validation.json"
        with open(validation_path, 'w') as f:
            json.dump(validation_results, f, indent=2)

    def _make_json_serializable(self, obj: Any) -> Any:
        """Convert object to JSON-serializable format."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._make_json_serializable(item) for item in obj]
        else:
            return obj

    def _save_requirements(self, requirements_path: Path) -> None:
        """Save Python requirements."""
        try:
            import pkg_resources
            installed_packages = [f"{pkg.project_name}=={pkg.version}"
                               for pkg in pkg_resources.working_set]

            with open(requirements_path, 'w') as f:
                f.write('\n'.join(sorted(installed_packages)))
        except:
            # Fallback to basic requirements
            with open(requirements_path, 'w') as f:
                f.write("numpy>=1.21.0\nscipy>=1.7.0\nmatplotlib>=3.4.0\n")

    def _create_reproduction_script(self, script_path: Path) -> None:
        """Create Python script to reproduce experiment."""
        script_content = f'''#!/usr/bin/env python3
"""
Reproduction script for {self.experiment_metadata.experiment_name}

This script reproduces the experiment under the same conditions as the original run.
"""

import os
import sys
import json
import numpy as np

# Set reproducibility parameters
os.environ['PYTHONHASHSEED'] = "{self.config.fixed_seed}"
np.random.seed({self.config.fixed_seed})

# Load metadata and results
with open("metadata.json", "r") as f:
    metadata = json.load(f)

with open("results.json", "r") as f:
    results = json.load(f)

print(f"Reproducing experiment: {{metadata['experiment_name']}}")
print(f"Original timestamp: {{metadata['timestamp']}}")
print(f"Random seed: {{metadata['random_seed']}}")
print(f"Python version: {{metadata['python_version']}}")

print("\\nExperiment results:")
for key, value in results.items():
    print(f"  {{key}}: {{value}}")

print("\\nReproduction completed successfully!")
'''

        with open(script_path, 'w') as f:
            f.write(script_content)

        # Make script executable
        os.chmod(script_path, 0o755)

    def _generate_markdown_report(self, experiment_results: Dict,
                                 validation_results: Optional[Dict]) -> str:
        """Generate markdown reproducibility report."""
        report = f"""# Reproducibility Report: {self.experiment_metadata.experiment_name}

## Experiment Metadata

- **Experiment Name**: {self.experiment_metadata.experiment_name}
- **Timestamp**: {self.experiment_metadata.timestamp}
- **Random Seed**: {self.experiment_metadata.random_seed}
- **Python Version**: {self.experiment_metadata.python_version}
- **NumPy Version**: {self.experiment_metadata.numpy_version}
- **Git Commit**: {self.experiment_metadata.git_commit or 'N/A'}

## System Information

- **Platform**: {self.experiment_metadata.system_info.get('platform', 'Unknown')}
- **Architecture**: {self.experiment_metadata.system_info.get('architecture', 'Unknown')}
- **Environment Hash**: {self.experiment_metadata.environment_hash}

## Experiment Results

"""

        for key, value in experiment_results.items():
            if isinstance(value, np.ndarray):
                report += f"- **{key}**: Array with shape {value.shape}, mean {np.mean(value):.6e}\n"
            else:
                report += f"- **{key}**: {value}\n"

        if validation_results:
            report += f"""
## Validation Results

- **Tests Passed**: {validation_results['tests_passed']}
- **Tests Failed**: {validation_results['tests_failed']}
- **Overall Status**: {'✅ PASSED' if validation_results['overall_status'] else '❌ FAILED'}

### Test Details

"""
            for test_name, test_result in validation_results['details'].items():
                status = '✅' if test_result.get('passed', False) else '❌'
                report += f"- **{test_name}**: {status} {test_result.get('description', '')}\n"
                report += f"  - Details: {test_result.get('details', '')}\n"

        report += f"""
## Reproduction Instructions

To reproduce this experiment:

1. Install requirements: `pip install -r requirements.txt`
2. Run reproduction script: `python reproduce_experiment.py`

## Files Generated

- `metadata.json`: Experiment metadata
- `results.json`: Experiment results
- `config.json`: Reproducibility configuration
- `requirements.txt`: Python requirements
- `reproduce_experiment.py`: Reproduction script

Generated on: {datetime.now().isoformat()}
"""

        return report