"""
Comprehensive uncertainty budget analysis for chronometric interferometry.

This module provides detailed uncertainty budget analysis following ISO GUM standards,
with specific focus on timing and frequency measurement uncertainties.
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Tuple, Optional, Union, Any
import warnings
from dataclasses import dataclass, field
from enum import Enum

from .uncertainty_analysis import UncertaintyComponent, UncertaintyBudget, UncertaintyType, DistributionType


class ErrorSourceType(Enum):
    """Types of error sources in chronometric interferometry."""
    THERMAL_NOISE = "thermal_noise"
    PHASE_NOISE = "phase_noise"
    QUANTIZATION = "quantization"
    CLOCK_JITTER = "clock_jitter"
    MULTIPATH = "multipath"
    CALIBRATION = "calibration"
    MODELING = "modeling"
    ENVIRONMENTAL = "environmental"
    SAMPLING = "sampling"
    ALGORITHMIC = "algorithmic"


@dataclass
class ErrorSource:
    """Individual error source with full characterization."""
    name: str
    source_type: ErrorSourceType
    nominal_value: float
    uncertainty: float
    distribution: DistributionType
    sensitivity_coefficient: float = 1.0
    degrees_of_freedom: Optional[int] = None
    correlation_group: Optional[str] = None
    category: UncertaintyType = UncertaintyType.TYPE_A
    mitigation_applied: bool = False
    mitigation_factor: float = 1.0


class UncertaintyBudgetAnalyzer:
    """
    Comprehensive uncertainty budget analysis for chronometric interferometry.

    Provides detailed uncertainty budget analysis with error source identification,
    sensitivity analysis, and optimization recommendations.
    """

    def __init__(self, confidence_level: float = 0.95):
        """
        Initialize uncertainty budget analyzer.

        Args:
            confidence_level: Confidence level for uncertainty analysis
        """
        self.confidence_level = confidence_level
        self.coverage_factor = stats.norm.ppf(0.5 + confidence_level / 2)

    def create_chronometric_interferometry_budget(self,
                                                 measurement_results: Dict[str, np.ndarray],
                                                 experimental_conditions: Dict) -> UncertaintyBudget:
        """
        Create comprehensive uncertainty budget for chronometric interferometry measurement.

        Args:
            measurement_results: Dictionary with measurement arrays
            experimental_conditions: Experimental setup parameters

        Returns:
            Complete uncertainty budget
        """
        components = []

        # Type A uncertainties from measurements
        for param_name, measurements in measurement_results.items():
            component = self._analyze_type_a_component(param_name, measurements)
            components.append(component)

        # Type B uncertainties - instrument and method
        type_b_components = self._analyze_type_b_components(experimental_conditions)
        components.extend(type_b_components)

        # Calculate correlation matrix
        correlation_matrix = self._calculate_correlation_matrix(components)

        # Combine uncertainties
        return self._combine_uncertainty_components(components, correlation_matrix)

    def _analyze_type_a_component(self, param_name: str, measurements: np.ndarray) -> UncertaintyComponent:
        """Analyze Type A uncertainty component."""
        n = len(measurements)
        mean_value = np.mean(measurements)
        std_dev = np.std(measurements, ddof=1)
        std_error = std_dev / np.sqrt(n)

        # Check for normality
        if n >= 8:
            _, p_value = stats.shapiro(measurements)
            if p_value < 0.05:
                warnings.warn(f"{param_name} may not be normally distributed (p={p_value:.3f})")

        return UncertaintyComponent(
            name=f"Type A - {param_name}",
            value=mean_value,
            uncertainty=std_error,
            distribution=DistributionType.GAUSSIAN,
            degrees_of_freedom=n - 1,
            coverage_factor=self.coverage_factor,
            category=UncertaintyType.TYPE_A
        )

    def _analyze_type_b_components(self, experimental_conditions: Dict) -> List[UncertaintyComponent]:
        """Analyze Type B uncertainty components."""
        components = []

        # Thermal noise uncertainty
        if 'noise_figure' in experimental_conditions and 'bandwidth' in experimental_conditions:
            nf_db = experimental_conditions['noise_figure']
            bandwidth_hz = experimental_conditions['bandwidth']
            temp_kelvin = experimental_conditions.get('temperature', 290)

            # Calculate thermal noise power
            k = 1.380649e-23  # Boltzmann constant
            noise_power = k * temp_kelvin * bandwidth_hz * 10**(nf_db/10)
            noise_uncertainty = noise_power * 0.1  # 10% uncertainty

            components.append(UncertaintyComponent(
                name="Type B - Thermal Noise",
                value=noise_power,
                uncertainty=noise_uncertainty,
                distribution=DistributionType.GAUSSIAN,
                category=UncertaintyType.TYPE_B
            ))

        # Oscillator phase noise
        if 'phase_noise' in experimental_conditions:
            phase_noise_dbc = experimental_conditions['phase_noise']
            phase_noise_linear = 10**(phase_noise_dbc/10)
            phase_uncertainty = phase_noise_linear * 0.05  # 5% uncertainty

            components.append(UncertaintyComponent(
                name="Type B - Phase Noise",
                value=phase_noise_linear,
                uncertainty=phase_uncertainty,
                distribution=DistributionType.GAUSSIAN,
                category=UncertaintyType.TYPE_B
            ))

        # Sampling clock jitter
        if 'clock_jitter' in experimental_conditions:
            jitter_ps = experimental_conditions['clock_jitter']
            jitter_uncertainty = jitter_ps * 0.02  # 2% uncertainty

            components.append(UncertaintyComponent(
                name="Type B - Clock Jitter",
                value=jitter_ps,
                uncertainty=jitter_uncertainty,
                distribution=DistributionType.UNIFORM,
                category=UncertaintyType.TYPE_B
            ))

        # ADC quantization
        if 'adc_bits' in experimental_conditions and 'v_range' in experimental_conditions:
            adc_bits = experimental_conditions['adc_bits']
            v_range = experimental_conditions['v_range']
            lsb = v_range / (2**adc_bits)
            quant_uncertainty = lsb / np.sqrt(12)  # RMS quantization noise

            components.append(UncertaintyComponent(
                name="Type B - ADC Quantization",
                value=quant_uncertainty,
                uncertainty=quant_uncertainty * 0.1,  # 10% uncertainty
                distribution=DistributionType.UNIFORM,
                category=UncertaintyType.TYPE_B
            ))

        # Calibration uncertainty
        if 'calibration_uncertainty' in experimental_conditions:
            cal_unc = experimental_conditions['calibration_uncertainty']
            components.append(UncertaintyComponent(
                name="Type B - Calibration",
                value=0,  # Calibration affects bias, not precision
                uncertainty=cal_unc,
                distribution=DistributionType.GAUSSIAN,
                category=UncertaintyType.TYPE_B
            ))

        return components

    def _calculate_correlation_matrix(self, components: List[UncertaintyComponent]) -> np.ndarray:
        """Calculate correlation matrix between uncertainty components."""
        n_components = len(components)
        correlation_matrix = np.eye(n_components)

        # Define correlations based on physical understanding
        for i, comp_i in enumerate(components):
            for j, comp_j in enumerate(components):
                if i != j:
                    # Thermal noise and phase noise are often correlated
                    if 'Thermal' in comp_i.name and 'Phase' in comp_j.name:
                        correlation_matrix[i, j] = 0.3
                    # Clock jitter affects timing-related measurements
                    elif 'Clock' in comp_i.name and 'timing' in comp_j.name.lower():
                        correlation_matrix[i, j] = 0.5
                    # Calibration affects all systematic errors
                    elif 'Calibration' in comp_i.name or 'Calibration' in comp_j.name:
                        correlation_matrix[i, j] = 0.2

        return correlation_matrix

    def _combine_uncertainty_components(self, components: List[UncertaintyComponent],
                                       correlation_matrix: np.ndarray) -> UncertaintyBudget:
        """Combine uncertainty components with correlations."""
        uncertainties = np.array([comp.uncertainty for comp in components])
        sensitivities = np.array([comp.sensitivity_coefficient for comp in components])

        # Apply sensitivity coefficients
        weighted_uncertainties = uncertainties * sensitivities

        # Calculate combined variance with correlations
        combined_var = np.dot(weighted_uncertainties.T,
                             np.dot(correlation_matrix, weighted_uncertainties))
        combined_uncertainty = np.sqrt(combined_var)

        # Calculate effective degrees of freedom
        effective_dof = self._calculate_welch_satterthwaite_dof(components)

        # Coverage factor based on degrees of freedom
        if effective_dof is not None and effective_dof < 30:
            coverage_factor = stats.t.ppf(0.5 + self.confidence_level / 2, effective_dof)
        else:
            coverage_factor = self.coverage_factor

        expanded_uncertainty = coverage_factor * combined_uncertainty

        return UncertaintyBudget(
            components=components,
            combined_uncertainty=combined_uncertainty,
            expanded_uncertainty=expanded_uncertainty,
            coverage_factor=coverage_factor,
            confidence_level=self.confidence_level,
            effective_degrees_of_freedom=effective_dof,
            correlation_matrix=correlation_matrix
        )

    def _calculate_welch_satterthwaite_dof(self, components: List[UncertaintyComponent]) -> Optional[float]:
        """Calculate effective degrees of freedom using Welch-Satterthwaite formula."""
        numerator = 0
        denominator = 0

        for comp in components:
            if comp.degrees_of_freedom is not None and comp.degrees_of_freedom > 0:
                contribution = (comp.uncertainty * comp.sensitivity_coefficient) ** 4
                numerator += contribution
                denominator += contribution / comp.degrees_of_freedom

        return numerator / denominator if denominator > 0 else None

    def sensitivity_analysis(self, budget: UncertaintyBudget,
                           parameter_variations: Dict[str, float] = None) -> Dict:
        """
        Perform sensitivity analysis of uncertainty budget.

        Args:
            budget: Uncertainty budget to analyze
            parameter_variations: Parameter variations for sensitivity analysis

        Returns:
            Dictionary with sensitivity analysis results
        """
        if parameter_variations is None:
            parameter_variations = {comp.name: 0.1 for comp in budget.components}  # 10% variation

        sensitivity_results = {}

        for comp_name, variation in parameter_variations.items():
            # Find component
            comp = next((c for c in budget.components if c.name == comp_name), None)
            if comp is None:
                continue

            # Calculate baseline contribution
            baseline_contribution = (comp.uncertainty * comp.sensitivity_coefficient) ** 2

            # Calculate perturbed contribution
            perturbed_uncertainty = comp.uncertainty * (1 + variation)
            perturbed_contribution = (perturbed_uncertainty * comp.sensitivity_coefficient) ** 2

            # Sensitivity coefficient
            sensitivity = (perturbed_contribution - baseline_contribution) / (variation * comp.uncertainty)

            sensitivity_results[comp_name] = {
                'baseline_contribution': baseline_contribution,
                'perturbed_contribution': perturbed_contribution,
                'sensitivity_coefficient': sensitivity,
                'percentage_contribution': baseline_contribution / budget.combined_uncertainty**2 * 100
            }

        return sensitivity_results

    def uncertainty_budget_table(self, budget: UncertaintyBudget) -> pd.DataFrame:
        """
        Create formatted uncertainty budget table.

        Args:
            budget: Uncertainty budget

        Returns:
            Pandas DataFrame with uncertainty budget table
        """
        data = []

        for comp in budget.components:
            contribution = (comp.uncertainty * comp.sensitivity_coefficient) ** 2
            percentage = contribution / budget.combined_uncertainty**2 * 100

            data.append({
                'Component': comp.name,
                'Value': comp.value,
                'Uncertainty': comp.uncertainty,
                'Sensitivity': comp.sensitivity_coefficient,
                'Contribution': np.sqrt(contribution),
                'Percentage': percentage,
                'Distribution': comp.distribution.value,
                'DOF': comp.degrees_of_freedom,
                'Type': comp.category.value
            })

        df = pd.DataFrame(data)

        # Add summary row
        total_row = {
            'Component': 'Combined',
            'Value': np.nan,
            'Uncertainty': budget.combined_uncertainty,
            'Sensitivity': np.nan,
            'Contribution': budget.combined_uncertainty,
            'Percentage': 100.0,
            'Distribution': 'Combined',
            'DOF': budget.effective_degrees_of_freedom,
            'Type': 'Combined'
        }

        df = pd.concat([df, pd.DataFrame([total_row])], ignore_index=True)

        return df

    def optimization_recommendations(self, budget: UncertaintyBudget,
                                   target_uncertainty: Optional[float] = None) -> List[Dict]:
        """
        Generate optimization recommendations for uncertainty reduction.

        Args:
            budget: Current uncertainty budget
            target_uncertainty: Target uncertainty level

        Returns:
            List of optimization recommendations
        """
        recommendations = []

        # Find largest contributors
        contributions = []
        for comp in budget.components:
            contribution = (comp.uncertainty * comp.sensitivity_coefficient) ** 2
            percentage = contribution / budget.combined_uncertainty**2 * 100
            contributions.append((comp.name, percentage, contribution))

        # Sort by contribution
        contributions.sort(key=lambda x: x[1], reverse=True)

        # Generate recommendations for top contributors
        for name, percentage, contribution in contributions[:3]:
            if percentage > 10:  # Only significant contributors
                if 'Thermal' in name:
                    recommendations.append({
                        'component': name,
                        'recommendation': 'Reduce measurement bandwidth or improve noise figure',
                        'potential_reduction': '20-30%',
                        'difficulty': 'Medium',
                        'impact': 'High'
                    })
                elif 'Phase' in name:
                    recommendations.append({
                        'component': name,
                        'recommendation': 'Use higher quality oscillator or phase noise cancellation',
                        'potential_reduction': '40-60%',
                        'difficulty': 'High',
                        'impact': 'High'
                    })
                elif 'Clock' in name:
                    recommendations.append({
                        'component': name,
                        'recommendation': 'Use low-jitter clock source or clock cleaning',
                        'potential_reduction': '30-50%',
                        'difficulty': 'Medium',
                        'impact': 'Medium'
                    })
                elif 'Quantization' in name:
                    recommendations.append({
                        'component': name,
                        'recommendation': 'Increase ADC resolution or use oversampling',
                        'potential_reduction': '10-20%',
                        'difficulty': 'Low',
                        'impact': 'Medium'
                    })

        # Add general recommendations
        if target_uncertainty and budget.combined_uncertainty > target_uncertainty:
            reduction_needed = (budget.combined_uncertainty - target_uncertainty) / budget.combined_uncertainty
            recommendations.append({
                'component': 'System Level',
                'recommendation': f'Overall uncertainty reduction of {reduction_needed*100:.1f}% required',
                'potential_reduction': f'{reduction_needed*100:.1f}%',
                'difficulty': 'Variable',
                'impact': 'Critical'
            })

        return recommendations