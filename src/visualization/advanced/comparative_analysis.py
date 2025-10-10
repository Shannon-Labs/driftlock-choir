"""
Comparative analysis visualization for chronometric interferometry research.

Provides sophisticated visualization comparing chronometric interferometry performance
against existing timing synchronization technologies for research publications.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from typing import Dict, List, Tuple, Optional, Union, Any
import warnings
from dataclasses import dataclass

from ..base.figure_generator import FigureGenerator
from ..base.styles import IEEEStyle, NASAStyle
from ..base.utils import format_scientific_notation, format_uncertainty


@dataclass
class TechnologyData:
    """Performance data for a timing synchronization technology."""
    name: str
    timing_precision: np.ndarray  # ps
    frequency_accuracy: np.ndarray  # ppb
    convergence_time: np.ndarray  # ms
    scalability: np.ndarray  # max nodes
    cost_index: float  # Relative cost (1-10)
    complexity_index: float  # Implementation complexity (1-10)
    maturity_level: str  # 'Research', 'Prototype', 'Commercial'
    color: str = None


@dataclass
class BenchmarkResults:
    """Results from comparative benchmarking experiments."""
    chronometric_interferometry: TechnologyData
    gps_disciplined_oscillator: TechnologyData
    ieee_1588_ptp: TechnologyData
    ntp: TechnologyData
    atomic_clock: TechnologyData
    additional_technologies: Dict[str, TechnologyData] = None


class ComparativeAnalysisPlotter:
    """
    Advanced comparative analysis visualization for chronometric interferometry.

    Creates publication-quality visualizations comparing performance against
    existing timing synchronization technologies.
    """

    def __init__(self, figure_generator: Optional[FigureGenerator] = None,
                 style: str = 'ieee'):
        """
        Initialize comparative analysis plotter.

        Args:
            figure_generator: Figure generator instance
            style: Publication style ('ieee' or 'nasa')
        """
        self.fig_gen = figure_generator or FigureGenerator(default_style=style)
        self.style = style

        # Default technology colors
        self.default_colors = {
            'Chronometric Interferometry': '#0072BD',
            'GPS-Disciplined Oscillator': '#D95319',
            'IEEE 1588 PTP': '#77AC30',
            'NTP': '#A2142F',
            'Atomic Clock': '#7E2F8E',
            'White Rabbit': '#4DBEEE',
            'MAC/White Rabbit': '#EDB120'
        }

    def plot_performance_comparison_radar(self, benchmark_data: BenchmarkResults,
                                         metrics: List[str] = None,
                                         title: str = "Technology Performance Comparison") -> plt.Figure:
        """
        Create radar chart comparing multiple technologies across different metrics.

        Args:
            benchmark_data: Benchmark results data
            metrics: List of metrics to compare
            title: Plot title

        Returns:
            Matplotlib figure
        """
        if metrics is None:
            metrics = ['Timing Precision', 'Frequency Accuracy', 'Convergence Speed',
                      'Scalability', 'Cost Efficiency', 'Implementation Simplicity']

        # Prepare data for radar chart
        technologies = [
            benchmark_data.chronometric_interferometry,
            benchmark_data.gps_disciplined_oscillator,
            benchmark_data.ieee_1588_ptp,
            benchmark_data.ntp,
            benchmark_data.atomic_clock
        ]

        # Normalize metrics for radar chart (0-1 scale)
        normalized_data = self._normalize_metrics_for_radar(technologies, metrics)

        spec = FigureSpec(
            title=title,
            width=7.0,
            height=6.0,
            style=self.style
        )

        fig = self.fig_gen.create_figure(spec)
        ax = fig.add_subplot(111, projection='polar')

        # Create radar chart
        self._create_radar_chart(ax, technologies, normalized_data, metrics)

        return fig

    def plot_timing_precision_comparison(self, benchmark_data: BenchmarkResults,
                                        show_distribution: bool = True,
                                        log_scale: bool = True,
                                        title: str = "Timing Precision Comparison") -> plt.Figure:
        """
        Create comprehensive timing precision comparison.

        Args:
            benchmark_data: Benchmark results data
            show_distribution: Whether to show full distributions
            log_scale: Whether to use log scale
            title: Plot title

        Returns:
            Matplotlib figure
        """
        spec = FigureSpec(
            title=title,
            width=7.0,
            height=5.0,
            style=self.style
        )

        fig = self.fig_gen.create_figure(spec)

        if show_distribution:
            # Create two subplots
            ax1 = self.fig_gen.add_subplot(fig, (0, 0), title='(a) Precision Distribution')
            ax2 = self.fig_gen.add_subplot(fig, (0, 1), title='(b) Performance Summary')

            self._plot_precision_distributions(ax1, benchmark_data)
            self._plot_precision_summary(ax2, benchmark_data, log_scale)
        else:
            # Single subplot
            ax = self.fig_gen.add_subplot(fig, 0)
            self._plot_precision_summary(ax, benchmark_data, log_scale)

        plt.tight_layout()
        return fig

    def plot_scalability_analysis(self, benchmark_data: BenchmarkResults,
                                 max_nodes: int = 1000,
                                 title: str = "Scalability Analysis") -> plt.Figure:
        """
        Create scalability analysis comparing node count capabilities.

        Args:
            benchmark_data: Benchmark results data
            max_nodes: Maximum number of nodes to display
            title: Plot title

        Returns:
            Matplotlib figure
        """
        spec = FigureSpec(
            title=title,
            width=7.0,
            height=5.0,
            style=self.style
        )

        fig = self.fig_gen.create_figure(spec)

        # Create subplots
        ax1 = self.fig_gen.add_subplot(fig, (0, 0), title='(a) Maximum Supported Nodes')
        ax2 = self.fig_gen.add_subplot(fig, (0, 1), title='(b) Precision vs. Node Count')

        # Plot maximum supported nodes
        technologies = [
            benchmark_data.chronometric_interferometry,
            benchmark_data.gps_disciplined_oscillator,
            benchmark_data.ieee_1588_ptp,
            benchmark_data.ntp,
            benchmark_data.atomic_clock
        ]

        tech_names = [tech.name for tech in technologies]
        max_nodes_supported = [np.mean(tech.scalability) for tech in technologies]
        colors = [self._get_technology_color(tech.name) for tech in technologies]

        bars = ax1.barh(tech_names, max_nodes_supported, color=colors, alpha=0.7)
        ax1.set_xlabel('Maximum Supported Nodes')
        ax1.set_xscale('log')
        ax1.grid(True, alpha=0.3)

        # Highlight chronometric interferometry
        ci_idx = tech_names.index('Chronometric Interferometry')
        bars[ci_idx].set_edgecolor('red')
        bars[ci_idx].set_linewidth(3)

        # Plot precision vs. node count (simulated data)
        node_counts = np.logspace(0, np.log10(max_nodes), 50)
        for tech in technologies:
            if 'Chronometric' in tech.name:
                # Chronometric interferometry maintains precision
                precision_vs_nodes = np.full_like(node_counts, np.mean(tech.timing_precision))
                label = tech.name
                style = '-'
                width = 3
            elif 'PTP' in tech.name:
                # PTP degrades with node count
                precision_vs_nodes = np.mean(tech.timing_precision) * np.sqrt(node_counts / 10)
                label = tech.name
                style = '--'
                width = 2
            elif 'NTP' in tech.name:
                # NTP degrades significantly
                precision_vs_nodes = np.mean(tech.timing_precision) * (node_counts / 10)**1.5
                label = tech.name
                style = ':'
                width = 2
            else:
                continue

            color = self._get_technology_color(tech.name)
            ax2.semilogx(node_counts, precision_vs_nodes, style, linewidth=width,
                        color=color, label=label)

        ax2.set_xlabel('Number of Nodes')
        ax2.set_ylabel('Timing Precision (ps)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def plot_cost_benefit_analysis(self, benchmark_data: BenchmarkResults,
                                  title: str = "Cost-Benefit Analysis") -> plt.Figure:
        """
        Create cost-benefit analysis comparing technologies.

        Args:
            benchmark_data: Benchmark results data
            title: Plot title

        Returns:
            Matplotlib figure
        """
        spec = FigureSpec(
            title=title,
            width=7.0,
            height=5.0,
            style=self.style
        )

        fig = self.fig_gen.create_figure(spec)

        # Create subplots
        ax1 = self.fig_gen.add_subplot(fig, (0, 0), title='(a) Cost vs. Performance')
        ax2 = self.fig_gen.add_subplot(fig, (0, 1), title='(b) Technology Bubble Chart')

        # Prepare data
        technologies = [
            benchmark_data.chronometric_interferometry,
            benchmark_data.gps_disciplined_oscillator,
            benchmark_data.ieee_1588_ptp,
            benchmark_data.ntp,
            benchmark_data.atomic_clock
        ]

        # Cost vs. Performance scatter plot
        costs = [tech.cost_index for tech in technologies]
        precisions = [np.mean(tech.timing_precision) for tech in technologies]
        colors = [self._get_technology_color(tech.name) for tech in technologies]
        sizes = [tech.scalability.mean() / 10 for tech in technologies]

        scatter = ax1.scatter(costs, precisions, s=sizes, c=colors, alpha=0.7,
                             edgecolors='black', linewidth=2)

        # Add technology labels
        for i, tech in enumerate(technologies):
            ax1.annotate(tech.name, (costs[i], precisions[i]),
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=8, fontweight='bold')

        ax1.set_xlabel('Cost Index (1=Low, 10=High)')
        ax1.set_ylabel('Timing Precision (ps)')
        ax1.grid(True, alpha=0.3)

        # Technology bubble chart
        x_positions = np.arange(len(technologies))
        bubble_sizes = [np.mean(tech.timing_precision) for tech in technologies]
        bubble_colors = [tech.cost_index for tech in technologies]

        bubbles = ax2.scatter(x_positions, bubble_sizes, c=bubble_colors,
                             s=sizes, alpha=0.7, cmap='RdYlBu_r',
                             edgecolors='black', linewidth=2,
                             vmin=1, vmax=10)

        ax2.set_xticks(x_positions)
        ax2.set_xticklabels([tech.name.replace(' ', '\n') for tech in technologies],
                           rotation=45, ha='right')
        ax2.set_ylabel('Timing Precision (ps)')

        # Add colorbar for cost
        cbar = fig.colorbar(bubbles, ax=ax2)
        cbar.set_label('Cost Index')

        plt.tight_layout()
        return fig

    def plot_deployment_scenarios(self, benchmark_data: BenchmarkResults,
                                 scenarios: Dict[str, Dict],
                                 title: str = "Deployment Scenario Analysis") -> plt.Figure:
        """
        Create deployment scenario analysis showing best technologies for different use cases.

        Args:
            benchmark_data: Benchmark results data
            scenarios: Dictionary of deployment scenarios
            title: Plot title

        Returns:
            Matplotlib figure
        """
        spec = FigureSpec(
            title=title,
            width=7.0,
            height=5.0,
            style=self.style
        )

        fig = self.fig_gen.create_figure(spec)

        # Create subplots
        ax1 = self.fig_gen.add_subplot(fig, (0, 0), title='(a) Scenario Suitability')
        ax2 = self.fig_gen.add_subplot(fig, (0, 1), title='(b) Technology Recommendations')

        # Scenario suitability heatmap
        technologies = [
            benchmark_data.chronometric_interferometry,
            benchmark_data.gps_disciplined_oscillator,
            benchmark_data.ieee_1588_ptp,
            benchmark_data.ntp,
            benchmark_data.atomic_clock
        ]

        tech_names = [tech.name for tech in technologies]
        scenario_names = list(scenarios.keys())

        # Calculate suitability scores
        suitability_matrix = np.zeros((len(tech_names), len(scenario_names)))
        for i, tech in enumerate(technologies):
            for j, (scenario_name, scenario_weights) in enumerate(scenarios.items()):
                suitability_matrix[i, j] = self._calculate_scenario_suitability(tech, scenario_weights)

        # Create heatmap
        im = ax1.imshow(suitability_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=100)

        # Add colorbar
        cbar = fig.colorbar(im, ax=ax1)
        cbar.set_label('Suitability Score (%)')

        # Set ticks and labels
        ax1.set_xticks(np.arange(len(scenario_names)))
        ax1.set_yticks(np.arange(len(tech_names)))
        ax1.set_xticklabels(scenario_names, rotation=45, ha='right')
        ax1.set_yticklabels(tech_names)

        # Add suitability values as text
        for i in range(len(tech_names)):
            for j in range(len(scenario_names)):
                text = ax1.text(j, i, f'{suitability_matrix[i, j]:.0f}',
                               ha="center", va="center", color="black", fontweight='bold')

        # Technology recommendations
        best_tech_indices = np.argmax(suitability_matrix, axis=0)
        recommended_techs = [tech_names[i] for i in best_tech_indices]
        scores = [suitability_matrix[i, j] for i, j in zip(best_tech_indices, range(len(scenario_names)))]

        bars = ax2.bar(scenario_names, scores, color='#0072BD', alpha=0.7)
        ax2.set_ylabel('Best Suitability Score (%)')
        ax2.set_ylim([0, 100])
        ax2.grid(True, alpha=0.3)

        # Add technology labels on bars
        for i, (bar, tech_name) in enumerate(zip(bars, recommended_techs)):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                    tech_name.split()[0], ha='center', va='bottom',
                    fontsize=8, fontweight='bold')

        plt.tight_layout()
        return fig

    def plot_research_roadmap(self, current_data: BenchmarkResults,
                            target_metrics: Dict[str, float],
                            timeline_years: List[int],
                            title: str = "Research Development Roadmap") -> plt.Figure:
        """
        Create research development roadmap showing improvement trajectory.

        Args:
            current_data: Current performance data
            target_metrics: Target performance metrics
            timeline_years: Timeline for development
            title: Plot title

        Returns:
            Matplotlib figure
        """
        spec = FigureSpec(
            title=title,
            width=7.0,
            height=5.0,
            style=self.style
        )

        fig = self.fig_gen.create_figure(spec)

        # Create subplots
        ax1 = self.fig_gen.add_subplot(fig, (0, 0), title='(a) Performance Improvement Trajectory')
        ax2 = self.fig_gen.add_subplot(fig, (0, 1), title='(b) Technology Maturity Timeline')

        # Performance improvement trajectory
        current_precision = np.mean(current_data.chronometric_interferometry.timing_precision)
        target_precision = target_metrics.get('timing_precision', current_precision / 10)

        years = np.array(timeline_years)
        precision_trajectory = self._generate_improvement_trajectory(
            current_precision, target_precision, years
        )

        ax1.semilogy(years, precision_trajectory, 'o-', linewidth=3, markersize=8,
                    color='#0072BD', label='Chronometric Interferometry')
        ax1.semilogy(years, [current_precision] * len(years), '--',
                    color='#D95319', label='Current GPS-Disciplined')
        ax1.semilogy(years, [target_precision] * len(years), '--',
                    color='#77AC30', label='Target Performance')

        ax1.set_xlabel('Year')
        ax1.set_ylabel('Timing Precision (ps)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Technology maturity timeline
        maturity_phases = ['Research', 'Prototype', 'Pilot', 'Commercial']
        phase_colors = ['#A2142F', '#EDB120', '#77AC30', '#0072BD']

        for i, (phase, color) in enumerate(zip(maturity_phases, phase_colors)):
            start_year = timeline_years[i * len(timeline_years) // len(maturity_phases)]
            if i < len(maturity_phases) - 1:
                end_year = timeline_years[(i + 1) * len(timeline_years) // len(maturity_phases) - 1]
            else:
                end_year = timeline_years[-1]

            ax2.barh(phase, end_year - start_year + 1, left=start_year,
                    height=0.8, color=color, alpha=0.7, label=phase)

        ax2.set_xlabel('Year')
        ax2.set_ylabel('Development Phase')
        ax2.set_xlim([timeline_years[0] - 1, timeline_years[-1] + 1])
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def _normalize_metrics_for_radar(self, technologies: List[TechnologyData],
                                   metrics: List[str]) -> Dict[str, np.ndarray]:
        """Normalize metrics for radar chart visualization."""
        normalized_data = {}

        for metric in metrics:
            values = []
            for tech in technologies:
                if 'Precision' in metric:
                    # Lower is better, normalize inversely
                    values.append(1.0 / np.mean(tech.timing_precision))
                elif 'Accuracy' in metric:
                    # Lower is better, normalize inversely
                    values.append(1.0 / np.mean(tech.frequency_accuracy))
                elif 'Convergence' in metric:
                    # Lower is better, normalize inversely
                    values.append(1.0 / np.mean(tech.convergence_time))
                elif 'Scalability' in metric:
                    # Higher is better
                    values.append(np.mean(tech.scalability))
                elif 'Cost' in metric:
                    # Higher is better (cost efficiency)
                    values.append(11.0 - tech.cost_index)
                elif 'Complexity' in metric:
                    # Higher is better (simplicity)
                    values.append(11.0 - tech.complexity_index)

            # Normalize to 0-1 scale
            values = np.array(values)
            normalized_data[metric] = (values - values.min()) / (values.max() - values.min())

        return normalized_data

    def _create_radar_chart(self, ax: plt.Axes, technologies: List[TechnologyData],
                          normalized_data: Dict[str, np.ndarray], metrics: List[str]) -> None:
        """Create radar chart."""
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle

        # Plot each technology
        for i, tech in enumerate(technologies):
            values = [normalized_data[metric][i] for metric in metrics]
            values += values[:1]  # Complete the circle

            color = self._get_technology_color(tech.name)
            ax.plot(angles, values, 'o-', linewidth=2, color=color, label=tech.name)
            ax.fill(angles, values, alpha=0.25, color=color)

        # Set labels
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics)
        ax.set_ylim([0, 1])
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))

    def _plot_precision_distributions(self, ax: plt.Axes, benchmark_data: BenchmarkResults) -> None:
        """Plot precision distributions for different technologies."""
        technologies = [
            benchmark_data.chronometric_interferometry,
            benchmark_data.gps_disciplined_oscillator,
            benchmark_data.ieee_1588_ptp,
            benchmark_data.ntp
        ]

        for tech in technologies:
            if len(tech.timing_precision) > 1:
                color = self._get_technology_color(tech.name)
                ax.hist(tech.timing_precision, bins=20, alpha=0.6, color=color,
                       density=True, label=tech.name)

        ax.set_xlabel('Timing Precision (ps)')
        ax.set_ylabel('Probability Density')
        ax.legend()
        ax.grid(True, alpha=0.3)

    def _plot_precision_summary(self, ax: plt.Axes, benchmark_data: BenchmarkResults,
                               log_scale: bool) -> None:
        """Plot precision summary comparison."""
        technologies = [
            benchmark_data.chronometric_interferometry,
            benchmark_data.gps_disciplined_oscillator,
            benchmark_data.ieee_1588_ptp,
            benchmark_data.ntp,
            benchmark_data.atomic_clock
        ]

        tech_names = [tech.name for tech in technologies]
        precisions = [np.mean(tech.timing_precision) for tech in technologies]
        errors = [np.std(tech.timing_precision) for tech in technologies]
        colors = [self._get_technology_color(tech.name) for tech in technologies]

        bars = ax.barh(tech_names, precisions, xerr=errors, capsize=5,
                      color=colors, alpha=0.7)

        if log_scale:
            ax.set_xscale('log')

        ax.set_xlabel('Timing Precision (ps)')
        ax.grid(True, alpha=0.3)

        # Highlight chronometric interferometry
        ci_idx = tech_names.index('Chronometric Interferometry')
        bars[ci_idx].set_edgecolor('red')
        bars[ci_idx].set_linewidth(3)

    def _get_technology_color(self, tech_name: str) -> str:
        """Get color for a technology."""
        return self.default_colors.get(tech_name, '#7F7F7F')

    def _calculate_scenario_suitability(self, tech: TechnologyData, weights: Dict[str, float]) -> float:
        """Calculate suitability score for a deployment scenario."""
        score = 0

        if 'timing_precision' in weights:
            precision_score = 1.0 / np.mean(tech.timing_precision)
            score += precision_score * weights['timing_precision']

        if 'frequency_accuracy' in weights:
            accuracy_score = 1.0 / np.mean(tech.frequency_accuracy)
            score += accuracy_score * weights['frequency_accuracy']

        if 'convergence_time' in weights:
            convergence_score = 1.0 / np.mean(tech.convergence_time)
            score += convergence_score * weights['convergence_time']

        if 'scalability' in weights:
            scalability_score = np.mean(tech.scalability)
            score += scalability_score * weights['scalability']

        if 'cost' in weights:
            cost_score = 11.0 - tech.cost_index
            score += cost_score * weights['cost']

        # Normalize to 0-100 scale
        return min(100, max(0, score * 10))

    def _generate_improvement_trajectory(self, current: float, target: float,
                                       years: np.ndarray) -> np.ndarray:
        """Generate realistic improvement trajectory."""
        # Exponential improvement model
        improvement_rate = np.log(target / current) / (years[-1] - years[0])
        trajectory = current * np.exp(improvement_rate * (years - years[0]))

        # Add some realistic variation
        noise = np.random.normal(0, 0.05, len(trajectory))
        trajectory *= (1 + noise)

        return trajectory


# Import FigureSpec for use in this module
from dataclasses import dataclass
from typing import Optional, List, Dict

@dataclass
class FigureSpec:
    """Figure specification for comparative analysis plots."""
    title: str = ""
    width: float = 7.0
    height: Optional[float] = None
    style: str = 'ieee'