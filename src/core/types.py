"""
Core data types and structures for Driftlock Choir.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, NewType, Optional, Union

import numpy as np

# Physical units with type safety
Seconds = NewType("Seconds", float)
Picoseconds = NewType("Picoseconds", float)
Hertz = NewType("Hertz", float)
PPM = NewType("PPM", float)
PPB = NewType("PPB", float)
Meters = NewType("Meters", float)
Decibels = NewType("Decibels", float)

from .constants import PhysicalConstants


class MeasurementQuality(Enum):
    """Quality indicators for measurements."""

    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"
    INVALID = "invalid"


@dataclass(frozen=True)
class Timestamp:
    """High-precision timestamp with uncertainty."""

    time: Seconds
    uncertainty: Picoseconds
    quality: MeasurementQuality

    def __post_init__(self):
        if self.uncertainty < 0:
            raise ValueError("Timestamp uncertainty must be non-negative")

    def to_ps(self) -> float:
        """Convert time to picoseconds."""
        return PhysicalConstants.seconds_to_ps(self.time)

    def to_datetime(self) -> datetime:
        """Convert to datetime object (assuming Unix epoch)."""
        return datetime.fromtimestamp(self.time, timezone.utc)

    @classmethod
    def from_ps(cls, picoseconds: float, uncertainty_ps: float = 0.0) -> "Timestamp":
        """Create timestamp from picoseconds."""
        return cls(
            time=Seconds(PhysicalConstants.ps_to_seconds(picoseconds)),
            uncertainty=Picoseconds(uncertainty_ps),
            quality=MeasurementQuality.EXCELLENT,
        )


@dataclass(frozen=True)
class Frequency:
    """Frequency measurement with uncertainty."""

    freq: Hertz
    uncertainty: Hertz
    quality: MeasurementQuality

    def __post_init__(self):
        if self.freq <= 0:
            raise ValueError("Frequency must be positive")
        if self.uncertainty < 0:
            raise ValueError("Frequency uncertainty must be non-negative")

    def to_mhz(self) -> float:
        """Convert frequency to MHz."""
        return PhysicalConstants.hz_to_mhz(self.freq)

    def to_ppm_at(self, reference: Hertz) -> float:
        """Convert frequency offset to PPM at reference frequency."""
        return PhysicalConstants.unit_to_ppm((self.freq - reference) / reference)

    def to_ppb_at(self, reference: Hertz) -> float:
        """Convert frequency offset to PPB at reference frequency."""
        return (self.freq - reference) / reference * PhysicalConstants.PPB_PER_UNIT


@dataclass(frozen=True)
class PhaseMeasurement:
    """Phase measurement at specific frequency and time."""

    phase: float  # radians
    frequency: Hertz
    timestamp: Timestamp
    uncertainty: float  # radians
    quality: MeasurementQuality

    def __post_init__(self):
        # Normalize phase to [-π, π]
        object.__setattr__(
            self, "phase", np.arctan2(np.sin(self.phase), np.cos(self.phase))
        )

    def unwrap_to(self, target_phase: float) -> float:
        """Unwrap phase to be continuous with target phase."""
        phase_diff = target_phase - self.phase
        wraps = np.round(phase_diff / (2 * np.pi))
        return self.phase + wraps * 2 * np.pi


@dataclass(frozen=True)
class BeatNoteData:
    """Complete beat-note measurement from two-way exchange."""

    tx_frequency: Hertz
    rx_frequency: Hertz
    sampling_rate: Hertz
    duration: Seconds
    waveform: np.ndarray  # Complex baseband samples
    timestamp: Timestamp
    snr: Decibels
    quality: MeasurementQuality
    measured_beat_frequency: Optional[Hertz] = None

    def __post_init__(self):
        if self.tx_frequency <= 0 or self.rx_frequency <= 0:
            raise ValueError("Frequencies must be positive")
        if self.sampling_rate <= 0:
            raise ValueError("Sampling rate must be positive")
        if self.duration <= 0:
            raise ValueError("Duration must be positive")
        expected_samples = int(self.sampling_rate * self.duration)
        if len(self.waveform) != expected_samples:
            raise ValueError(
                f"Waveform length {len(self.waveform)} doesn't match expected {expected_samples}"
            )

        if self.measured_beat_frequency is None:
            object.__setattr__(
                self,
                "measured_beat_frequency",
                Hertz(abs(self.tx_frequency - self.rx_frequency)),
            )

    def get_beat_frequency(self) -> Hertz:
        """Get the beat frequency (absolute difference)."""
        return self.measured_beat_frequency  # type: ignore[return-value]

    def get_time_vector(self) -> np.ndarray:
        """Get time vector for the waveform."""
        n_samples = len(self.waveform)
        return np.arange(n_samples) / self.sampling_rate + self.timestamp.time

    def get_analytic_signal(self) -> np.ndarray:
        """Get the analytic signal (Hilbert transform)."""
        from scipy.signal import hilbert

        return hilbert(self.waveform.real)


@dataclass(frozen=True)
class EstimationResult:
    """Result of τ/Δf estimation from beat-note data."""

    tau: Picoseconds  # Time-of-flight
    tau_uncertainty: Picoseconds
    delta_f: Hertz  # Frequency offset
    delta_f_uncertainty: Hertz
    covariance: np.ndarray  # 2x2 covariance matrix
    likelihood: float
    quality: MeasurementQuality
    method: str  # Estimation method used
    timestamp: Timestamp

    def __post_init__(self):
        if self.tau_uncertainty < 0 or self.delta_f_uncertainty < 0:
            raise ValueError("Uncertainties must be non-negative")
        if self.covariance.shape != (2, 2):
            raise ValueError("Covariance must be 2x2 matrix")
        if not (0 <= self.likelihood <= 1):
            raise ValueError("Likelihood must be between 0 and 1")

    def get_tau_seconds(self) -> float:
        """Convert tau to seconds."""
        return PhysicalConstants.ps_to_seconds(self.tau)

    def get_tau_meters(self) -> float:
        """Convert tau to distance (assuming speed of light)."""
        return PhysicalConstants.ps_to_meters(self.tau)

    def get_confidence_interval_95(self) -> Dict[str, tuple]:
        """Get 95% confidence intervals."""
        z_score = 1.96  # 95% confidence
        return {
            "tau": (
                self.tau - z_score * self.tau_uncertainty,
                self.tau + z_score * self.tau_uncertainty,
            ),
            "delta_f": (
                self.delta_f - z_score * self.delta_f_uncertainty,
                self.delta_f + z_score * self.delta_f_uncertainty,
            ),
        }


@dataclass(frozen=True)
class NodeState:
    """Complete state of a network node."""

    node_id: int
    clock_bias: Picoseconds
    clock_bias_uncertainty: Picoseconds
    frequency_offset: PPB
    frequency_offset_uncertainty: PPB
    last_update: Timestamp
    quality: MeasurementQuality
    # Extended fields for adaptive consensus
    local_spectral_gap: Optional[float] = None
    local_step_size: Optional[float] = None
    variance_warning_level: Optional[str] = None
    ml_prediction_confidence: Optional[float] = None

    def get_frequency_hz(self, nominal_freq: Hertz) -> Hertz:
        """Get actual frequency in Hz."""
        return Hertz(nominal_freq * (1.0 + self.frequency_offset * 1e-9))

    def get_clock_bias_seconds(self) -> float:
        """Get clock bias in seconds."""
        return PhysicalConstants.ps_to_seconds(self.clock_bias)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "node_id": self.node_id,
            "clock_bias": self.clock_bias,
            "clock_bias_uncertainty": self.clock_bias_uncertainty,
            "frequency_offset": self.frequency_offset,
            "frequency_offset_uncertainty": self.frequency_offset_uncertainty,
            "last_update": self._timestamp_to_dict(self.last_update),
            "quality": self.quality.value,
            "local_spectral_gap": self.local_spectral_gap,
            "local_step_size": self.local_step_size,
            "variance_warning_level": self.variance_warning_level,
            "ml_prediction_confidence": self.ml_prediction_confidence,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "NodeState":
        """Create from dictionary for deserialization."""
        # Handle backward compatibility
        local_spectral_gap = data.get("local_spectral_gap")
        local_step_size = data.get("local_step_size")
        variance_warning_level = data.get("variance_warning_level")
        ml_prediction_confidence = data.get("ml_prediction_confidence")

        return cls(
            node_id=data["node_id"],
            clock_bias=Picoseconds(data["clock_bias"]),
            clock_bias_uncertainty=Picoseconds(data["clock_bias_uncertainty"]),
            frequency_offset=PPB(data["frequency_offset"]),
            frequency_offset_uncertainty=PPB(data["frequency_offset_uncertainty"]),
            last_update=cls._timestamp_from_dict(data["last_update"]),
            quality=MeasurementQuality(data["quality"]),
            local_spectral_gap=local_spectral_gap,
            local_step_size=local_step_size,
            variance_warning_level=variance_warning_level,
            ml_prediction_confidence=ml_prediction_confidence,
        )

    @staticmethod
    def _timestamp_to_dict(timestamp: "Timestamp") -> Dict[str, Any]:
        """Convert Timestamp to dictionary."""
        return {
            "time": timestamp.time,
            "uncertainty": timestamp.uncertainty,
            "quality": timestamp.quality.value,
        }

    @staticmethod
    def _timestamp_from_dict(data: Dict[str, Any]) -> "Timestamp":
        """Create Timestamp from dictionary."""
        return Timestamp(
            time=Seconds(data["time"]),
            uncertainty=Picoseconds(data["uncertainty"]),
            quality=MeasurementQuality(data["quality"]),
        )


@dataclass(frozen=True)
class NetworkTopology:
    """Network connectivity information."""

    adjacency_matrix: np.ndarray
    node_ids: List[int]
    laplacian: np.ndarray
    spectral_gap: float
    is_connected: bool

    def __post_init__(self):
        n = len(self.node_ids)
        if self.adjacency_matrix.shape != (n, n):
            raise ValueError("Adjacency matrix dimensions don't match number of nodes")
        if self.laplacian.shape != (n, n):
            raise ValueError("Laplacian matrix dimensions don't match number of nodes")

    def get_neighbors(self, node_id: int) -> List[int]:
        """Get list of neighbor node IDs."""
        if node_id not in self.node_ids:
            raise ValueError(f"Node {node_id} not in topology")
        idx = self.node_ids.index(node_id)
        return [
            self.node_ids[i]
            for i, connected in enumerate(self.adjacency_matrix[idx])
            if connected > 0
        ]

    def get_degree(self, node_id: int) -> int:
        """Get degree of a node."""
        return len(self.get_neighbors(node_id))


@dataclass(frozen=True)
class ConsensusState:
    """State of consensus algorithm across network."""

    iteration: int
    node_states: List[NodeState]
    topology: NetworkTopology
    weight_matrix: np.ndarray
    convergence_metric: float
    timestamp: Timestamp
    # Extended fields for adaptive consensus
    spectral_gap: Optional[float] = None
    step_size: Optional[float] = None
    variance_regulation: Optional[Dict[str, Any]] = None
    ml_predictions: Optional[Dict[str, Any]] = None
    adaptation_history: Optional[List[Dict[str, Any]]] = None

    def get_state_by_id(self, node_id: int) -> Optional[NodeState]:
        """Get node state by ID."""
        for state in self.node_states:
            if state.node_id == node_id:
                return state
        return None

    def get_max_frequency_error(self) -> PPB:
        """Get maximum frequency error across all nodes."""
        if not self.node_states:
            return PPB(0.0)
        return max(abs(state.frequency_offset) for state in self.node_states)

    def get_max_timing_error(self) -> Picoseconds:
        """Get maximum timing error across all nodes."""
        if not self.node_states:
            return Picoseconds(0.0)
        return max(abs(state.clock_bias) for state in self.node_states)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "iteration": self.iteration,
            "node_states": [
                self._node_state_to_dict(state) for state in self.node_states
            ],
            "topology": self._topology_to_dict(self.topology),
            "weight_matrix": self.weight_matrix.tolist(),
            "convergence_metric": self.convergence_metric,
            "timestamp": self._timestamp_to_dict(self.timestamp),
            "spectral_gap": self.spectral_gap,
            "step_size": self.step_size,
            "variance_regulation": self.variance_regulation,
            "ml_predictions": self.ml_predictions,
            "adaptation_history": self.adaptation_history,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ConsensusState":
        """Create from dictionary for deserialization."""
        # Handle backward compatibility
        spectral_gap = data.get("spectral_gap")
        step_size = data.get("step_size")
        variance_regulation = data.get("variance_regulation")
        ml_predictions = data.get("ml_predictions")
        adaptation_history = data.get("adaptation_history")

        return cls(
            iteration=data["iteration"],
            node_states=[
                cls._node_state_from_dict(state_data)
                for state_data in data["node_states"]
            ],
            topology=cls._topology_from_dict(data["topology"]),
            weight_matrix=np.array(data["weight_matrix"]),
            convergence_metric=data["convergence_metric"],
            timestamp=cls._timestamp_from_dict(data["timestamp"]),
            spectral_gap=spectral_gap,
            step_size=step_size,
            variance_regulation=variance_regulation,
            ml_predictions=ml_predictions,
            adaptation_history=adaptation_history,
        )

    @staticmethod
    def _node_state_to_dict(state: "NodeState") -> Dict[str, Any]:
        """Convert NodeState to dictionary."""
        return {
            "node_id": state.node_id,
            "clock_bias": state.clock_bias,
            "clock_bias_uncertainty": state.clock_bias_uncertainty,
            "frequency_offset": state.frequency_offset,
            "frequency_offset_uncertainty": state.frequency_offset_uncertainty,
            "last_update": ConsensusState._timestamp_to_dict(state.last_update),
            "quality": state.quality.value,
            # Handle extended fields
            "local_spectral_gap": getattr(state, "local_spectral_gap", None),
            "local_step_size": getattr(state, "local_step_size", None),
            "variance_warning_level": getattr(state, "variance_warning_level", None),
            "ml_prediction_confidence": getattr(
                state, "ml_prediction_confidence", None
            ),
        }

    @staticmethod
    def _node_state_from_dict(data: Dict[str, Any]) -> "NodeState":
        """Create NodeState from dictionary."""
        # Handle backward compatibility
        local_spectral_gap = data.get("local_spectral_gap")
        local_step_size = data.get("local_step_size")
        variance_warning_level = data.get("variance_warning_level")
        ml_prediction_confidence = data.get("ml_prediction_confidence")

        # Create base NodeState
        node_state = NodeState(
            node_id=data["node_id"],
            clock_bias=Picoseconds(data["clock_bias"]),
            clock_bias_uncertainty=Picoseconds(data["clock_bias_uncertainty"]),
            frequency_offset=PPB(data["frequency_offset"]),
            frequency_offset_uncertainty=PPB(data["frequency_offset_uncertainty"]),
            last_update=ConsensusState._timestamp_from_dict(data["last_update"]),
            quality=MeasurementQuality(data["quality"]),
        )

        # Add extended fields if they exist
        if local_spectral_gap is not None:
            object.__setattr__(node_state, "local_spectral_gap", local_spectral_gap)
        if local_step_size is not None:
            object.__setattr__(node_state, "local_step_size", local_step_size)
        if variance_warning_level is not None:
            object.__setattr__(
                node_state, "variance_warning_level", variance_warning_level
            )
        if ml_prediction_confidence is not None:
            object.__setattr__(
                node_state, "ml_prediction_confidence", ml_prediction_confidence
            )

        return node_state

    @staticmethod
    def _topology_to_dict(topology: "NetworkTopology") -> Dict[str, Any]:
        """Convert NetworkTopology to dictionary."""
        return {
            "adjacency_matrix": topology.adjacency_matrix.tolist(),
            "node_ids": topology.node_ids,
            "laplacian": topology.laplacian.tolist(),
            "spectral_gap": topology.spectral_gap,
            "is_connected": topology.is_connected,
        }

    @staticmethod
    def _topology_from_dict(data: Dict[str, Any]) -> "NetworkTopology":
        """Create NetworkTopology from dictionary."""
        return NetworkTopology(
            adjacency_matrix=np.array(data["adjacency_matrix"]),
            node_ids=data["node_ids"],
            laplacian=np.array(data["laplacian"]),
            spectral_gap=data["spectral_gap"],
            is_connected=data["is_connected"],
        )

    @staticmethod
    def _timestamp_to_dict(timestamp: "Timestamp") -> Dict[str, Any]:
        """Convert Timestamp to dictionary."""
        return {
            "time": timestamp.time,
            "uncertainty": timestamp.uncertainty,
            "quality": timestamp.quality.value,
        }

    @staticmethod
    def _timestamp_from_dict(data: Dict[str, Any]) -> "Timestamp":
        """Create Timestamp from dictionary."""
        return Timestamp(
            time=Seconds(data["time"]),
            uncertainty=Picoseconds(data["uncertainty"]),
            quality=MeasurementQuality(data["quality"]),
        )


@dataclass(frozen=True)
class AdaptiveConsensusState(ConsensusState):
    """Extended consensus state with adaptive-specific fields."""

    spectral_margin_estimator_state: Optional[Dict[str, Any]] = None
    step_size_tuner_state: Optional[Dict[str, Any]] = None
    variance_safeguard_state: Optional[Dict[str, Any]] = None
    ml_hooks_state: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        base_dict = super().to_dict()
        base_dict.update(
            {
                "spectral_margin_estimator_state": self.spectral_margin_estimator_state,
                "step_size_tuner_state": self.step_size_tuner_state,
                "variance_safeguard_state": self.variance_safeguard_state,
                "ml_hooks_state": self.ml_hooks_state,
                "type": "AdaptiveConsensusState",  # Add type marker for deserialization
            }
        )
        return base_dict

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AdaptiveConsensusState":
        """Create from dictionary for deserialization."""
        # Remove type marker if present
        if "type" in data:
            data = {k: v for k, v in data.items() if k != "type"}

        # Create base ConsensusState first
        consensus_state = ConsensusState.from_dict(data)

        # Create AdaptiveConsensusState with additional fields
        return cls(
            iteration=consensus_state.iteration,
            node_states=consensus_state.node_states,
            topology=consensus_state.topology,
            weight_matrix=consensus_state.weight_matrix,
            convergence_metric=consensus_state.convergence_metric,
            timestamp=consensus_state.timestamp,
            spectral_gap=consensus_state.spectral_gap,
            step_size=consensus_state.step_size,
            variance_regulation=consensus_state.variance_regulation,
            ml_predictions=consensus_state.ml_predictions,
            adaptation_history=consensus_state.adaptation_history,
            spectral_margin_estimator_state=data.get("spectral_margin_estimator_state"),
            step_size_tuner_state=data.get("step_size_tuner_state"),
            variance_safeguard_state=data.get("variance_safeguard_state"),
            ml_hooks_state=data.get("ml_hooks_state"),
        )


@dataclass(frozen=True)
class KalmanState:
    """Kalman filter state for local estimation."""

    state_vector: np.ndarray  # [bias, frequency, drift_rate, ...]
    covariance_matrix: np.ndarray
    process_noise: np.ndarray
    measurement_noise: np.ndarray
    last_update: Timestamp

    def __post_init__(self):
        n_states = len(self.state_vector)
        if self.covariance_matrix.shape != (n_states, n_states):
            raise ValueError("Covariance matrix dimensions don't match state vector")
        if len(self.process_noise) != n_states:
            raise ValueError("Process noise dimensions don't match state vector")
        if len(self.measurement_noise) != n_states:
            raise ValueError("Measurement noise dimensions don't match state vector")

    def get_bias(self) -> Picoseconds:
        """Get clock bias from state vector."""
        return Picoseconds(self.state_vector[0])

    def get_frequency_offset(self) -> PPB:
        """Get frequency offset from state vector."""
        return PPB(self.state_vector[1])

    def get_bias_uncertainty(self) -> Picoseconds:
        """Get clock bias uncertainty."""
        return Picoseconds(np.sqrt(self.covariance_matrix[0, 0]))

    def get_frequency_uncertainty(self) -> PPB:
        """Get frequency offset uncertainty."""
        return PPB(np.sqrt(self.covariance_matrix[1, 1]))


@dataclass(frozen=True)
class OscillatorModel:
    """Model of an oscillator with phase noise characteristics."""

    nominal_frequency: Hertz
    phase_noise_profile: Dict[float, Decibels]  # offset_freq -> noise_level
    temperature_coefficient: float  # ppm/°C
    aging_rate: float  # ppb/day
    current_temperature: float  # °C
    current_age_days: float

    @property
    def frequency(self) -> Hertz:
        """Backward-compatible alias for ``nominal_frequency``."""
        return self.nominal_frequency

    @property
    def phase_noise_enabled(self) -> bool:
        """Return True when a phase-noise profile is configured."""
        return bool(self.phase_noise_profile)

    def get_phase_noise_at(self, offset_freq: Hertz) -> Decibels:
        """Interpolate phase noise at specific offset frequency."""
        if not self.phase_noise_profile:
            return Decibels(-100.0)  # Default value

        # Simple linear interpolation in log-log space
        freqs = np.array(list(self.phase_noise_profile.keys()))
        noise_levels = np.array(list(self.phase_noise_profile.values()))

        if offset_freq <= freqs[0]:
            return self.phase_noise_profile[freqs[0]]
        if offset_freq >= freqs[-1]:
            return self.phase_noise_profile[freqs[-1]]

        # Log-log interpolation
        log_freqs = np.log10(freqs)
        log_noise = np.array(noise_levels)
        log_offset = np.log10(offset_freq)

        interpolated_noise = np.interp(log_offset, log_freqs, log_noise)
        return Decibels(interpolated_noise)


@dataclass(frozen=True)
class ChannelModel:
    """Multipath channel model."""

    delay_spread: Picoseconds
    path_delays: List[Picoseconds]
    path_gains: List[float]  # Linear scale
    doppler_shift: Hertz
    temperature: float  # °C
    humidity: float  # %

    def __post_init__(self):
        if len(self.path_delays) != len(self.path_gains):
            raise ValueError("Path delays and gains must have same length")
        if self.delay_spread < 0:
            raise ValueError("Delay spread must be non-negative")

    def get_impulse_response(self, sampling_rate: Hertz) -> np.ndarray:
        """Generate discrete-time impulse response."""
        numeric_sampling_rate = float(sampling_rate)
        if numeric_sampling_rate <= 0:
            raise ValueError("Sampling rate must be positive")

        sample_delays = [float(delay) * numeric_sampling_rate * 1e-12 for delay in self.path_delays]

        if not sample_delays:
            return np.array([1.0 + 0j])

        max_delay = int(np.ceil(max(sample_delays)))
        ir = np.zeros(max_delay + 2, dtype=complex)

        for delay_samples, gain in zip(sample_delays, self.path_gains):
            integer_part = int(np.floor(delay_samples))
            fractional_part = delay_samples - integer_part

            if fractional_part < 1e-9:
                ir[integer_part] += gain
            else:
                weight_next = fractional_part
                weight_current = 1.0 - fractional_part
                ir[integer_part] += gain * weight_current
                ir[integer_part + 1] += gain * weight_next

        return ir


@dataclass(frozen=True)
class RFConfig:
    """RF front-end configuration."""

    carrier_frequency: Hertz
    bandwidth: Hertz
    transmit_power: Decibels
    noise_figure: Decibels
    adc_resolution: int  # bits
    sampling_rate: Hertz

    def get_noise_floor(self) -> Decibels:
        """Calculate thermal noise floor."""
        thermal_noise = PhysicalConstants.THERMAL_NOISE_DBM_PER_HZ + 10 * np.log10(
            self.bandwidth
        )
        return Decibels(thermal_noise + self.noise_figure)

    def get_quantization_noise(self) -> Decibels:
        """Estimate quantization noise level."""
        # Full-scale sinusoid power
        full_scale_power = 0.0  # dBm (normalized)
        quantization_noise = full_scale_power - 6.02 * self.adc_resolution - 1.76
        return Decibels(quantization_noise)


@dataclass(frozen=True)
class ExperimentConfig:
    """Configuration for an experiment."""

    experiment_id: str
    description: str
    parameters: Dict[str, Union[str, int, float, bool]]
    seed: Optional[int]
    start_time: Timestamp
    expected_duration: Seconds

    def get_parameter(self, name: str, default: Any = None) -> Any:
        """Get parameter value with default."""
        return self.parameters.get(name, default)


@dataclass(frozen=True)
class PerformanceMetrics:
    """Performance metrics for algorithm evaluation."""

    rmse_timing: Picoseconds
    rmse_frequency: PPB
    convergence_time: Seconds
    iterations_to_convergence: int
    final_spectral_gap: float
    communication_overhead: int  # bytes exchanged
    computation_time: Seconds

    def get_timing_rmse_ns(self) -> float:
        """Get timing RMSE in nanoseconds."""
        return self.rmse_timing / 1000.0

    def get_convergence_time_minutes(self) -> float:
        """Get convergence time in minutes."""
        return self.convergence_time / 60.0


@dataclass
class ExperimentResult:
    """Complete results from an experiment."""

    config: ExperimentConfig
    metrics: PerformanceMetrics
    telemetry: List[Any]  # Simplified for now
    final_state: Optional[ConsensusState]
    success: bool
    error_message: Optional[str]
    completion_time: Timestamp
    # Extended fields for adaptive consensus
    adaptive_metrics: Optional[Dict[str, Any]] = None
    spectral_gap_history: Optional[List[float]] = None
    step_size_history: Optional[List[float]] = None
    variance_regulation_history: Optional[List[Dict[str, Any]]] = None

    def __post_init__(self):
        if self.metrics is None:
            raise ValueError("Metrics cannot be None")
        if self.completion_time is None:
            raise ValueError("Completion time cannot be None")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "config": self._config_to_dict(self.config),
            "metrics": self._metrics_to_dict(self.metrics),
            "telemetry": self.telemetry,
            "final_state": self.final_state.to_dict() if self.final_state else None,
            "success": self.success,
            "error_message": self.error_message,
            "completion_time": self._timestamp_to_dict(self.completion_time),
            "adaptive_metrics": self.adaptive_metrics,
            "spectral_gap_history": self.spectral_gap_history,
            "step_size_history": self.step_size_history,
            "variance_regulation_history": self.variance_regulation_history,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExperimentResult":
        """Create from dictionary for deserialization."""
        # Handle backward compatibility
        adaptive_metrics = data.get("adaptive_metrics")
        spectral_gap_history = data.get("spectral_gap_history")
        step_size_history = data.get("step_size_history")
        variance_regulation_history = data.get("variance_regulation_history")

        # Determine if final_state is AdaptiveConsensusState
        final_state_data = data.get("final_state")
        final_state = None
        if final_state_data:
            if final_state_data.get("type") == "AdaptiveConsensusState":
                final_state = AdaptiveConsensusState.from_dict(final_state_data)
            else:
                final_state = ConsensusState.from_dict(final_state_data)

        return cls(
            config=cls._config_from_dict(data["config"]),
            metrics=cls._metrics_from_dict(data["metrics"]),
            telemetry=data["telemetry"],
            final_state=final_state,
            success=data["success"],
            error_message=data["error_message"],
            completion_time=cls._timestamp_from_dict(data["completion_time"]),
            adaptive_metrics=adaptive_metrics,
            spectral_gap_history=spectral_gap_history,
            step_size_history=step_size_history,
            variance_regulation_history=variance_regulation_history,
        )

    @staticmethod
    def _config_to_dict(config: "ExperimentConfig") -> Dict[str, Any]:
        """Convert ExperimentConfig to dictionary."""
        return {
            "experiment_id": config.experiment_id,
            "description": config.description,
            "parameters": config.parameters,
            "seed": config.seed,
            "start_time": ExperimentResult._timestamp_to_dict(config.start_time),
            "expected_duration": config.expected_duration,
        }

    @staticmethod
    def _config_from_dict(data: Dict[str, Any]) -> "ExperimentConfig":
        """Create ExperimentConfig from dictionary."""
        return ExperimentConfig(
            experiment_id=data["experiment_id"],
            description=data["description"],
            parameters=data["parameters"],
            seed=data["seed"],
            start_time=ExperimentResult._timestamp_from_dict(data["start_time"]),
            expected_duration=Seconds(data["expected_duration"]),
        )

    @staticmethod
    def _metrics_to_dict(metrics: "PerformanceMetrics") -> Dict[str, Any]:
        """Convert PerformanceMetrics to dictionary."""
        return {
            "rmse_timing": metrics.rmse_timing,
            "rmse_frequency": metrics.rmse_frequency,
            "convergence_time": metrics.convergence_time,
            "iterations_to_convergence": metrics.iterations_to_convergence,
            "final_spectral_gap": metrics.final_spectral_gap,
            "communication_overhead": metrics.communication_overhead,
            "computation_time": metrics.computation_time,
        }

    @staticmethod
    def _metrics_from_dict(data: Dict[str, Any]) -> "PerformanceMetrics":
        """Create PerformanceMetrics from dictionary."""
        return PerformanceMetrics(
            rmse_timing=Picoseconds(data["rmse_timing"]),
            rmse_frequency=PPB(data["rmse_frequency"]),
            convergence_time=Seconds(data["convergence_time"]),
            iterations_to_convergence=data["iterations_to_convergence"],
            final_spectral_gap=data["final_spectral_gap"],
            communication_overhead=data["communication_overhead"],
            computation_time=Seconds(data["computation_time"]),
        )

    @staticmethod
    def _timestamp_to_dict(timestamp: "Timestamp") -> Dict[str, Any]:
        """Convert Timestamp to dictionary."""
        return {
            "time": timestamp.time,
            "uncertainty": timestamp.uncertainty,
            "quality": timestamp.quality.value,
        }

    @staticmethod
    def _timestamp_from_dict(data: Dict[str, Any]) -> "Timestamp":
        """Create Timestamp from dictionary."""
        return Timestamp(
            time=Seconds(data["time"]),
            uncertainty=Picoseconds(data["uncertainty"]),
            quality=MeasurementQuality(data["quality"]),
        )
