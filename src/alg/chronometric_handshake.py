"""Chronometric handshake primitives shared across simulation phases.

This module encapsulates the Chronometric Interferometry two-way handshake and
extends it with practical enhancements inspired by carrier-phase time-transfer
literature:

- Optional coarse wideband delay estimates to remove the 2π ambiguity.
- Multi-carrier (retuned) exchanges that provide a synthetic wavelength for
  robust phase unwrapping.
- Lightweight two-ray multipath modelling to stress bias behaviour.

Higher-layer simulations re-use the API to obtain τ̂, Δf̂, covariance estimates,
coarse delay hints, and per-carrier diagnostics for Monte Carlo sweeps.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy import signal

from phy.noise import NoiseGenerator, NoiseParams
from phy.formants import FormantSynthesisConfig
from phy.preamble import Preamble, build_preamble, estimate_delay
from phy.pathfinder import PathfinderConfig, PathfinderResult, find_first_arrival
from mac.scheduler import MacSlots
from chan.tdl import TappedDelayLine, tdl_from_profile
from phy.impairments import ImpairmentConfig, apply_amplifier_nonlinearity, generate_phase_noise

# Physical constant
C = 299_792_458.0  # Speed of light (m/s)

# Numerical safeguards
_EPS = np.finfo(float).eps


@dataclass
class ChronometricNodeConfig:
    """Configuration/state snapshot for a node participating in the handshake."""

    node_id: int
    carrier_freq_hz: float
    phase_offset_rad: float
    clock_bias_s: float
    freq_error_ppm: float


@dataclass(frozen=True)
class ChronometricNode:
    """Thin wrapper providing convenience properties used by the simulator."""

    config: ChronometricNodeConfig

    @property
    def carrier_freq_hz(self) -> float:
        return self.config.carrier_freq_hz

    @property
    def phase_offset_rad(self) -> float:
        return self.config.phase_offset_rad

    @property
    def clock_bias_s(self) -> float:
        return self.config.clock_bias_s

    @property
    def freq_error_ppm(self) -> float:
        return self.config.freq_error_ppm

    @property
    def node_id(self) -> int:
        return self.config.node_id


@dataclass
class HandshakeTrace:
    """Captured intermediate waveforms useful for plotting and debugging."""

    time_us: NDArray[np.float64]
    beat_raw: NDArray[np.complex128]
    beat_filtered: NDArray[np.complex128]
    adc_time_us: NDArray[np.float64]
    adc_samples: NDArray[np.complex128]
    unwrapped_phase: NDArray[np.float64]
    phase_fit: NDArray[np.float64]


@dataclass
class DirectionalMeasurement:
    """Result of a single directed (tx→rx) Chronometric handshake."""

    tau_est_s: float
    tau_true_s: float
    tau_raw_s: float
    delta_f_est_hz: float
    delta_f_true_hz: float
    clock_offset_true_s: float
    residual_phase_rms: float
    covariance: NDArray[np.float64]
    effective_tau_variance_s2: float
    running_variance_tau: Optional[float]
    running_variance_delta_f: Optional[float]
    coarse_tau_est_s: Optional[float]
    coarse_locked: Optional[bool]
    coarse_guard_hit: Optional[bool]
    carrier_frequencies_hz: Tuple[float, ...]
    tau_unwrapped_candidates_s: Optional[Tuple[float, ...]]
    alias_resolved: Optional[bool] = None
    phase_bias_rad: Optional[float] = None
    hardware_delay_s: float = 0.0
    calibration_offset_s: float = 0.0
    trace: Optional[HandshakeTrace] = None
    pathfinder: Optional[PathfinderResult] = None


@dataclass
class TwoWayHandshakeResult:
    """Combined outcome of the BEACON/RESPONSE exchange."""

    forward: DirectionalMeasurement
    reverse: DirectionalMeasurement
    tof_est_s: float
    tof_true_s: float
    clock_offset_est_s: float
    clock_offset_true_s: float
    delta_f_est_hz: float
    delta_f_true_hz: float
    tof_variance_s2: float
    clock_offset_variance_s2: float
    delta_f_variance_hz2: float
    coarse_tof_est_s: Optional[float]
    reciprocity_bias_s: float
    delta_t_schedule_us: Tuple[float, ...]


@dataclass
class ChronometricHandshakeConfig:
    """Handshake configuration shared across simulations."""

    beat_duration_s: float = 20e-6
    baseband_rate_factor: float = 20.0
    min_baseband_rate_hz: float = 200_000.0
    min_adc_rate_hz: float = 20_000.0
    filter_relative_bw: float = 1.4
    phase_noise_psd: float = -80.0 # Legacy, kept for baseline comparison
    jitter_rms_s: float = 1e-12
    retune_offsets_hz: Tuple[float, ...] = ()
    coarse_enabled: bool = False
    coarse_bandwidth_hz: float = 20e6
    coarse_duration_s: float = 5e-6
    coarse_variance_floor_ps: float = 50.0
    coarse_preamble_mode: str = 'zadoff'
    coarse_formant_profile: Optional[str] = None
    coarse_formant_fundamental_hz: float = 25_000.0
    coarse_formant_harmonic_count: int = 12
    coarse_formant_include_fundamental: bool = False
    coarse_formant_scale: float = 1_000.0
    coarse_formant_phase_jitter: float = 0.0
    coarse_formant_missing_fundamental: bool = True
    channel_profile: Optional[str] = None  # TDL profile (e.g. "INDOOR_OFFICE")
    impairments: Optional[ImpairmentConfig] = None # Physical hardware impairments
    delta_t_schedule_us: Tuple[float, ...] = (0.0,)
    d_tx_ns: Optional[Dict[int, float]] = None
    d_rx_ns: Optional[Dict[int, float]] = None
    calibration_mode: str = 'off'
    loopback_cal_noise_ps: float = 10.0
    mac: Optional[MacSlots] = None
    use_phase_slope_fit: bool = False  # Opt-in phase-slope fusion for multi-carrier tau estimation
    use_theoretical_variance: bool = False  # Use theoretical phase variance for covariance
    pathfinder_enabled: bool = True
    pathfinder_relative_threshold_db: float = -12.0
    pathfinder_noise_guard_multiplier: float = 6.0
    pathfinder_smoothing_kernel: int = 5
    pathfinder_guard_interval_s: float = 30e-9
    pathfinder_pre_guard_ns: float = 0.0
    pathfinder_aperture_duration_ns: float = 100.0
    pathfinder_use_simple_search: bool = True
    pathfinder_first_path_blend: float = 0.05
    pathfinder_aperture_lead_ns: float = 60.0
    pathfinder_aperture_trail_ns: float = 40.0
    pathfinder_threshold_relax_db: float = 6.0
    pathfinder_local_percentile: float = 60.0
    pathfinder_fractional_oversample: int = 4
    pathfinder_min_peak_ratio: float = 0.18
    pathfinder_fractional_refine: bool = False
    pathfinder_refine_window_ns: float = 20.0
    pathfinder_refine_grid: int = 32
    multipath_two_ray_alpha: Optional[float] = None
    pathfinder_alpha: float = 0.3



@dataclass
class _RunningVariance:
    """Numerically stable running variance tracker (Welford)."""

    mean: float = 0.0
    m2: float = 0.0
    count: int = 0

    def update(self, value: float) -> None:
        self.count += 1
        delta = value - self.mean
        self.mean += delta / self.count
        delta2 = value - self.mean
        self.m2 += delta * delta2

    def variance(self) -> Optional[float]:
        if self.count < 2:
            return None
        return self.m2 / (self.count - 1)


@dataclass
class _DirectionalStats:
    """Aggregated error statistics for a directed handshake configuration."""

    tau_stats: _RunningVariance = field(default_factory=_RunningVariance)
    delta_f_stats: _RunningVariance = field(default_factory=_RunningVariance)


class ChronometricHandshakeSimulator:
    """Implements beat synthesis, filtering, estimation, and statistics."""

    def __init__(self, config: ChronometricHandshakeConfig):
        self.cfg = config
        self._filter_cache: Dict[Tuple[float, float], NDArray[np.float64]] = {}
        self._stats_cache: Dict[Tuple[str, float, float, float, int], _DirectionalStats] = {}
        self._coarse_waveform_cache: Dict[Tuple[int, float], Preamble] = {}
        self._delta_t_schedule_us = tuple(config.delta_t_schedule_us) if config.delta_t_schedule_us else (0.0,)
        self._calibration_mode = config.calibration_mode.lower()
        self._tx_delay_s = {
            int(node_id): float(delay_ns) * 1e-9
            for node_id, delay_ns in (config.d_tx_ns or {}).items()
        }
        self._rx_delay_s = {
            int(node_id): float(delay_ns) * 1e-9
            for node_id, delay_ns in (config.d_rx_ns or {}).items()
        }
        all_nodes = set(self._tx_delay_s.keys()) | set(self._rx_delay_s.keys())
        rng = np.random.default_rng(314159)
        noise_std = float(config.loopback_cal_noise_ps) * 1e-12
        self._tx_calibration = {}
        self._rx_calibration = {}
        for node in all_nodes:
            tx_delay = self._tx_delay_s.get(node, 0.0)
            rx_delay = self._rx_delay_s.get(node, 0.0)
            if self._calibration_mode == 'perfect':
                self._tx_calibration[node] = tx_delay
                self._rx_calibration[node] = rx_delay
            elif self._calibration_mode == 'loopback':
                self._tx_calibration[node] = tx_delay + rng.normal(0.0, noise_std)
                self._rx_calibration[node] = rx_delay + rng.normal(0.0, noise_std)
            else:
                self._tx_calibration[node] = 0.0
                self._rx_calibration[node] = 0.0
        self._mac_slots = config.mac
        self._channel_profile = config.channel_profile
        self._impairments = config.impairments
        if config.pathfinder_enabled:
            kernel = int(max(1, config.pathfinder_smoothing_kernel))
            if kernel % 2 == 0:
                kernel += 1
            self._pathfinder_cfg: Optional[PathfinderConfig] = PathfinderConfig(
                relative_threshold_db=float(config.pathfinder_relative_threshold_db),
                noise_guard_multiplier=float(config.pathfinder_noise_guard_multiplier),
                smoothing_kernel=kernel,
                guard_interval_s=float(config.pathfinder_guard_interval_s),
                pre_guard_interval_s=float(config.pathfinder_pre_guard_ns) * 1e-9,
                aperture_duration_ns=float(config.pathfinder_aperture_duration_ns),
                use_simple_search=bool(config.pathfinder_use_simple_search),
                aperture_lead_ns=float(config.pathfinder_aperture_lead_ns),
                aperture_trail_ns=float(config.pathfinder_aperture_trail_ns),
                threshold_relax_db=float(config.pathfinder_threshold_relax_db),
                local_floor_percentile=float(config.pathfinder_local_percentile),
                fractional_oversample=max(int(config.pathfinder_fractional_oversample), 1),
                minimum_peak_ratio=self._profile_min_peak_ratio(float(config.pathfinder_min_peak_ratio)),
            )
        else:
            self._pathfinder_cfg = None
        self._fractional_refine_enabled = bool(config.pathfinder_fractional_refine)
        self._fractional_refine_window_ns = float(config.pathfinder_refine_window_ns)
        self._fractional_refine_grid = max(int(config.pathfinder_refine_grid), 1)

    # Public API -------------------------------------------------------

    def run_two_way(
        self,
        node_a: ChronometricNode,
        node_b: ChronometricNode,
        distance_m: float,
        snr_db: float,
        rng: np.random.Generator,
        capture_trace: bool = False,
        retune_offsets_hz: Optional[Sequence[float]] = None,
        delta_t_schedule_us: Optional[Sequence[float]] = None,
    ) -> Tuple[TwoWayHandshakeResult, Optional[Dict[str, HandshakeTrace]]]:
        """Execute BEACON/RESPONSE handshake for a node pair."""

        offsets = tuple(retune_offsets_hz) if retune_offsets_hz is not None else self.cfg.retune_offsets_hz
        schedule = (
            tuple(float(val) for val in delta_t_schedule_us)
            if delta_t_schedule_us is not None
            else self._delta_t_schedule_us
        )

        forward, forward_trace = self._simulate_direction(
            tx=node_a,
            rx=node_b,
            distance_m=distance_m,
            snr_db=snr_db,
            rng=rng,
            capture_trace=capture_trace,
            retune_offsets_hz=offsets,
        )
        reverse, reverse_trace = self._simulate_direction(
            tx=node_b,
            rx=node_a,
            distance_m=distance_m,
            snr_db=snr_db,
            rng=rng,
            capture_trace=capture_trace,
            retune_offsets_hz=offsets,
        )

        tof_est = 0.5 * (forward.tau_est_s + reverse.tau_est_s)
        tof_true = 0.5 * (forward.tau_true_s + reverse.tau_true_s)
        clock_offset_est = 0.5 * (forward.tau_est_s - reverse.tau_est_s)
        clock_offset_true = 0.5 * (forward.tau_true_s - reverse.tau_true_s)
        delta_f_est = 0.5 * (forward.delta_f_est_hz - reverse.delta_f_est_hz)
        delta_f_true = forward.delta_f_true_hz

        forward_var_tau = _select_variance(forward)
        reverse_var_tau = _select_variance(reverse)
        forward_var_df = _select_variance(forward, tau=False)
        reverse_var_df = _select_variance(reverse, tau=False)

        tof_variance = 0.25 * (forward_var_tau + reverse_var_tau)
        clock_offset_variance = 0.25 * (forward_var_tau + reverse_var_tau)
        delta_f_variance = 0.25 * (forward_var_df + reverse_var_df)

        coarse_values = [
            val
            for val in (forward.coarse_tau_est_s, reverse.coarse_tau_est_s)
            if val is not None
        ]
        coarse_tof_est = float(np.mean(coarse_values)) if coarse_values else None

        trace_dict: Optional[Dict[str, HandshakeTrace]] = None
        if capture_trace:
            trace_dict = {
                'forward': forward_trace,
                'reverse': reverse_trace,
            }

        residual_forward = forward.hardware_delay_s - forward.calibration_offset_s
        residual_reverse = reverse.hardware_delay_s - reverse.calibration_offset_s
        reciprocity_bias = residual_forward - residual_reverse

        return (
            TwoWayHandshakeResult(
                forward=forward,
                reverse=reverse,
                tof_est_s=tof_est,
                tof_true_s=tof_true,
                clock_offset_est_s=clock_offset_est,
                clock_offset_true_s=clock_offset_true,
                delta_f_est_hz=delta_f_est,
                delta_f_true_hz=delta_f_true,
                tof_variance_s2=tof_variance,
                clock_offset_variance_s2=clock_offset_variance,
                delta_f_variance_hz2=delta_f_variance,
                coarse_tof_est_s=coarse_tof_est,
                reciprocity_bias_s=reciprocity_bias,
                delta_t_schedule_us=schedule,
            ),
            trace_dict,
        )

    # Internal helpers -------------------------------------------------

    def _sample_channel(self, rng: np.random.Generator) -> Optional[TappedDelayLine]:
        if not self._channel_profile:
            return None
        return tdl_from_profile(self._channel_profile, rng)

    def _condition_channel(
        self,
        channel: Optional[TappedDelayLine],
        tau_true: float,
        pathfinder: Optional[PathfinderResult],
    ) -> Optional[TappedDelayLine]:
        if channel is None or pathfinder is None or self._pathfinder_cfg is None:
            return channel
        relative_cut = max(pathfinder.first_path_s - tau_true, 0.0) + self._pathfinder_cfg.guard_interval_s
        return channel.window(relative_cut)

    def _simulate_direction(
        self,
        tx: ChronometricNode,
        rx: ChronometricNode,
        distance_m: float,
        snr_db: float,
        rng: np.random.Generator,
        capture_trace: bool,
        retune_offsets_hz: Tuple[float, ...],
    ) -> Tuple[DirectionalMeasurement, Optional[HandshakeTrace]]:
        """Simulate a single directed handshake measurement."""

        tau_true = self._apparent_tau(tx, rx, distance_m)
        channel = self._sample_channel(rng)
        coarse_estimate = self._coarse_delay_estimate(tau_true, snr_db, rng, channel)
        tau_hint = coarse_estimate.tau_est_s if coarse_estimate.tau_est_s is not None else tau_true
        conditioned_channel = self._condition_channel(channel, tau_true, coarse_estimate.pathfinder)
        hardware_delay = self._hardware_delay(tx.node_id, rx.node_id)

        # Base (primary) carrier
        (
            base_measurement,
            base_trace,
            base_raw_tau,
            base_len_adc,
            base_decimation,
            base_phase_bias,
            base_intercept,
            base_theoretical_var,
        ) = self._single_carrier_measurement(
            tx=tx,
            rx=rx,
            distance_m=distance_m,
            tau_true=tau_true,
            tau_hint=tau_hint,
            snr_db=snr_db,
            rng=rng,
            capture_trace=capture_trace,
            channel=conditioned_channel,
            pathfinder=coarse_estimate.pathfinder,
            coarse_tau_est=coarse_estimate.tau_est_s,
        )

        carrier_frequencies = [tx.carrier_freq_hz]
        intercepts = [base_intercept]
        raw_candidates = [(tx.carrier_freq_hz, base_raw_tau)]
        traces: Optional[HandshakeTrace] = base_trace
        all_covariances = [base_measurement.covariance]
        all_delta_f_est = [base_measurement.delta_f_est_hz]
        all_residual_rms = [base_measurement.residual_phase_rms]
        all_theoretical_vars: List[float] = []
        if base_theoretical_var is not None:
            all_theoretical_vars.append(base_theoretical_var)

        # Multi-carrier retunes (comb tones)
        for offset in retune_offsets_hz:
            if np.isclose(offset, 0.0):
                continue
            retuned_tx = self._retune_node(tx, offset)
            retuned_rx = self._retune_node(rx, offset)
            (
                measurement,
                _,
                raw_tau,
                _,
                _,
                _,
                intercept,
                theoretical_var,
            ) = self._single_carrier_measurement(
                tx=retuned_tx,
                rx=retuned_rx,
                distance_m=distance_m,
                tau_true=tau_true,
                tau_hint=tau_hint,
                snr_db=snr_db,
                rng=rng,
                capture_trace=False,
                channel=conditioned_channel,
                pathfinder=coarse_estimate.pathfinder,
                coarse_tau_est=coarse_estimate.tau_est_s,
            )
            carrier_frequencies.append(retuned_tx.carrier_freq_hz)
            intercepts.append(intercept)
            raw_candidates.append((retuned_tx.carrier_freq_hz, raw_tau))
            all_covariances.append(measurement.covariance)
            all_delta_f_est.append(measurement.delta_f_est_hz)
            all_residual_rms.append(measurement.residual_phase_rms)
            if theoretical_var is not None:
                all_theoretical_vars.append(theoretical_var)

        # Coarse correction for primary if available
        if coarse_estimate.tau_est_s is not None and raw_candidates:
            primary_freq, primary_raw = raw_candidates[0]
            correction_cycles = np.round((tau_hint - primary_raw) * primary_freq)
            tau_hint = primary_raw + correction_cycles / max(primary_freq, 1.0)

        # Multi-carrier estimation: phase slope fit if multiple carriers
        num_carriers = len(carrier_frequencies)
        use_phase_slope = (
            hasattr(self.cfg, 'use_phase_slope_fit') and self.cfg.use_phase_slope_fit and num_carriers > 1
        )
        if use_phase_slope and num_carriers > 1:
            # Fit intercept_k = theta - 2 pi f_k tau + noise
            # Linear: intercept ~ f, slope = -2 pi tau
            f_arr = np.array(carrier_frequencies, dtype=float)
            intercept_arr = np.array(intercepts, dtype=float)
            A_slope = np.vstack([f_arr, np.ones_like(f_arr)]).T
            slope, _ = np.linalg.lstsq(A_slope, intercept_arr, rcond=None)[0]
            tau_est_from_slope = -slope / (2.0 * np.pi)

            # Variance of slope: assume independent intercepts, var_intercept_k ≈ sigma_phase_k^2 / N_k
            # Use theoretical if available, else average residual
            if self.cfg.use_theoretical_variance and all_theoretical_vars:
                var_intercept_avg = float(np.mean(all_theoretical_vars))
            else:
                avg_sigma_phase_sq = np.mean([rms**2 * (base_len_adc - 2) for rms in all_residual_rms])
                var_intercept_avg = avg_sigma_phase_sq
            # LS var(slope) = sigma^2 / sum( (f_i - fbar)^2 )
            f_mean = np.mean(f_arr)
            sum_dev_sq = np.sum((f_arr - f_mean) ** 2)
            if sum_dev_sq <= 0:
                var_slope = np.inf
                tau_var_from_slope = np.inf
            else:
                var_slope = var_intercept_avg / sum_dev_sq
                tau_var_from_slope = var_slope / ((2.0 * np.pi) ** 2)

            # Update base_measurement with slope-based tau
            tau_est_final = float(tau_est_from_slope)
            # Combine delta_f: average
            delta_f_est_final = float(np.mean(all_delta_f_est))
            # Covariance: approximate diagonal for tau, average for delta_f
            avg_delta_f_var = np.mean([cov[1,1] for cov in all_covariances])
            covariance_final = np.diag([tau_var_from_slope, avg_delta_f_var])
            tau_unwrapped_candidates = None  # Slope fit resolves ambiguity globally
        else:
            # Fallback to existing weighted tau average
            tau_est_final, tau_unwrapped_candidates = self._unwrap_candidates(
                raw_candidates,
                tau_hint,
            )
            # Average delta_f and covariance
            delta_f_est_final = float(np.mean(all_delta_f_est))
            avg_tau_var = np.mean([cov[0, 0] for cov in all_covariances])
            avg_delta_f_var = np.mean([cov[1, 1] for cov in all_covariances])
            covariance_final = np.diag([avg_tau_var / num_carriers, avg_delta_f_var / num_carriers])  # Weighted approx

        calibration_offset = self._calibration_offset(tx.node_id, rx.node_id)
        if tau_unwrapped_candidates:
            tau_unwrapped_candidates = [candidate - calibration_offset for candidate in tau_unwrapped_candidates]
        tau_est_final -= calibration_offset
        tau_true_calibrated = tau_true - calibration_offset

        candidate_array = tuple(tau_unwrapped_candidates) if tau_unwrapped_candidates else None

        base_measurement.tau_est_s = tau_est_final
        base_measurement.tau_unwrapped_candidates_s = candidate_array
        base_measurement.carrier_frequencies_hz = tuple(carrier_frequencies)
        base_measurement.coarse_tau_est_s = coarse_estimate.tau_est_s
        base_measurement.coarse_locked = coarse_estimate.locked
        base_measurement.coarse_guard_hit = coarse_estimate.guard_hit
        base_measurement.alias_resolved = (
            abs(tau_est_final - tau_true_calibrated) <= abs(base_raw_tau - tau_true)
        )
        base_measurement.phase_bias_rad = base_phase_bias
        base_measurement.delta_f_est_hz = delta_f_est_final
        base_measurement.hardware_delay_s = hardware_delay
        base_measurement.calibration_offset_s = calibration_offset
        base_measurement.covariance = covariance_final
        base_measurement.effective_tau_variance_s2 = float(covariance_final[0, 0])

        cache_key = self._stats_cache_key(
            tx_id=tx.node_id,
            rx_id=rx.node_id,
            distance_m=distance_m,
            snr_db=snr_db,
            delta_f_hz=base_measurement.delta_f_true_hz,
            n_adc=base_len_adc,
            decimation=base_decimation,
        )
        stats = self._stats_cache.setdefault(cache_key, _DirectionalStats())
        stats.tau_stats.update(base_measurement.tau_est_s - tau_true)
        stats.delta_f_stats.update(base_measurement.delta_f_est_hz - base_measurement.delta_f_true_hz)

        base_measurement.running_variance_tau = stats.tau_stats.variance()
        base_measurement.running_variance_delta_f = stats.delta_f_stats.variance()

        return base_measurement, traces

    def _single_carrier_measurement(
        self,
        tx: ChronometricNode,
        rx: ChronometricNode,
        distance_m: float,
        tau_true: float,
        tau_hint: float,
        snr_db: float,
        rng: np.random.Generator,
        capture_trace: bool,
        channel: Optional[TappedDelayLine],
        pathfinder: Optional[PathfinderResult],
        coarse_tau_est: Optional[float],
    ) -> Tuple[DirectionalMeasurement, Optional[HandshakeTrace], float, int, int, Optional[float], float, Optional[float]]:
        delta_f_true = rx.carrier_freq_hz - tx.carrier_freq_hz
        baseband_rate = max(
            self.cfg.baseband_rate_factor * max(abs(delta_f_true), 1.0),
            self.cfg.min_baseband_rate_hz,
        )
        n_samples = max(int(self.cfg.beat_duration_s * baseband_rate), 256)
        t = np.arange(n_samples) / baseband_rate

        # --- Physically Accurate Signal Generation ---
        # Instead of an ideal beat phase, we now model the full TX/RX chain

        # 1. Generate phase noise for each oscillator if impairments are enabled
        tx_phase_noise = np.zeros(n_samples)
        rx_phase_noise = np.zeros(n_samples)
        if self._impairments and self._impairments.phase_noise_h:
            tx_phase_noise = generate_phase_noise(
                n_samples, baseband_rate, self._impairments.phase_noise_h, rng
            )
            rx_phase_noise = generate_phase_noise(
                n_samples, baseband_rate, self._impairments.phase_noise_h, rng
            )

        # 2. Create TX and RX LO signals with their own phase noise
        tx_lo_phase = 2 * np.pi * tx.carrier_freq_hz * t + tx.phase_offset_rad + tx_phase_noise
        rx_lo_phase = 2 * np.pi * rx.carrier_freq_hz * t + rx.phase_offset_rad + rx_phase_noise

        # 3. Create transmitted signal (assuming ideal modulation for now)
        tx_signal = np.exp(1j * tx_lo_phase)

        # 4. Apply transmitter impairments (e.g., amplifier non-linearity)
        if self._impairments:
            tx_signal = apply_amplifier_nonlinearity(
                tx_signal, self._impairments.amp_c1, self._impairments.amp_c3
            )

        # 5. Propagate signal through channel (delay + multipath)
        # We model delay by time-shifting the signal before mixing
        delayed_tx_signal = self._fractional_delay(
            tx_signal,
            tau_true,
            baseband_rate,
            carrier_freq_hz=tx.carrier_freq_hz,
        )

        # 6. Create the beat signal by mixing at the receiver
        rx_lo_signal = np.exp(1j * rx_lo_phase)
        beat_clean = delayed_tx_signal * np.conj(rx_lo_signal)

        beat_clean, phase_bias = self._apply_channel_effects(
            beat_clean,
            baseband_rate,
            tx.carrier_freq_hz,
            rng,
            channel=channel,
        )

        noise_gen = NoiseGenerator(
            NoiseParams(
                snr_db=snr_db,
                phase_noise_psd=self.cfg.phase_noise_psd,
                jitter_rms=self.cfg.jitter_rms_s,
            ),
            sample_rate=baseband_rate,
        )
        beat_noisy = noise_gen.add_awgn(beat_clean, rng=rng)

        # Theoretical phase variance for covariance (integrated over beat BW ≈ baseband_rate / 2)
        theoretical_phase_var = None
        if self.cfg.use_theoretical_variance:
            beat_bw = baseband_rate / 2.0  # Approximate beat bandwidth
            theoretical_phase_var = noise_gen.integrated_phase_variance(beat_bw, tx.carrier_freq_hz)

        beat_filtered = self._bandpass_filter(beat_noisy, baseband_rate, delta_f_true)
        adc_time, adc_samples, decimation = self._downsample_adc(beat_filtered, baseband_rate, delta_f_true)

        (
            tau_est,
            tau_raw,
            delta_f_est,
            residual_rms,
            fitted_phase,
            unwrapped_phase,
            covariance,
            intercept,
        ) = self._estimate_parameters(
            tx=tx,
            rx=rx,
            adc_time=adc_time,
            adc_samples=adc_samples,
            tau_hint=tau_hint,
            theoretical_phase_var=theoretical_phase_var,
        )

        # The phase slope from mixing yields f_tx - f_rx; flip sign so the
        # reported estimate tracks Δf = f_rx - f_tx used elsewhere.
        delta_f_est = -delta_f_est
        covariance = covariance.copy()
        covariance[0, 1] *= -1.0
        covariance[1, 0] *= -1.0

        trace: Optional[HandshakeTrace] = None
        if capture_trace:
            max_vis_samples = min(2000, len(t))
            trace = HandshakeTrace(
                time_us=t[:max_vis_samples] * 1e6,
                beat_raw=beat_noisy[:max_vis_samples],
                beat_filtered=beat_filtered[:max_vis_samples],
                adc_time_us=adc_time * 1e6,
                adc_samples=adc_samples,
                unwrapped_phase=unwrapped_phase,
                phase_fit=fitted_phase,
            )

        measurement = DirectionalMeasurement(
            tau_est_s=tau_est,
            tau_true_s=tau_true,
            tau_raw_s=tau_raw,
            delta_f_est_hz=delta_f_est,
            delta_f_true_hz=delta_f_true,
            clock_offset_true_s=rx.clock_bias_s - tx.clock_bias_s,
            residual_phase_rms=residual_rms,
            covariance=covariance,
            effective_tau_variance_s2=float(covariance[0, 0]),
            running_variance_tau=None,
            running_variance_delta_f=None,
            coarse_tau_est_s=None if coarse_tau_est is None else float(coarse_tau_est),
            coarse_locked=None,
            coarse_guard_hit=None,
            carrier_frequencies_hz=(tx.carrier_freq_hz,),
            tau_unwrapped_candidates_s=None,
            phase_bias_rad=phase_bias,
            pathfinder=pathfinder,
            trace=trace,
        )

        return measurement, trace, tau_raw, len(adc_samples), decimation, phase_bias, intercept, theoretical_phase_var

    def _apparent_tau(self, tx: ChronometricNode, rx: ChronometricNode, distance_m: float) -> float:
        """Propagation delay plus relative clock bias perceived during measurement."""
        geometric = distance_m / C
        clock_skew = rx.clock_bias_s - tx.clock_bias_s
        hardware = self._hardware_delay(tx.node_id, rx.node_id)
        return geometric + clock_skew + hardware


    def _bandpass_filter(
        self,
        signal_in: NDArray[np.complex128],
        sample_rate: float,
        delta_f_hz: float,
    ) -> NDArray[np.complex128]:
        """Apply a narrow band-pass around the beat tone."""
        centre = max(abs(delta_f_hz), 1.0)
        cache_key = (sample_rate, centre)
        sos = self._filter_cache.get(cache_key)
        if sos is None:
            half_bw = max(
                centre * (self.cfg.filter_relative_bw - 1.0) / 2.0,
                0.2 * centre,
            )
            low = max(1.0, centre - half_bw)
            high = min(sample_rate / 2.0 - 1.0, centre + half_bw)
            if high <= low:
                return signal_in
            sos = signal.butter(
                N=2,
                Wn=[low, high],
                btype='bandpass',
                fs=sample_rate,
                output='sos',
            )
            self._filter_cache[cache_key] = sos
        return signal.sosfilt(sos, signal_in)

    def _downsample_adc(
        self,
        signal_in: NDArray[np.complex128],
        sample_rate: float,
        delta_f_hz: float,
    ) -> Tuple[NDArray[np.float64], NDArray[np.complex128], int]:
        """Sub-sample the beat at ~4×Δf to avoid Nyquist ambiguity."""
        target_rate = max(4.0 * max(abs(delta_f_hz), 1.0), self.cfg.min_adc_rate_hz)
        decimation = max(int(sample_rate // target_rate), 1)
        adc_samples = signal_in[::decimation]
        adc_time = np.arange(len(adc_samples)) * decimation / sample_rate
        return adc_time.astype(float), adc_samples, decimation

    def _apply_channel_effects(
        self,
        signal_in: NDArray[np.complex128],
        sample_rate: float,
        carrier_freq_hz: float,
        rng: np.random.Generator,
        channel: Optional[TappedDelayLine],
    ) -> Tuple[NDArray[np.complex128], Optional[float]]:
        if channel is None:
            if not self._channel_profile:
                return signal_in, None
            channel = tdl_from_profile(self._channel_profile, rng)
        processed = channel.apply_to_waveform(signal_in, sample_rate)
        response = channel.narrowband_response(carrier_freq_hz)
        phase_bias = float(np.angle(response)) if response != 0.0 else None
        return processed, phase_bias

    def _estimate_parameters(
        self,
        tx: ChronometricNode,
        rx: ChronometricNode,
        adc_time: NDArray[np.float64],
        adc_samples: NDArray[np.complex128],
        tau_hint: float,
        theoretical_phase_var: Optional[float] = None,
    ) -> Tuple[float, float, float, float, NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], float]:
        """Closed-form τ / Δf estimator from phase samples using optional hint."""
        if len(adc_samples) < 8:
            raise RuntimeError('Insufficient ADC samples for estimation')

        unwrapped_phase = np.unwrap(np.angle(adc_samples))
        A = np.vstack([adc_time, np.ones_like(adc_time)]).T
        xtx = A.T @ A
        xtx_inv = np.linalg.inv(xtx)
        slope, intercept = np.linalg.lstsq(A, unwrapped_phase, rcond=None)[0]
        delta_f_est = slope / (2.0 * np.pi)

        theta_diff = tx.phase_offset_rad - rx.phase_offset_rad
        tau_candidate = (theta_diff - intercept) / (2.0 * np.pi * tx.carrier_freq_hz)

        tau_est = self._unwrap_single_tau(
            tau_candidate=tau_candidate,
            tau_hint=tau_hint,
            carrier_freq_hz=tx.carrier_freq_hz,
        )

        fitted_phase = (A @ np.array([slope, intercept])).astype(float)
        residual = unwrapped_phase - fitted_phase
        residual_rms = float(np.sqrt(np.mean(residual**2)))

        # Use theoretical phase variance if configured and provided
        if (
            hasattr(self.cfg, 'use_theoretical_variance')
            and self.cfg.use_theoretical_variance
            and theoretical_phase_var is not None
        ):
            sigma_phase_sq = float(theoretical_phase_var)
        else:
            sigma_phase_sq = float(np.dot(residual, residual) / max(len(adc_samples) - 2, 1))
        
        cov_params = sigma_phase_sq * xtx_inv

        var_slope = cov_params[0, 0]
        var_intercept = cov_params[1, 1]
        cov_slope_intercept = cov_params[0, 1]

        delta_f_variance = var_slope / (2.0 * np.pi) ** 2
        tau_variance = var_intercept / (2.0 * np.pi * tx.carrier_freq_hz) ** 2
        cov_tau_delta_f = (
            -cov_slope_intercept
            / ((2.0 * np.pi) ** 2 * tx.carrier_freq_hz)
        )

        covariance = np.array(
            [
                [tau_variance, cov_tau_delta_f],
                [cov_tau_delta_f, delta_f_variance],
            ],
            dtype=float,
        )

        return (
            tau_est,
            float(tau_candidate),
            float(delta_f_est),
            residual_rms,
            fitted_phase,
            unwrapped_phase,
            covariance,
            float(intercept),
        )

    def _unwrap_single_tau(
        self,
        tau_candidate: float,
        tau_hint: float,
        carrier_freq_hz: float,
    ) -> float:
        cycles = np.round((tau_hint - tau_candidate) * carrier_freq_hz)
        tau_est = tau_candidate + cycles / carrier_freq_hz
        return float(tau_est)

    def _unwrap_candidates(
        self,
        candidates: Sequence[Tuple[float, float]],
        tau_hint: float,
    ) -> Tuple[float, List[float]]:
        if not candidates:
            raise ValueError('At least one carrier measurement is required')

        unwrapped = []
        current_hint = tau_hint
        for freq_hz, tau_raw in candidates:
            tau_unwrapped = self._unwrap_single_tau(
                tau_candidate=tau_raw,
                tau_hint=current_hint,
                carrier_freq_hz=freq_hz,
            )
            unwrapped.append((freq_hz, tau_unwrapped))
            current_hint = tau_unwrapped

        weights = np.array([freq ** 2 for freq, _ in unwrapped], dtype=float)
        taus = np.array([tau for _, tau in unwrapped], dtype=float)
        tau_final = float(np.average(taus, weights=weights)) if np.any(weights) else float(np.mean(taus))
        return tau_final, [tau for _, tau in unwrapped]

    def _coarse_delay_estimate(
        self,
        tau_true: float,
        snr_db: float,
        rng: np.random.Generator,
        channel: Optional[TappedDelayLine],
    ) -> CoarseEstimate:
        if not self.cfg.coarse_enabled:
            return CoarseEstimate(None, None)
        sample_rate = max(self.cfg.coarse_bandwidth_hz, self.cfg.min_baseband_rate_hz)
        if self._mac_slots is not None:
            n_samples = max(self._mac_slots.preamble_len, 128)
        else:
            n_samples = max(int(self.cfg.coarse_duration_s * sample_rate), 128)
        preamble = self._get_coarse_preamble(n_samples, sample_rate)
        transmitted = preamble.samples
        received = self._fractional_delay(transmitted, tau_true, sample_rate)
        if channel is not None:
            received = channel.apply_to_waveform(received, sample_rate)
        signal_power = float(np.mean(np.abs(received) ** 2) + _EPS)
        snr_linear = max(10 ** (snr_db / 10.0), _EPS)
        noise_power = signal_power / snr_linear
        noise_std = np.sqrt(noise_power / 2.0)
        noise = rng.normal(0.0, noise_std, size=received.shape) + 1j * rng.normal(0.0, noise_std, size=received.shape)
        noisy = received + noise
        upsample = 128
        effective_rate = sample_rate
        corr_signal = noisy
        corr_preamble = preamble
        if upsample > 1:
            hi_length = n_samples * upsample
            transmitted_hi = signal.resample(transmitted, hi_length)
            noisy_hi = signal.resample(noisy, hi_length)
            matched_hi = np.conj(transmitted_hi[::-1])
            preamble_hi = Preamble(
                samples=transmitted_hi.astype(np.complex128),
                matched_filter=matched_hi.astype(np.complex128),
                metadata=preamble.metadata,
            )
            corr_signal = noisy_hi
            corr_preamble = preamble_hi
            effective_rate = sample_rate * upsample
        
        corr = signal.convolve(corr_signal, corr_preamble.matched_filter, mode='full')
        magnitude = np.abs(corr)
        
        use_pathfinder = self._pathfinder_cfg is not None and channel is not None
        if use_pathfinder:
            pf_result = find_first_arrival(corr_signal, corr_preamble, effective_rate, self._pathfinder_cfg)
            tau_est = None
            if pf_result is not None:
                peak_s = float(pf_result.peak_path_s)
                first_s = float(pf_result.first_path_s)
                delta = float(peak_s - first_s)
                blend_eff = self._effective_first_path_blend(delta, pf_result)
                if blend_eff > 0.0 and np.isfinite(delta):
                    tau_est = peak_s - blend_eff * delta
                else:
                    tau_est = peak_s
        else:
            tau_est = estimate_delay(corr_signal, corr_preamble, effective_rate)
            pf_result = None
        guard_hit = False
        tau_value: Optional[float]
        if tau_est is None or not np.isfinite(tau_est):
            tau_value = None
        else:
            # Refine around the selected tau estimate (first-path or peak)
            tau_value, guard_hit = self._refine_coarse_tau(
                tau_est,
                magnitude,
                effective_rate,
                corr_preamble.length,
            )
            if (
                tau_value is not None
                and self._fractional_refine_enabled
                and self._pathfinder_cfg is not None
                and pf_result is not None
            ):
                tau_value = self._fractional_peak_search(
                    corr_signal,
                    corr_preamble,
                    effective_rate,
                    tau_value,
                    self._fractional_refine_window_ns,
                    self._fractional_refine_grid,
                )
        locked: Optional[bool]
        if tau_value is None:
            locked = None
        else:
            tolerance = self._pathfinder_cfg.guard_interval_s if self._pathfinder_cfg else (2.0 / effective_rate)
            locked = abs(tau_value - tau_true) <= tolerance
        return CoarseEstimate(tau_value, pf_result, guard_hit=guard_hit, locked=locked)


    def _get_coarse_preamble(self, n_samples: int, sample_rate: float) -> Preamble:
        mode = (self.cfg.coarse_preamble_mode or 'zadoff').lower()
        if mode == 'formant':
            descriptor_key = (
                'formant',
                (self.cfg.coarse_formant_profile or 'A').upper(),
                round(float(self.cfg.coarse_formant_fundamental_hz), 3),
                int(max(self.cfg.coarse_formant_harmonic_count, 1)),
                bool(self.cfg.coarse_formant_include_fundamental),
                round(float(self.cfg.coarse_formant_scale), 3),
                round(float(self.cfg.coarse_formant_phase_jitter), 3),
                bool(self.cfg.coarse_formant_missing_fundamental),
            )
        else:
            descriptor_key = ('zadoff', 1)

        cache_key = (n_samples, round(float(sample_rate), 6), descriptor_key)
        preamble = self._coarse_waveform_cache.get(cache_key)
        if preamble is None:
            if mode == 'formant':
                profile = (self.cfg.coarse_formant_profile or 'A').upper()
                formant_cfg = FormantSynthesisConfig(
                    profile=profile,
                    fundamental_hz=float(self.cfg.coarse_formant_fundamental_hz),
                    harmonic_count=int(max(self.cfg.coarse_formant_harmonic_count, 1)),
                    include_fundamental=bool(self.cfg.coarse_formant_include_fundamental),
                    formant_scale=float(self.cfg.coarse_formant_scale),
                    phase_jitter=float(self.cfg.coarse_formant_phase_jitter),
                )
                preamble, _ = build_preamble(
                    length=n_samples,
                    sample_rate=sample_rate,
                    bandwidth_hz=self.cfg.coarse_bandwidth_hz,
                    mode='formant',
                    formant_config=formant_cfg,
                )
                if not self.cfg.coarse_formant_missing_fundamental:
                    metadata = dict(preamble.metadata or {})
                    metadata['formant_analyze'] = False
                    preamble = Preamble(
                        samples=preamble.samples,
                        matched_filter=preamble.matched_filter,
                        metadata=metadata,
                    )
            else:
                preamble, _ = build_preamble(
                    length=n_samples,
                    sample_rate=sample_rate,
                    bandwidth_hz=self.cfg.coarse_bandwidth_hz,
                )
            self._coarse_waveform_cache[cache_key] = preamble
        return preamble

    def _profile_min_peak_ratio(self, base_ratio: float) -> float:
        profile = (self._channel_profile or '').upper()
        ratio = float(np.clip(base_ratio, 0.0, 1.0))
        if profile.startswith('INDOOR'):
            ratio = max(ratio, 0.24)
        elif profile.startswith('URBAN'):
            ratio = min(ratio, 0.16)
        return float(np.clip(ratio, 0.05, 0.6))

    def _effective_first_path_blend(
        self,
        delta_s: float,
        pf_result: Optional[PathfinderResult],
    ) -> float:
        base = float(np.clip(self.cfg.pathfinder_first_path_blend, 0.0, 1.0))
        if base <= 0.0 or not np.isfinite(delta_s) or delta_s <= 0.0:
            return 0.0

        guard_s = self._pathfinder_cfg.guard_interval_s if (self._pathfinder_cfg and self._pathfinder_cfg.guard_interval_s > 0.0) else None
        window_scale = 1.0
        if guard_s:
            ratio = float(delta_s / guard_s)
            if ratio <= 1.0:
                window_scale = max(ratio, 0.0)
            else:
                window_scale = 1.0 / (1.0 + max(ratio - 1.0, 0.0))

        profile = (self._channel_profile or '').upper()
        if profile.startswith('INDOOR'):
            profile_scale = 1.0
        elif profile.startswith('URBAN'):
            profile_scale = 0.35
        else:
            profile_scale = 0.6

        amplitude_boost = 1.0
        if pf_result is not None and np.isfinite(pf_result.peak_to_first_ratio):
            amp_ratio = float(np.clip(pf_result.peak_to_first_ratio, 0.0, 1.0))
            # When the first path is weak relative to the peak, emphasize the blend.
            amplitude_boost = 1.0 + 1.5 * (1.0 - amp_ratio)

        effective = base * window_scale * profile_scale * amplitude_boost
        return float(np.clip(effective, 0.0, 0.95))

    def _refine_coarse_tau(
        self,
        tau_initial: float,
        magnitude: NDArray[np.float64],
        sample_rate: float,
        preamble_len: int,
    ) -> Tuple[float, bool]:
        if magnitude.size < 3 or sample_rate <= 0.0:
            return float(tau_initial), False

        initial_index = tau_initial * sample_rate + (preamble_len - 1)
        guard_hit = bool(initial_index <= 1 or initial_index >= magnitude.size - 2)
        clamped_index = int(np.clip(round(initial_index), 1, magnitude.size - 2))

        y_prev = float(magnitude[clamped_index - 1])
        y_curr = float(magnitude[clamped_index])
        y_next = float(magnitude[clamped_index + 1])
        denom = (y_prev - 2.0 * y_curr + y_next)
        delta = 0.0
        if abs(denom) > 1e-12:
            delta = 0.5 * (y_prev - y_next) / denom
            delta = float(np.clip(delta, -0.5, 0.5))

        refined_lag = (clamped_index + delta) - (preamble_len - 1)
        tau_refined = refined_lag / sample_rate
        return float(tau_refined), guard_hit

    def _fractional_peak_search(
        self,
        corr_signal: NDArray[np.complex128],
        preamble: Preamble,
        sample_rate: float,
        tau_initial: float,
        window_ns: float,
        grid_points: int,
    ) -> float:
        if grid_points <= 1:
            return float(tau_initial)
        half_window_s = max(window_ns * 0.5e-9, 1.0 / sample_rate)
        start_tau = float(tau_initial - half_window_s)
        stop_tau = float(tau_initial + half_window_s)
        taus = np.linspace(start_tau, stop_tau, grid_points)
        best_tau = float(tau_initial)
        best_val = -np.inf
        for tau in taus:
            shift_samples = int(np.floor(tau * sample_rate))
            frac = tau * sample_rate - shift_samples
            if shift_samples < 0 or shift_samples + preamble.length >= corr_signal.size:
                continue
            segment = corr_signal[shift_samples:shift_samples + preamble.length]
            weight = np.exp(-2j * np.pi * frac * np.arange(preamble.length) / preamble.length)
            value = np.abs(np.vdot(segment * weight, preamble.matched_filter[:preamble.length]))
            if value > best_val:
                best_val = value
                best_tau = tau
        return float(best_tau)

    def _fractional_delay(
        self,
        signal_in: NDArray[np.complex128],
        delay_s: float,
        sample_rate: float,
        carrier_freq_hz: Optional[float] = None,
    ) -> NDArray[np.complex128]:
        """High-fidelity fractional delay that preserves carrier phase."""
        if signal_in.size == 0 or np.isclose(delay_s, 0.0):
            return signal_in.copy()

        samples = np.asarray(signal_in, dtype=np.complex128)
        delay_samples = float(delay_s * sample_rate)

        # Separate integer and fractional portions of the desired delay.
        frac_delay, integer_delay = np.modf(delay_samples)
        integer_delay = int(integer_delay)
        if frac_delay < 0.0:
            frac_delay += 1.0
            integer_delay -= 1

        # Optionally mix to baseband to avoid aliasing error when delaying a high-frequency tone.
        if carrier_freq_hz is not None:
            time_axis = np.arange(samples.size, dtype=float) / sample_rate
            baseband_phasor = np.exp(-1j * 2.0 * np.pi * carrier_freq_hz * time_axis)
            baseband_signal = samples * baseband_phasor
        else:
            baseband_signal = samples

        # Windowed-sinc kernel centred on the fractional remainder.
        num_taps = 129  # 64 samples on each side keeps sub-ps error at MHz rates.
        half_len = num_taps // 2
        tap_index = np.arange(num_taps, dtype=float)
        kernel = np.sinc(tap_index - half_len - frac_delay)
        kernel *= signal.windows.kaiser(num_taps, beta=8.6)
        kernel /= np.sum(kernel)

        filtered = signal.fftconvolve(baseband_signal, kernel, mode='full')
        filtered = filtered[half_len : half_len + samples.size]

        if integer_delay > 0:
            output = np.zeros_like(samples, dtype=np.complex128)
            if integer_delay < samples.size:
                output[integer_delay:] = filtered[: samples.size - integer_delay]
        elif integer_delay < 0:
            output = np.zeros_like(samples, dtype=np.complex128)
            shift = min(samples.size, -integer_delay)
            if shift < samples.size:
                output[: samples.size - shift] = filtered[shift:]
        else:
            output = filtered.copy()

        if carrier_freq_hz is not None:
            # Re-apply the carrier, accounting for the absolute propagation delay.
            time_axis = np.arange(samples.size, dtype=float) / sample_rate
            remod_phasor = np.exp(1j * 2.0 * np.pi * carrier_freq_hz * (time_axis - delay_s))
            output *= remod_phasor

        return output

    def _hardware_delay(self, tx_id: int, rx_id: int) -> float:
        return self._tx_delay_s.get(tx_id, 0.0) + self._rx_delay_s.get(rx_id, 0.0)

    def _calibration_offset(self, tx_id: int, rx_id: int) -> float:
        offset_tx = self._tx_calibration.get(tx_id, 0.0)
        offset_rx = self._rx_calibration.get(rx_id, 0.0)
        return offset_tx + offset_rx

    def _retune_node(self, node: ChronometricNode, offset_hz: float) -> ChronometricNode:
        cfg = node.config
        return ChronometricNode(
            ChronometricNodeConfig(
                node_id=cfg.node_id,
                carrier_freq_hz=cfg.carrier_freq_hz + offset_hz,
                phase_offset_rad=cfg.phase_offset_rad,
                clock_bias_s=cfg.clock_bias_s,
                freq_error_ppm=cfg.freq_error_ppm,
            )
        )

    def _stats_cache_key(
        self,
        tx_id: int,
        rx_id: int,
        distance_m: float,
        snr_db: float,
        delta_f_hz: float,
        n_adc: int,
        decimation: int,
    ) -> Tuple[str, float, float, float, int]:
        direction_id = f"{tx_id}->{rx_id}"
        return (
            direction_id,
            round(distance_m, 6),
            round(abs(delta_f_hz), 3),
            round(snr_db, 2),
            max(n_adc, 1) * max(decimation, 1),
        )


def _select_variance(measurement: DirectionalMeasurement, tau: bool = True) -> float:
    """Helper selecting the best variance estimate available."""
    running = measurement.running_variance_tau if tau else measurement.running_variance_delta_f
    if running is not None and running > 0.0:
        return running
    if tau:
        variance = float(measurement.effective_tau_variance_s2)
    else:
        variance = float(measurement.covariance[1, 1])
    return max(variance, _EPS)


@dataclass(frozen=True)
class CoarseEstimate:
    """Bundle describing the coarse delay hint and guard status."""

    tau_est_s: Optional[float]
    pathfinder: Optional[PathfinderResult]
    guard_hit: bool = False
    locked: Optional[bool] = None


def simulate_handshake_pair(
    node_a: ChronometricNode,
    node_b: ChronometricNode,
    distance_m: float,
    snr_db: float,
    rng: np.random.Generator,
    simulator: Optional[ChronometricHandshakeSimulator] = None,
    config: Optional[ChronometricHandshakeConfig] = None,
    capture_trace: bool = False,
    retune_offsets_hz: Optional[Sequence[float]] = None,
    delta_t_schedule_us: Optional[Sequence[float]] = None,
) -> Tuple[TwoWayHandshakeResult, Optional[Dict[str, HandshakeTrace]]]:
    """Convenience wrapper returning handshake result + optional trace."""
    if simulator is None:
        if config is None:
            config = ChronometricHandshakeConfig()
        simulator = ChronometricHandshakeSimulator(config)
    return simulator.run_two_way(
        node_a=node_a,
        node_b=node_b,
        distance_m=distance_m,
        snr_db=snr_db,
        rng=rng,
        capture_trace=capture_trace,
        retune_offsets_hz=retune_offsets_hz,
        delta_t_schedule_us=delta_t_schedule_us,
    )
