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
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy import signal

from phy.noise import NoiseGenerator, NoiseParams
from phy.preamble import Preamble, build_preamble, estimate_delay
from mac.scheduler import MacSlots
from chan.tdl import TappedDelayLine

# Physical constant
C = 299_792_458.0  # Speed of light (m/s)

# Numerical safeguards
_EPS = np.finfo(float).eps


@dataclass(frozen=True)
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
    carrier_frequencies_hz: Tuple[float, ...]
    tau_unwrapped_candidates_s: Optional[Tuple[float, ...]]
    alias_resolved: Optional[bool] = None
    phase_bias_rad: Optional[float] = None
    hardware_delay_s: float = 0.0
    calibration_offset_s: float = 0.0
    trace: Optional[HandshakeTrace] = None


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
    phase_noise_psd: float = -80.0
    jitter_rms_s: float = 1e-12
    retune_offsets_hz: Tuple[float, ...] = ()
    coarse_enabled: bool = False
    coarse_bandwidth_hz: float = 20e6
    coarse_duration_s: float = 5e-6
    coarse_variance_floor_ps: float = 50.0
    multipath_two_ray_alpha: Optional[float] = None
    multipath_two_ray_delay_s: Optional[float] = None
    delta_t_schedule_us: Tuple[float, ...] = (0.0,)
    d_tx_ns: Optional[Dict[int, float]] = None
    d_rx_ns: Optional[Dict[int, float]] = None
    calibration_mode: str = 'off'
    loopback_cal_noise_ps: float = 10.0
    mac: Optional[MacSlots] = None
    channel_model: Optional[TappedDelayLine] = None
    use_phase_slope_fit: bool = True  # Use phase slope for multi-carrier tau estimation
    use_theoretical_variance: bool = False  # Use theoretical phase variance for covariance


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
        tau_coarse = self._coarse_delay_estimate(tau_true, snr_db, rng)
        tau_hint = tau_coarse if tau_coarse is not None else tau_true
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
        if tau_coarse is not None and raw_candidates:
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
            slope, theta_est = np.linalg.lstsq(A_slope, intercept_arr, rcond=None)[0]
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
        base_measurement.coarse_tau_est_s = tau_coarse
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
    ) -> Tuple[DirectionalMeasurement, Optional[HandshakeTrace], float, int, int, Optional[float], float, Optional[float]]:
        delta_f_true = rx.carrier_freq_hz - tx.carrier_freq_hz
        baseband_rate = max(
            self.cfg.baseband_rate_factor * max(abs(delta_f_true), 1.0),
            self.cfg.min_baseband_rate_hz,
        )
        n_samples = max(int(self.cfg.beat_duration_s * baseband_rate), 256)
        t = np.arange(n_samples) / baseband_rate

        theta_diff = tx.phase_offset_rad - rx.phase_offset_rad
        beat_phase = self._build_beat_phase(
            t=t,
            delta_f_hz=delta_f_true,
            theta_diff=theta_diff,
            carrier_freq_hz=tx.carrier_freq_hz,
            tau_true=tau_true,
        )
        beat_clean = np.exp(1j * beat_phase)
        beat_clean, phase_bias = self._apply_channel_effects(
            beat_clean,
            baseband_rate,
            tau_true,
            tx.carrier_freq_hz,
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
            coarse_tau_est_s=None,
            carrier_frequencies_hz=(tx.carrier_freq_hz,),
            tau_unwrapped_candidates_s=None,
            phase_bias_rad=phase_bias,
            trace=trace,
        )

        return measurement, trace, tau_raw, len(adc_samples), decimation, phase_bias, intercept, theoretical_phase_var

    def _apparent_tau(self, tx: ChronometricNode, rx: ChronometricNode, distance_m: float) -> float:
        """Propagation delay plus relative clock bias perceived during measurement."""
        geometric = distance_m / C
        clock_skew = rx.clock_bias_s - tx.clock_bias_s
        hardware = self._hardware_delay(tx.node_id, rx.node_id)
        return geometric + clock_skew + hardware

    def _build_beat_phase(
        self,
        t: NDArray[np.float64],
        delta_f_hz: float,
        theta_diff: float,
        carrier_freq_hz: float,
        tau_true: float,
    ) -> NDArray[np.float64]:
        carrier_term = -2.0 * np.pi * carrier_freq_hz * tau_true
        beat_phase = 2.0 * np.pi * delta_f_hz * t + theta_diff + carrier_term
        return beat_phase

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
        tau_true: float,
        carrier_freq_hz: float,
    ) -> Tuple[NDArray[np.complex128], Optional[float]]:
        phase_bias = None
        if self.cfg.channel_model is not None:
            channel = self.cfg.channel_model
            processed = channel.apply_to_waveform(signal_in, sample_rate)
            response = channel.narrowband_response(carrier_freq_hz)
            phase_bias = float(np.angle(response)) if response != 0.0 else None
            return processed, phase_bias
        processed = self._apply_two_ray_multipath(signal_in, sample_rate, tau_true)
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
    ) -> Optional[float]:
        if not self.cfg.coarse_enabled:
            return None
        sample_rate = max(self.cfg.coarse_bandwidth_hz, self.cfg.min_baseband_rate_hz)
        if self._mac_slots is not None:
            n_samples = max(self._mac_slots.preamble_len, 128)
        else:
            n_samples = max(int(self.cfg.coarse_duration_s * sample_rate), 128)
        preamble = self._get_coarse_preamble(n_samples, sample_rate)
        transmitted = preamble.samples
        received = self._fractional_delay(transmitted, tau_true, sample_rate)
        if self.cfg.channel_model is not None:
            received = self.cfg.channel_model.apply_to_waveform(received, sample_rate)
        else:
            received = self._apply_two_ray_multipath(received, sample_rate, tau_true)
        signal_power = float(np.mean(np.abs(received) ** 2) + _EPS)
        snr_linear = max(10 ** (snr_db / 10.0), _EPS)
        noise_power = signal_power / snr_linear
        noise_std = np.sqrt(noise_power / 2.0)
        noise = rng.normal(0.0, noise_std, size=received.shape) + 1j * rng.normal(0.0, noise_std, size=received.shape)
        noisy = received + noise
        upsample = 128
        if upsample > 1:
            hi_length = n_samples * upsample
            transmitted_hi = signal.resample(transmitted, hi_length)
            noisy_hi = signal.resample(noisy, hi_length)
            matched_hi = np.conj(transmitted_hi[::-1])
            preamble_hi = Preamble(samples=transmitted_hi.astype(np.complex128), matched_filter=matched_hi.astype(np.complex128))
            tau_est = estimate_delay(noisy_hi, preamble_hi, sample_rate * upsample)
        else:
            tau_est = estimate_delay(noisy, preamble, sample_rate)
        return float(tau_est)

    def _apply_two_ray_multipath(
        self,
        signal_in: NDArray[np.complex128],
        sample_rate: float,
        tau_true: float,
    ) -> NDArray[np.complex128]:
        alpha = self.cfg.multipath_two_ray_alpha
        if alpha is None or np.isclose(alpha, 0.0):
            return signal_in
        delay = self.cfg.multipath_two_ray_delay_s or (tau_true * 0.5)
        delay_samples = delay * sample_rate
        integer = int(np.floor(delay_samples))
        frac = delay_samples - integer
        echoed = np.zeros_like(signal_in)
        if integer < len(signal_in):
            echoed[integer:] = signal_in[: len(signal_in) - integer]
        if frac > 1e-6:
            echoed[integer:-1] = (
                (1.0 - frac) * echoed[integer:-1]
                + frac * echoed[integer + 1 :]
            )
        return signal_in + alpha * echoed

    def _get_coarse_preamble(self, n_samples: int, sample_rate: float) -> Preamble:
        cache_key = (n_samples, round(float(sample_rate), 6))
        preamble = self._coarse_waveform_cache.get(cache_key)
        if preamble is None:
            preamble, _ = build_preamble(
                length=n_samples,
                sample_rate=sample_rate,
                bandwidth_hz=self.cfg.coarse_bandwidth_hz,
            )
            self._coarse_waveform_cache[cache_key] = preamble
        return preamble

    def _fractional_delay(
        self,
        signal_in: NDArray[np.complex128],
        delay_s: float,
        sample_rate: float,
    ) -> NDArray[np.complex128]:
        if np.isclose(delay_s, 0.0):
            return signal_in
        t_in = np.arange(len(signal_in)) / sample_rate
        t_out = t_in
        real = np.interp(t_out - delay_s, t_in, signal_in.real, left=0.0, right=0.0)
        imag = np.interp(t_out - delay_s, t_in, signal_in.imag, left=0.0, right=0.0)
        return real + 1j * imag

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
