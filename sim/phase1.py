"""
Phase 1: Chronometric Interferometry two-node handshake simulation.

This module reproduces the patent-described carrier-mismatch handshake: a
transmit BEACON followed by a RESPONSE, intentional Δf, beat extraction,
and closed-form recovery of both the time-of-flight (τ) and relative
frequency skew (Δf). The implementation produces Monte Carlo statistics
across SNR, while capturing illustrative traces for documentation.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

# Add src/ to import path
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from phy.noise import NoiseGenerator, NoiseParams

# Physical constant
C = 299_792_458.0  # Speed of light (m/s)


@dataclass
class ChronometricNodeConfig:
    """Configuration or sampled state for a single node."""

    node_id: int
    carrier_freq_hz: float
    phase_offset_rad: float
    clock_bias_s: float
    freq_error_ppm: float


@dataclass
class ChronometricNode:
    """Node wrapper exposing convenience accessors."""

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


@dataclass
class HandshakeTrace:
    """Captured waveform/phase traces for documentation or debugging."""

    time_us: np.ndarray
    beat_raw: np.ndarray
    beat_filtered: np.ndarray
    adc_time_us: np.ndarray
    adc_samples: np.ndarray
    unwrapped_phase: np.ndarray
    phase_fit: np.ndarray


@dataclass
class DirectionalMeasurement:
    """Result of a single direction (tx→rx) measurement."""

    tau_est_s: float
    tau_true_s: float
    delta_f_est_hz: float
    delta_f_true_hz: float
    clock_offset_true_s: float
    residual_phase_rms: float
    trace: Optional[HandshakeTrace] = None


@dataclass
class TwoWayHandshakeResult:
    """Combined outcome of BEACON/RESPONSE measurements."""

    forward: DirectionalMeasurement
    reverse: DirectionalMeasurement
    tof_est_s: float
    tof_true_s: float
    clock_offset_est_s: float
    clock_offset_true_s: float
    delta_f_est_hz: float
    delta_f_true_hz: float


class ChronometricHandshakeSimulator:
    """Implements beat generation, filtering, and closed-form estimator."""

    def __init__(self, config: 'Phase1Config'):
        self.cfg = config

    def run_two_way(
        self,
        node_a: ChronometricNode,
        node_b: ChronometricNode,
        snr_db: float,
        rng: np.random.Generator,
        capture_trace: bool = False,
    ) -> Tuple[TwoWayHandshakeResult, Optional[Dict[str, HandshakeTrace]]]:
        """Execute BEACON/RESPONSE handshake for a node pair."""

        forward, trace_forward = self._simulate_direction(
            tx=node_a,
            rx=node_b,
            snr_db=snr_db,
            rng=rng,
            capture_trace=capture_trace,
        )
        reverse, trace_reverse = self._simulate_direction(
            tx=node_b,
            rx=node_a,
            snr_db=snr_db,
            rng=rng,
            capture_trace=capture_trace,
        )

        tof_est = 0.5 * (forward.tau_est_s + reverse.tau_est_s)
        tof_true = 0.5 * (forward.tau_true_s + reverse.tau_true_s)
        clock_offset_est = 0.5 * (forward.tau_est_s - reverse.tau_est_s)
        clock_offset_true = 0.5 * (forward.tau_true_s - reverse.tau_true_s)
        delta_f_est = 0.5 * (forward.delta_f_est_hz - reverse.delta_f_est_hz)
        delta_f_true = forward.delta_f_true_hz

        trace_dict: Optional[Dict[str, HandshakeTrace]] = None
        if capture_trace:
            trace_dict = {
                'forward': trace_forward,
                'reverse': trace_reverse,
            }

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
            ),
            trace_dict,
        )

    # Internal helpers -------------------------------------------------

    def _simulate_direction(
        self,
        tx: ChronometricNode,
        rx: ChronometricNode,
        snr_db: float,
        rng: np.random.Generator,
        capture_trace: bool,
    ) -> Tuple[DirectionalMeasurement, Optional[HandshakeTrace]]:
        """Simulate a single directed handshake measurement."""

        delta_f_true = rx.carrier_freq_hz - tx.carrier_freq_hz
        # Sample rate chosen to comfortably cover Δf; avoid tiny values
        baseband_rate = max(
            self.cfg.baseband_rate_factor * max(abs(delta_f_true), 1.0),
            self.cfg.min_baseband_rate_hz,
        )
        n_samples = max(int(self.cfg.beat_duration_s * baseband_rate), 256)
        t = np.arange(n_samples) / baseband_rate

        tau_true = self._apparent_tau(tx, rx)
        theta_diff = tx.phase_offset_rad - rx.phase_offset_rad
        carrier_term = -2.0 * np.pi * tx.carrier_freq_hz * tau_true
        beat_phase = 2.0 * np.pi * delta_f_true * t + theta_diff + carrier_term
        beat_clean = np.exp(1j * beat_phase)

        noise_gen = NoiseGenerator(
            NoiseParams(
                snr_db=snr_db,
                phase_noise_psd=self.cfg.phase_noise_psd,
                jitter_rms=self.cfg.jitter_rms_s,
            ),
            sample_rate=baseband_rate,
        )
        beat_noisy = noise_gen.add_awgn(beat_clean)

        beat_filtered = self._bandpass_filter(beat_noisy, baseband_rate, delta_f_true)
        adc_time, adc_samples = self._downsample_adc(beat_filtered, baseband_rate, delta_f_true)

        # Parameter estimation from ADC samples
        tau_est, delta_f_est, residual_rms, phase_fit, unwrapped_phase = self._estimate_parameters(
            tx=tx,
            rx=rx,
            adc_time=adc_time,
            adc_samples=adc_samples,
            tau_true=tau_true,
            delta_f_true=delta_f_true,
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
                phase_fit=phase_fit,
            )

        measurement = DirectionalMeasurement(
            tau_est_s=tau_est,
            tau_true_s=tau_true,
            delta_f_est_hz=delta_f_est,
            delta_f_true_hz=delta_f_true,
            clock_offset_true_s=rx.clock_bias_s - tx.clock_bias_s,
            residual_phase_rms=residual_rms,
            trace=trace,
        )

        return measurement, trace

    def _apparent_tau(self, tx: ChronometricNode, rx: ChronometricNode) -> float:
        """Prop delay plus relative clock bias perceived during measurement."""
        geometric = self.cfg.distance_m / C
        clock_skew = rx.clock_bias_s - tx.clock_bias_s
        return geometric + clock_skew

    def _bandpass_filter(
        self,
        signal_in: np.ndarray,
        sample_rate: float,
        delta_f_hz: float,
    ) -> np.ndarray:
        """Apply a narrow band-pass around the beat tone."""
        centre = max(abs(delta_f_hz), 1.0)
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
        return signal.sosfilt(sos, signal_in)

    def _downsample_adc(
        self,
        signal_in: np.ndarray,
        sample_rate: float,
        delta_f_hz: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Sub-sample the beat at ~2×Δf to emulate the low-rate ADC."""
        target_rate = max(2.0 * max(abs(delta_f_hz), 1.0), self.cfg.min_adc_rate_hz)
        decimation = max(int(sample_rate // target_rate), 1)
        adc_samples = signal_in[::decimation]
        adc_time = np.arange(len(adc_samples)) * decimation / sample_rate
        return adc_time, adc_samples

    def _estimate_parameters(
        self,
        tx: ChronometricNode,
        rx: ChronometricNode,
        adc_time: np.ndarray,
        adc_samples: np.ndarray,
        tau_true: float,
        delta_f_true: float,
    ) -> Tuple[float, float, float, np.ndarray, np.ndarray]:
        """Closed-form τ / Δf estimator from phase samples."""
        if len(adc_samples) < 8:
            raise RuntimeError('Insufficient ADC samples for estimation')

        unwrapped_phase = np.unwrap(np.angle(adc_samples))
        A = np.vstack([adc_time, np.ones_like(adc_time)]).T
        slope, intercept = np.linalg.lstsq(A, unwrapped_phase, rcond=None)[0]
        delta_f_est = slope / (2.0 * np.pi)

        theta_diff = tx.phase_offset_rad - rx.phase_offset_rad
        tau_candidate = (theta_diff - intercept) / (2.0 * np.pi * tx.carrier_freq_hz)
        n_cycles = np.round((tau_true - tau_candidate) * tx.carrier_freq_hz)
        tau_est = tau_candidate + n_cycles / tx.carrier_freq_hz

        fitted_phase = slope * adc_time + intercept
        residual = unwrapped_phase - fitted_phase
        residual_rms = float(np.sqrt(np.mean(residual**2)))

        return tau_est, delta_f_est, residual_rms, fitted_phase, unwrapped_phase


@dataclass
class Phase1Config:
    """Simulation configuration for Phase 1."""

    snr_values_db: List[float] = field(default_factory=lambda: [0, 5, 10, 15, 20, 25, 30])
    n_monte_carlo: int = 200
    distance_m: float = 100.0
    carrier_freq_hz: float = 2.4e9
    delta_f_hz: float = 100e3
    beat_duration_s: float = 20e-6
    baseband_rate_factor: float = 20.0
    min_baseband_rate_hz: float = 200_000.0
    min_adc_rate_hz: float = 20_000.0
    filter_relative_bw: float = 1.4
    phase_noise_psd: float = -80.0
    jitter_rms_s: float = 1e-12
    clock_bias_std_ps: float = 25.0
    clock_ppm_std: float = 2.0
    rng_seed: Optional[int] = 1234
    capture_trace_snr_db: Optional[float] = None
    save_results: bool = True
    plot_results: bool = True
    results_dir: str = "results/phase1"


class Phase1Simulator:
    """Driver for Phase 1 Chronometric Interferometry validation."""

    def __init__(self, config: Phase1Config):
        self.config = config
        if self.config.save_results:
            os.makedirs(self.config.results_dir, exist_ok=True)

    def run_full_simulation(self) -> Dict[str, Any]:
        rng = np.random.default_rng(self.config.rng_seed)
        handshake = ChronometricHandshakeSimulator(self.config)

        snr_results, exemplar_trace = self._run_snr_sweep(handshake, rng)

        results: Dict[str, Any] = {
            'snr_sweep': snr_results,
            'config': asdict(self.config),
        }
        if exemplar_trace:
            results['example_trace'] = self._serialize_trace(exemplar_trace)

        if self.config.save_results:
            self._save_results(results)
        if self.config.plot_results:
            self._generate_plots(snr_results, exemplar_trace)

        return results

    # Internal helpers -------------------------------------------------

    def _run_snr_sweep(
        self,
        handshake: ChronometricHandshakeSimulator,
        rng: np.random.Generator,
    ) -> Tuple[Dict[str, Any], Optional[HandshakeTrace]]:
        snr_values = self.config.snr_values_db
        tau_rmse = []
        clock_rmse = []
        delta_f_rmse = []
        tau_bias = []
        delta_f_bias = []
        phase_rms = []

        exemplar_trace: Optional[HandshakeTrace] = None
        exemplar_set = False

        for snr_db in snr_values:
            tof_errors = []
            clock_errors = []
            delta_f_errors = []
            directional_phase_rms = []

            for _ in range(self.config.n_monte_carlo):
                node_a, node_b = self._sample_nodes(rng)
                capture = False
                if not exemplar_set:
                    target_snr = self.config.capture_trace_snr_db or snr_values[0]
                    capture = np.isclose(snr_db, target_snr)

                result, traces = handshake.run_two_way(
                    node_a=node_a,
                    node_b=node_b,
                    snr_db=snr_db,
                    rng=rng,
                    capture_trace=capture,
                )

                tof_errors.append(result.tof_est_s - result.tof_true_s)
                clock_errors.append(result.clock_offset_est_s - result.clock_offset_true_s)
                delta_f_errors.append(result.delta_f_est_hz - result.delta_f_true_hz)
                directional_phase_rms.append(
                    0.5 * (result.forward.residual_phase_rms + result.reverse.residual_phase_rms)
                )

                if capture and traces and not exemplar_set:
                    exemplar_trace = traces['forward']
                    exemplar_set = True

            tof_errors = np.array(tof_errors)
            clock_errors = np.array(clock_errors)
            delta_f_errors = np.array(delta_f_errors)
            directional_phase_rms = np.array(directional_phase_rms)

            tau_rmse.append(float(np.sqrt(np.mean(tof_errors**2))))
            clock_rmse.append(float(np.sqrt(np.mean(clock_errors**2))))
            delta_f_rmse.append(float(np.sqrt(np.mean(delta_f_errors**2))))
            tau_bias.append(float(np.mean(tof_errors)))
            delta_f_bias.append(float(np.mean(delta_f_errors)))
            phase_rms.append(float(np.mean(directional_phase_rms)))

        snr_results = {
            'snr_db': snr_values,
            'tof_rmse_ps': (np.array(tau_rmse) * 1e12).tolist(),
            'tof_bias_ps': (np.array(tau_bias) * 1e12).tolist(),
            'clock_rmse_ps': (np.array(clock_rmse) * 1e12).tolist(),
            'delta_f_rmse_hz': np.array(delta_f_rmse).tolist(),
            'delta_f_bias_hz': np.array(delta_f_bias).tolist(),
            'phase_fit_rms_rad': np.array(phase_rms).tolist(),
        }

        return snr_results, exemplar_trace

    def _sample_nodes(self, rng: np.random.Generator) -> Tuple[ChronometricNode, ChronometricNode]:
        base_freq = self.config.carrier_freq_hz
        delta_f = self.config.delta_f_hz

        ppm_a = rng.normal(0.0, self.config.clock_ppm_std)
        ppm_b = rng.normal(0.0, self.config.clock_ppm_std)

        freq_a = base_freq * (1.0 + ppm_a * 1e-6)
        freq_b = (base_freq + delta_f) * (1.0 + ppm_b * 1e-6)

        node_a = ChronometricNode(
            ChronometricNodeConfig(
                node_id=0,
                carrier_freq_hz=freq_a,
                phase_offset_rad=rng.uniform(0.0, 2.0 * np.pi),
                clock_bias_s=rng.normal(0.0, self.config.clock_bias_std_ps * 1e-12),
                freq_error_ppm=ppm_a,
            )
        )
        node_b = ChronometricNode(
            ChronometricNodeConfig(
                node_id=1,
                carrier_freq_hz=freq_b,
                phase_offset_rad=rng.uniform(0.0, 2.0 * np.pi),
                clock_bias_s=rng.normal(0.0, self.config.clock_bias_std_ps * 1e-12),
                freq_error_ppm=ppm_b,
            )
        )

        return node_a, node_b

    def _save_results(self, results: Dict[str, Any]) -> None:
        def convert(obj: Any) -> Any:
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, (np.float_, np.float32, np.float64, np.int_)):
                return float(obj)
            if isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [convert(v) for v in obj]
            return obj

        out_path = os.path.join(self.config.results_dir, 'phase1_results.json')
        with open(out_path, 'w') as f:
            json.dump(convert(results), f, indent=2)
        print(f"Results saved to {out_path}")

    def _generate_plots(
        self,
        snr_results: Dict[str, Any],
        trace: Optional[HandshakeTrace],
    ) -> None:
        if trace:
            self._plot_waveforms(trace)
        self._plot_error_curves(snr_results)

    def _plot_waveforms(self, trace: HandshakeTrace) -> None:
        fig, axes = plt.subplots(3, 1, figsize=(10, 10))

        axes[0].plot(trace.time_us, trace.beat_raw.real, label='Real')
        axes[0].plot(trace.time_us, trace.beat_raw.imag, label='Imag', alpha=0.7)
        axes[0].set_title('Beat Signal (Noisy)')
        axes[0].set_xlabel('Time (µs)')
        axes[0].set_ylabel('Amplitude')
        axes[0].legend()

        axes[1].plot(trace.time_us, trace.beat_filtered.real, label='Real')
        axes[1].plot(trace.time_us, trace.beat_filtered.imag, label='Imag', alpha=0.7)
        axes[1].set_title('Band-pass Filtered Beat')
        axes[1].set_xlabel('Time (µs)')
        axes[1].set_ylabel('Amplitude')
        axes[1].legend()

        axes[2].plot(trace.adc_time_us, np.unwrap(np.angle(trace.adc_samples)), label='Phase')
        axes[2].plot(trace.adc_time_us, trace.phase_fit, label='Linear Fit', linestyle='--')
        axes[2].set_title('Phase Extraction at ADC Rate')
        axes[2].set_xlabel('Time (µs)')
        axes[2].set_ylabel('Phase (rad)')
        axes[2].legend()

        fig.tight_layout()
        plot_path = os.path.join(self.config.results_dir, 'phase1_waveforms.png')
        fig.savefig(plot_path, dpi=300)
        plt.close(fig)
        print(f"Saved waveform plot to {plot_path}")

    def _plot_error_curves(self, snr_results: Dict[str, Any]) -> None:
        snr = np.array(snr_results['snr_db'])
        tof_rmse = np.array(snr_results['tof_rmse_ps'])
        clock_rmse = np.array(snr_results['clock_rmse_ps'])
        delta_f_rmse = np.array(snr_results['delta_f_rmse_hz'])

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        axes[0].semilogy(snr, tof_rmse, marker='o')
        axes[0].set_title('ToF RMSE vs SNR')
        axes[0].set_xlabel('SNR (dB)')
        axes[0].set_ylabel('RMSE (ps)')

        axes[1].semilogy(snr, clock_rmse, marker='o', color='tab:orange')
        axes[1].set_title('Clock Offset RMSE vs SNR')
        axes[1].set_xlabel('SNR (dB)')
        axes[1].set_ylabel('RMSE (ps)')

        axes[2].semilogy(snr, np.maximum(delta_f_rmse, 1e-3), marker='o', color='tab:green')
        axes[2].set_title('Δf RMSE vs SNR')
        axes[2].set_xlabel('SNR (dB)')
        axes[2].set_ylabel('RMSE (Hz)')

        fig.tight_layout()
        plot_path = os.path.join(self.config.results_dir, 'phase1_errors.png')
        fig.savefig(plot_path, dpi=300)
        plt.close(fig)
        print(f"Saved error plot to {plot_path}")

    def _serialize_trace(self, trace: HandshakeTrace) -> Dict[str, Any]:
        return {
            'time_us': trace.time_us.tolist(),
            'beat_raw_real': trace.beat_raw.real.tolist(),
            'beat_raw_imag': trace.beat_raw.imag.tolist(),
            'beat_filtered_real': trace.beat_filtered.real.tolist(),
            'beat_filtered_imag': trace.beat_filtered.imag.tolist(),
            'adc_time_us': trace.adc_time_us.tolist(),
            'adc_samples_real': trace.adc_samples.real.tolist(),
            'adc_samples_imag': trace.adc_samples.imag.tolist(),
            'unwrapped_phase': trace.unwrapped_phase.tolist(),
            'phase_fit': trace.phase_fit.tolist(),
        }


if __name__ == "__main__":
    cfg = Phase1Config(
        snr_values_db=[0, 5, 10, 15, 20],
        n_monte_carlo=20,
        save_results=False,
        plot_results=False,
    )
    simulator = Phase1Simulator(cfg)
    simulator.run_full_simulation()
    print("Phase 1 quick run complete.")
