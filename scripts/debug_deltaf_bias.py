#!/usr/bin/env python3
"""Synthetic delta-f estimator sanity-check harness."""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from typing import Tuple

import numpy as np

from alg.chronometric_handshake import (
    ChronometricHandshakeConfig,
    ChronometricHandshakeSimulator,
    ChronometricNode,
    ChronometricNodeConfig,
)


@dataclass
class SynthConfig:
    delta_f_hz: float = 50.0
    tau_true_s: float = 80e-9
    sample_rate: float = 200_000.0
    beat_duration_s: float = 20e-6
    snr_db: float = 60.0
    seed: int = 2025


def synthesize_samples(cfg: SynthConfig, tx_freq_hz: float) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(cfg.seed)
    n_samples = max(int(cfg.beat_duration_s * cfg.sample_rate), 256)
    adc_time = np.arange(n_samples, dtype=float) / cfg.sample_rate

    phase = (2.0 * np.pi * cfg.delta_f_hz * adc_time) - (2.0 * np.pi * tx_freq_hz * cfg.tau_true_s)
    samples = np.exp(1j * phase)

    noise_phase = rng.normal(0.0, 1.0 / np.sqrt(cfg.snr_db), size=n_samples)
    samples *= np.exp(1j * noise_phase)

    return adc_time.astype(float), samples.astype(np.complex128)


def run_single(cfg: SynthConfig, debug: bool) -> None:
    logging.basicConfig(level=logging.DEBUG if debug else logging.INFO, format='[%(levelname)s] %(message)s')

    handshake_cfg = ChronometricHandshakeConfig(
        beat_duration_s=cfg.beat_duration_s,
        baseband_rate_factor=1.0,
        min_baseband_rate_hz=cfg.sample_rate,
        min_adc_rate_hz=cfg.sample_rate,
        retune_offsets_hz=(),
        coarse_enabled=False,
        debug_logging=debug,
    )
    simulator = ChronometricHandshakeSimulator(handshake_cfg)

    tx = ChronometricNode(ChronometricNodeConfig(0, 2.4e9, 0.0, 0.0, 0.0))
    rx = ChronometricNode(ChronometricNodeConfig(1, 2.4e9 + cfg.delta_f_hz, 0.0, 0.0, 0.0))

    adc_time, adc_samples = synthesize_samples(cfg, tx_freq_hz=tx.carrier_freq_hz)
    tau_est, tau_raw, delta_f_est, residual_rms, fitted_phase, unwrapped_phase, covariance, intercept = simulator._estimate_parameters(  # pylint: disable=protected-access
        tx,
        rx,
        adc_time,
        adc_samples,
        tau_hint=cfg.tau_true_s,
        theoretical_phase_var=None,
    )

    logging.info('True delta_f: %.6f Hz | Estimated: %.6f Hz | Error: %.3f Hz', cfg.delta_f_hz, delta_f_est, delta_f_est - cfg.delta_f_hz)
    logging.info('True tau: %.6e s | Estimated: %.6e s | Error: %.3e s', cfg.tau_true_s, tau_est, tau_est - cfg.tau_true_s)
    logging.info('Residual RMS: %.3e rad', residual_rms)
    logging.info('Covariance matrix:\n%s', covariance)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Debug delta-f estimator using synthetic data.')
    parser.add_argument('--delta-f-hz', type=float, default=50.0)
    parser.add_argument('--tau-ns', type=float, default=80.0)
    parser.add_argument('--sample-rate', type=float, default=200_000.0)
    parser.add_argument('--beat-duration-us', type=float, default=20.0)
    parser.add_argument('--snr-db', type=float, default=60.0)
    parser.add_argument('--seed', type=int, default=2025)
    parser.add_argument('--debug', action='store_true', help='Enable verbose estimator logging.')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = SynthConfig(
        delta_f_hz=args.delta_f_hz,
        tau_true_s=args.tau_ns * 1e-9,
        sample_rate=args.sample_rate,
        beat_duration_s=args.beat_duration_us * 1e-6,
        snr_db=args.snr_db,
        seed=args.seed,
    )
    run_single(cfg, debug=args.debug)


if __name__ == '__main__':
    main()
