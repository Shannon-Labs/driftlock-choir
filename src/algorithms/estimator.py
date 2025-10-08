"""
τ/Δf estimation algorithms for chronometric interferometry.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm

from ..core.constants import PhysicalConstants
from ..core.types import (BeatNoteData, EstimationResult, Hertz,
                          MeasurementQuality, PhaseMeasurement, Picoseconds,
                          Timestamp)


class TauDeltaEstimator(ABC):
    """
    Abstract base class for τ/Δf estimators.

    This class defines the interface for various estimation algorithms
    that extract time-of-flight (τ) and frequency offset (Δf) from beat-note data.
    """

    @abstractmethod
    def estimate(self, beat_note: BeatNoteData) -> EstimationResult:
        """
        Estimate τ and Δf from beat-note data.

        Args:
            beat_note: Beat-note measurement data

        Returns:
            Estimation result with τ, Δf, and uncertainties
        """
        pass

    @abstractmethod
    def compute_uncertainty(self, measurements: List[PhaseMeasurement]) -> np.ndarray:
        """
        Compute measurement uncertainty covariance.

        Args:
            measurements: List of phase measurements

        Returns:
            Covariance matrix
        """
        pass

    @abstractmethod
    def validate_result(self, result: EstimationResult) -> bool:
        """
        Validate estimation result quality.

        Args:
            result: Estimation result to validate

        Returns:
            True if result is valid
        """
        pass


class PhaseSlopeEstimator(TauDeltaEstimator):
    """
    Phase slope estimator for τ/Δf estimation.

    This estimator uses the linear relationship between phase and frequency
    to estimate time-of-flight from the phase slope across multiple frequencies.
    """

    def __init__(self, frequency_list: Optional[List[Hertz]] = None):
        """
        Initialize phase slope estimator.

        Args:
            frequency_list: List of frequencies for multi-frequency estimation
        """
        self.frequency_list = frequency_list

    def estimate(self, beat_note: BeatNoteData) -> EstimationResult:
        """
        Estimate τ and Δf using phase slope method.

        Args:
            beat_note: Beat-note measurement data

        Returns:
            Estimation result
        """
        if self.frequency_list and len(self.frequency_list) > 1:
            return self._estimate_multi_frequency(beat_note)
        else:
            return self._estimate_single_frequency(beat_note)

    def _estimate_single_frequency(self, beat_note: BeatNoteData) -> EstimationResult:
        """
        Single-frequency estimation (simplified).

        Args:
            beat_note: Beat-note data

        Returns:
            Estimation result
        """
        from ..signal_processing.beat_note import BeatNoteProcessor

        processor = BeatNoteProcessor(beat_note.sampling_rate)

        try:
            time_vector, instantaneous_phase = processor.extract_instantaneous_phase(
                beat_note
            )
        except Exception:
            return processor._estimate_fft_peak(beat_note)

        if len(time_vector) < 3:
            return processor._estimate_fft_peak(beat_note)

        time_vector = time_vector - time_vector[0]

        design_matrix = np.column_stack(
            [time_vector, np.ones_like(time_vector)]
        )

        try:
            solution, residuals, rank, _ = np.linalg.lstsq(
                design_matrix, instantaneous_phase, rcond=None
            )
        except np.linalg.LinAlgError:
            return processor._estimate_fft_peak(beat_note)

        if rank < 2 or not np.all(np.isfinite(solution)):
            return processor._estimate_fft_peak(beat_note)

        slope, intercept = solution
        expected_beat_freq = beat_note.get_beat_frequency()
        beat_freq = float(expected_beat_freq)

        if beat_freq <= 0:
            return processor._estimate_fft_peak(beat_note)

        if residuals.size > 0:
            sigma_squared = residuals[0] / (len(time_vector) - design_matrix.shape[1])
        else:
            fitted_phase = design_matrix @ solution
            residual = instantaneous_phase - fitted_phase
            dof = max(len(time_vector) - design_matrix.shape[1], 1)
            sigma_squared = np.sum(residual**2) / dof

        try:
            cov_phase = sigma_squared * np.linalg.inv(design_matrix.T @ design_matrix)
        except np.linalg.LinAlgError:
            cov_phase = np.eye(2) * sigma_squared

        slope_var = max(cov_phase[0, 0], 1e-18)
        intercept_var = max(cov_phase[1, 1], 1e-18)

        estimated_frequency = slope / (2 * np.pi)
        delta_f = estimated_frequency - beat_freq
        delta_f_uncertainty_hz = np.sqrt(slope_var) / (2 * np.pi)

        sampling_rate_hz = float(beat_note.sampling_rate)
        tx_alias = np.mod(float(beat_note.tx_frequency), sampling_rate_hz)
        rx_alias = np.mod(float(beat_note.rx_frequency), sampling_rate_hz)
        alias_sum = tx_alias + rx_alias

        if alias_sum < 1e-6:
            alias_sum = float(beat_note.rx_frequency)

        tau_seconds = abs(intercept) / (2 * np.pi * alias_sum)
        tau_ps = PhysicalConstants.seconds_to_ps(tau_seconds)
        tau_uncertainty_seconds = np.sqrt(intercept_var) / (2 * np.pi * alias_sum)
        tau_uncertainty_ps = PhysicalConstants.seconds_to_ps(tau_uncertainty_seconds)

        covariance = np.array(
            [
                [tau_uncertainty_ps**2, 0.0],
                [0.0, delta_f_uncertainty_hz**2],
            ]
        )

        residual_std = np.sqrt(sigma_squared)
        likelihood = float(np.clip(np.exp(-residual_std), 0.0, 1.0))

        return EstimationResult(
            tau=Picoseconds(tau_ps),
            tau_uncertainty=Picoseconds(tau_uncertainty_ps),
            delta_f=Hertz(delta_f),
            delta_f_uncertainty=Hertz(delta_f_uncertainty_hz),
            covariance=covariance,
            likelihood=likelihood,
            quality=beat_note.quality,
            method="phase_slope_single",
            timestamp=beat_note.timestamp,
        )

    def _estimate_multi_frequency(self, beat_note: BeatNoteData) -> EstimationResult:
        """
        Multi-frequency estimation using phase slope.

        This is a placeholder for multi-frequency estimation.
        Full implementation would require beat-note data at multiple frequencies.

        Args:
            beat_note: Beat-note data

        Returns:
            Estimation result
        """
        # For now, fall back to single-frequency
        return self._estimate_single_frequency(beat_note)

    def compute_uncertainty(self, measurements: List[PhaseMeasurement]) -> np.ndarray:
        """
        Compute measurement uncertainty covariance.

        Args:
            measurements: List of phase measurements

        Returns:
            2x2 covariance matrix
        """
        if len(measurements) < 2:
            return np.eye(2) * 1e6  # Large uncertainty for insufficient data

        # Extract phases and frequencies
        phases = np.array([m.phase for m in measurements])
        frequencies = np.array([m.frequency for m in measurements])
        phase_uncertainties = np.array([m.uncertainty for m in measurements])

        # Weighted linear regression
        weights = 1.0 / (phase_uncertainties**2)

        # Design matrix for phase = 2π*f*τ + φ₀
        A = np.column_stack([2 * np.pi * frequencies, np.ones_like(frequencies)])

        # Weighted least squares
        W = np.diag(weights)
        ATA = A.T @ W @ A
        covariance = (
            np.linalg.inv(ATA) if np.linalg.cond(ATA) < 1e12 else np.eye(2) * 1e6
        )

        return covariance

    def validate_result(self, result: EstimationResult) -> bool:
        """
        Validate estimation result.

        Args:
            result: Estimation result

        Returns:
            True if result is valid
        """
        # Check for reasonable values
        if abs(result.tau) > 1e9:  # > 1 second
            return False

        if abs(result.delta_f) > 1e6:  # > 1 MHz
            return False

        # Check uncertainties
        if result.tau_uncertainty <= 0 or result.delta_f_uncertainty <= 0:
            return False

        # Check likelihood
        if result.likelihood < 0 or result.likelihood > 1:
            return False

        return True


class MaximumLikelihoodEstimator(TauDeltaEstimator):
    """
    Maximum likelihood estimator for τ/Δf.

    This estimator uses numerical optimization to find the τ and Δf
    that maximize the likelihood function.
    """

    def __init__(self, method: str = "L-BFGS-B"):
        """
        Initialize ML estimator.

        Args:
            method: Optimization method
        """
        self.method = method

    def estimate(self, beat_note: BeatNoteData) -> EstimationResult:
        """
        Estimate τ and Δf using maximum likelihood.

        Args:
            beat_note: Beat-note data

        Returns:
            Estimation result
        """
        # Initial guess
        beat_freq = beat_note.get_beat_frequency()
        initial_tau = 0.0  # picoseconds
        initial_delta_f = 0.0  # Hz

        # Define negative likelihood function
        def negative_likelihood(params):
            tau_ps, delta_f_hz = params

            # Generate expected beat signal with these parameters
            expected_signal = self._generate_expected_signal(
                beat_note, tau_ps, delta_f_hz
            )

            # Compute likelihood (simplified)
            residual = beat_note.waveform - expected_signal
            likelihood = -np.sum(np.abs(residual) ** 2)

            return -likelihood

        # Optimize
        result = minimize(
            negative_likelihood,
            x0=[initial_tau, initial_delta_f],
            method=self.method,
            bounds=[(-1e6, 1e6), (-1e3, 1e3)],  # Reasonable bounds
        )

        if result.success:
            tau_ps, delta_f_hz = result.x

            # Estimate uncertainties from Hessian with robust handling
            try:
                if hasattr(result, "hess_inv") and result.hess_inv is not None:
                    # Handle different types of hess_inv objects
                    if hasattr(result.hess_inv, "todense"):
                        # Sparse matrix
                        covariance = np.array(result.hess_inv.todense())
                    elif hasattr(result.hess_inv, "__array__"):
                        # Array-like object
                        covariance = np.array(result.hess_inv)
                    elif isinstance(result.hess_inv, np.ndarray):
                        # Already numpy array
                        covariance = result.hess_inv
                    else:
                        # For LbfgsInvHessProduct or other special objects
                        # Try to approximate by evaluating on identity
                        try:
                            covariance = np.array(
                                [result.hess_inv @ np.eye(2)[:, i] for i in range(2)]
                            ).T
                        except:
                            # Final fallback
                            covariance = np.eye(2) * 100.0

                    # Validate covariance matrix
                    if covariance.shape == (2, 2) and np.all(np.isfinite(covariance)):
                        tau_uncertainty_ps = np.sqrt(abs(covariance[0, 0]))
                        delta_f_uncertainty_hz = np.sqrt(abs(covariance[1, 1]))
                    else:
                        # Invalid covariance, use fallback
                        tau_uncertainty_ps = 100.0
                        delta_f_uncertainty_hz = 1.0
                        covariance = np.eye(2) * 100.0
                else:
                    # No hessian available
                    tau_uncertainty_ps = 100.0
                    delta_f_uncertainty_hz = 1.0
                    covariance = np.eye(2) * 100.0
            except Exception:
                # Any error in uncertainty estimation
                tau_uncertainty_ps = 100.0
                delta_f_uncertainty_hz = 1.0
                covariance = np.eye(2) * 100.0

            # Likelihood
            likelihood = np.exp(-result.fun / len(beat_note.waveform))

            return EstimationResult(
                tau=Picoseconds(tau_ps),
                tau_uncertainty=Picoseconds(tau_uncertainty_ps),
                delta_f=Hertz(delta_f_hz),
                delta_f_uncertainty=Hertz(delta_f_uncertainty_hz),
                covariance=covariance if hasattr(result, "hess_inv") else np.eye(2),
                likelihood=min(1.0, likelihood),
                quality=beat_note.quality,
                method="maximum_likelihood",
                timestamp=beat_note.timestamp,
            )
        else:
            # Optimization failed, return fallback result
            return self._fallback_estimation(beat_note)

    def _generate_expected_signal(
        self, beat_note: BeatNoteData, tau_ps: float, delta_f_hz: float
    ) -> np.ndarray:
        """
        Generate expected beat signal given τ and Δf.

        Args:
            beat_note: Beat-note data
            tau_ps: Time-of-flight in picoseconds
            delta_f_hz: Frequency offset in Hz

        Returns:
            Expected beat signal
        """
        n_samples = len(beat_note.waveform)
        t = np.arange(n_samples) / beat_note.sampling_rate

        # Convert tau to seconds
        tau_seconds = PhysicalConstants.ps_to_seconds(tau_ps)

        # Expected beat frequency with offset
        beat_freq = beat_note.get_beat_frequency() + delta_f_hz

        # Generate expected signal
        expected_signal = np.exp(
            1j * (2 * np.pi * beat_freq * t + 2 * np.pi * beat_freq * tau_seconds)
        )

        return expected_signal

    def _fallback_estimation(self, beat_note: BeatNoteData) -> EstimationResult:
        """
        Fallback estimation if optimization fails.

        Args:
            beat_note: Beat-note data

        Returns:
            Fallback estimation result
        """
        # Use simple FFT-based estimation as fallback
        from ..signal_processing.beat_note import BeatNoteProcessor

        processor = BeatNoteProcessor(beat_note.sampling_rate)
        return processor._estimate_fft_peak(beat_note)

    def compute_uncertainty(self, measurements: List[PhaseMeasurement]) -> np.ndarray:
        """
        Compute measurement uncertainty covariance.

        Args:
            measurements: List of phase measurements

        Returns:
            2x2 covariance matrix
        """
        # Use same method as phase slope estimator
        slope_estimator = PhaseSlopeEstimator()
        return slope_estimator.compute_uncertainty(measurements)

    def validate_result(self, result: EstimationResult) -> bool:
        """
        Validate estimation result.

        Args:
            result: Estimation result

        Returns:
            True if result is valid
        """
        # Use same validation as phase slope estimator
        slope_estimator = PhaseSlopeEstimator()
        return slope_estimator.validate_result(result)


class CRLBEstimator(TauDeltaEstimator):
    """
    Cramér-Rao Lower Bound estimator.

    This estimator provides theoretical performance bounds and can be used
    to validate other estimators.
    """

    def __init__(self):
        """Initialize CRLB estimator."""
        pass

    def estimate(self, beat_note: BeatNoteData) -> EstimationResult:
        """
        Compute CRLB-based estimation.

        Args:
            beat_note: Beat-note data

        Returns:
            CRLB estimation result
        """
        # Compute CRLB for τ and Δf
        crlb_tau, crlb_delta_f = self._compute_crlb(beat_note)

        # Use any estimator for the actual values
        from ..signal_processing.beat_note import BeatNoteProcessor

        processor = BeatNoteProcessor(beat_note.sampling_rate)
        base_result = processor._estimate_fft_peak(beat_note)

        # Replace uncertainties with CRLB
        return EstimationResult(
            tau=base_result.tau,
            tau_uncertainty=Picoseconds(crlb_tau),
            delta_f=base_result.delta_f,
            delta_f_uncertainty=Hertz(crlb_delta_f),
            covariance=np.diag([crlb_tau**2, crlb_delta_f**2]),
            likelihood=base_result.likelihood,
            quality=base_result.quality,
            method="crlb_bound",
            timestamp=base_result.timestamp,
        )

    def _compute_crlb(self, beat_note: BeatNoteData) -> Tuple[float, float]:
        """
        Compute Cramér-Rao lower bounds.

        Args:
            beat_note: Beat-note data

        Returns:
            Tuple of (crlb_tau_ps, crlb_delta_f_hz)
        """
        # Simplified CRLB calculation
        # Full derivation would be more complex

        n_samples = len(beat_note.waveform)
        snr_linear = 10 ** (beat_note.snr / 10)
        beat_freq = beat_note.get_beat_frequency()

        # CRLB for frequency estimation
        crlb_delta_f = 3.0 / (
            2 * np.pi**2 * snr_linear * n_samples * (n_samples**2 - 1)
        )

        # CRLB for time delay estimation
        if beat_freq > 0:
            crlb_tau = (
                1.0
                / (8 * np.pi**2 * snr_linear * beat_freq**2 * n_samples)
                * PhysicalConstants.PS_PER_SEC
            )
        else:
            crlb_tau = float("inf")

        return crlb_tau, crlb_delta_f

    def compute_uncertainty(self, measurements: List[PhaseMeasurement]) -> np.ndarray:
        """
        Compute CRLB uncertainty covariance.

        Args:
            measurements: List of phase measurements

        Returns:
            2x2 CRLB covariance matrix
        """
        if not measurements:
            return np.eye(2) * 1e6

        # Simplified CRLB calculation
        n = len(measurements)
        avg_phase_uncertainty = np.mean([m.uncertainty for m in measurements])
        avg_frequency = np.mean([m.frequency for m in measurements])

        if avg_frequency > 0:
            crlb_tau = avg_phase_uncertainty / (2 * np.pi * avg_frequency)
            crlb_delta_f = avg_phase_uncertainty / (2 * np.pi * n)
        else:
            crlb_tau = float("inf")
            crlb_delta_f = float("inf")

        return np.diag([crlb_tau**2, crlb_delta_f**2])

    def validate_result(self, result: EstimationResult) -> bool:
        """
        Validate CRLB result.

        Args:
            result: Estimation result

        Returns:
            True if result is valid
        """
        # CRLB should always be positive
        if result.tau_uncertainty <= 0 or result.delta_f_uncertainty <= 0:
            return False

        return True


class EstimatorFactory:
    """
    Factory class for creating estimators.
    """

    @staticmethod
    def create_estimator(estimator_type: str, **kwargs) -> TauDeltaEstimator:
        """
        Create estimator of specified type.

        Args:
            estimator_type: Type of estimator
            **kwargs: Estimator-specific parameters

        Returns:
            Estimator instance
        """
        if estimator_type == "phase_slope":
            return PhaseSlopeEstimator(**kwargs)
        elif estimator_type == "maximum_likelihood":
            return MaximumLikelihoodEstimator(**kwargs)
        elif estimator_type == "crlb":
            return CRLBEstimator(**kwargs)
        else:
            raise ValueError(f"Unknown estimator type: {estimator_type}")

    @staticmethod
    def get_available_estimators() -> List[str]:
        """
        Get list of available estimator types.

        Returns:
            List of estimator type names
        """
        return ["phase_slope", "maximum_likelihood", "crlb"]
