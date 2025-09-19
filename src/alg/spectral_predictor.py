"""Utilities for predicting consensus iteration counts from spectral data."""

from __future__ import annotations

import math


def predict_iterations_to_rmse(
    target_rmse_ps: float,
    initial_rmse_ps: float,
    lambda2: float,
    epsilon: float,
) -> float:
    """Return the predicted iterations needed to reach ``target_rmse_ps``.

    Parameters
    ----------
    target_rmse_ps : float
        Desired steady-state timing RMSE in picoseconds.
    initial_rmse_ps : float
        Initial timing RMSE in picoseconds.
    lambda2 : float
        Algebraic connectivity (second-smallest Laplacian eigenvalue).
    epsilon : float
        Consensus step-size.

    Notes
    -----
    Follows the linear model ||e_k||_2 ≈ ||e_0||_2 (1 - ε λ₂)^k. Degenerate
    parameters return ``math.inf``.
    """

    if target_rmse_ps <= 0.0:
        return 0.0
    if initial_rmse_ps <= 0.0:
        return 0.0
    if target_rmse_ps >= initial_rmse_ps:
        return 0.0
    if epsilon <= 0.0 or lambda2 <= 0.0:
        return math.inf

    decay = 1.0 - epsilon * lambda2
    if decay <= 0.0:
        return 1.0
    if decay >= 1.0:
        return math.inf

    ratio = target_rmse_ps / initial_rmse_ps
    ratio = max(ratio, 1e-12)
    decay = max(decay, 1e-12)

    iterations = math.log(ratio) / math.log(decay)
    return max(iterations, 0.0)
