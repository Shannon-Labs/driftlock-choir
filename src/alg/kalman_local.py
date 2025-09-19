"""Lightweight local Kalman filtering for per-node synchronisation states."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class LocalKFConfig:
    """Static configuration for the two-state Kalman filter."""

    dt: float
    sigma_T: float
    sigma_f: float


class LocalTwoStateKF:
    """Minimal 2×2 Kalman filter tracking clock bias and frequency offset."""

    def __init__(self, cfg: LocalKFConfig, x0: NDArray[np.float64], P0: NDArray[np.float64]):
        if x0.shape != (2,):
            raise ValueError("x0 must have shape (2,)")
        if P0.shape != (2, 2):
            raise ValueError("P0 must have shape (2, 2)")
        self.cfg = cfg
        self._x = np.array(x0, dtype=float)
        self._P = np.array(P0, dtype=float)
        self._F = np.array([[1.0, cfg.dt], [0.0, 1.0]], dtype=float)
        q_diag = np.array([cfg.sigma_T ** 2, cfg.sigma_f ** 2], dtype=float)
        self._Q = np.diag(q_diag)
        self._H = np.eye(2, dtype=float)

    def predict(self) -> None:
        """Time update step."""
        self._x = self._F @ self._x
        self._P = self._F @ self._P @ self._F.T + self._Q

    def update_with_neighbor(self, z: NDArray[np.float64], R: NDArray[np.float64]) -> None:
        """Fuse a pseudo-measurement from a neighbour."""
        if z.shape != (2,):
            raise ValueError("z must have shape (2,)")
        if R.shape != (2, 2):
            raise ValueError("R must have shape (2, 2)")
        S = self._H @ self._P @ self._H.T + R
        K = self._P @ self._H.T @ np.linalg.inv(S)
        innovation = z - self._H @ self._x
        self._x = self._x + K @ innovation
        I = np.eye(2)
        self._P = (I - K @ self._H) @ self._P

    def get_posterior(self) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return posterior state and covariance."""
        return self._x.copy(), self._P.copy()
