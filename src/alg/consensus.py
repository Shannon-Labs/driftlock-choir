"""Consensus primitives for chronometric synchronization."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import networkx as nx
import numpy as np
from numpy.typing import NDArray
from scipy.sparse.linalg import ArpackNoConvergence, eigsh


@dataclass
class ConsensusOptions:
    """Configuration for decentralized chronometric consensus."""

    max_iterations: int = 1000
    epsilon: Optional[float] = None
    tolerance_ps: float = 100.0
    asynchronous: bool = False
    rng_seed: Optional[int] = None
    enforce_zero_mean: bool = True
    spectral_margin: float = 0.8


@dataclass
class ConsensusResult:
    """Telemetry from the consensus solver."""

    state_history: NDArray[np.float64]
    timing_rms_ps: NDArray[np.float64]
    frequency_rms_hz: NDArray[np.float64]
    converged: bool
    convergence_iteration: Optional[int]
    epsilon: float
    asynchronous: bool
    edge_residuals: Dict[Tuple[int, int], NDArray[np.float64]]
    lambda_max: float
    lambda_2: float
    spectral_gap: float


class DecentralizedChronometricConsensus:
    """Implements Equation (4) variance-weighted consensus updates."""

    def __init__(self, graph: nx.Graph, options: ConsensusOptions):
        if graph.number_of_nodes() == 0:
            raise ValueError("Consensus graph must contain at least one node")
        self.graph = graph
        self.options = options
        self._rng = np.random.default_rng(options.rng_seed)
        self._last_lambda_max: float = 1.0
        self._last_lambda_2: float = 0.0

    # Public API -------------------------------------------------------

    def run(
        self,
        initial_state: NDArray[np.float64],
        true_state: Optional[NDArray[np.float64]] = None,
        measurement_attr: str = "measurement",
        weight_attr: str = "weight_matrix",
    ) -> ConsensusResult:
        """Execute the decentralized consensus iterations."""
        state = np.array(initial_state, dtype=float)
        n_nodes, n_params = state.shape
        if n_params != 2:
            raise ValueError("State must provide [ΔT, Δf] per node (shape: N×2)")
        if n_nodes != self.graph.number_of_nodes():
            raise ValueError("Initial state size does not match graph nodes")

        if true_state is not None:
            true_state = np.array(true_state, dtype=float)
            if true_state.shape != state.shape:
                raise ValueError("True state must match initial state shape")
            if self.options.enforce_zero_mean:
                true_state = self._project_zero_mean(true_state)

        epsilon = self._resolve_step_size()

        state_history = [state.copy()]
        timing_hist = []
        freq_hist = []
        timing_rms, freq_rms = self._compute_rms(state, true_state)
        timing_hist.append(timing_rms * 1e12)
        freq_hist.append(freq_rms)

        converged = timing_hist[-1] <= self.options.tolerance_ps
        convergence_iteration: Optional[int] = 0 if converged else None

        for iteration in range(1, self.options.max_iterations + 1):
            if self.options.asynchronous:
                state = self._step_asynchronous(state, epsilon, measurement_attr, weight_attr)
            else:
                state = self._step_synchronous(state, epsilon, measurement_attr, weight_attr)

            if self.options.enforce_zero_mean:
                state = self._project_zero_mean(state)

            state_history.append(state.copy())
            timing_rms, freq_rms = self._compute_rms(state, true_state)
            timing_hist.append(timing_rms * 1e12)
            freq_hist.append(freq_rms)

            if timing_hist[-1] <= self.options.tolerance_ps:
                converged = True
                convergence_iteration = iteration
                break

        residuals = self._edge_residuals(state_history[-1], measurement_attr)

        return ConsensusResult(
            state_history=np.stack(state_history, axis=0),
            timing_rms_ps=np.array(timing_hist),
            frequency_rms_hz=np.array(freq_hist),
            converged=converged,
            convergence_iteration=convergence_iteration,
            epsilon=epsilon,
            asynchronous=self.options.asynchronous,
            edge_residuals=residuals,
            lambda_max=self._last_lambda_max,
            lambda_2=self._last_lambda_2,
            spectral_gap=self._last_lambda_2,
        )

    # Internal helpers -------------------------------------------------

    def _resolve_step_size(self) -> float:
        if self.options.epsilon is not None:
            return float(self.options.epsilon)

        laplacian = nx.laplacian_matrix(self.graph).astype(float)
        if laplacian.shape == (0, 0) or laplacian.nnz == 0:
            self._last_lambda_max = 1.0
            self._last_lambda_2 = 0.0
            return 1.0

        try:
            largest = eigsh(
                laplacian,
                k=1,
                which='LM',
                return_eigenvectors=False,
            )
            lambda_max = float(np.real(largest[0]))
            if laplacian.shape[0] > 1:
                smallest = eigsh(
                    laplacian,
                    k=2,
                    which='SM',
                    return_eigenvectors=False,
                )
                lambda_2 = float(np.real(smallest[1]))
            else:
                lambda_2 = 0.0
        except ArpackNoConvergence as exc:
            eigen_real = np.real(exc.eigenvalues) if exc.eigenvalues.size else np.array([1.0])
            lambda_max = float(np.max(eigen_real))
            if exc.eigenvalues.size > 1:
                lambda_2 = float(sorted(eigen_real)[1])
            else:
                lambda_2 = 0.0

        if lambda_max <= 0:
            self._last_lambda_max = 1.0
            self._last_lambda_2 = 0.0
            return 1.0

        margin = self.options.spectral_margin
        self._last_lambda_max = lambda_max
        self._last_lambda_2 = max(lambda_2, 0.0)
        return float(margin / lambda_max)

    def _step_synchronous(
        self,
        state: NDArray[np.float64],
        epsilon: float,
        measurement_attr: str,
        weight_attr: str,
    ) -> NDArray[np.float64]:
        gradient = np.zeros_like(state)
        for u, v, data in self.graph.edges(data=True):
            measurement = self._directed_measurement(u, v, data, measurement_attr)
            weight = np.array(data[weight_attr], dtype=float)
            diff = (state[v] - state[u]) - measurement
            update = weight @ diff
            gradient[u] += update
            gradient[v] -= update
        return state + epsilon * gradient

    def _step_asynchronous(
        self,
        state: NDArray[np.float64],
        epsilon: float,
        measurement_attr: str,
        weight_attr: str,
    ) -> NDArray[np.float64]:
        node = int(self._rng.integers(self.graph.number_of_nodes()))
        update = np.zeros(2, dtype=float)
        for neighbor, data in self.graph[node].items():
            measurement = self._directed_measurement(node, neighbor, data, measurement_attr)
            weight = np.array(data[weight_attr], dtype=float)
            diff = (state[neighbor] - state[node]) - measurement
            update += weight @ diff
        state[node] = state[node] + epsilon * update
        return state

    def _directed_measurement(
        self,
        source: int,
        target: int,
        data: Dict[str, object],
        measurement_attr: str,
    ) -> NDArray[np.float64]:
        measurement = np.array(data[measurement_attr], dtype=float)
        orientation = data.get('orientation')
        if orientation is None:
            orientation = (source, target)
        if orientation[0] == source and orientation[1] == target:
            return measurement
        if orientation[0] == target and orientation[1] == source:
            return -measurement
        # Fallback to sorted orientation if metadata missing
        if source < target:
            return measurement
        return -measurement

    def _project_zero_mean(self, state: NDArray[np.float64]) -> NDArray[np.float64]:
        centred = state.copy()
        centred[:, 0] -= np.mean(centred[:, 0])
        centred[:, 1] -= np.mean(centred[:, 1])
        return centred

    def _compute_rms(
        self,
        state: NDArray[np.float64],
        true_state: Optional[NDArray[np.float64]],
    ) -> Tuple[float, float]:
        if true_state is None:
            error = state - np.mean(state, axis=0, keepdims=True)
        else:
            error = state - true_state
        timing_rms = float(np.sqrt(np.mean(error[:, 0] ** 2)))
        freq_rms = float(np.sqrt(np.mean(error[:, 1] ** 2)))
        return timing_rms, freq_rms

    def _edge_residuals(
        self,
        state: NDArray[np.float64],
        measurement_attr: str,
    ) -> Dict[Tuple[int, int], NDArray[np.float64]]:
        residuals: Dict[Tuple[int, int], NDArray[np.float64]] = {}
        for u, v, data in self.graph.edges(data=True):
            measurement = self._directed_measurement(u, v, data, measurement_attr)
            diff = (state[v] - state[u]) - measurement
            residuals[(u, v)] = diff.astype(float)
        return residuals
