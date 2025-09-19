"""Edge weighting heuristics for consensus on synchronisation graphs."""

from __future__ import annotations

import math
from typing import Callable

import networkx as nx
import numpy as np
from numpy.typing import NDArray


WeightFunction = Callable[[nx.Graph, int, int], float]


def metropolis_weight(graph: nx.Graph, u: int, v: int) -> float:
    deg_u = max(graph.degree[u], 1)
    deg_v = max(graph.degree[v], 1)
    return 1.0 / (1.0 + max(deg_u, deg_v))


def metropolis_variance_weight(graph: nx.Graph, u: int, v: int, variance_scale: float) -> float:
    return metropolis_weight(graph, u, v) * variance_scale


def bx_surrogate_weight(graph: nx.Graph, u: int, v: int) -> float:
    deg_u = max(graph.degree[u], 1)
    deg_v = max(graph.degree[v], 1)
    geometric = math.sqrt(deg_u * deg_v)
    return 1.0 / (1.0 + geometric)


def build_weight_matrix(
    strategy: str,
    graph: nx.Graph,
    u: int,
    v: int,
    sigma_tau_sq: float,
    sigma_df_sq: float,
) -> NDArray[np.float64]:
    base_sigma_tau = max(sigma_tau_sq, 1e-30)
    base_sigma_df = max(sigma_df_sq, 1e-30)

    if strategy == 'metropolis':
        scalar = metropolis_weight(graph, u, v)
        return scalar * np.eye(2, dtype=float)

    if strategy == 'metropolis_var':
        scalar = metropolis_weight(graph, u, v)
        diag = np.array([1.0 / base_sigma_tau, 1.0 / base_sigma_df], dtype=float)
        return scalar * np.diag(diag)

    if strategy == 'bx_surrogate':
        scalar = bx_surrogate_weight(graph, u, v)
        return scalar * np.eye(2, dtype=float)

    diag = np.array([1.0 / base_sigma_tau, 1.0 / base_sigma_df], dtype=float)
    return np.diag(diag)
