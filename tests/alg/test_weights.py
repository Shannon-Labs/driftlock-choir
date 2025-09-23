import networkx as nx
import numpy as np
import pytest

from alg.weights import (
    metropolis_weight,
    metropolis_variance_weight,
    bx_surrogate_weight,
    build_weight_matrix,
)


@pytest.fixture
def sample_graph():
    """A simple graph for testing weight functions."""
    G = nx.Graph()
    G.add_edges_from([(0, 1), (1, 2), (1, 3)])
    return G


def test_metropolis_weight(sample_graph):
    # deg(0)=1, deg(1)=3 -> max(1,3)=3 -> 1/(1+3) = 0.25
    assert metropolis_weight(sample_graph, 0, 1) == pytest.approx(0.25)
    # deg(2)=1, deg(1)=3 -> max(1,3)=3 -> 1/(1+3) = 0.25
    assert metropolis_weight(sample_graph, 1, 2) == pytest.approx(0.25)


def test_metropolis_variance_weight(sample_graph):
    variance_scale = 0.5
    expected_weight = metropolis_weight(sample_graph, 0, 1) * variance_scale
    assert metropolis_variance_weight(sample_graph, 0, 1, variance_scale) == pytest.approx(expected_weight)


def test_bx_surrogate_weight(sample_graph):
    # deg(0)=1, deg(1)=3 -> sqrt(1*3)=1.732 -> 1/(1+1.732) = 0.366
    assert bx_surrogate_weight(sample_graph, 0, 1) == pytest.approx(1 / (1 + np.sqrt(3)))
    # deg(2)=1, deg(1)=3 -> sqrt(1*3)=1.732 -> 1/(1+1.732) = 0.366
    assert bx_surrogate_weight(sample_graph, 1, 2) == pytest.approx(1 / (1 + np.sqrt(3)))


def test_build_weight_matrix_metropolis(sample_graph):
    weight_matrix = build_weight_matrix("metropolis", sample_graph, 0, 1, 1.0, 1.0)
    expected_scalar = metropolis_weight(sample_graph, 0, 1)
    np.testing.assert_allclose(weight_matrix, expected_scalar * np.eye(2))


def test_build_weight_matrix_metropolis_var(sample_graph):
    sigma_tau_sq = 4.0
    sigma_df_sq = 2.0
    weight_matrix = build_weight_matrix("metropolis_var", sample_graph, 0, 1, sigma_tau_sq, sigma_df_sq)
    scalar = metropolis_weight(sample_graph, 0, 1)
    expected_diag = np.array([1.0 / sigma_tau_sq, 1.0 / sigma_df_sq])
    np.testing.assert_allclose(weight_matrix, scalar * np.diag(expected_diag))


def test_build_weight_matrix_bx_surrogate(sample_graph):
    weight_matrix = build_weight_matrix("bx_surrogate", sample_graph, 0, 1, 1.0, 1.0)
    expected_scalar = bx_surrogate_weight(sample_graph, 0, 1)
    np.testing.assert_allclose(weight_matrix, expected_scalar * np.eye(2))


def test_build_weight_matrix_inverse_variance(sample_graph):
    sigma_tau_sq = 4.0
    sigma_df_sq = 2.0
    weight_matrix = build_weight_matrix("inverse_variance", sample_graph, 0, 1, sigma_tau_sq, sigma_df_sq)
    expected_diag = np.array([1.0 / sigma_tau_sq, 1.0 / sigma_df_sq])
    np.testing.assert_allclose(weight_matrix, np.diag(expected_diag))

def test_build_weight_matrix_default_is_inverse_variance(sample_graph):
    sigma_tau_sq = 4.0
    sigma_df_sq = 2.0
    # Use a strategy that doesn't exist to trigger the default case
    weight_matrix = build_weight_matrix("non_existent_strategy", sample_graph, 0, 1, sigma_tau_sq, sigma_df_sq)
    expected_diag = np.array([1.0 / sigma_tau_sq, 1.0 / sigma_df_sq])
    np.testing.assert_allclose(weight_matrix, np.diag(expected_diag))
