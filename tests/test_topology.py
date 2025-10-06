"""
Unit tests for topology generators.

This module tests the topology generators for line, ring, and random
geometric topologies, as well as the analysis utilities.
"""

import unittest
import warnings

import numpy as np
import pytest

try:
    from src.simulation.topology import TopologyAnalysis, TopologyGenerator
except ModuleNotFoundError:  # pragma: no cover - optional feature
    pytest.skip(
        "Topology simulation utilities are not part of the open-source release.",
        allow_module_level=True,
    )

from src.core.types import NetworkTopology


class TestTopologyGenerator(unittest.TestCase):
    """Test cases for TopologyGenerator class."""

    def setUp(self):
        """Set up test fixtures."""
        self.generator = TopologyGenerator()

    def test_generate_line_topology_single_node(self):
        """Test line topology generation with a single node."""
        topology = self.generator.generate_line_topology(1)

        self.assertEqual(len(topology.node_ids), 1)
        self.assertEqual(topology.adjacency_matrix.shape, (1, 1))
        self.assertEqual(topology.adjacency_matrix[0, 0], 0)
        self.assertTrue(topology.is_connected)
        self.assertEqual(topology.spectral_gap, 0)  # Single node has no spectral gap

    def test_generate_line_topology_two_nodes(self):
        """Test line topology generation with two nodes."""
        topology = self.generator.generate_line_topology(2)

        self.assertEqual(len(topology.node_ids), 2)
        self.assertEqual(topology.adjacency_matrix.shape, (2, 2))
        self.assertEqual(topology.adjacency_matrix[0, 1], 1)
        self.assertEqual(topology.adjacency_matrix[1, 0], 1)
        self.assertEqual(topology.adjacency_matrix[0, 0], 0)
        self.assertEqual(topology.adjacency_matrix[1, 1], 0)
        self.assertTrue(topology.is_connected)
        self.assertGreater(topology.spectral_gap, 0)

    def test_generate_line_topology_multiple_nodes(self):
        """Test line topology generation with multiple nodes."""
        topology = self.generator.generate_line_topology(5)

        self.assertEqual(len(topology.node_ids), 5)
        self.assertEqual(topology.adjacency_matrix.shape, (5, 5))

        # Check that end nodes have degree 1
        self.assertEqual(topology.get_degree(0), 1)
        self.assertEqual(topology.get_degree(4), 1)

        # Check that middle nodes have degree 2
        for i in range(1, 4):
            self.assertEqual(topology.get_degree(i), 2)

        # Check connectivity
        self.assertTrue(topology.is_connected)
        self.assertGreater(topology.spectral_gap, 0)

    def test_generate_line_topology_invalid_nodes(self):
        """Test line topology generation with invalid number of nodes."""
        with self.assertRaises(ValueError):
            self.generator.generate_line_topology(0)

        with self.assertRaises(ValueError):
            self.generator.generate_line_topology(-1)

    def test_generate_ring_topology_single_node(self):
        """Test ring topology generation with a single node."""
        topology = self.generator.generate_ring_topology(1)

        self.assertEqual(len(topology.node_ids), 1)
        self.assertEqual(topology.adjacency_matrix.shape, (1, 1))
        self.assertEqual(topology.adjacency_matrix[0, 0], 0)
        self.assertTrue(topology.is_connected)
        self.assertEqual(topology.spectral_gap, 0)

    def test_generate_ring_topology_two_nodes(self):
        """Test ring topology generation with two nodes."""
        topology = self.generator.generate_ring_topology(2)

        self.assertEqual(len(topology.node_ids), 2)
        self.assertEqual(topology.adjacency_matrix.shape, (2, 2))
        self.assertEqual(topology.adjacency_matrix[0, 1], 1)
        self.assertEqual(topology.adjacency_matrix[1, 0], 1)
        self.assertEqual(topology.adjacency_matrix[0, 0], 0)
        self.assertEqual(topology.adjacency_matrix[1, 1], 0)
        self.assertTrue(topology.is_connected)
        self.assertGreater(topology.spectral_gap, 0)

    def test_generate_ring_topology_multiple_nodes(self):
        """Test ring topology generation with multiple nodes."""
        topology = self.generator.generate_ring_topology(5)

        self.assertEqual(len(topology.node_ids), 5)
        self.assertEqual(topology.adjacency_matrix.shape, (5, 5))

        # Check that all nodes have degree 2
        for i in range(5):
            self.assertEqual(topology.get_degree(i), 2)

        # Check connectivity
        self.assertTrue(topology.is_connected)
        self.assertGreater(topology.spectral_gap, 0)

    def test_generate_ring_topology_invalid_nodes(self):
        """Test ring topology generation with invalid number of nodes."""
        with self.assertRaises(ValueError):
            self.generator.generate_ring_topology(0)

        with self.assertRaises(ValueError):
            self.generator.generate_ring_topology(-1)

    def test_generate_random_geometric_topology_single_node(self):
        """Test random geometric topology generation with a single node."""
        topology = self.generator.generate_random_geometric_topology(1, 0.5)

        self.assertEqual(len(topology.node_ids), 1)
        self.assertEqual(topology.adjacency_matrix.shape, (1, 1))
        self.assertEqual(topology.adjacency_matrix[0, 0], 0)
        self.assertTrue(topology.is_connected)
        self.assertEqual(topology.spectral_gap, 0)

    def test_generate_random_geometric_topology_valid(self):
        """Test random geometric topology generation with valid parameters."""
        topology = self.generator.generate_random_geometric_topology(10, 0.5, seed=42)

        self.assertEqual(len(topology.node_ids), 10)
        self.assertEqual(topology.adjacency_matrix.shape, (10, 10))

        # Check that adjacency matrix is symmetric
        self.assertTrue(
            np.allclose(topology.adjacency_matrix, topology.adjacency_matrix.T)
        )

        # Check that diagonal is zero
        self.assertTrue(np.all(np.diag(topology.adjacency_matrix) == 0))

        # Check that values are 0 or 1
        unique_values = np.unique(topology.adjacency_matrix)
        self.assertTrue(np.all(np.isin(unique_values, [0, 1])))

    def test_generate_random_geometric_topology_invalid_nodes(self):
        """Test random geometric topology generation with invalid number of nodes."""
        with self.assertRaises(ValueError):
            self.generator.generate_random_geometric_topology(0, 0.5)

        with self.assertRaises(ValueError):
            self.generator.generate_random_geometric_topology(-1, 0.5)

    def test_generate_random_geometric_topology_invalid_radius(self):
        """Test random geometric topology generation with invalid radius."""
        with self.assertRaises(ValueError):
            self.generator.generate_random_geometric_topology(5, 0.0)

        with self.assertRaises(ValueError):
            self.generator.generate_random_geometric_topology(5, -0.1)

        with self.assertRaises(ValueError):
            self.generator.generate_random_geometric_topology(5, 1.1)

    def test_generate_random_geometric_topology_disconnected(self):
        """Test random geometric topology generation that may be disconnected."""
        # Use very small radius to likely create disconnected graph
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            topology = self.generator.generate_random_geometric_topology(
                20, 0.1, seed=42
            )

            # May or may not be connected depending on random placement
            # Just check that the warning is issued if disconnected
            if not topology.is_connected:
                self.assertEqual(len(w), 1)
                self.assertIn("disconnected", str(w[0].message))

    def test_create_network_topology_valid(self):
        """Test creating network topology from valid adjacency matrix."""
        adjacency_matrix = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
        node_ids = [0, 1, 2]

        topology = self.generator.create_network_topology(adjacency_matrix, node_ids)

        self.assertEqual(topology.node_ids, node_ids)
        self.assertTrue(np.array_equal(topology.adjacency_matrix, adjacency_matrix))
        self.assertTrue(topology.is_connected)
        self.assertGreater(topology.spectral_gap, 0)

    def test_create_network_topology_invalid_dimensions(self):
        """Test creating network topology with invalid dimensions."""
        adjacency_matrix = np.array([[0, 1], [1, 0]])
        node_ids = [0, 1, 2]  # Mismatch

        with self.assertRaises(ValueError):
            self.generator.create_network_topology(adjacency_matrix, node_ids)

    def test_create_network_topology_asymmetric(self):
        """Test creating network topology with asymmetric adjacency matrix."""
        adjacency_matrix = np.array([[0, 1, 1], [0, 0, 1], [0, 0, 0]])
        node_ids = [0, 1, 2]

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            topology = self.generator.create_network_topology(
                adjacency_matrix, node_ids
            )

            # Should issue warning about asymmetric matrix
            self.assertEqual(len(w), 1)
            self.assertIn("not symmetric", str(w[0].message))

            # Should make it symmetric
            expected_matrix = (adjacency_matrix + adjacency_matrix.T) / 2
            self.assertTrue(np.array_equal(topology.adjacency_matrix, expected_matrix))

    def test_analyze_topology_line(self):
        """Test analysis of line topology."""
        topology = self.generator.generate_line_topology(5)
        analysis = self.generator.analyze_topology(topology)

        self.assertIsInstance(analysis, TopologyAnalysis)
        self.assertEqual(analysis.degree_distribution, {1: 2, 2: 3})
        self.assertEqual(analysis.clustering_coefficient, 0.0)  # Line has no triangles
        self.assertEqual(analysis.diameter, 4)  # Distance between end nodes
        self.assertTrue(analysis.is_connected)
        self.assertEqual(analysis.num_components, 1)
        self.assertGreater(analysis.spectral_gap, 0)

    def test_analyze_topology_ring(self):
        """Test analysis of ring topology."""
        topology = self.generator.generate_ring_topology(5)
        analysis = self.generator.analyze_topology(topology)

        self.assertIsInstance(analysis, TopologyAnalysis)
        self.assertEqual(analysis.degree_distribution, {2: 5})
        self.assertEqual(analysis.clustering_coefficient, 0.0)  # Ring has no triangles
        self.assertEqual(analysis.diameter, 2)  # Maximum distance in 5-node ring
        self.assertTrue(analysis.is_connected)
        self.assertEqual(analysis.num_components, 1)
        self.assertGreater(analysis.spectral_gap, 0)

    def test_get_topology_consensus_properties(self):
        """Test getting consensus-relevant properties."""
        topology = self.generator.generate_line_topology(5)
        properties = self.generator.get_topology_consensus_properties(topology)

        self.assertIn("num_nodes", properties)
        self.assertIn("avg_degree", properties)
        self.assertIn("max_degree", properties)
        self.assertIn("min_degree", properties)
        self.assertIn("spectral_gap", properties)
        self.assertIn("algebraic_connectivity", properties)
        self.assertIn("clustering_coefficient", properties)
        self.assertIn("diameter", properties)
        self.assertIn("average_path_length", properties)
        self.assertIn("is_connected", properties)
        self.assertIn("num_components", properties)
        self.assertIn("convergence_time_estimate", properties)
        self.assertIn("mixing_time_estimate", properties)
        self.assertIn("consensus_difficulty", properties)

        self.assertEqual(properties["num_nodes"], 5)
        self.assertEqual(properties["max_degree"], 2)
        self.assertEqual(properties["min_degree"], 1)
        self.assertTrue(properties["is_connected"])
        self.assertEqual(properties["num_components"], 1)

    def test_estimate_consensus_difficulty_disconnected(self):
        """Test consensus difficulty estimation for disconnected topology."""
        # Create a disconnected topology
        adjacency_matrix = np.array(
            [[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]]
        )
        node_ids = [0, 1, 2, 3]
        topology = self.generator.create_network_topology(adjacency_matrix, node_ids)

        difficulty = self.generator._estimate_consensus_difficulty(topology)
        self.assertEqual(difficulty, "Impossible (disconnected)")

    def test_estimate_consensus_difficulty_connected(self):
        """Test consensus difficulty estimation for connected topologies."""
        # Well-connected topology
        topology = self.generator.generate_ring_topology(5)
        difficulty = self.generator._estimate_consensus_difficulty(topology)
        self.assertIn(
            difficulty, ["Easy (well-connected)", "Moderate (reasonably connected)"]
        )

        # Poorly connected topology (line)
        topology = self.generator.generate_line_topology(10)
        difficulty = self.generator._estimate_consensus_difficulty(topology)
        self.assertIn(
            difficulty,
            ["Moderate (reasonably connected)", "Challenging (poorly connected)"],
        )


class TestTopologyAnalysis(unittest.TestCase):
    """Test cases for TopologyAnalysis class."""

    def test_topology_analysis_creation(self):
        """Test creation of TopologyAnalysis object."""
        analysis = TopologyAnalysis(
            degree_distribution={1: 2, 2: 3},
            clustering_coefficient=0.5,
            average_path_length=2.5,
            diameter=4,
            is_connected=True,
            num_components=1,
            spectral_gap=0.2,
            algebraic_connectivity=0.2,
        )

        self.assertEqual(analysis.degree_distribution, {1: 2, 2: 3})
        self.assertEqual(analysis.clustering_coefficient, 0.5)
        self.assertEqual(analysis.average_path_length, 2.5)
        self.assertEqual(analysis.diameter, 4)
        self.assertTrue(analysis.is_connected)
        self.assertEqual(analysis.num_components, 1)
        self.assertEqual(analysis.spectral_gap, 0.2)
        self.assertEqual(analysis.algebraic_connectivity, 0.2)


if __name__ == "__main__":
    unittest.main()
