"""
Random-geometric graph builder for network topology generation.

This module creates realistic network topologies based on geometric
constraints and random connectivity patterns.
"""

import numpy as np
from typing import List, Tuple, Dict, Optional, Any
from dataclasses import dataclass
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform


@dataclass
class TopologyParams:
    """Parameters for network topology generation."""
    n_nodes: int             # Number of nodes
    area_size: float         # Network area size (m)
    comm_range: float        # Communication range (m)
    connectivity_prob: float = 1.0  # Probability of connection within range
    min_degree: int = 2      # Minimum node degree
    topology_type: str = 'rgg'  # 'rgg', 'unit_disk', 'small_world'


class RandomGeometricGraph:
    """Random geometric graph generator for wireless networks."""
    
    def __init__(self, params: TopologyParams):
        self.params = params
        self.node_positions = None
        self.adjacency_matrix = None
        self.distance_matrix = None
        
    def generate_topology(self, seed: Optional[int] = None) -> Dict[str, Any]:
        """
        Generate random geometric graph topology.
        
        Args:
            seed: Random seed for reproducibility
            
        Returns:
            Dictionary with topology information
        """
        if seed is not None:
            np.random.seed(seed)
            
        # Generate node positions
        self.node_positions = self._generate_node_positions()
        
        # Compute distance matrix
        self.distance_matrix = self._compute_distances()
        
        # Create adjacency matrix based on communication range
        self.adjacency_matrix = self._create_adjacency_matrix()
        
        # Ensure minimum connectivity
        self._ensure_connectivity()
        
        return {
            'node_positions': self.node_positions,
            'adjacency_matrix': self.adjacency_matrix,
            'distance_matrix': self.distance_matrix,
            'topology_metrics': self._compute_topology_metrics()
        }
        
    def _generate_node_positions(self) -> np.ndarray:
        """Generate random node positions in the network area."""
        if self.params.topology_type == 'rgg':
            # Uniform random positions
            positions = np.random.uniform(
                0, self.params.area_size, 
                (self.params.n_nodes, 2)
            )
        elif self.params.topology_type == 'clustered':
            # Clustered positions (simplified)
            n_clusters = max(1, self.params.n_nodes // 10)
            cluster_centers = np.random.uniform(
                self.params.area_size * 0.2, 
                self.params.area_size * 0.8, 
                (n_clusters, 2)
            )
            
            positions = []
            nodes_per_cluster = self.params.n_nodes // n_clusters
            
            for i, center in enumerate(cluster_centers):
                if i == len(cluster_centers) - 1:
                    # Last cluster gets remaining nodes
                    n_nodes_cluster = self.params.n_nodes - len(positions)
                else:
                    n_nodes_cluster = nodes_per_cluster
                    
                cluster_positions = center + np.random.normal(
                    0, self.params.comm_range / 3, 
                    (n_nodes_cluster, 2)
                )
                positions.extend(cluster_positions)
                
            positions = np.array(positions[:self.params.n_nodes])
        else:
            # Default to uniform random
            positions = np.random.uniform(
                0, self.params.area_size, 
                (self.params.n_nodes, 2)
            )
            
        return positions
        
    def _compute_distances(self) -> np.ndarray:
        """Compute pairwise distances between all nodes."""
        distances = squareform(pdist(self.node_positions, 'euclidean'))
        return distances
        
    def _create_adjacency_matrix(self) -> np.ndarray:
        """Create adjacency matrix based on communication range."""
        adjacency = np.zeros((self.params.n_nodes, self.params.n_nodes))
        
        for i in range(self.params.n_nodes):
            for j in range(i + 1, self.params.n_nodes):
                distance = self.distance_matrix[i, j]
                
                # Check if within communication range
                if distance <= self.params.comm_range:
                    # Apply connectivity probability
                    if np.random.random() < self.params.connectivity_prob:
                        adjacency[i, j] = 1
                        adjacency[j, i] = 1
                        
        return adjacency
        
    def _ensure_connectivity(self):
        """Ensure minimum connectivity requirements."""
        degrees = np.sum(self.adjacency_matrix, axis=1)
        
        for i, degree in enumerate(degrees):
            if degree < self.params.min_degree:
                # Find closest unconnected nodes
                distances_i = self.distance_matrix[i, :]
                connected_mask = self.adjacency_matrix[i, :] > 0
                connected_mask[i] = True  # Exclude self
                
                # Get indices of unconnected nodes sorted by distance
                unconnected_indices = np.where(~connected_mask)[0]
                if len(unconnected_indices) > 0:
                    unconnected_distances = distances_i[unconnected_indices]
                    sorted_indices = unconnected_indices[np.argsort(unconnected_distances)]
                    
                    # Connect to closest nodes to meet minimum degree
                    needed_connections = self.params.min_degree - int(degree)
                    for j in sorted_indices[:needed_connections]:
                        self.adjacency_matrix[i, j] = 1
                        self.adjacency_matrix[j, i] = 1
                        
    def _compute_topology_metrics(self) -> Dict[str, float]:
        """Compute various topology metrics."""
        # Basic metrics
        n_edges = np.sum(self.adjacency_matrix) // 2
        avg_degree = np.mean(np.sum(self.adjacency_matrix, axis=1))
        
        # Clustering coefficient
        clustering_coeff = self._compute_clustering_coefficient()
        
        # Path length metrics
        path_length_metrics = self._compute_path_lengths()
        
        # Connectivity metrics
        connectivity_metrics = self._compute_connectivity_metrics()
        
        return {
            'n_nodes': self.params.n_nodes,
            'n_edges': n_edges,
            'avg_degree': avg_degree,
            'clustering_coefficient': clustering_coeff,
            'avg_path_length': path_length_metrics['avg_path_length'],
            'diameter': path_length_metrics['diameter'],
            'algebraic_connectivity': connectivity_metrics['algebraic_connectivity'],
            'is_connected': connectivity_metrics['is_connected']
        }
        
    def _compute_clustering_coefficient(self) -> float:
        """Compute average clustering coefficient."""
        clustering_coeffs = []
        
        for i in range(self.params.n_nodes):
            neighbors = np.where(self.adjacency_matrix[i, :] > 0)[0]
            k = len(neighbors)
            
            if k < 2:
                clustering_coeffs.append(0)
                continue
                
            # Count edges between neighbors
            edges_between_neighbors = 0
            for j in neighbors:
                for l in neighbors:
                    if j < l and self.adjacency_matrix[j, l] > 0:
                        edges_between_neighbors += 1
                        
            clustering_coeff = 2 * edges_between_neighbors / (k * (k - 1))
            clustering_coeffs.append(clustering_coeff)
            
        return np.mean(clustering_coeffs)
        
    def _compute_path_lengths(self) -> Dict[str, float]:
        """Compute shortest path lengths using Floyd-Warshall."""
        n = self.params.n_nodes
        dist = np.full((n, n), np.inf)
        
        # Initialize distances
        for i in range(n):
            dist[i, i] = 0
            for j in range(n):
                if self.adjacency_matrix[i, j] > 0:
                    dist[i, j] = 1
                    
        # Floyd-Warshall algorithm
        for k in range(n):
            for i in range(n):
                for j in range(n):
                    if dist[i, k] + dist[k, j] < dist[i, j]:
                        dist[i, j] = dist[i, k] + dist[k, j]
                        
        # Extract metrics
        finite_distances = dist[dist < np.inf]
        finite_distances = finite_distances[finite_distances > 0]  # Exclude self-distances
        
        if len(finite_distances) > 0:
            avg_path_length = np.mean(finite_distances)
            diameter = np.max(finite_distances)
        else:
            avg_path_length = np.inf
            diameter = np.inf
            
        return {
            'avg_path_length': avg_path_length,
            'diameter': diameter
        }
        
    def _compute_connectivity_metrics(self) -> Dict[str, Any]:
        """Compute connectivity-related metrics."""
        # Laplacian matrix
        degree_matrix = np.diag(np.sum(self.adjacency_matrix, axis=1))
        laplacian = degree_matrix - self.adjacency_matrix
        
        # Compute eigenvalues
        eigenvals = np.linalg.eigvals(laplacian)
        eigenvals = np.sort(eigenvals)
        
        # Algebraic connectivity (second smallest eigenvalue)
        algebraic_connectivity = eigenvals[1] if len(eigenvals) > 1 else 0
        
        # Check if graph is connected
        is_connected = algebraic_connectivity > 1e-10
        
        return {
            'algebraic_connectivity': algebraic_connectivity,
            'is_connected': is_connected,
            'laplacian_eigenvalues': eigenvals
        }
        
    def visualize_topology(self, save_path: Optional[str] = None):
        """Visualize the network topology."""
        if self.node_positions is None:
            raise ValueError("Topology must be generated first")
            
        plt.figure(figsize=(10, 10))
        
        # Plot edges
        for i in range(self.params.n_nodes):
            for j in range(i + 1, self.params.n_nodes):
                if self.adjacency_matrix[i, j] > 0:
                    x_coords = [self.node_positions[i, 0], self.node_positions[j, 0]]
                    y_coords = [self.node_positions[i, 1], self.node_positions[j, 1]]
                    plt.plot(x_coords, y_coords, 'b-', alpha=0.5, linewidth=0.5)
                    
        # Plot nodes
        plt.scatter(self.node_positions[:, 0], self.node_positions[:, 1], 
                   c='red', s=50, zorder=5)
        
        # Add node labels
        for i, pos in enumerate(self.node_positions):
            plt.annotate(str(i), (pos[0], pos[1]), xytext=(5, 5), 
                        textcoords='offset points', fontsize=8)
            
        plt.xlim(0, self.params.area_size)
        plt.ylim(0, self.params.area_size)
        plt.xlabel('X Position (m)')
        plt.ylabel('Y Position (m)')
        plt.title(f'Network Topology ({self.params.n_nodes} nodes)')
        plt.grid(True, alpha=0.3)
        plt.axis('equal')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
