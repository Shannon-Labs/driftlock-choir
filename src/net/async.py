"""
Random transmission offset and packet drop model for asynchronous networks.

This module implements realistic asynchronous behavior including random
transmission timing, packet drops, and network delays.
"""

import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from collections import defaultdict
import heapq


@dataclass
class AsyncParams:
    """Parameters for asynchronous network behavior."""
    tx_offset_mean: float = 0.0      # Mean transmission offset (s)
    tx_offset_std: float = 0.01      # Std deviation of transmission offset (s)
    packet_drop_rate: float = 0.05   # Packet drop probability
    delay_mean: float = 0.001        # Mean network delay (s)
    delay_std: float = 0.0005        # Std deviation of network delay (s)
    jitter_std: float = 0.0001       # Timing jitter std deviation (s)
    burst_drop_prob: float = 0.01    # Probability of burst drops
    burst_drop_length: int = 5       # Length of burst drops


class NetworkDelayModel:
    """Model for network delays and jitter."""
    
    def __init__(self, params: AsyncParams):
        self.params = params
        
    def get_transmission_delay(self, distance: float = 0.0) -> float:
        """Get transmission delay including propagation and processing."""
        # Propagation delay (speed of light)
        prop_delay = distance / 3e8
        
        # Processing delay (random)
        proc_delay = np.random.normal(self.params.delay_mean, self.params.delay_std)
        proc_delay = max(0, proc_delay)  # Ensure non-negative
        
        # Jitter
        jitter = np.random.normal(0, self.params.jitter_std)
        
        return prop_delay + proc_delay + jitter
        
    def get_transmission_offset(self) -> float:
        """Get random transmission timing offset."""
        return np.random.normal(self.params.tx_offset_mean, self.params.tx_offset_std)


class PacketDropModel:
    """Model for packet drops including burst errors."""
    
    def __init__(self, params: AsyncParams):
        self.params = params
        self.burst_state = False
        self.burst_remaining = 0
        
    def should_drop_packet(self) -> bool:
        """Determine if packet should be dropped."""
        # Check burst drop state
        if self.burst_state:
            self.burst_remaining -= 1
            if self.burst_remaining <= 0:
                self.burst_state = False
            return True
            
        # Check for new burst
        if np.random.random() < self.params.burst_drop_prob:
            self.burst_state = True
            self.burst_remaining = self.params.burst_drop_length - 1
            return True
            
        # Regular random drop
        return np.random.random() < self.params.packet_drop_rate


@dataclass
class AsyncEvent:
    """Asynchronous network event."""
    timestamp: float
    event_type: str
    source_node: int
    dest_node: Optional[int]
    packet_id: int
    payload: Dict[str, Any]


class AsynchronousNetwork:
    """Asynchronous network simulation with realistic imperfections."""
    
    def __init__(self, params: AsyncParams, n_nodes: int, 
                 adjacency_matrix: np.ndarray, distance_matrix: np.ndarray):
        self.params = params
        self.n_nodes = n_nodes
        self.adjacency_matrix = adjacency_matrix
        self.distance_matrix = distance_matrix
        
        # Network models
        self.delay_model = NetworkDelayModel(params)
        self.drop_models = [PacketDropModel(params) for _ in range(n_nodes)]
        
        # Event queue
        self.event_queue = []
        self.current_time = 0.0
        
        # Statistics
        self.stats = {
            'packets_transmitted': 0,
            'packets_delivered': 0,
            'packets_dropped': 0,
            'total_delay': 0.0,
            'delay_samples': [],
            'per_node_stats': defaultdict(lambda: {
                'tx_count': 0, 'rx_count': 0, 'drop_count': 0
            })
        }
        
    def schedule_transmission(self, source_node: int, dest_nodes: List[int], 
                            packet_id: int, payload: Dict[str, Any], 
                            base_time: float):
        """Schedule packet transmission with asynchronous behavior."""
        # Add transmission offset
        tx_offset = self.delay_model.get_transmission_offset()
        actual_tx_time = base_time + tx_offset
        
        # Schedule transmission event
        event = AsyncEvent(
            timestamp=actual_tx_time,
            event_type='transmission',
            source_node=source_node,
            dest_node=None,
            packet_id=packet_id,
            payload=payload
        )
        
        heapq.heappush(self.event_queue, (actual_tx_time, event))
        
        # Schedule reception events for each destination
        for dest_node in dest_nodes:
            if self.adjacency_matrix[source_node, dest_node] > 0:
                self._schedule_reception(source_node, dest_node, packet_id, 
                                       payload, actual_tx_time)
                
    def _schedule_reception(self, source_node: int, dest_node: int, 
                          packet_id: int, payload: Dict[str, Any], tx_time: float):
        """Schedule packet reception with delays and drops."""
        # Check if packet should be dropped
        if self.drop_models[dest_node].should_drop_packet():
            self.stats['packets_dropped'] += 1
            self.stats['per_node_stats'][dest_node]['drop_count'] += 1
            return
            
        # Calculate reception time
        distance = self.distance_matrix[source_node, dest_node]
        delay = self.delay_model.get_transmission_delay(distance)
        reception_time = tx_time + delay
        
        # Schedule reception event
        event = AsyncEvent(
            timestamp=reception_time,
            event_type='reception',
            source_node=source_node,
            dest_node=dest_node,
            packet_id=packet_id,
            payload=payload
        )
        
        heapq.heappush(self.event_queue, (reception_time, event))
        
    def run_simulation(self, duration: float) -> Dict[str, Any]:
        """Run asynchronous network simulation."""
        end_time = self.current_time + duration
        processed_events = []
        
        while self.event_queue and self.current_time < end_time:
            event_time, event = heapq.heappop(self.event_queue)
            self.current_time = event_time
            
            if event_time > end_time:
                break
                
            processed_events.append(event)
            self._process_event(event)
            
        return self._generate_simulation_results(processed_events)
        
    def _process_event(self, event: AsyncEvent):
        """Process a network event."""
        if event.event_type == 'transmission':
            self.stats['packets_transmitted'] += 1
            self.stats['per_node_stats'][event.source_node]['tx_count'] += 1
            
        elif event.event_type == 'reception':
            self.stats['packets_delivered'] += 1
            self.stats['per_node_stats'][event.dest_node]['rx_count'] += 1
            
            # Calculate end-to-end delay
            if 'tx_timestamp' in event.payload:
                delay = event.timestamp - event.payload['tx_timestamp']
                self.stats['total_delay'] += delay
                self.stats['delay_samples'].append(delay)
                
    def _generate_simulation_results(self, events: List[AsyncEvent]) -> Dict[str, Any]:
        """Generate comprehensive simulation results."""
        # Basic statistics
        total_tx = self.stats['packets_transmitted']
        total_rx = self.stats['packets_delivered']
        total_drops = self.stats['packets_dropped']
        
        # Delivery ratio
        delivery_ratio = total_rx / total_tx if total_tx > 0 else 0
        drop_rate = total_drops / (total_tx + total_drops) if (total_tx + total_drops) > 0 else 0
        
        # Delay statistics
        delay_samples = self.stats['delay_samples']
        if delay_samples:
            avg_delay = np.mean(delay_samples)
            delay_std = np.std(delay_samples)
            delay_percentiles = np.percentile(delay_samples, [50, 90, 95, 99])
        else:
            avg_delay = 0
            delay_std = 0
            delay_percentiles = [0, 0, 0, 0]
            
        # Per-node statistics
        per_node_delivery_ratio = {}
        per_node_drop_rate = {}
        
        for node_id in range(self.n_nodes):
            stats = self.stats['per_node_stats'][node_id]
            tx_count = stats['tx_count']
            rx_count = stats['rx_count']
            drop_count = stats['drop_count']
            
            per_node_delivery_ratio[node_id] = rx_count / tx_count if tx_count > 0 else 0
            per_node_drop_rate[node_id] = drop_count / (tx_count + drop_count) if (tx_count + drop_count) > 0 else 0
            
        # Event timeline analysis
        tx_events = [e for e in events if e.event_type == 'transmission']
        rx_events = [e for e in events if e.event_type == 'reception']
        
        # Throughput over time (packets per second)
        time_bins = np.arange(0, self.current_time, 1.0)  # 1-second bins
        tx_throughput = np.histogram([e.timestamp for e in tx_events], bins=time_bins)[0]
        rx_throughput = np.histogram([e.timestamp for e in rx_events], bins=time_bins)[0]
        
        return {
            'simulation_duration': self.current_time,
            'total_events': len(events),
            'packet_statistics': {
                'transmitted': total_tx,
                'delivered': total_rx,
                'dropped': total_drops,
                'delivery_ratio': delivery_ratio,
                'drop_rate': drop_rate
            },
            'delay_statistics': {
                'average_delay': avg_delay,
                'delay_std': delay_std,
                'median_delay': delay_percentiles[0],
                'p90_delay': delay_percentiles[1],
                'p95_delay': delay_percentiles[2],
                'p99_delay': delay_percentiles[3]
            },
            'per_node_statistics': {
                'delivery_ratio': per_node_delivery_ratio,
                'drop_rate': per_node_drop_rate,
                'detailed_stats': dict(self.stats['per_node_stats'])
            },
            'throughput_analysis': {
                'time_bins': time_bins[:-1],  # Remove last bin edge
                'tx_throughput': tx_throughput,
                'rx_throughput': rx_throughput,
                'avg_tx_throughput': np.mean(tx_throughput),
                'avg_rx_throughput': np.mean(rx_throughput)
            },
            'network_efficiency': {
                'channel_utilization': total_tx / (self.current_time * self.n_nodes) if self.current_time > 0 else 0,
                'effective_throughput': total_rx / self.current_time if self.current_time > 0 else 0
            }
        }
        
    def get_real_time_statistics(self) -> Dict[str, Any]:
        """Get current real-time network statistics."""
        return {
            'current_time': self.current_time,
            'pending_events': len(self.event_queue),
            'packets_in_flight': len([e for _, e in self.event_queue if e.event_type == 'reception']),
            'instantaneous_stats': {
                'tx_rate': self.stats['per_node_stats'][0]['tx_count'] / max(self.current_time, 1e-6),
                'rx_rate': self.stats['per_node_stats'][0]['rx_count'] / max(self.current_time, 1e-6),
                'current_delivery_ratio': (self.stats['packets_delivered'] / 
                                         max(self.stats['packets_transmitted'], 1))
            }
        }
