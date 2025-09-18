"""
BEACON / RESPONSE packet simulation for MAC layer protocols.

This module implements MAC layer protocols for synchronization beacon
transmission and response handling in wireless networks.
"""

import numpy as np
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import heapq
from collections import defaultdict


class PacketType(Enum):
    """Types of MAC packets."""
    BEACON = "beacon"
    RESPONSE = "response"
    DATA = "data"
    ACK = "ack"


@dataclass
class MACPacket:
    """MAC layer packet structure."""
    packet_id: int
    packet_type: PacketType
    source_id: int
    dest_id: int
    timestamp: float
    payload: Dict[str, Any] = field(default_factory=dict)
    size_bytes: int = 64
    priority: int = 0
    retries: int = 0
    max_retries: int = 3


@dataclass
class MACParams:
    """Parameters for MAC layer simulation."""
    beacon_interval: float = 1.0      # Beacon transmission interval (s)
    response_timeout: float = 0.1     # Response timeout (s)
    backoff_min: float = 0.001        # Minimum backoff time (s)
    backoff_max: float = 0.01         # Maximum backoff time (s)
    packet_error_rate: float = 0.01   # Packet error rate
    channel_access_delay: float = 0.001  # Channel access delay (s)
    max_queue_size: int = 100         # Maximum queue size per node


class ChannelState(Enum):
    """Channel state for CSMA/CA."""
    IDLE = "idle"
    BUSY = "busy"
    COLLISION = "collision"


class MACProtocol:
    """MAC protocol implementation for synchronization networks."""
    
    def __init__(self, params: MACParams, n_nodes: int, adjacency_matrix: np.ndarray):
        self.params = params
        self.n_nodes = n_nodes
        self.adjacency_matrix = adjacency_matrix
        
        # Node state
        self.node_queues = [[] for _ in range(n_nodes)]
        self.node_states = ['idle'] * n_nodes
        self.last_beacon_time = [0.0] * n_nodes
        
        # Channel state
        self.channel_state = ChannelState.IDLE
        self.current_transmissions = []
        
        # Event queue (priority queue for discrete event simulation)
        self.event_queue = []
        self.current_time = 0.0
        
        # Statistics
        self.stats = {
            'packets_sent': defaultdict(int),
            'packets_received': defaultdict(int),
            'packets_dropped': defaultdict(int),
            'collisions': 0,
            'channel_utilization': 0.0
        }
        
    def schedule_event(self, event_time: float, event_type: str, 
                      node_id: int, packet: Optional[MACPacket] = None):
        """Schedule a MAC event."""
        heapq.heappush(self.event_queue, (event_time, event_type, node_id, packet))
        
    def run_simulation(self, duration: float) -> Dict[str, Any]:
        """
        Run MAC layer simulation for specified duration.
        
        Args:
            duration: Simulation duration (s)
            
        Returns:
            Dictionary with simulation results and statistics
        """
        # Initialize beacon scheduling
        for node_id in range(self.n_nodes):
            beacon_time = node_id * self.params.beacon_interval / self.n_nodes
            self.schedule_event(beacon_time, 'beacon_trigger', node_id)
            
        # Main simulation loop
        while self.event_queue and self.current_time < duration:
            event_time, event_type, node_id, packet = heapq.heappop(self.event_queue)
            self.current_time = event_time
            
            self._handle_event(event_type, node_id, packet)
            
        return self._generate_results()
        
    def _handle_event(self, event_type: str, node_id: int, packet: Optional[MACPacket]):
        """Handle MAC events."""
        if event_type == 'beacon_trigger':
            self._handle_beacon_trigger(node_id)
        elif event_type == 'packet_transmission':
            self._handle_packet_transmission(node_id, packet)
        elif event_type == 'packet_reception':
            self._handle_packet_reception(node_id, packet)
        elif event_type == 'response_timeout':
            self._handle_response_timeout(node_id, packet)
        elif event_type == 'channel_idle':
            self._handle_channel_idle()
        elif event_type == 'backoff_complete':
            self._handle_backoff_complete(node_id, packet)
            
    def _handle_beacon_trigger(self, node_id: int):
        """Handle beacon transmission trigger."""
        # Create beacon packet
        beacon_packet = MACPacket(
            packet_id=self._generate_packet_id(),
            packet_type=PacketType.BEACON,
            source_id=node_id,
            dest_id=-1,  # Broadcast
            timestamp=self.current_time,
            payload={
                'sync_params': {
                    'delay_estimate': np.random.randn() * 1e-9,
                    'freq_estimate': np.random.randn() * 1e-6
                },
                'sequence_number': len(self.stats['packets_sent'])
            }
        )
        
        # Add to transmission queue
        self._enqueue_packet(node_id, beacon_packet)
        
        # Schedule next beacon
        next_beacon_time = self.current_time + self.params.beacon_interval
        self.schedule_event(next_beacon_time, 'beacon_trigger', node_id)
        
    def _enqueue_packet(self, node_id: int, packet: MACPacket):
        """Enqueue packet for transmission."""
        if len(self.node_queues[node_id]) >= self.params.max_queue_size:
            self.stats['packets_dropped'][node_id] += 1
            return
            
        self.node_queues[node_id].append(packet)
        
        # Try to transmit if channel is idle
        if self.channel_state == ChannelState.IDLE and self.node_states[node_id] == 'idle':
            self._attempt_transmission(node_id)
            
    def _attempt_transmission(self, node_id: int):
        """Attempt packet transmission with CSMA/CA."""
        if not self.node_queues[node_id]:
            return
            
        packet = self.node_queues[node_id][0]
        
        # Check if channel is idle
        if self.channel_state == ChannelState.IDLE:
            # Start transmission immediately
            self._start_transmission(node_id, packet)
        else:
            # Perform backoff
            backoff_time = np.random.uniform(self.params.backoff_min, self.params.backoff_max)
            self.schedule_event(self.current_time + backoff_time, 'backoff_complete', 
                              node_id, packet)
            
    def _start_transmission(self, node_id: int, packet: MACPacket):
        """Start packet transmission."""
        # Remove packet from queue
        self.node_queues[node_id].pop(0)
        
        # Update node and channel state
        self.node_states[node_id] = 'transmitting'
        self.channel_state = ChannelState.BUSY
        self.current_transmissions.append((node_id, packet))
        
        # Calculate transmission time
        data_rate = 1e6  # 1 Mbps
        tx_time = packet.size_bytes * 8 / data_rate + self.params.channel_access_delay
        
        # Schedule transmission completion
        self.schedule_event(self.current_time + tx_time, 'packet_transmission', 
                           node_id, packet)
        
        self.stats['packets_sent'][node_id] += 1
        
    def _handle_packet_transmission(self, node_id: int, packet: MACPacket):
        """Handle completion of packet transmission."""
        # Remove from current transmissions
        self.current_transmissions = [
            (nid, pkt) for nid, pkt in self.current_transmissions 
            if not (nid == node_id and pkt.packet_id == packet.packet_id)
        ]
        
        # Update node state
        self.node_states[node_id] = 'idle'
        
        # Check for collisions
        collision_detected = len([
            (nid, pkt) for nid, pkt in self.current_transmissions 
            if pkt.timestamp == packet.timestamp
        ]) > 0
        
        if collision_detected:
            self.stats['collisions'] += 1
            self.channel_state = ChannelState.COLLISION
            # Handle collision recovery
            self._handle_collision(node_id, packet)
        else:
            # Successful transmission - deliver to neighbors
            self._deliver_packet(node_id, packet)
            
        # Update channel state
        if not self.current_transmissions:
            self.channel_state = ChannelState.IDLE
            self.schedule_event(self.current_time + 1e-6, 'channel_idle', 0)
            
        # Try to transmit next packet in queue
        if self.node_queues[node_id]:
            self._attempt_transmission(node_id)
            
    def _deliver_packet(self, sender_id: int, packet: MACPacket):
        """Deliver packet to neighboring nodes."""
        # Determine receivers based on adjacency matrix
        if packet.dest_id == -1:  # Broadcast
            receivers = np.where(self.adjacency_matrix[sender_id, :] > 0)[0]
        else:  # Unicast
            if self.adjacency_matrix[sender_id, packet.dest_id] > 0:
                receivers = [packet.dest_id]
            else:
                receivers = []  # No direct connection
                
        for receiver_id in receivers:
            # Apply packet error rate
            if np.random.random() > self.params.packet_error_rate:
                # Successful reception
                reception_delay = 1e-6  # Small propagation delay
                self.schedule_event(self.current_time + reception_delay, 
                                  'packet_reception', receiver_id, packet)
                
    def _handle_packet_reception(self, node_id: int, packet: MACPacket):
        """Handle packet reception at a node."""
        self.stats['packets_received'][node_id] += 1
        
        # Generate response for beacon packets
        if packet.packet_type == PacketType.BEACON:
            response_packet = MACPacket(
                packet_id=self._generate_packet_id(),
                packet_type=PacketType.RESPONSE,
                source_id=node_id,
                dest_id=packet.source_id,
                timestamp=self.current_time,
                payload={
                    'beacon_id': packet.packet_id,
                    'sync_params': {
                        'delay_estimate': np.random.randn() * 1e-9,
                        'freq_estimate': np.random.randn() * 1e-6
                    }
                }
            )
            
            # Add response delay
            response_delay = np.random.uniform(0.001, 0.01)
            self.schedule_event(self.current_time + response_delay, 'backoff_complete', 
                              node_id, response_packet)
            
    def _handle_collision(self, node_id: int, packet: MACPacket):
        """Handle packet collision."""
        if packet.retries < packet.max_retries:
            packet.retries += 1
            # Exponential backoff
            backoff_time = self.params.backoff_min * (2 ** packet.retries)
            backoff_time = min(backoff_time, self.params.backoff_max)
            backoff_time += np.random.uniform(0, backoff_time)
            
            self.schedule_event(self.current_time + backoff_time, 'backoff_complete', 
                              node_id, packet)
        else:
            # Drop packet after max retries
            self.stats['packets_dropped'][node_id] += 1
            
    def _handle_backoff_complete(self, node_id: int, packet: MACPacket):
        """Handle backoff completion."""
        if packet not in self.node_queues[node_id]:
            self.node_queues[node_id].insert(0, packet)
        self._attempt_transmission(node_id)
        
    def _handle_channel_idle(self):
        """Handle channel becoming idle."""
        # Try to start pending transmissions
        for node_id in range(self.n_nodes):
            if (self.node_queues[node_id] and 
                self.node_states[node_id] == 'idle' and 
                self.channel_state == ChannelState.IDLE):
                self._attempt_transmission(node_id)
                break  # Only one transmission at a time
                
    def _generate_packet_id(self) -> int:
        """Generate unique packet ID."""
        return int(self.current_time * 1e6) + np.random.randint(0, 1000)
        
    def _generate_results(self) -> Dict[str, Any]:
        """Generate simulation results and statistics."""
        total_packets_sent = sum(self.stats['packets_sent'].values())
        total_packets_received = sum(self.stats['packets_received'].values())
        total_packets_dropped = sum(self.stats['packets_dropped'].values())
        
        # Calculate channel utilization
        busy_time = sum([pkt.size_bytes * 8 / 1e6 for pkt in 
                        [pkt for _, pkt in self.current_transmissions]])
        channel_utilization = busy_time / self.current_time if self.current_time > 0 else 0
        
        return {
            'simulation_time': self.current_time,
            'total_packets_sent': total_packets_sent,
            'total_packets_received': total_packets_received,
            'total_packets_dropped': total_packets_dropped,
            'packet_delivery_ratio': (total_packets_received / total_packets_sent 
                                    if total_packets_sent > 0 else 0),
            'collision_rate': self.stats['collisions'] / total_packets_sent if total_packets_sent > 0 else 0,
            'channel_utilization': channel_utilization,
            'per_node_stats': {
                'packets_sent': dict(self.stats['packets_sent']),
                'packets_received': dict(self.stats['packets_received']),
                'packets_dropped': dict(self.stats['packets_dropped'])
            }
        }
