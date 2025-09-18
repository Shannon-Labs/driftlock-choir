"""
IQ imbalance and DC offset modeling for quadrature transceivers.

This module implements realistic IQ imbalance effects including amplitude
and phase imbalance, as well as DC offset in I and Q branches.
"""

import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass


@dataclass
class IQImbalanceParams:
    """Parameters for IQ imbalance modeling."""
    amplitude_imbalance_db: float  # Amplitude imbalance (dB)
    phase_imbalance_deg: float     # Phase imbalance (degrees)
    dc_offset_i: float             # DC offset in I branch (V)
    dc_offset_q: float             # DC offset in Q branch (V)


class IQImbalance:
    """Model for IQ imbalance and DC offset in quadrature systems."""
    
    def __init__(self, imbalance_db: float = 0.1, phase_imbalance_deg: float = 1.0,
                 dc_offset_i: float = 0.01, dc_offset_q: float = 0.01):
        self.params = IQImbalanceParams(
            imbalance_db, phase_imbalance_deg, dc_offset_i, dc_offset_q
        )
        
        # Precompute imbalance coefficients
        self._compute_imbalance_matrix()
        
    def _compute_imbalance_matrix(self):
        """Compute the IQ imbalance transformation matrix."""
        # Convert parameters to linear scale
        amp_imbalance = 10 ** (self.params.amplitude_imbalance_db / 20)
        phase_imbalance_rad = np.deg2rad(self.params.phase_imbalance_deg)
        
        # IQ imbalance matrix for transmitter
        self.tx_matrix = np.array([
            [1.0, 0.0],
            [np.sin(phase_imbalance_rad), amp_imbalance * np.cos(phase_imbalance_rad)]
        ])
        
        # IQ imbalance matrix for receiver (inverse of transmitter)
        det = self.tx_matrix[0, 0] * self.tx_matrix[1, 1] - self.tx_matrix[0, 1] * self.tx_matrix[1, 0]
        self.rx_matrix = np.array([
            [self.tx_matrix[1, 1], -self.tx_matrix[0, 1]],
            [-self.tx_matrix[1, 0], self.tx_matrix[0, 0]]
        ]) / det
        
    def apply_tx_imbalance(self, signal: np.ndarray) -> np.ndarray:
        """Apply transmitter IQ imbalance to complex signal."""
        if not np.iscomplexobj(signal):
            raise ValueError("Input signal must be complex")
            
        # Extract I and Q components
        i_component = np.real(signal)
        q_component = np.imag(signal)
        
        # Apply IQ imbalance transformation
        iq_vector = np.array([i_component, q_component])
        imbalanced_iq = self.tx_matrix @ iq_vector
        
        # Add DC offsets
        imbalanced_iq[0] += self.params.dc_offset_i
        imbalanced_iq[1] += self.params.dc_offset_q
        
        # Reconstruct complex signal
        return imbalanced_iq[0] + 1j * imbalanced_iq[1]
        
    def apply_rx_imbalance(self, signal: np.ndarray) -> np.ndarray:
        """Apply receiver IQ imbalance to complex signal."""
        if not np.iscomplexobj(signal):
            raise ValueError("Input signal must be complex")
            
        # Extract I and Q components
        i_component = np.real(signal)
        q_component = np.imag(signal)
        
        # Remove DC offsets first
        i_component -= self.params.dc_offset_i
        q_component -= self.params.dc_offset_q
        
        # Apply IQ imbalance correction
        iq_vector = np.array([i_component, q_component])
        corrected_iq = self.rx_matrix @ iq_vector
        
        # Reconstruct complex signal
        return corrected_iq[0] + 1j * corrected_iq[1]
        
    def get_image_rejection_ratio(self) -> float:
        """Calculate image rejection ratio in dB."""
        # Simplified calculation based on amplitude and phase imbalance
        amp_imbalance_linear = 10 ** (self.params.amplitude_imbalance_db / 20)
        phase_imbalance_rad = np.deg2rad(self.params.phase_imbalance_deg)
        
        # IRR calculation (approximate)
        epsilon_a = (amp_imbalance_linear - 1) / (amp_imbalance_linear + 1)
        epsilon_p = phase_imbalance_rad / 2
        
        irr_linear = 1 / (epsilon_a ** 2 + epsilon_p ** 2)
        return 10 * np.log10(irr_linear)
