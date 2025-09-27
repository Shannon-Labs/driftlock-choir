"""
TransceiverNode class for modeling hardware-level transceiver behavior.

This module implements a comprehensive transceiver node model that combines
various hardware imperfections including LO drift, ADC effects, and IQ imbalance.
"""

import numpy as np
from scipy import signal
from typing import Optional, Dict, Any
from dataclasses import dataclass

from ..phy.osc import OscillatorParams, AllanDeviationGenerator
from .lo import LocalOscillator
from .adc import ADCModel
from .iq import IQImbalance


@dataclass
class TrxParams:
    """Configuration parameters for transceiver node."""
    node_id: int
    carrier_freq: float      # Carrier frequency (Hz)
    sample_rate: float       # Sampling rate (Hz)
    tx_power: float          # Transmit power (dBm)
    osc_params: OscillatorParams
    adc_bits: int           # ADC resolution (bits)
    iq_imbalance_db: float  # IQ imbalance (dB)
    channel_bandwidth_hz: float = 25000.0
    iip3_dbm: float = 0.0
    p1db_dbm: float = -10.0


class Transceiver:
    """Hardware-level transceiver node with realistic imperfections."""
    
    def __init__(self, config: TrxParams):
        self.params = config
        self.node_id = config.node_id
        
        # Initialize hardware components
        self.oscillator = AllanDeviationGenerator(config.osc_params, config.sample_rate)
        self.lo = LocalOscillator(config.carrier_freq, config.osc_params)
        self.adc = ADCModel(config.adc_bits, config.sample_rate)
        self.iq_imbalance = IQImbalance(config.iq_imbalance_db)

        # Initialize digital BPF for receiver selectivity (baseband equivalent)
        nyquist = self.params.sample_rate / 2.0
        cutoff_norm = self.params.channel_bandwidth_hz / 2.0 / nyquist
        self.bpf_b, self.bpf_a = signal.butter(5, cutoff_norm, btype='low', analog=False)
        
        # Internal state
        self.phase_offset = 0.0
        self.frequency_offset = 0.0
        
    def _apply_non_linearity(self, signal_in: np.ndarray) -> np.ndarray:
        """
        Apply non-linear distortion (compression/intermodulation) using a simple
        soft-limiter model based on P1dB and IIP3.
        
        We use a simplified tanh-based soft limiter for complex baseband signals.
        The input power is normalized relative to P1dB.
        """
        # Convert P1dB from dBm to linear power (W) relative to 1 Ohm load (arbitrary)
        # P1dB_lin = 10^(P1dB_dBm / 10) * 1e-3
        # Since we are dealing with normalized complex baseband signals, we can
        # use the P1dB_dBm directly to scale the input magnitude.
        
        # Convert P1dB_dBm to linear magnitude scale factor (V/sqrt(Ohm))
        # P1dB_mag = sqrt(10^(P1dB_dBm / 10) * 1e-3)
        # For simulation purposes, we normalize the input magnitude relative to P1dB
        
        p1db_mag = np.sqrt(10**(self.params.p1db_dbm / 10.0) * 1e-3)
        
        # Normalize input magnitude by P1dB magnitude
        mag_in = np.abs(signal_in)
        normalized_mag = mag_in / p1db_mag
        
        # Apply soft limiting (tanh approximation)
        # The output magnitude is limited by the P1dB point.
        # We use a simple compression model: mag_out = mag_in / (1 + (mag_in/P1dB_mag)^2)
        # A more accurate model would involve IIP3, but for simplicity and stability,
        # we use a magnitude-based compression that ensures output power saturation.
        
        # Simplified compression:
        compression_factor = 1.0 / (1.0 + normalized_mag**2)
        
        # Apply compression to the magnitude
        mag_out = mag_in * compression_factor
        
        # Reconstruct the complex signal
        signal_out = mag_out * (signal_in / mag_in)
        
        # Handle zero magnitude case to avoid division by zero
        signal_out[mag_in == 0] = 0
        
        return signal_out
        
    def transmit(self, baseband_signal: np.ndarray, timestamp: float) -> np.ndarray:
        """Transmit baseband signal with hardware imperfections."""
        # Apply IQ imbalance
        signal_iq = self.iq_imbalance.apply_tx_imbalance(baseband_signal)
        
        # Apply LO phase noise and drift
        lo_phase = self.lo.get_phase_at_time(timestamp, len(signal_iq))
        signal_upconverted = signal_iq * np.exp(1j * lo_phase)
        
        return signal_upconverted
        
    def receive(self, rf_signal: np.ndarray, timestamp: float) -> np.ndarray:
        """Receive RF signal and convert to baseband with hardware imperfections."""
        
        # 1. Apply non-linear distortion (RF stage)
        signal_non_linear = self._apply_non_linearity(rf_signal)
        
        # 2. Apply LO phase noise and drift for downconversion
        lo_phase = self.lo.get_phase_at_time(timestamp, len(signal_non_linear))
        signal_downconverted = signal_non_linear * np.exp(-1j * lo_phase)
        
        # 3. Apply digital BPF (Receiver selectivity)
        # Note: signal.lfilter handles complex signals
        signal_filtered = signal.lfilter(self.bpf_b, self.bpf_a, signal_downconverted)
        
        # 4. Apply ADC effects
        signal_digitized = self.adc.digitize(signal_filtered)
        
        # 5. Apply IQ imbalance
        signal_corrected = self.iq_imbalance.apply_rx_imbalance(signal_digitized)
        
        return signal_corrected
        
    def get_current_frequency_offset(self) -> float:
        """Get current frequency offset from nominal."""
        return self.lo.get_frequency_offset()
        
    def get_current_phase_offset(self) -> float:
        """Get current phase offset."""
        return self.lo.get_phase_offset()
