"""
TransceiverNode class for modeling hardware-level transceiver behavior.

This module implements a comprehensive transceiver node model that combines
various hardware imperfections including LO drift, ADC effects, and IQ imbalance.
"""

import numpy as np
from typing import Optional, Dict, Any
from dataclasses import dataclass

from phy.osc import OscillatorParams, AllanDeviationGenerator
from hw.lo import LocalOscillator
from hw.adc import ADCModel
from hw.iq import IQImbalance


@dataclass
class TransceiverConfig:
    """Configuration parameters for transceiver node."""
    node_id: int
    carrier_freq: float      # Carrier frequency (Hz)
    sample_rate: float       # Sampling rate (Hz)
    tx_power: float          # Transmit power (dBm)
    osc_params: OscillatorParams
    adc_bits: int           # ADC resolution (bits)
    iq_imbalance_db: float  # IQ imbalance (dB)


class TransceiverNode:
    """Hardware-level transceiver node with realistic imperfections."""
    
    def __init__(self, config: TransceiverConfig):
        self.config = config
        self.node_id = config.node_id
        
        # Initialize hardware components
        self.oscillator = AllanDeviationGenerator(config.osc_params, config.sample_rate)
        self.lo = LocalOscillator(config.carrier_freq, config.osc_params)
        self.adc = ADCModel(config.adc_bits, config.sample_rate)
        self.iq_imbalance = IQImbalance(config.iq_imbalance_db)
        
        # Internal state
        self.phase_offset = 0.0
        self.frequency_offset = 0.0
        
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
        # Apply LO phase noise and drift for downconversion
        lo_phase = self.lo.get_phase_at_time(timestamp, len(rf_signal))
        signal_downconverted = rf_signal * np.exp(-1j * lo_phase)
        
        # Apply ADC effects
        signal_digitized = self.adc.digitize(signal_downconverted)
        
        # Apply IQ imbalance
        signal_corrected = self.iq_imbalance.apply_rx_imbalance(signal_digitized)
        
        return signal_corrected
        
    def get_current_frequency_offset(self) -> float:
        """Get current frequency offset from nominal."""
        return self.lo.get_frequency_offset()
        
    def get_current_phase_offset(self) -> float:
        """Get current phase offset."""
        return self.lo.get_phase_offset()
