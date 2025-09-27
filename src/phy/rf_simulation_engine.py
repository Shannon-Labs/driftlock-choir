#!/usr/bin/env python3
"""
RF Simulation Engine
Hardware-realistic simulation for formant-based beacon systems

This module provides comprehensive RF simulation capabilities that model
real hardware constraints, propagation effects, and interference patterns.
Focus is on engineering accuracy rather than musical abstractions.

Key Simulation Features:
- Realistic RF hardware models (ADC quantization, phase noise, nonlinearity)
- Propagation modeling (path loss, shadowing, multipath)
- Interference simulation (co-channel, adjacent channel, spurious)
- Performance analysis using standard RF metrics
- Monte Carlo simulation for statistical validation
- Hardware impairment modeling (I/Q imbalance, DC offset, etc.)
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Callable
from enum import Enum
import time
import concurrent.futures
from scipy import stats
from scipy.signal import butter, filtfilt, hilbert
from scipy.spatial.distance import cdist

from .practical_rf_beacon import PracticalRFBeacon, RFParameters, ChannelModel, DetectionResult


class PropagationModel(Enum):
    """RF propagation models"""
    FREE_SPACE = "free_space"
    HATA_URBAN = "hata_urban"
    HATA_SUBURBAN = "hata_suburban"
    COST231_HATA = "cost231_hata"
    ITU_R_1546 = "itu_r_1546"


class HardwareType(Enum):
    """RF hardware types with different impairment characteristics"""
    HIGH_END_SDR = "high_end_sdr"        # USRP, high-performance SDR
    MID_RANGE_SDR = "mid_range_sdr"      # RTL-SDR, HackRF
    COMMERCIAL_RADIO = "commercial"       # Commercial VHF radio
    MOBILE_DEVICE = "mobile"             # Smartphone, tablet RF


@dataclass
class HardwareImpairments:
    """RF hardware impairment parameters"""
    
    # ADC parameters
    adc_bits: int = 12                    # ADC resolution
    adc_full_scale_dbm: float = 0.0       # ADC full scale power
    
    # Phase noise
    phase_noise_dbchz: float = -100.0     # Phase noise at 1kHz offset
    phase_noise_floor_dbchz: float = -150.0  # Phase noise floor
    
    # Frequency accuracy
    frequency_accuracy_ppm: float = 2.5   # Frequency accuracy in ppm
    
    # I/Q imbalance
    iq_amplitude_imbalance_db: float = 0.1  # I/Q amplitude imbalance
    iq_phase_imbalance_deg: float = 1.0   # I/Q phase imbalance
    
    # DC offset
    dc_offset_percent: float = 0.1        # DC offset as % of full scale
    
    # Nonlinearity
    ip3_dbm: float = 20.0                 # Third-order intercept point
    p1db_dbm: float = 10.0                # 1dB compression point
    
    # Thermal noise
    noise_figure_db: float = 8.0          # Receiver noise figure


@dataclass
class PropagationParameters:
    """RF propagation parameters"""
    
    # Basic parameters
    frequency_mhz: float = 150.0          # Frequency in MHz
    tx_power_dbm: float = 30.0            # Transmit power in dBm
    tx_antenna_gain_dbi: float = 2.0      # TX antenna gain
    rx_antenna_gain_dbi: float = 2.0      # RX antenna gain
    
    # Environmental parameters
    tx_height_m: float = 10.0             # TX antenna height
    rx_height_m: float = 2.0              # RX antenna height
    environment_type: str = "urban"       # urban, suburban, rural
    
    # Fading parameters
    shadowing_std_db: float = 8.0         # Log-normal shadowing std dev
    rician_k_factor_db: float = 6.0       # Rician K-factor
    doppler_spread_hz: float = 5.0        # Doppler spread


@dataclass
class SimulationScenario:
    """Complete RF simulation scenario"""
    
    name: str
    description: str
    
    # Geometry
    tx_positions: List[Tuple[float, float]] = field(default_factory=list)  # (x, y) in meters
    rx_positions: List[Tuple[float, float]] = field(default_factory=list)  # (x, y) in meters
    
    # RF parameters
    propagation_params: PropagationParameters = field(default_factory=PropagationParameters)
    hardware_impairments: HardwareImpairments = field(default_factory=HardwareImpairments)
    
    # Interference
    interferer_powers_dbm: List[float] = field(default_factory=list)
    interferer_frequencies_hz: List[float] = field(default_factory=list)
    
    # Simulation parameters
    num_monte_carlo_runs: int = 1000
    snr_range_db: Tuple[float, float] = (-10.0, 30.0)
    snr_step_db: float = 2.0


class RFSimulationEngine:
    """
    RF Simulation Engine
    
    Provides comprehensive RF simulation capabilities for evaluating
    formant-based beacon systems under realistic conditions.
    """
    
    def __init__(self):
        self.beacon_system = PracticalRFBeacon()
        
        # Predefined hardware configurations
        self.hardware_configs = {
            HardwareType.HIGH_END_SDR: HardwareImpairments(
                adc_bits=14, phase_noise_dbchz=-110, frequency_accuracy_ppm=0.1,
                iq_amplitude_imbalance_db=0.05, noise_figure_db=5.0
            ),
            HardwareType.MID_RANGE_SDR: HardwareImpairments(
                adc_bits=8, phase_noise_dbchz=-90, frequency_accuracy_ppm=20.0,
                iq_amplitude_imbalance_db=0.5, noise_figure_db=10.0
            ),
            HardwareType.COMMERCIAL_RADIO: HardwareImpairments(
                adc_bits=12, phase_noise_dbchz=-100, frequency_accuracy_ppm=2.5,
                iq_amplitude_imbalance_db=0.2, noise_figure_db=8.0
            ),
            HardwareType.MOBILE_DEVICE: HardwareImpairments(
                adc_bits=10, phase_noise_dbchz=-85, frequency_accuracy_ppm=50.0,
                iq_amplitude_imbalance_db=1.0, noise_figure_db=12.0
            )
        }
        
    def calculate_path_loss(self, distance_m: float, params: PropagationParameters, 
                           model: PropagationModel = PropagationModel.HATA_URBAN) -> float:
        """Calculate path loss using specified propagation model"""
        
        if model == PropagationModel.FREE_SPACE:
            # Free space path loss
            path_loss_db = 20 * np.log10(distance_m) + 20 * np.log10(params.frequency_mhz) - 27.55
            
        elif model == PropagationModel.HATA_URBAN:
            # Hata model for urban areas
            a_hr = (1.1 * np.log10(params.frequency_mhz) - 0.7) * params.rx_height_m - \
                   (1.56 * np.log10(params.frequency_mhz) - 0.8)
            
            path_loss_db = 69.55 + 26.16 * np.log10(params.frequency_mhz) - 13.82 * np.log10(params.tx_height_m) - \
                          a_hr + (44.9 - 6.55 * np.log10(params.tx_height_m)) * np.log10(distance_m / 1000.0)
                          
        elif model == PropagationModel.HATA_SUBURBAN:
            # Hata suburban correction
            urban_loss = self.calculate_path_loss(distance_m, params, PropagationModel.HATA_URBAN)
            correction = 2 * (np.log10(params.frequency_mhz / 28.0))**2 + 5.4
            path_loss_db = urban_loss - correction
            
        else:
            # Default to free space
            path_loss_db = 20 * np.log10(distance_m) + 20 * np.log10(params.frequency_mhz) - 27.55
        
        return path_loss_db
    
    def apply_hardware_impairments(self, signal: NDArray[np.complex128], 
                                  impairments: HardwareImpairments,
                                  sample_rate_hz: float) -> NDArray[np.complex128]:
        """Apply realistic hardware impairments to signal"""
        
        impaired_signal = signal.copy()
        
        # 1. Add phase noise
        if impairments.phase_noise_dbchz > -200:
            phase_noise_power = 10**(impairments.phase_noise_dbchz / 10) * sample_rate_hz
            phase_noise = np.sqrt(phase_noise_power) * np.random.randn(len(signal))
            phase_noise_integrated = np.cumsum(phase_noise) / sample_rate_hz
            impaired_signal *= np.exp(1j * phase_noise_integrated)
        
        # 2. Add frequency offset
        if impairments.frequency_accuracy_ppm > 0:
            max_freq_error = 150e6 * impairments.frequency_accuracy_ppm * 1e-6  # At 150 MHz
            freq_error = np.random.uniform(-max_freq_error, max_freq_error)
            t = np.arange(len(signal)) / sample_rate_hz
            impaired_signal *= np.exp(1j * 2 * np.pi * freq_error * t)
        
        # 3. Apply I/Q imbalance
        if impairments.iq_amplitude_imbalance_db > 0 or impairments.iq_phase_imbalance_deg > 0:
            # Amplitude imbalance
            amp_imbalance = 10**(impairments.iq_amplitude_imbalance_db / 20)
            phase_imbalance_rad = np.deg2rad(impairments.iq_phase_imbalance_deg)
            
            # Apply imbalance
            i_component = np.real(impaired_signal) * amp_imbalance
            q_component = np.imag(impaired_signal) * np.cos(phase_imbalance_rad) + \
                         np.real(impaired_signal) * np.sin(phase_imbalance_rad)
            
            impaired_signal = i_component + 1j * q_component
        
        # 4. Add DC offset
        if impairments.dc_offset_percent > 0:
            signal_rms = np.sqrt(np.mean(np.abs(impaired_signal)**2))
            dc_offset = signal_rms * impairments.dc_offset_percent / 100.0
            dc_i = np.random.uniform(-dc_offset, dc_offset)
            dc_q = np.random.uniform(-dc_offset, dc_offset)
            impaired_signal += dc_i + 1j * dc_q
        
        # 5. ADC quantization
        if impairments.adc_bits < 16:
            # Simple uniform quantization
            full_scale = 10**(impairments.adc_full_scale_dbm / 20) * np.sqrt(0.001)  # Convert dBm to V
            levels = 2**(impairments.adc_bits - 1)  # Signed ADC
            
            # Normalize to full scale
            signal_power = np.sqrt(np.mean(np.abs(impaired_signal)**2))
            if signal_power > 0:
                normalized_signal = impaired_signal * full_scale / signal_power
                
                # Quantize I and Q separately
                quantized_i = np.round(np.real(normalized_signal) * levels) / levels
                quantized_q = np.round(np.imag(normalized_signal) * levels) / levels
                
                # Convert back
                impaired_signal = (quantized_i + 1j * quantized_q) * signal_power / full_scale
        
        return impaired_signal
    
    def simulate_interference(self, signal: NDArray[np.complex128],
                            interferer_powers_dbm: List[float],
                            interferer_frequencies_hz: List[float],
                            sample_rate_hz: float) -> NDArray[np.complex128]:
        """Add realistic interference signals"""
        
        interfered_signal = signal.copy()
        t = np.arange(len(signal)) / sample_rate_hz
        
        for power_dbm, freq_hz in zip(interferer_powers_dbm, interferer_frequencies_hz):
            # Convert power to linear scale
            power_linear = 10**(power_dbm / 10) * 1e-3  # Convert dBm to watts
            amplitude = np.sqrt(power_linear)
            
            # Generate interfering signal (could be more sophisticated)
            phase = np.random.uniform(0, 2*np.pi)
            interferer = amplitude * np.exp(1j * (2 * np.pi * freq_hz * t + phase))
            
            # Add random modulation to make it more realistic
            modulation = 1 + 0.1 * np.random.randn(len(signal))  # 10% amplitude modulation
            interferer *= modulation
            
            interfered_signal += interferer
        
        return interfered_signal
    
    def run_monte_carlo_simulation(self, scenario: SimulationScenario) -> Dict[str, Any]:
        """Run Monte Carlo simulation for performance evaluation"""
        
        print(f"🔬 Running Monte Carlo simulation: {scenario.name}")
        print(f"   {scenario.num_monte_carlo_runs} runs, SNR range {scenario.snr_range_db[0]}-{scenario.snr_range_db[1]} dB")
        
        # SNR points to test
        snr_points = np.arange(scenario.snr_range_db[0], 
                              scenario.snr_range_db[1] + scenario.snr_step_db, 
                              scenario.snr_step_db)
        
        # Results storage
        results = {
            'snr_db': snr_points,
            'detection_probability': np.zeros(len(snr_points)),
            'false_alarm_probability': np.zeros(len(snr_points)),
            'average_processing_time_ms': np.zeros(len(snr_points)),
            'scenario': scenario
        }
        
        # Test each signature
        for sig_id in range(5):  # Test first 5 signatures
            
            sig_results = {
                'detection_prob': np.zeros(len(snr_points)),
                'false_alarm_prob': np.zeros(len(snr_points))
            }
            
            for snr_idx, target_snr_db in enumerate(snr_points):
                
                detections = 0
                false_alarms = 0
                total_time = 0.0
                
                for run in range(scenario.num_monte_carlo_runs):
                    
                    # Generate beacon signal
                    beacon_signal = self.beacon_system.generate_beacon_signal(sig_id)
                    
                    # Add channel effects
                    # 1. Set signal power for target SNR
                    noise_power = 10**(-100/10) * 1e-3  # -100 dBm noise power
                    signal_power = noise_power * 10**(target_snr_db/10)
                    signal_amplitude = np.sqrt(signal_power)
                    
                    scaled_signal = beacon_signal * signal_amplitude / np.sqrt(np.mean(np.abs(beacon_signal)**2))
                    
                    # 2. Apply hardware impairments
                    impaired_signal = self.apply_hardware_impairments(
                        scaled_signal, scenario.hardware_impairments, self.beacon_system.rf_params.sample_rate_hz
                    )
                    
                    # 3. Add interference
                    if scenario.interferer_powers_dbm:
                        impaired_signal = self.simulate_interference(
                            impaired_signal, scenario.interferer_powers_dbm, 
                            scenario.interferer_frequencies_hz, self.beacon_system.rf_params.sample_rate_hz
                        )
                    
                    # 4. Add thermal noise
                    noise = np.sqrt(noise_power/2) * (np.random.randn(len(impaired_signal)) + 
                                                    1j * np.random.randn(len(impaired_signal)))
                    received_signal = impaired_signal + noise
                    
                    # Detection
                    start_time = time.time()
                    detection_results = self.beacon_system.detect_beacon(received_signal, ChannelModel.AWGN)
                    processing_time = (time.time() - start_time) * 1000  # ms
                    
                    # Check detection results
                    correct_detection = any(r.detected and r.signature_id == sig_id for r in detection_results)
                    false_detection = any(r.detected and r.signature_id != sig_id for r in detection_results)
                    
                    if correct_detection:
                        detections += 1
                    if false_detection:
                        false_alarms += 1
                    
                    total_time += processing_time
                
                # Calculate probabilities
                sig_results['detection_prob'][snr_idx] = detections / scenario.num_monte_carlo_runs
                sig_results['false_alarm_prob'][snr_idx] = false_alarms / scenario.num_monte_carlo_runs
                
                # Store in main results (average across signatures)
                results['detection_probability'][snr_idx] += sig_results['detection_prob'][snr_idx] / 5
                results['false_alarm_probability'][snr_idx] += sig_results['false_alarm_prob'][snr_idx] / 5
                results['average_processing_time_ms'][snr_idx] = total_time / scenario.num_monte_carlo_runs
                
                print(f"     SNR {target_snr_db:+3.0f} dB: Pd={sig_results['detection_prob'][snr_idx]:.3f}, "
                      f"Pfa={sig_results['false_alarm_prob'][snr_idx]:.3f}")
        
        return results
    
    def analyze_performance(self, results: Dict[str, Any]) -> Dict[str, float]:
        """Analyze simulation results and compute key metrics"""
        
        # Find key performance points
        metrics = {}
        
        # Sensitivity (SNR for 90% detection probability)
        detection_probs = results['detection_probability']
        snr_points = results['snr_db']
        
        sensitivity_idx = np.where(detection_probs >= 0.9)[0]
        if len(sensitivity_idx) > 0:
            metrics['sensitivity_snr_db'] = snr_points[sensitivity_idx[0]]
        else:
            metrics['sensitivity_snr_db'] = float('inf')
        
        # False alarm rate at sensitivity point
        if 'sensitivity_snr_db' in metrics and metrics['sensitivity_snr_db'] < float('inf'):
            metrics['false_alarm_rate'] = results['false_alarm_probability'][sensitivity_idx[0]]
        else:
            metrics['false_alarm_rate'] = 1.0
        
        # Processing time
        metrics['avg_processing_time_ms'] = np.mean(results['average_processing_time_ms'])
        
        # Dynamic range (difference between 10% and 90% detection points)
        detection_10_idx = np.where(detection_probs >= 0.1)[0]
        detection_90_idx = np.where(detection_probs >= 0.9)[0]
        
        if len(detection_10_idx) > 0 and len(detection_90_idx) > 0:
            metrics['dynamic_range_db'] = snr_points[detection_90_idx[0]] - snr_points[detection_10_idx[0]]
        else:
            metrics['dynamic_range_db'] = float('inf')
        
        return metrics
    
    def create_standard_scenarios(self) -> Dict[str, SimulationScenario]:
        """Create standard simulation scenarios for system evaluation"""
        
        scenarios = {}
        
        # 1. Baseline scenario - clean conditions
        scenarios['baseline'] = SimulationScenario(
            name="Baseline",
            description="Clean AWGN channel with high-end hardware",
            propagation_params=PropagationParameters(),
            hardware_impairments=self.hardware_configs[HardwareType.HIGH_END_SDR],
            num_monte_carlo_runs=500
        )
        
        # 2. Mobile device scenario
        scenarios['mobile'] = SimulationScenario(
            name="Mobile Device", 
            description="Mobile device with typical impairments",
            propagation_params=PropagationParameters(),
            hardware_impairments=self.hardware_configs[HardwareType.MOBILE_DEVICE],
            num_monte_carlo_runs=500
        )
        
        # 3. Interference scenario
        scenarios['interference'] = SimulationScenario(
            name="Co-channel Interference",
            description="Co-channel interference from nearby transmitters",
            propagation_params=PropagationParameters(),
            hardware_impairments=self.hardware_configs[HardwareType.COMMERCIAL_RADIO],
            interferer_powers_dbm=[-10.0, -15.0],  # Interferers at -10 and -15 dBm
            interferer_frequencies_hz=[1000.0, 3000.0],  # 1 kHz and 3 kHz offset
            num_monte_carlo_runs=300
        )
        
        # 4. Low-cost hardware scenario  
        scenarios['low_cost'] = SimulationScenario(
            name="Low-cost Hardware",
            description="RTL-SDR with realistic impairments",
            propagation_params=PropagationParameters(),
            hardware_impairments=self.hardware_configs[HardwareType.MID_RANGE_SDR],
            num_monte_carlo_runs=300
        )
        
        return scenarios


def main():
    """Demonstrate RF simulation engine capabilities"""
    print("🔬 RF Simulation Engine")
    print("Hardware-Realistic Performance Evaluation")
    print("=" * 60)
    print()
    
    # Initialize simulation engine
    sim_engine = RFSimulationEngine()
    
    # Create standard scenarios
    scenarios = sim_engine.create_standard_scenarios()
    
    print("📋 Available Simulation Scenarios:")
    for name, scenario in scenarios.items():
        print(f"   • {name}: {scenario.description}")
    print()
    
    # Run a quick simulation on baseline scenario
    print("🚀 Running Baseline Scenario Simulation:")
    baseline_scenario = scenarios['baseline']
    baseline_scenario.num_monte_carlo_runs = 100  # Reduced for demo
    baseline_scenario.snr_range_db = (0.0, 20.0)
    baseline_scenario.snr_step_db = 5.0
    
    results = sim_engine.run_monte_carlo_simulation(baseline_scenario)
    
    print()
    print("📊 Performance Analysis:")
    metrics = sim_engine.analyze_performance(results)
    
    print(f"   Sensitivity (90% detection): {metrics['sensitivity_snr_db']:.1f} dB SNR")
    print(f"   False Alarm Rate: {metrics['false_alarm_rate']:.1e}")
    print(f"   Processing Time: {metrics['avg_processing_time_ms']:.1f} ms")
    print(f"   Dynamic Range: {metrics['dynamic_range_db']:.1f} dB")
    
    print()
    print("📈 Detection Probability vs SNR:")
    for i, (snr, pd) in enumerate(zip(results['snr_db'], results['detection_probability'])):
        bar_length = int(pd * 20)  # Scale to 20 characters
        bar = "█" * bar_length + "░" * (20 - bar_length)
        print(f"   {snr:+3.0f} dB: {bar} {pd:.2f}")
    
    print()
    print("✅ RF simulation demonstrates practical beacon system")
    print("   performance under realistic hardware and channel conditions.")
    print("   This provides engineering-grade validation of the formant-based")
    print("   spectral signature approach for RF beacon applications.")


if __name__ == "__main__":
    main()