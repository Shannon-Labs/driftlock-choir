"""Interference-as-Data Architecture - Revolutionary Musical-RF Intelligence System.

This module implements the core breakthrough of Musical-RF Architecture:
treating interference as a valuable data source instead of noise to be eliminated.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from numpy.typing import NDArray

# Musical interval ratios for consonant interference patterns
MUSICAL_INTERVALS = {
    "unison": 1.0,
    "minor_second": 16/15,
    "major_second": 9/8,
    "minor_third": 6/5,
    "major_third": 5/4,
    "perfect_fourth": 4/3,
    "tritone": 45/32,  # Diminished fifth
    "perfect_fifth": 3/2,
    "minor_sixth": 8/5,
    "major_sixth": 5/3,
    "minor_seventh": 16/9,
    "major_seventh": 15/8,
    "octave": 2.0,
}

# Consonance scores for different intervals (higher = more consonant)
CONSONANCE_SCORES = {
    "unison": 1.0,
    "octave": 0.95,
    "perfect_fifth": 0.9,
    "perfect_fourth": 0.85,
    "major_third": 0.8,
    "minor_third": 0.75,
    "major_sixth": 0.7,
    "minor_sixth": 0.65,
    "major_second": 0.4,
    "major_seventh": 0.35,
    "minor_second": 0.3,
    "minor_seventh": 0.25,
    "tritone": 0.1,  # Most dissonant
}


@dataclass
class InterferencePattern:
    """Represents an interference pattern between two RF signals."""
    
    signal1_freq: float
    signal2_freq: float
    beat_frequency: float
    musical_interval: str
    consonance_score: float
    timing_offset: float
    phase_relationship: float
    spatial_separation: Optional[float] = None


@dataclass
class MultiApertureObservation:
    """Observation from a single aperture/device in the constellation."""
    
    device_id: str
    position: Tuple[float, float, float]  # 3D coordinates
    observed_frequencies: List[float]
    signal_strengths: List[float]
    harmonic_components: Dict[float, List[float]]  # freq -> harmonics
    timestamp: float
    local_interference_patterns: List[InterferencePattern]


class InterferenceIntelligenceEngine:
    """Core engine for Musical-RF interference-as-data processing."""
    
    def __init__(self, fundamental_frequency: float = 50e6):
        self.fundamental_frequency = fundamental_frequency
        self.active_observations: Dict[str, MultiApertureObservation] = {}
        self.global_interference_map: List[InterferencePattern] = []
        self.missing_fundamental_cache: Dict[str, float] = {}
    
    def generate_consonant_interference(
        self, 
        base_frequencies: List[float], 
        target_interval: str = "perfect_fifth"
    ) -> List[float]:
        """Generate frequencies that create consonant interference patterns."""
        
        if target_interval not in MUSICAL_INTERVALS:
            raise ValueError(f"Unknown musical interval: {target_interval}")
        
        interval_ratio = MUSICAL_INTERVALS[target_interval]
        consonant_frequencies = []
        
        for base_freq in base_frequencies:
            # Generate frequency that creates the desired interval
            consonant_freq = base_freq * interval_ratio
            consonant_frequencies.append(consonant_freq)
        
        return consonant_frequencies
    
    def extract_beat_patterns(
        self, 
        signal1: NDArray[np.complex128], 
        signal2: NDArray[np.complex128],
        sample_rate: float
    ) -> InterferencePattern:
        """Extract beat pattern information from two interfering signals."""
        
        # Create interference by combining signals
        combined_signal = signal1 + signal2
        
        # Analyze the resulting beat pattern
        envelope = np.abs(combined_signal)
        
        # Find beat frequency through envelope analysis
        envelope_fft = np.fft.fft(envelope)
        freqs = np.fft.fftfreq(len(envelope), 1/sample_rate)
        
        # Beat frequency is the dominant low-frequency component
        low_freq_mask = (freqs > 0) & (freqs < sample_rate/10)
        low_freq_spectrum = np.abs(envelope_fft[low_freq_mask])
        low_freqs = freqs[low_freq_mask]
        
        if len(low_freq_spectrum) > 0:
            beat_freq_idx = np.argmax(low_freq_spectrum)
            beat_frequency = low_freqs[beat_freq_idx]
        else:
            beat_frequency = 0.0
        
        # Determine carrier frequencies (simplified - would need more sophisticated analysis)
        signal1_spectrum = np.fft.fft(signal1)
        signal2_spectrum = np.fft.fft(signal2)
        
        freq_axis = np.fft.fftfreq(len(signal1), 1/sample_rate)
        pos_freqs = freq_axis[freq_axis > 0]
        
        # Find dominant frequencies
        sig1_dominant_idx = np.argmax(np.abs(signal1_spectrum[freq_axis > 0]))
        sig2_dominant_idx = np.argmax(np.abs(signal2_spectrum[freq_axis > 0]))
        
        freq1 = pos_freqs[sig1_dominant_idx]
        freq2 = pos_freqs[sig2_dominant_idx]
        
        # Identify musical interval
        freq_ratio = max(freq1, freq2) / min(freq1, freq2)
        musical_interval = self._identify_musical_interval(freq_ratio)
        consonance_score = CONSONANCE_SCORES.get(musical_interval, 0.0)
        
        # Calculate phase relationship
        phase1 = np.angle(signal1_spectrum[np.argmax(np.abs(signal1_spectrum))])
        phase2 = np.angle(signal2_spectrum[np.argmax(np.abs(signal2_spectrum))])
        phase_relationship = phase2 - phase1
        
        # Calculate timing offset from phase relationship
        timing_offset = phase_relationship / (2 * np.pi * beat_frequency) if beat_frequency > 0 else 0.0
        
        return InterferencePattern(
            signal1_freq=freq1,
            signal2_freq=freq2,
            beat_frequency=beat_frequency,
            musical_interval=musical_interval,
            consonance_score=consonance_score,
            timing_offset=timing_offset,
            phase_relationship=phase_relationship
        )
    
    def _identify_musical_interval(self, frequency_ratio: float) -> str:
        """Identify the closest musical interval for a given frequency ratio."""
        
        best_interval = "unison"
        min_error = float('inf')
        
        for interval_name, ratio in MUSICAL_INTERVALS.items():
            error = abs(frequency_ratio - ratio)
            if error < min_error:
                min_error = error
                best_interval = interval_name
        
        return best_interval
    
    def reconstruct_missing_fundamental(
        self, 
        observations: List[MultiApertureObservation]
    ) -> Dict[str, float]:
        """Reconstruct missing fundamentals using distributed harmonic information."""
        
        reconstructed_fundamentals = {}
        
        # Collect all observed harmonics across all devices
        all_harmonics = {}
        for obs in observations:
            for freq, harmonics in obs.harmonic_components.items():
                if freq not in all_harmonics:
                    all_harmonics[freq] = []
                all_harmonics[freq].extend(harmonics)
        
        # For each potential fundamental frequency
        for base_freq, harmonic_list in all_harmonics.items():
            if len(harmonic_list) >= 2:  # Need at least 2 harmonics for reconstruction
                # Find the greatest common divisor of harmonic frequencies
                # This gives us the missing fundamental
                harmonic_array = np.array(harmonic_list)
                
                # Simple GCD approach for missing fundamental
                potential_fundamental = self._find_fundamental_from_harmonics(harmonic_array)
                
                if potential_fundamental > 0:
                    device_key = f"reconstructed_{len(reconstructed_fundamentals)}"
                    reconstructed_fundamentals[device_key] = potential_fundamental
                    
                    # Cache the result
                    cache_key = f"{base_freq}_{len(harmonic_list)}"
                    self.missing_fundamental_cache[cache_key] = potential_fundamental
        
        return reconstructed_fundamentals
    
    def _find_fundamental_from_harmonics(self, harmonics: NDArray[np.float64]) -> float:
        """Find the fundamental frequency from a set of harmonics using GCD approach."""
        
        # Convert to integers for GCD calculation (scaled up to preserve precision)
        scale_factor = 1000
        scaled_harmonics = (harmonics * scale_factor).astype(int)
        
        # Find GCD of all harmonics
        if len(scaled_harmonics) == 0:
            return 0.0
        
        result = scaled_harmonics[0]
        for harmonic in scaled_harmonics[1:]:
            result = np.gcd(result, harmonic)
        
        # Convert back to frequency
        fundamental = result / scale_factor
        
        # Validate that this is reasonable (within expected range)
        if fundamental < 1e6 or fundamental > 100e6:  # 1 MHz to 100 MHz range
            return 0.0
        
        return fundamental
    
    def optimize_constellation_harmony(
        self, 
        observations: List[MultiApertureObservation]
    ) -> Dict[str, float]:
        """Optimize the constellation for maximum harmonic consonance."""
        
        optimization_scores = {}
        
        # Calculate consonance matrix between all device pairs
        n_devices = len(observations)
        consonance_matrix = np.zeros((n_devices, n_devices))
        
        for i, obs1 in enumerate(observations):
            for j, obs2 in enumerate(observations):
                if i != j:
                    # Calculate average consonance between device observations
                    consonance_score = self._calculate_device_pair_consonance(obs1, obs2)
                    consonance_matrix[i, j] = consonance_score
        
        # Overall constellation harmony score
        overall_consonance = np.mean(consonance_matrix[consonance_matrix > 0])
        optimization_scores['overall_consonance'] = overall_consonance
        
        # Identify optimal frequency assignments for maximum harmony
        optimal_assignments = self._find_optimal_frequency_assignments(observations)
        optimization_scores['optimal_assignments'] = optimal_assignments
        
        # Calculate interference pattern diversity
        total_patterns = sum(len(obs.local_interference_patterns) for obs in observations)
        unique_intervals = set()
        for obs in observations:
            for pattern in obs.local_interference_patterns:
                unique_intervals.add(pattern.musical_interval)
        
        optimization_scores['pattern_diversity'] = len(unique_intervals)
        optimization_scores['total_patterns'] = total_patterns
        
        return optimization_scores
    
    def _calculate_device_pair_consonance(
        self, 
        obs1: MultiApertureObservation, 
        obs2: MultiApertureObservation
    ) -> float:
        """Calculate consonance score between two device observations."""
        
        total_consonance = 0.0
        pair_count = 0
        
        # Compare all frequency pairs between devices
        for freq1 in obs1.observed_frequencies:
            for freq2 in obs2.observed_frequencies:
                if freq1 > 0 and freq2 > 0:
                    ratio = max(freq1, freq2) / min(freq1, freq2)
                    interval = self._identify_musical_interval(ratio)
                    consonance = CONSONANCE_SCORES.get(interval, 0.0)
                    total_consonance += consonance
                    pair_count += 1
        
        return total_consonance / pair_count if pair_count > 0 else 0.0
    
    def _find_optimal_frequency_assignments(
        self, 
        observations: List[MultiApertureObservation]
    ) -> Dict[str, List[float]]:
        """Find optimal frequency assignments for each device to maximize harmony."""
        
        optimal_assignments = {}
        
        for obs in observations:
            device_id = obs.device_id
            current_freqs = obs.observed_frequencies
            
            # For each current frequency, find the most consonant alternatives
            optimized_freqs = []
            for freq in current_freqs:
                # Generate consonant alternatives
                best_freq = freq
                best_consonance = 0.0
                
                for interval, ratio in MUSICAL_INTERVALS.items():
                    consonance_score = CONSONANCE_SCORES[interval]
                    if consonance_score > best_consonance:
                        candidate_freq = freq * ratio
                        # Check if this frequency creates good harmony with other devices
                        if self._validate_frequency_choice(candidate_freq, observations, device_id):
                            best_freq = candidate_freq
                            best_consonance = consonance_score
                
                optimized_freqs.append(best_freq)
            
            optimal_assignments[device_id] = optimized_freqs
        
        return optimal_assignments
    
    def _validate_frequency_choice(
        self, 
        candidate_freq: float, 
        observations: List[MultiApertureObservation],
        exclude_device: str
    ) -> bool:
        """Validate if a frequency choice creates good harmony with the constellation."""
        
        # Simple validation - check if frequency creates consonant ratios with other devices
        consonant_count = 0
        total_comparisons = 0
        
        for obs in observations:
            if obs.device_id != exclude_device:
                for other_freq in obs.observed_frequencies:
                    if other_freq > 0:
                        ratio = max(candidate_freq, other_freq) / min(candidate_freq, other_freq)
                        interval = self._identify_musical_interval(ratio)
                        consonance = CONSONANCE_SCORES.get(interval, 0.0)
                        
                        if consonance > 0.5:  # Threshold for "consonant"
                            consonant_count += 1
                        total_comparisons += 1
        
        consonance_ratio = consonant_count / total_comparisons if total_comparisons > 0 else 0.0
        return consonance_ratio > 0.3  # At least 30% consonant relationships


def create_musical_interference_demo() -> Dict:
    """Demonstrate the Musical-RF interference-as-data concept."""
    
    print("MUSICAL-RF INTERFERENCE-AS-DATA DEMONSTRATION")
    print("=" * 60)
    
    engine = InterferenceIntelligenceEngine()
    
    # Create sample multi-aperture observations
    observations = [
        MultiApertureObservation(
            device_id="device_1",
            position=(0.0, 0.0, 0.0),
            observed_frequencies=[65.9e6, 98.1e6, 219.7e6],  # A vowel formants
            signal_strengths=[1.0, 0.8, 0.6],
            harmonic_components={
                65.9e6: [65.9e6, 131.8e6, 197.7e6],
                98.1e6: [98.1e6, 196.2e6, 294.3e6],
            },
            timestamp=0.0,
            local_interference_patterns=[]
        ),
        MultiApertureObservation(
            device_id="device_2", 
            position=(100.0, 0.0, 0.0),
            observed_frequencies=[36.0e6, 153.0e6, 234.0e6],  # E vowel formants
            signal_strengths=[1.0, 0.9, 0.7],
            harmonic_components={
                36.0e6: [36.0e6, 72.0e6, 108.0e6],
                153.0e6: [153.0e6, 306.0e6],
            },
            timestamp=0.0,
            local_interference_patterns=[]
        ),
        MultiApertureObservation(
            device_id="device_3",
            position=(50.0, 86.6, 0.0),  # Triangular arrangement
            observed_frequencies=[31.5e6, 54.0e6, 243.0e6],  # U vowel formants  
            signal_strengths=[1.0, 0.8, 0.5],
            harmonic_components={
                31.5e6: [31.5e6, 63.0e6, 94.5e6, 126.0e6],
                54.0e6: [54.0e6, 108.0e6, 162.0e6],
            },
            timestamp=0.0,
            local_interference_patterns=[]
        )
    ]
    
    # Demonstrate consonant interference generation
    print("1. CONSONANT INTERFERENCE GENERATION")
    print("-" * 40)
    
    base_freqs = [65.9e6, 98.1e6]  # A vowel formants
    consonant_freqs = engine.generate_consonant_interference(base_freqs, "perfect_fifth")
    
    print(f"Base frequencies: {[f/1e6 for f in base_freqs]} MHz")
    print(f"Perfect fifth harmonics: {[f/1e6 for f in consonant_freqs]} MHz")
    print(f"Frequency ratios: {[cf/bf for bf, cf in zip(base_freqs, consonant_freqs)]}")
    
    # Demonstrate missing fundamental reconstruction
    print("\\n2. MISSING FUNDAMENTAL RECONSTRUCTION")
    print("-" * 40)
    
    reconstructed = engine.reconstruct_missing_fundamental(observations)
    print(f"Reconstructed fundamentals: {len(reconstructed)} found")
    for key, freq in reconstructed.items():
        print(f"  {key}: {freq/1e6:.2f} MHz")
    
    # Demonstrate constellation harmony optimization
    print("\\n3. CONSTELLATION HARMONY OPTIMIZATION")
    print("-" * 40)
    
    harmony_scores = engine.optimize_constellation_harmony(observations)
    print(f"Overall constellation consonance: {harmony_scores['overall_consonance']:.3f}")
    print(f"Interference pattern diversity: {harmony_scores['pattern_diversity']} unique intervals")
    print(f"Total interference patterns: {harmony_scores['total_patterns']}")
    
    # Show optimal frequency assignments
    print("\\nOptimal frequency assignments:")
    for device_id, freqs in harmony_scores['optimal_assignments'].items():
        print(f"  {device_id}: {[f/1e6 for f in freqs[:3]]} MHz")
    
    return {
        'engine': engine,
        'observations': observations,
        'consonant_demo': consonant_freqs,
        'reconstructed_fundamentals': reconstructed,
        'harmony_optimization': harmony_scores
    }


if __name__ == "__main__":
    demo_results = create_musical_interference_demo()
    
    print("\\n🎼 INTERFERENCE-AS-DATA ARCHITECTURE SUCCESSFULLY IMPLEMENTED!")
    print("🎵 Musical-RF: Transforming Interference into Intelligence!")