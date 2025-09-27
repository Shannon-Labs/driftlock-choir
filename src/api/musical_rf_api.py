#!/usr/bin/env python3
"""
Musical-RF Development API
Comprehensive API Framework for Musical-RF Integration
Part of Musical-RF Architecture: First Systematic Application of Acoustic Science to RF Engineering

This module provides the complete API framework for developers to integrate
Musical-RF technology into their applications. Enables easy access to the
revolutionary spectrum intelligence capabilities through simple, intuitive APIs.

Key Features:
- Signal Generation API: Create Musical-RF signals from acoustic parameters
- Detection & Analysis API: Analyze RF signals for Musical-RF characteristics
- Cultural Extension API: Add new languages and musical traditions
- Orchestral Coordination API: Manage multiple simultaneous RF signals
- Multi-Aperture Intelligence API: Distributed missing-fundamental reconstruction

API Design Philosophy:
- Simple: Easy-to-use interfaces for common operations
- Powerful: Full access to advanced Musical-RF capabilities
- Extensible: Plugin architecture for cultural and musical extensions
- Efficient: Optimized for real-time RF signal processing
- Cross-Platform: Works across different RF hardware platforms

The Musical-RF API democratizes access to spectrum intelligence technology,
enabling developers worldwide to build applications leveraging centuries
of acoustic science applied to RF engineering.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from enum import Enum
import numpy as np
from numpy.typing import NDArray
import asyncio
import threading
import json
import time
from pathlib import Path

# Import core Musical-RF components
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from phy.formants import (
    FormantDescriptor, FormantSynthesisConfig, FormantAnalysisResult,
    synthesize_formant_preamble, analyze_missing_fundamental, build_formant_library
)
from phy.orchestral_rf import OrchestralRFSynthesizer, OrchestralSection, InstrumentFamily
from phy.formants_extended import CHINESE_VOWEL_FORMANTS, MANDARIN_TONE_CONTOURS
from phy.interference_intelligence import InterferenceIntelligenceEngine


class MusicalRFAPIVersion(Enum):
    """API version management"""
    V1_0 = "1.0"
    V1_1 = "1.1"
    V2_0 = "2.0"  # Future version with advanced features


class SignalType(Enum):
    """Types of Musical-RF signals"""
    VOCAL = "vocal"              # Vowel-based signals (original)
    CULTURAL = "cultural"        # Cultural language extensions
    ORCHESTRAL = "orchestral"    # Orchestral instrument signals
    INTERFERENCE = "interference" # Interference-as-data signals
    CUSTOM = "custom"            # User-defined acoustic signatures


class APIResponse(Enum):
    """API response status codes"""
    SUCCESS = "success"
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


@dataclass
class MusicalRFConfig:
    """Configuration for Musical-RF API operations"""
    
    # Core RF parameters
    fundamental_hz: float = 50_000_000.0    # 50 MHz VHF carrier
    formant_scale: float = 90_000.0         # VHF scaling factor
    sample_rate: float = 1_000_000.0        # 1 MHz sampling
    
    # Signal generation parameters
    signal_length: int = 1024               # Default signal length
    harmonic_count: int = 12                # Number of harmonics
    include_fundamental: bool = False       # Include carrier fundamental
    
    # Detection parameters
    detection_threshold: float = 0.7        # Minimum confidence for detection
    false_positive_threshold: float = 0.01  # Maximum false positive rate
    
    # Cultural extensions
    enabled_cultures: List[str] = field(default_factory=lambda: ["italian", "chinese"])
    cultural_weight: float = 1.0            # Cultural signal weighting
    
    # Performance parameters
    max_concurrent_signals: int = 128       # Maximum simultaneous signals
    processing_threads: int = 4             # Parallel processing threads
    cache_size: int = 1000                  # Signal cache size


@dataclass
class SignalRequest:
    """Request for Musical-RF signal generation"""
    
    signal_type: SignalType
    parameters: Dict[str, Any]
    config: Optional[MusicalRFConfig] = None
    metadata: Dict[str, str] = field(default_factory=dict)


@dataclass
class SignalResponse:
    """Response from Musical-RF signal generation"""
    
    status: APIResponse
    signal: Optional[NDArray[np.complex128]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None
    processing_time: float = 0.0


@dataclass
class AnalysisRequest:
    """Request for Musical-RF signal analysis"""
    
    signal: NDArray[np.complex128]
    analysis_type: str = "full"  # "full", "detection", "classification"
    config: Optional[MusicalRFConfig] = None
    cultural_context: Optional[str] = None


@dataclass
class AnalysisResponse:
    """Response from Musical-RF signal analysis"""
    
    status: APIResponse
    detected_signals: List[Dict[str, Any]] = field(default_factory=list)
    confidence_scores: List[float] = field(default_factory=list)
    cultural_classification: Optional[str] = None
    interference_patterns: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None
    processing_time: float = 0.0


class CulturalExtension(ABC):
    """Abstract base class for cultural extensions"""
    
    @abstractmethod
    def get_name(self) -> str:
        """Get the name of this cultural extension"""
        pass
        
    @abstractmethod
    def get_vowel_formants(self) -> Dict[str, Tuple[float, float, float]]:
        """Get vowel formant definitions for this culture"""
        pass
        
    @abstractmethod
    def get_tone_contours(self) -> Dict[str, List[float]]:
        """Get tone contour definitions (if applicable)"""
        pass
        
    @abstractmethod
    def validate_acoustic_signature(self, formants: Tuple[float, ...]) -> bool:
        """Validate if acoustic signature belongs to this culture"""
        pass


class MusicalRFAPI:
    """
    Main Musical-RF API Class
    
    Provides comprehensive interface for all Musical-RF capabilities including
    signal generation, analysis, cultural extensions, and orchestral coordination.
    """
    
    def __init__(self, config: Optional[MusicalRFConfig] = None):
        self.config = config or MusicalRFConfig()
        
        # Initialize core components
        self.orchestral_synthesizer = OrchestralRFSynthesizer(self.config.sample_rate)
        self.interference_engine = InterferenceIntelligenceEngine()
        
        # Cultural extensions registry
        self.cultural_extensions: Dict[str, CulturalExtension] = {}
        self._register_default_cultures()
        
        # Signal cache for performance
        self.signal_cache: Dict[str, NDArray[np.complex128]] = {}
        self.cache_timestamps: Dict[str, float] = {}
        
        # Performance metrics
        self.generation_count = 0
        self.analysis_count = 0
        self.start_time = time.time()
        
        # Threading for concurrent operations
        self.thread_pool = None
        self._initialize_threading()
    
    def _register_default_cultures(self):
        """Register default cultural extensions"""
        # Register Italian (original) culture
        self.cultural_extensions["italian"] = ItalianCulturalExtension()
        
        # Register Chinese culture  
        self.cultural_extensions["chinese"] = ChineseCulturalExtension()
    
    def _initialize_threading(self):
        """Initialize thread pool for concurrent operations"""
        if self.config.processing_threads > 1:
            # Thread pool would be initialized here for production use
            pass
    
    # ===========================================
    # CORE SIGNAL GENERATION API
    # ===========================================
    
    def generate_signal(self, request: SignalRequest) -> SignalResponse:
        """
        Generate Musical-RF signal from acoustic parameters
        
        Args:
            request: Signal generation request with parameters
            
        Returns:
            SignalResponse with generated RF signal and metadata
        """
        start_time = time.time()
        
        try:
            # Use request config or default
            config = request.config or self.config
            
            # Check cache first
            cache_key = self._generate_cache_key(request)
            if cache_key in self.signal_cache:
                cached_signal = self.signal_cache[cache_key]
                return SignalResponse(
                    status=APIResponse.SUCCESS,
                    signal=cached_signal,
                    metadata={"cached": True, "cache_key": cache_key},
                    processing_time=time.time() - start_time
                )
            
            # Generate signal based on type
            if request.signal_type == SignalType.VOCAL:
                signal = self._generate_vocal_signal(request.parameters, config)
            elif request.signal_type == SignalType.CULTURAL:
                signal = self._generate_cultural_signal(request.parameters, config)
            elif request.signal_type == SignalType.ORCHESTRAL:
                signal = self._generate_orchestral_signal(request.parameters, config)
            elif request.signal_type == SignalType.INTERFERENCE:
                signal = self._generate_interference_signal(request.parameters, config)
            elif request.signal_type == SignalType.CUSTOM:
                signal = self._generate_custom_signal(request.parameters, config)
            else:
                raise ValueError(f"Unsupported signal type: {request.signal_type}")
            
            # Cache the result
            self._cache_signal(cache_key, signal)
            
            # Update metrics
            self.generation_count += 1
            
            return SignalResponse(
                status=APIResponse.SUCCESS,
                signal=signal,
                metadata={
                    "signal_type": request.signal_type.value,
                    "signal_length": len(signal),
                    "generation_id": self.generation_count
                },
                processing_time=time.time() - start_time
            )
            
        except Exception as e:
            return SignalResponse(
                status=APIResponse.ERROR,
                error_message=str(e),
                processing_time=time.time() - start_time
            )
    
    def _generate_vocal_signal(self, params: Dict[str, Any], config: MusicalRFConfig) -> NDArray[np.complex128]:
        """Generate vocal-based Musical-RF signal"""
        vowel = params.get("vowel", "A").upper()
        dynamics = params.get("dynamics", "mf")
        articulation = params.get("articulation", "legato")
        
        # Use core formant synthesis
        synthesis_config = FormantSynthesisConfig(
            profile=vowel,
            fundamental_hz=config.fundamental_hz,
            formant_scale=config.formant_scale,
            harmonic_count=config.harmonic_count,
            include_fundamental=config.include_fundamental
        )
        
        signal, _ = synthesize_formant_preamble(
            config.signal_length, config.sample_rate, synthesis_config
        )
        
        # Apply dynamics scaling
        dynamic_scales = {"pp": 0.2, "p": 0.4, "mp": 0.6, "mf": 0.8, "f": 1.0, "ff": 1.3}
        scale = dynamic_scales.get(dynamics, 0.8)
        signal = signal * scale
        
        return signal
    
    def _generate_cultural_signal(self, params: Dict[str, Any], config: MusicalRFConfig) -> NDArray[np.complex128]:
        """Generate cultural Musical-RF signal"""
        culture = params.get("culture", "chinese")
        vowel = params.get("vowel", "a_zh")
        tone = params.get("tone", None)
        
        if culture not in self.cultural_extensions:
            raise ValueError(f"Unsupported culture: {culture}")
        
        extension = self.cultural_extensions[culture]
        vowel_formants = extension.get_vowel_formants()
        
        if vowel not in vowel_formants:
            raise ValueError(f"Unsupported vowel '{vowel}' for culture '{culture}'")
        
        # Generate base signal from cultural formants
        formants = vowel_formants[vowel]
        
        # Create custom formant synthesis
        time_axis = np.arange(config.signal_length, dtype=float) / config.sample_rate
        signal = np.zeros(config.signal_length, dtype=np.complex128)
        
        # Add formant frequencies as harmonics
        for i, formant_freq in enumerate(formants):
            rf_freq = formant_freq * config.formant_scale
            amplitude = 1.0 / (i + 1)  # Decreasing amplitude
            signal += amplitude * np.exp(1j * 2 * np.pi * rf_freq * time_axis)
        
        # Apply tone contour if specified
        if tone and culture in ["chinese"]:
            tone_contours = extension.get_tone_contours()
            if tone in tone_contours:
                contour = tone_contours[tone]
                # Apply frequency modulation based on tone contour
                # This is a simplified implementation
                tone_mod = np.interp(np.linspace(0, len(contour)-1, len(signal)), 
                                   range(len(contour)), contour)
                phase_mod = np.cumsum(tone_mod) * 0.01  # Small phase modulation
                signal *= np.exp(1j * phase_mod)
        
        # Normalize
        signal = signal / np.max(np.abs(signal)) if np.max(np.abs(signal)) > 0 else signal
        
        return signal
    
    def _generate_orchestral_signal(self, params: Dict[str, Any], config: MusicalRFConfig) -> NDArray[np.complex128]:
        """Generate orchestral Musical-RF signal"""
        instrument = params.get("instrument", "violin_e")
        frequency = params.get("frequency", 659.3)
        dynamics = params.get("dynamics", "mf")
        articulation = params.get("articulation", "legato")
        
        return self.orchestral_synthesizer.synthesize_orchestral_signal(
            instrument, frequency, config.signal_length, dynamics, articulation
        )
    
    def _generate_interference_signal(self, params: Dict[str, Any], config: MusicalRFConfig) -> NDArray[np.complex128]:
        """Generate interference-as-data Musical-RF signal"""
        signal_count = params.get("signal_count", 2)
        consonance_target = params.get("consonance", 0.8)
        
        # Use interference intelligence engine
        return self.interference_engine.generate_consonant_interference(
            signal_count, consonance_target, config.signal_length
        )
    
    def _generate_custom_signal(self, params: Dict[str, Any], config: MusicalRFConfig) -> NDArray[np.complex128]:
        """Generate custom Musical-RF signal from user-defined parameters"""
        formants = params.get("formants", [650.0, 1080.0, 2650.0])  # Default to A vowel
        amplitudes = params.get("amplitudes", None)
        
        time_axis = np.arange(config.signal_length, dtype=float) / config.sample_rate
        signal = np.zeros(config.signal_length, dtype=np.complex128)
        
        for i, formant_freq in enumerate(formants):
            rf_freq = formant_freq * config.formant_scale
            amplitude = amplitudes[i] if amplitudes and i < len(amplitudes) else 1.0 / (i + 1)
            signal += amplitude * np.exp(1j * 2 * np.pi * rf_freq * time_axis)
        
        return signal
    
    # ===========================================
    # SIGNAL ANALYSIS API
    # ===========================================
    
    def analyze_signal(self, request: AnalysisRequest) -> AnalysisResponse:
        """
        Analyze RF signal for Musical-RF characteristics
        
        Args:
            request: Analysis request with signal and parameters
            
        Returns:
            AnalysisResponse with detection results and metadata
        """
        start_time = time.time()
        
        try:
            config = request.config or self.config
            
            # Build formant library for analysis
            library = build_formant_library(
                fundamental_hz=config.fundamental_hz,
                harmonic_count=config.harmonic_count,
                include_fundamental=config.include_fundamental,
                formant_scale=config.formant_scale
            )
            
            # Perform missing-fundamental analysis
            result = analyze_missing_fundamental(
                request.signal, config.sample_rate, list(library.values())
            )
            
            detected_signals = []
            confidence_scores = []
            
            if result:
                detected_signals.append({
                    "label": result.label,
                    "dominant_frequency": result.dominant_hz,
                    "missing_fundamental": result.missing_fundamental_hz,
                    "score": result.score
                })
                
                # Convert score to confidence (higher score = lower confidence in this implementation)
                confidence = max(0.0, 1.0 - result.score)
                confidence_scores.append(confidence)
            
            # Cultural classification if requested
            cultural_classification = None
            if request.cultural_context:
                cultural_classification = self._classify_cultural_signal(
                    request.signal, request.cultural_context, config
                )
            
            # Interference pattern analysis
            interference_patterns = self._analyze_interference_patterns(request.signal, config)
            
            self.analysis_count += 1
            
            return AnalysisResponse(
                status=APIResponse.SUCCESS,
                detected_signals=detected_signals,
                confidence_scores=confidence_scores,
                cultural_classification=cultural_classification,
                interference_patterns=interference_patterns,
                metadata={
                    "analysis_type": request.analysis_type,
                    "signal_length": len(request.signal),
                    "analysis_id": self.analysis_count
                },
                processing_time=time.time() - start_time
            )
            
        except Exception as e:
            return AnalysisResponse(
                status=APIResponse.ERROR,
                error_message=str(e),
                processing_time=time.time() - start_time
            )
    
    def _classify_cultural_signal(self, signal: NDArray[np.complex128], 
                                 culture: str, config: MusicalRFConfig) -> Optional[str]:
        """Classify signal according to cultural acoustic patterns"""
        if culture not in self.cultural_extensions:
            return None
        
        extension = self.cultural_extensions[culture]
        
        # Extract formant-like features from signal
        # This is a simplified implementation
        spectrum = np.fft.fft(signal)
        freqs = np.fft.fftfreq(len(signal), d=1.0/config.sample_rate)
        
        # Find dominant frequencies
        dominant_indices = np.argsort(np.abs(spectrum))[-3:]  # Top 3 frequencies
        dominant_freqs = freqs[dominant_indices]
        
        # Convert back to acoustic domain
        acoustic_formants = tuple(abs(f) / config.formant_scale for f in dominant_freqs)
        
        # Validate against cultural patterns
        if extension.validate_acoustic_signature(acoustic_formants):
            return culture
        
        return None
    
    def _analyze_interference_patterns(self, signal: NDArray[np.complex128], 
                                     config: MusicalRFConfig) -> List[Dict[str, Any]]:
        """Analyze interference patterns in signal"""
        # This would implement more sophisticated interference analysis
        # For now, return basic pattern detection
        patterns = []
        
        # Simple beat pattern detection
        envelope = np.abs(signal)
        if len(envelope) > 100:
            # Look for amplitude modulation patterns
            autocorr = np.correlate(envelope, envelope, mode='full')
            autocorr = autocorr[autocorr.size // 2:]
            
            # Find peaks that might indicate beat patterns
            peaks = []
            for i in range(1, min(100, len(autocorr)-1)):
                if autocorr[i] > autocorr[i-1] and autocorr[i] > autocorr[i+1]:
                    if autocorr[i] > 0.5 * np.max(autocorr):  # Significant peak
                        beat_frequency = config.sample_rate / i
                        peaks.append(beat_frequency)
            
            if peaks:
                patterns.append({
                    "type": "beat_pattern",
                    "frequencies": peaks[:3],  # Top 3 beat frequencies
                    "confidence": 0.7
                })
        
        return patterns
    
    # ===========================================
    # UTILITY FUNCTIONS
    # ===========================================
    
    def _generate_cache_key(self, request: SignalRequest) -> str:
        """Generate cache key for signal request"""
        key_data = {
            "type": request.signal_type.value,
            "params": request.parameters,
            "config_hash": hash(str(request.config)) if request.config else 0
        }
        return str(hash(str(key_data)))
    
    def _cache_signal(self, key: str, signal: NDArray[np.complex128]):
        """Cache generated signal"""
        if len(self.signal_cache) >= self.config.cache_size:
            # Remove oldest entry
            oldest_key = min(self.cache_timestamps.keys(), 
                           key=lambda k: self.cache_timestamps[k])
            del self.signal_cache[oldest_key]
            del self.cache_timestamps[oldest_key]
        
        self.signal_cache[key] = signal.copy()
        self.cache_timestamps[key] = time.time()
    
    def get_api_status(self) -> Dict[str, Any]:
        """Get comprehensive API status information"""
        uptime = time.time() - self.start_time
        
        return {
            "version": MusicalRFAPIVersion.V1_0.value,
            "uptime_seconds": uptime,
            "signals_generated": self.generation_count,
            "signals_analyzed": self.analysis_count,
            "cache_size": len(self.signal_cache),
            "cache_hit_ratio": 0.0,  # Would track this in production
            "supported_cultures": list(self.cultural_extensions.keys()),
            "orchestral_instruments": len(self.orchestral_synthesizer.instruments),
            "config": {
                "fundamental_hz": self.config.fundamental_hz,
                "formant_scale": self.config.formant_scale,
                "max_concurrent_signals": self.config.max_concurrent_signals
            }
        }
    
    def register_cultural_extension(self, extension: CulturalExtension):
        """Register new cultural extension"""
        name = extension.get_name()
        self.cultural_extensions[name] = extension
        
    def list_available_signals(self) -> Dict[str, List[str]]:
        """List all available signal types and their variants"""
        return {
            "vocal": ["A", "E", "I", "O", "U"],
            "cultural_chinese": list(CHINESE_VOWEL_FORMANTS.keys()),
            "orchestral": list(self.orchestral_synthesizer.instruments.keys()),
            "interference": ["consonant", "dissonant", "beat_pattern"],
            "custom": ["user_defined"]
        }


# ===========================================
# CULTURAL EXTENSION IMPLEMENTATIONS
# ===========================================

class ItalianCulturalExtension(CulturalExtension):
    """Italian cultural extension (original Musical-RF vowels)"""
    
    def get_name(self) -> str:
        return "italian"
    
    def get_vowel_formants(self) -> Dict[str, Tuple[float, float, float]]:
        return {
            "A": (650.0, 1080.0, 2650.0),
            "E": (400.0, 1700.0, 2600.0),
            "I": (340.0, 1870.0, 2800.0),
            "O": (400.0, 800