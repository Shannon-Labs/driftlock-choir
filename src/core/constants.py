"""
Physical constants and unit conversion factors for Driftlock Choir.
"""

from dataclasses import dataclass

# Physical constants
C = 299792458.0  # Speed of light in m/s
K_B = 1.380649e-23  # Boltzmann constant in J/K


@dataclass(frozen=True)
class PhysicalConstants:
    """Immutable physical constants for unit conversions."""
    
    # Time conversions
    PS_PER_SEC: float = 1e12
    SEC_PER_PS: float = 1e-12
    NS_PER_SEC: float = 1e9
    SEC_PER_NS: float = 1e-9
    
    # Frequency conversions
    HZ_PER_MHZ: float = 1e6
    MHZ_PER_HZ: float = 1e-6
    HZ_PER_GHZ: float = 1e9
    GHZ_PER_HZ: float = 1e-9
    
    # Relative frequency units
    PPM_PER_UNIT: float = 1e6
    PPB_PER_UNIT: float = 1e9
    
    # Distance conversions
    M_PER_KM: float = 1000.0
    KM_PER_M: float = 1e-3
    
    # Propagation constants
    SEC_PER_M: float = 1.0 / C  # seconds per meter
    PS_PER_M: float = 1e12 / C  # picoseconds per meter
    
    # Thermal noise
    THERMAL_NOISE_DBW_PER_HZ: float = -174.0  # dBW/Hz at room temperature
    THERMAL_NOISE_DBM_PER_HZ: float = -144.0  # dBm/Hz at room temperature
    
    @classmethod
    def seconds_to_ps(cls, seconds: float) -> float:
        """Convert seconds to picoseconds."""
        return seconds * cls.PS_PER_SEC
    
    @classmethod
    def ps_to_seconds(cls, picoseconds: float) -> float:
        """Convert picoseconds to seconds."""
        return picoseconds * cls.SEC_PER_PS
    
    @classmethod
    def hz_to_mhz(cls, hz: float) -> float:
        """Convert Hz to MHz."""
        return hz * cls.MHZ_PER_HZ
    
    @classmethod
    def mhz_to_hz(cls, mhz: float) -> float:
        """Convert MHz to Hz."""
        return mhz * cls.HZ_PER_MHZ
    
    @classmethod
    def ppm_to_unit(cls, ppm: float) -> float:
        """Convert PPM to unitless ratio."""
        return ppm / cls.PPM_PER_UNIT
    
    @classmethod
    def unit_to_ppm(cls, ratio: float) -> float:
        """Convert unitless ratio to PPM."""
        return ratio * cls.PPM_PER_UNIT
    
    @classmethod
    def meters_to_ps(cls, meters: float) -> float:
        """Convert distance in meters to propagation delay in picoseconds."""
        return meters * cls.PS_PER_M
    
    @classmethod
    def ps_to_meters(cls, picoseconds: float) -> float:
        """Convert propagation delay in picoseconds to distance in meters."""
        return picoseconds * cls.SEC_PER_PS * C