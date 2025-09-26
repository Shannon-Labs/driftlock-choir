# Critical Frequency Scaling Issue - Requires Immediate Attention

## Problem Identified

The current formant-based beacon system uses **inappropriate frequency scaling** for practical RF applications:

**Current (Problematic) Configuration:**
- F₀ (carrier): 25 kHz
- Formant scale: 1000×  
- Actual RF range: 350 kHz - 3.4 MHz (HF band)

**Issues:**
1. **25 kHz is too low** for most RF coordination applications
2. **HF band (3-30 MHz)** has poor/unreliable propagation for local coordination
3. **Bandwidth inefficient** - formants spread over huge frequency range
4. **Not aligned with common RF bands** (VHF/UHF/ISM)

## Recommended Solutions

### Option 1: VHF Coordination Band (50 MHz)
```python
DEFAULT_FUNDAMENTAL_HZ = 50_000_000.0  # 50 MHz
DEFAULT_FORMANT_SCALE = 2_000.0        # Maintains formant ratios
```

**Results:**
- F₀: 50 MHz (excellent VHF propagation)  
- Formant range: 35-340 MHz (all VHF, good propagation)
- I vowel F2: 270 MHz (easily achievable with modern RF)

### Option 2: UHF ISM Band (2.4 GHz) 
```python
DEFAULT_FUNDAMENTAL_HZ = 2_400_000_000.0  # 2.4 GHz
DEFAULT_FORMANT_SCALE = 96_000.0          # Scales appropriately
```

**Results:**
- F₀: 2.4 GHz (ISM band, no licensing required)
- Formant range: 2.43-2.73 GHz (within ISM allocation)
- Higher bandwidth available for multiple beacons

### Option 3: Conservative VHF (100 MHz)
```python  
DEFAULT_FUNDAMENTAL_HZ = 100_000_000.0  # 100 MHz
DEFAULT_FORMANT_SCALE = 4_000.0         # 4000× scaling
```

**Results:**
- F₀: 100 MHz (good VHF propagation)
- Formant range: 70-680 MHz (VHF/low UHF)
- Practical for coordination applications

## Recommendation: Option 1 (50 MHz VHF)

**Rationale:**
1. **Excellent propagation** for coordination (line-of-sight + some NLOS)
2. **Reasonable bandwidth requirements** (35-340 MHz span)
3. **Achievable with standard RF hardware** (SDRs, etc.)
4. **Good compromise** between propagation and bandwidth efficiency

## Implementation Changes Required

### Core Frequency Parameters
```python
# src/phy/formants.py
DEFAULT_FUNDAMENTAL_HZ = 50_000_000.0  # 50 MHz (was 25 kHz)
DEFAULT_FORMANT_SCALE = 2_000.0        # 2000× (was 1000×)
```

### Updated Documentation
All frequency examples in documentation need updating:
- RF mapping examples
- Missing-fundamental calculations  
- Harmonic analysis descriptions
- Performance validation frequency ranges

### Hardware Implications
- **SDR Requirements**: Need VHF-capable transceivers (most modern SDRs support this)
- **Antenna Design**: VHF antennas (much more practical than HF)
- **Propagation Models**: Update multipath simulations for VHF characteristics

## Impact on Current Results

**Good News**: The **formant optimization work remains valid**! 
- Acoustic separation principles still apply
- Missing-fundamental detection algorithm unchanged
- Italian vowel formant ratios preserved
- Only the RF carrier frequency changes

**Validation Required**: 
- Re-run performance tests with corrected frequencies
- Validate multipath models for VHF propagation  
- Confirm bandwidth requirements within practical limits

## Action Items for Next Agent

1. **Update frequency parameters** in `src/phy/formants.py`
2. **Revalidate beacon performance** with realistic RF frequencies
3. **Update all documentation** to reflect VHF operation
4. **Consider antenna and propagation implications** for hardware validation

This is a **critical fix** that transforms the system from an academic exercise to a **practical RF coordination solution**.

## Frequency Band Comparison

| Band | Frequency | Propagation | Licensing | Hardware Cost | Recommended |
|------|-----------|-------------|-----------|---------------|-------------|
| HF (current) | 3-30 MHz | Poor/unreliable | Required | Low | ❌ No |
| **VHF (proposed)** | **30-300 MHz** | **Excellent** | **Amateur/ISM** | **Moderate** | **✅ Yes** |
| UHF | 300 MHz-3 GHz | Good | Mixed | Moderate | ✅ Alternative |
| SHF | 3-30 GHz | Line-of-sight | ISM available | Higher | 🤔 Future |

The **VHF solution (50 MHz carrier)** provides the optimal balance of propagation characteristics, hardware practicality, and bandwidth efficiency for formant-based RF coordination.