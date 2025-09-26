# URGENT: Fix Frequency Scaling from Hz to kHz for Practical RF Operation

## Critical Issue Summary

The formant-based spectrum beacon system currently operates at **impractically low frequencies** that prevent real-world RF deployment. You must fix the frequency scaling to enable practical RF coordination applications.

## Current Problem (Broken Configuration)

```python
# Current values in src/phy/formants.py
DEFAULT_FUNDAMENTAL_HZ = 25_000.0      # 25 kHz - TOO LOW for RF
DEFAULT_FORMANT_SCALE = 1_000.0        # Results in kHz formants, not MHz

# This creates formant frequencies like:
# I vowel: F1=300 kHz, F2=2.7 MHz, F3=3.4 MHz
# Operating in HF band (3-30 MHz) with poor propagation
```

**Problems:**
- 25 kHz carrier is too low for practical RF coordination
- HF band (300 kHz - 3.4 MHz) has unreliable propagation characteristics  
- Not suitable for cognitive radio, IoT, or military applications
- Standard RF hardware (SDRs) work better at higher frequencies

## Required Solution (VHF Operation)

**Update frequency parameters for VHF band operation:**

```python
# NEW values needed in src/phy/formants.py
DEFAULT_FUNDAMENTAL_HZ = 50_000_000.0  # 50 MHz carrier (VHF band)
DEFAULT_FORMANT_SCALE = 2_000.0        # Scale to put formants in VHF range

# This will create practical RF formant frequencies:
# I vowel: F1=600 kHz, F2=5.4 MHz, F3=6.8 MHz  
# A vowel: F1=1.4 MHz, F2=2.44 MHz, F3=5.2 MHz
# All in VHF band (30-300 MHz) with excellent propagation
```

## Specific Implementation Tasks

### 1. Update Core Frequency Parameters
**File: `src/phy/formants.py`**

**Find these lines (~23-25):**
```python
DEFAULT_FUNDAMENTAL_HZ = 25_000.0
DEFAULT_FORMANT_SCALE = 1_000.0
```

**Replace with:**
```python
DEFAULT_FUNDAMENTAL_HZ = 50_000_000.0  # 50 MHz carrier for VHF operation
DEFAULT_FORMANT_SCALE = 2_000.0        # 2000× scaling for practical RF frequencies
```

### 2. Validate the Frequency Mapping
**Run this check to confirm correct scaling:**

```bash
cd /path/to/driftlock-choir
PYTHONPATH=src python -c "
from phy.formants import VOWEL_FORMANT_TABLE, DEFAULT_FUNDAMENTAL_HZ, DEFAULT_FORMANT_SCALE

print('Updated RF Frequency Mapping:')
print(f'Carrier (F₀): {DEFAULT_FUNDAMENTAL_HZ/1e6:.0f} MHz')
print(f'Formant Scale: {DEFAULT_FORMANT_SCALE:.0f}×')
print()

for vowel, (f1, f2, f3) in VOWEL_FORMANT_TABLE.items():
    rf_f1 = f1 * DEFAULT_FORMANT_SCALE / 1e6  # Convert to MHz
    rf_f2 = f2 * DEFAULT_FORMANT_SCALE / 1e6
    rf_f3 = f3 * DEFAULT_FORMANT_SCALE / 1e6
    print(f'{vowel}: F1={rf_f1:.1f} MHz, F2={rf_f2:.1f} MHz, F3={rf_f3:.1f} MHz')

print()
print('✅ All formants should be in VHF range (30-300 MHz)')
print('✅ Good propagation characteristics for RF coordination')
"
```

**Expected output after fix:**
```
Updated RF Frequency Mapping:
Carrier (F₀): 50 MHz
Formant Scale: 2000×

A: F1=1.4 MHz, F2=2.4 MHz, F3=5.2 MHz
E: F1=0.9 MHz, F2=4.2 MHz, F3=5.8 MHz  
I: F1=0.6 MHz, F2=5.4 MHz, F3=6.8 MHz
O: F1=1.0 MHz, F2=1.8 MHz, F3=4.8 MHz
U: F1=0.7 MHz, F2=1.5 MHz, F3=4.4 MHz
```

### 3. Revalidate System Performance
**After making frequency changes, run full validation:**

```bash
# Test core functionality still works
python -m pytest tests/test_formants.py tests/test_chronometric_handshake.py

# Validate beacon performance with new frequencies
python scripts/run_spectrum_beacon_sim.py \
    --profiles A E I O U \
    --num-trials 100 \
    --snr-db 25 30 \
    --output vhf_validation.json

# Check results
python scripts/analyze_beacon_performance.py \
    --beacon-summary vhf_validation.json \
    --output vhf_analysis.json
```

### 4. Update Documentation
**Files needing frequency updates:**

1. **`docs/formant_beacon_architecture.md`** - Update all frequency examples
2. **`docs/italian_vowel_optimization.md`** - Update RF frequency mappings  
3. **`README.md`** - Update technical implementation section
4. **Comments in code** - Search for frequency references

**Find and replace these patterns:**
- "25 kHz" → "50 MHz"
- "25_000" → "50_000_000" 
- "1000× scaling" → "2000× scaling"
- "kHz - MHz range" → "MHz range (VHF band)"

## Why This Frequency Range Works

### VHF Band Advantages (30-300 MHz)
- **Excellent propagation**: Line-of-sight + some beyond-horizon capability
- **Standard RF hardware**: All modern SDRs support VHF operation  
- **Regulatory availability**: Amateur radio and ISM band allocations
- **Practical antennas**: VHF antennas are reasonable size for portable applications

### RF Engineering Benefits
- **Bandwidth efficiency**: Formants span 0.6-6.8 MHz (practical for coordination)
- **Multipath characteristics**: VHF multipath behavior well-understood
- **Hardware implementation**: Standard RF components available
- **Power requirements**: Reasonable transmission power for coordination ranges

## Expected Impact

### Performance Should Remain Excellent
The **formant optimization work remains completely valid**:
- ✅ Acoustic separation principles unchanged
- ✅ Missing-fundamental algorithm identical  
- ✅ Italian vowel ratios preserved
- ✅ Only carrier frequency scales up

### New Capabilities Enabled  
- **Real RF deployment**: Practical frequency range for coordination
- **Standard hardware**: Compatible with USRP, BladeRF, HackRF, etc.
- **Regulatory compliance**: Alignable with amateur/ISM allocations
- **Commercial viability**: Suitable for cognitive radio, IoT, military applications

## Validation Checklist

After implementing the frequency fix:

- [ ] ✅ Core tests pass (`pytest tests/`)
- [ ] ✅ Beacon simulation runs without errors
- [ ] ✅ All vowels maintain >50% accuracy  
- [ ] ✅ Zero false positive rate preserved
- [ ] ✅ Multi-receiver consensus >99% consistency
- [ ] ✅ Frequencies are in VHF range (30-300 MHz)
- [ ] ✅ Documentation updated with new frequency values

## Success Criteria

**Before Fix:** Academic system operating at impractical HF frequencies  
**After Fix:** Production-ready RF coordination system operating in practical VHF band

This frequency scaling correction transforms the system from a **proof-of-concept** to a **deployable RF solution** suitable for real-world coordination applications.

---

**PRIORITY**: This is the **most critical fix** needed for the system to have practical RF applications. Everything else is optimization - this is foundational.