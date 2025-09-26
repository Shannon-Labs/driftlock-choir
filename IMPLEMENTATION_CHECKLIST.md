# Frequency Scaling Implementation Checklist

## Quick Reference: Exact Changes Needed

### Step 1: Update Core Parameters (CRITICAL)

**File:** `src/phy/formants.py`  
**Lines:** ~23-25

**CHANGE FROM:**
```python
DEFAULT_FUNDAMENTAL_HZ = 25_000.0
DEFAULT_FORMANT_SCALE = 1_000.0
```

**CHANGE TO:**
```python
DEFAULT_FUNDAMENTAL_HZ = 50_000_000.0  # 50 MHz carrier for VHF operation  
DEFAULT_FORMANT_SCALE = 2_000.0        # 2000× scaling for practical RF frequencies
```

### Step 2: Verify Frequency Mapping

**Expected Results After Change:**
```
Carrier: 50 MHz (was 25 kHz)
Formants: 0.6 - 6.8 MHz range (was 0.3 - 3.4 MHz)
Band: VHF (was HF)
Propagation: Excellent (was poor)
```

### Step 3: Test & Validate

**Run These Commands:**
```bash
# 1. Core functionality test
python -m pytest tests/test_formants.py

# 2. Frequency validation  
PYTHONPATH=src python -c "
from phy.formants import DEFAULT_FUNDAMENTAL_HZ, DEFAULT_FORMANT_SCALE
print(f'Carrier: {DEFAULT_FUNDAMENTAL_HZ/1e6:.0f} MHz')
print(f'Scale: {DEFAULT_FORMANT_SCALE:.0f}×')
print('✅ Should show 50 MHz carrier, 2000× scale')
"

# 3. Beacon performance test
python scripts/run_spectrum_beacon_sim.py --profiles A I --num-trials 20 --output test.json

# 4. Check results
python scripts/analyze_beacon_performance.py --beacon-summary test.json
```

### Step 4: Documentation Updates (Secondary)

**Files to Update:**
- `README.md` - Technical implementation section
- `docs/formant_beacon_architecture.md` - RF frequency examples
- `RH_AGENT_BRIEFING.md` - Frequency specifications

**Search/Replace Patterns:**
- "25 kHz" → "50 MHz"
- "25_000" → "50_000_000"
- "1000×" → "2000×"

## Troubleshooting

### If Tests Fail After Change:
1. **Check syntax**: Ensure no typos in the large numbers
2. **Verify imports**: Make sure PYTHONPATH includes src/
3. **Run individual tests**: Test formant synthesis first, then detection

### If Performance Degrades:
- **Expected**: Some accuracy changes due to different harmonic structure
- **Acceptable**: ±5% accuracy variation  
- **Concerning**: >10% accuracy loss or detection failures
- **Solution**: The formant ratios are preserved, so performance should be similar

### If Frequency Validation Fails:
```python
# Debug frequency calculation
from phy.formants import VOWEL_FORMANT_TABLE, DEFAULT_FORMANT_SCALE

for vowel, (f1, f2, f3) in VOWEL_FORMANT_TABLE.items():
    print(f"{vowel}: {f1*DEFAULT_FORMANT_SCALE/1e6:.1f} MHz")
    
# All values should be in 0.6 - 6.8 MHz range
```

## Success Indicators

✅ **Frequencies in VHF range** (30-300 MHz)  
✅ **Tests pass** (functionality preserved)  
✅ **Performance maintained** (±5% accuracy acceptable)  
✅ **No errors** in beacon simulation  

This simple change enables practical RF deployment of the formant-based coordination system.

---

**Time Estimate:** 15-30 minutes for experienced RF engineer  
**Difficulty:** Low (just parameter changes)  
**Impact:** Transforms system from academic to practical RF application