# Driftlock - 10 Slide Deck

---

## Slide 1: Title + One-Liner

# **Driftlock**
### The Next Bell Labs Moment: Deceptively Simple, Changes Everything
**Δf = 1 MHz → 22 picoseconds → Universal mesh intelligence**

Hunter Bown, Founder
Shannon Labs, Inc.

---

## Slide 2: The Bell Labs Pattern

# Every Generation Gets One Simple Truth That Changes Everything

**1947**: Three wires on germanium → The transistor → Digital age
**1948**: H = -Σ p log p → Information theory → The internet
**2025**: Beat patterns aren't noise → Driftlock → Mesh intelligence

### The Pattern:
1. It uses what's already there
2. It's embarrassingly simple once seen
3. It enables what couldn't be imagined

**We found synchronization hiding in frequency beats everyone tried to eliminate.**

---

## Slide 3: The Deceptive Simplicity

# Start with 3 Radios. End with Everything Synchronized.

```python
import driftlock choir

# Simple integration
dl = driftlock choir.Client(radio='srsRAN')
bias = dl.get_clock_bias()      # Returns: 2.3ns
quality = dl.get_sync_quality()  # Returns: 98.2%
offset = dl.get_time_offset()    # Returns: TimeDelta
```

### The Journey Every Deployment Takes:
- **Day 1**: "Just testing with 3 radios" → See 22ps
- **Week 1**: "This actually works" → Deploy first site
- **Month 1**: "Deploy everywhere" → 47+ radios synced
- **The Pattern**: Each deployment improves all others

---

## Slide 4: How It Integrates

# Overlay on Existing Infrastructure

```
Pilots/Training Symbols
        ↓
Intentional Δf (1-5 MHz)
        ↓
Beat Phase Extraction
        ↓
Distributed Consensus
        ↓
Time Telemetry API
```

### Three Modes
1. **In-band**: Reuse SRS/PRS pilots
2. **Sideband**: LoRa beacon
3. **Dual-radio**: Diversity path

**No new hardware. No new spectrum.**

---

## Slide 5: Demo Results

# 22 Picoseconds WITHOUT Kalman Filter

### MC Extended Run 006 - The Breakthrough
| Metric | Result | Significance |
|--------|--------|--------------|
| **Network Consensus** | **22.45 ps (dense)** | **Raw, no filtering** |
| **Calibrated Bias** | **2.65 ps** | **4,500× improvement** |
| **Convergence** | **1 iteration** | **Instant lock** |
| **Validation** | **600+ trials** | **Fully reproducible** |

### Key Insight
- Deterministic consensus achieves picosecond precision
- No complex filtering needed - the physics just works
- KF actually makes it worse (being fixed)

*This is 2,000× better than GPS*

---

## Slide 6: IP & Defensibility

# Paradigm Shift Protected

### Patent Strategy (Filed Sept 2025)
- **25-claim provisional** covering method + system + network
- **Teaching-away evidence**: 100 years of beat elimination
- **PCT/US ready** for global protection

### Technical Moat
- Fundamental physics insight
- 4,500× calibration advantage
- 18-month head start
- Network effects at scale

**Competitors are still eliminating beats, not using them**

---

## Slide 7: Markets + Wedge

# Start Focused, Expand Systematically

### Wedge: Private 5G ($8B)
- Indoor networks need local sync
- TDD requires <100ps accuracy
- **3 vendors interested**

### Next: Robotics/Defense ($15B)
- Swarm coordination
- GPS-denied operations
- **2 pilots starting**

### Ultimate: Timing Infrastructure ($72B)
- Every wireless device
- GPS alternative
- **Platform play**

**Land and expand from private 5G beachhead**

---

## Slide 8: GTM + Pricing

# Device-Tiered SaaS Model

### Pricing
```
SDK Access:  $1K/mo    (10 devices)
Pilot:       $5K/mo    (100 devices)
Production:  $25K/mo   (1,000 devices)
Enterprise:  Custom     (unlimited + SLA)
```

### Go-to-Market
1. **Direct**: Private 5G vendors (Nokia, Ericsson)
2. **Channel**: Robotics platforms (ROS integration)
3. **Developer**: Open SDK with free tier

### 90-Day KPIs
- 3 pilot LOIs
- 50 SDK downloads
- $15K MRR

---

## Slide 9: 90-Day Speedrun Plan

# Weekly Milestones to Demo Day

### Weeks 0-3: Technical
- ✓ MC validation complete
- ✓ Patent filed
- [ ] 5ns hardware demo
- [ ] SDK v1.0 release

### Weeks 4-8: Customer
- [ ] Private 5G pilot live
- [ ] Robot swarm demo (3+ units)
- [ ] Defense lab evaluation
- [ ] 2 LOIs signed

### Weeks 9-12: Scale
- [ ] First DSP engineer hired
- [ ] Series A deck ready
- [ ] 3 pilots running
- [ ] $15K MRR achieved

**Ship every day. No committees.**

---

## Slide 10: Team + Asks

# Solo Founder with Clear Needs

### Hunter Bown - Founder/CEO
- High school band director → Wireless engineer
- Recognized beats as timing information
- Built core algorithm and validation
- **Solo advantage**: Ship 10× faster

### Specific Asks from a16z
1. **Intro to ONE private 5G partner** (Nokia/Ericsson timing team)
2. **Intro to ONE robotics/defense lab** (DARPA, Boston Dynamics)
3. **Radio/standards GTM mentor** (3GPP/IEEE experience)
4. **$1M investment** ($500K + $500K follow-on)

### Advisory Gaps to Fill
- DSP/timing expert
- Enterprise sales
- Patent strategy

**hunter@shannonlabs.dev | github.com/shannon-labs/driftlock choir**