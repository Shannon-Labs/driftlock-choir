# a16z Speedrun Application - Driftlock (Final)

## One-Liner
**"Software-only time telemetry overlay for existing radios. We exploit tiny, intentional Δf to synchronize distributed devices."**

---

## Company Details
- **Company**: Shannon Labs, Inc.
- **Founder**: Hunter Bown (Solo Technical Founder)
- **Stage**: Pre-seed, Patent Pending
- **Contact**: hunter@shannonlabs.dev
- **GitHub**: https://github.com/shannon-labs/driftlock choir
- **Demo**: Run `python experiment/run_demo.py` for instant results

---

## The Problem
Private 5G, robotics, and edge AI need microsecond-to-nanosecond synchronization. GPS doesn't work indoors, gets jammed, and requires clear sky view. Existing alternatives need new hardware, fiber, or atomic clocks.

## The Solution
We turn intentional frequency offsets (Δf) into timing information. Beat patterns encode phase that reveals propagation delay with picosecond precision. Software-only, works on existing radios, no new spectrum.

## Technical Results (mc_runs extended_006)

### Alias Calibration
- **Off**: ~-12 ns bias
- **Loopback**: ~+2.65 ps bias
- **Improvement**: 4,500×

### Consensus Performance
- **64-node** (metropolis): Converges in 1 iteration to ~22 ps RMSE
- **25-node** (inverse-variance): Converges in 1 iteration to ~24 ps RMSE

*Simulation results demonstrate theoretical capability; hardware pilots in progress achieving 10ns.*

---

## IP Moat
- **Provisional Patent**: 25-claim set filed Sept 2025
- **Teaching-Away**: 100+ years of interference elimination
- **PCT/US Drafts**: Ready for filing
- **Paradigm Shift**: Beat patterns as information carriers

---

## Solo Founder Advantage

### Hunter Bown
- **Background**: High school band director → Wireless engineer
- **Unique Insight**: Recognized beats as timing information
- **Domain Expertise**: Music + DSP + wireless systems
- **Execution Speed**: No coordination overhead, ship daily

### Why Solo Works Here
- Core algorithm complete and validated
- Infrastructure play = technical depth over breadth
- Hiring plan clear: DSP engineer first, then platform team
- Advisory board covers gaps (timing systems, GTM, patents)

---

## Business Model

### Device-Tiered SaaS
```
SDK Access:    $1K/mo     (10 devices)
Pilot Tier:    $5K/mo     (100 devices)
Production:    $25K/mo    (1,000 devices)
Enterprise:    Custom      (unlimited + SLA)
```

### 90-Day Targets
- 3 pilot LOIs signed
- 50+ SDK downloads
- 5 telemetry dashboards deployed
- First $15K MRR

---

## Market & Wedge

### Wedge: Private 5G Networks ($8B)
- Need local synchronization for TDD
- Can't rely on GPS indoors
- Willing to pay for software overlay

### Expand: Robotics & Industrial IoT
- Swarm coordination
- Factory automation
- Warehouse AGVs

### Ultimate: GPS Alternative
- Every edge device
- Resilient timing infrastructure
- $72B TAM

---

## 90-Day Speedrun Plan

### Weeks 0-3: Technical Validation
- [ ] Hardware demo to 5ns
- [ ] Loopback calibration in production
- [ ] SDK v1.0 release

### Weeks 4-8: Customer Validation
- [ ] Private 5G pilot (Nokia/Ericsson)
- [ ] Robotics demo (3+ unit swarm)
- [ ] Defense program introduction

### Weeks 9-12: Scale Preparation
- [ ] First DSP engineer hired
- [ ] 3 LOIs signed
- [ ] Series A deck ready

---

## The Ask

### Investment
- $500K for 10% (SAFE) + $500K follow-on = **$1M total**
- Plus $5M+ cloud/AI credits

### Specific Intros Needed
1. **One private 5G partner** (Nokia/Ericsson timing team)
2. **One robotics/defense lab** (swarm coordination use case)
3. **Radio/standards GTM mentor** (3GPP/IEEE experience)

### Use of Funds
- 40% Hardware validation & lab equipment
- 30% Patent portfolio (PCT + US filing)
- 20% First technical hire
- 10% Operating expenses

---

## Why Now

1. **Edge AI explosion** needs microsecond coordination
2. **Private 5G spending** accelerating ($8B market)
3. **GPS vulnerabilities** exposed (Ukraine, Red Sea)
4. **Modern radios** have CFO/phase estimators we reuse
5. **Solo founder speed** - ship faster than committees

---

## Demo Bundle Ready

### 60-Second Video
- Shows bench setup with 2 Feathers TX, RTL-SDR
- Visualizes 1 kHz beat pattern
- Moves node 30cm → Δτ ≈ 1000 ps updates
- SDK prints bias + quality
- Grafana panel shows green

### GitHub Repository
```bash
git clone https://github.com/shannon-labs/driftlock choir
cd driftlock choir/experiment
python run_demo.py  # Single command
# Outputs: timing.npy + beat_pattern.png
```

### Live Demo
Available for in-person demonstration at a16z offices

---

## Why This Wins

1. **Technical moat**: 4,500× improvement with calibration
2. **Market timing**: Private 5G needs this NOW
3. **Solo advantage**: Ship 10× faster
4. **Clear wedge**: Start with private 5G, expand from there
5. **Infrastructure play**: Platform with network effects

---

## Contact

Hunter Bown
- Email: hunter@shannonlabs.dev
- Phone: [Available on request]
- GitHub: https://github.com/shannon-labs/driftlock choir
- Location: [Willing to relocate to SF]

**Ready to turn beat patterns into the timing layer for everything.**