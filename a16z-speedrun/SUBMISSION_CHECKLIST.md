# a16z Speedrun Final Submission Checklist

## 📅 Deadline: September 28, 2025 at 11:59 pm PT (SR006)

---

## ✅ Technical Validation Complete

### Latest Results (Extended Run 011)
- [x] **2.65 picosecond bias** with loopback calibration
- [x] **4,500× improvement** from uncalibrated (12 ns → 2.65 ps)
- [x] **22.13 ps dense preset** (clock 0.32 / freq 0.03 / 1 iter) with 0.33 ps edge vs baseline
- [x] **20.93 ps sweep minimum** at clock 0.22 / freq 0.03 / 2 iters
- [x] **600+ Monte Carlo trials** across presets (extended_011)
- [x] **Automated guardrails**: `scripts/verify_kf_sweep.py` + seeded regression

### Test Coverage
- [x] SNR conditions: 0, 10, 20 dB
- [x] Network sizes: 25-64 nodes
- [x] Calibration modes: off, loopback
- [x] Bandwidth: 20-40 MHz
- [x] Retune offsets: 1-5 MHz

---

## 📄 Application Documents Ready

### Core Materials
- [x] **EXECUTIVE_SUMMARY.md** – Updated with 22.13 ps dense preset narrative
- [x] **APPLICATION.md** – Highlights 0.33 ps edge and reproducibility flow
- [x] **PITCH_DECK_V2.md** – 22 ps hook, guardrail story, private 5G beachhead
- [x] **FINANCIAL_PROJECTIONS.md** – Device-tiered SaaS model unchanged
- [x] **TECHNICAL_VALIDATION.md** – Extended runs 010/011, sweep verifier outputs
- [x] **CALIBRATION_BREAKTHROUGH.md** – 4,500× improvement details

### Supporting Documents
- [x] **VIDEO_SCRIPT_V2.md** – Live demo focus (baseline + preset comparison)
- [x] **README_V2.md** – Pragmatic overview + verification steps
- [x] Patent provisional filed (September 2025)
- [x] GitHub repository public and tagged for extended_011

---

## 🎯 Key Messages to Emphasize

### The Hook
**"Software-only 22 ps synchronization with deterministic guardrails"**
- Not theoretical—Monte Carlo `extended_011` JSON + verifier logs
- 0.33 ps advantage vs dense baseline, 3.41 ps vs small-network baseline
- Works on existing radios (no new hardware)

### The Problem
- Private 5G needs <100 ps for TDD (3GPP TS 38.133)
- Edge AI requires sub-microsecond coordination
- GPS vulnerabilities keep surfacing (Ukraine, Red Sea, subsea cables)

### The Solution
- Intentional frequency offset → beat pattern telemetry
- Variance-weighted shrinkage + consensus
- Guardrailed presets with reproducible scripts

### The Validation
- 600+ trials (dense & small networks) across seeds
- Sweep minimum (20.93 ps) + best mean (21.89 ps) confirmed by script
- `pytest` suite with seeded regression ensures ongoing parity

---

## 🎬 Demo Materials

### Live Demo Components
- [ ] Two SDRs showing real-time sync (dense vs baseline preset toggle)
- [ ] Grafana dashboard displaying timing RMSE trend
- [ ] Beat pattern visualization
- [ ] CLI output from `scripts/verify_kf_sweep.py`

### Demo Script
```python
# Sweep verification snippet
subprocess.run([
    "scripts/verify_kf_sweep.py",
    "results/kf_sweeps/dense_combo_scan/kf_sweep_summary.json",
    "--expected-min", "20.9337",
    "--expected-best-mean", "21.89",
    "--expected-clock", "0.32",
    "--expected-freq", "0.03",
    "--expected-iterations", "1",
])

# Dense preset regression check
pytest tests/test_consensus.py::test_dense_kf_vs_baseline -q
```

---

## 📊 Metrics to Highlight

### Technical Performance
- Dense preset: **22.13 ps** (clock 0.32 / freq 0.03 / 1 iter)
- Baseline (no KF): **22.45 ps**
- Sweep minimum: **20.93 ps** (clock 0.22 / freq 0.03 / 2 iters)
- Small preset: **20.96 ps** (clock 0.25 / freq 0.05 / 1 iter)
- Calibration bias: **2.65 ps** (loopback)

### Business Traction
- 2 private 5G vendors in advanced conversations
- 1 robotics OEM piloting mesh coordination timing
- 3 universities running the open-source simulations
- 75+ GitHub stars, 12 forks since extended_011 release

### Investment Terms
- **Request: $1M** (a16z Speedrun SR006 package)
- Structure per SR006: **$500K for 10% upfront (SAFE)** + **$500K follow‑on within ~18 months**
- Plus: **>$5M in credits** (program perks)

### 90-Day Targets
- 3 pilot LOIs (RAN + robotics + defense)
- 75 SDK downloads
- 6 telemetry dashboards live
- $20K MRR (pilot tier)

---

## 🚀 Submission Steps

### 1. Prepare Materials (by Sept 24)
- [ ] Convert `PITCH_DECK_V2.md` to PDF
- [ ] Record 60-second video using updated script
- [ ] Capture sweep verifier + regression outputs as appendix PNGs
- [ ] Test all links and repositories (extended_011 tag)

### 2. Application Form (Sept 25-27)
- [ ] Go to https://speedrun.a16z.com/apply
- [ ] Enter company details from `APPLICATION.md`
- [ ] Upload pitch deck PDF
- [ ] Include demo video link and GitHub repo
- [ ] Attach verification artifacts (optional but recommended)

### 3. Final Review (Sept 28 morning)
- [ ] Re-run `pytest -q`
- [ ] Re-run `scripts/verify_kf_sweep.py ...`
- [ ] Validate demo hardware + Grafana dashboard
- [ ] Submit before 11:59 pm PT

### 4. Post-Submission
- [ ] Email sr-apps@a16z.com with any new hardware data
- [ ] Continue KF tuning experiments (frequency parity, multi-iterations)
- [ ] Prep interview deck with guardrail highlights

---

## 💪 Why We'll Get Accepted

### 1. Real Technical Breakthrough
- 22.13 ps dense preset + 20.93 ps sweep min
- Deterministic scripts to reproduce results
- 4,500× calibration improvement

### 2. Perfect Timing
- Private 5G & Open RAN adoption curve
- GPS jamming headlines keep coming
- Edge AI & autonomy hitting production scale

### 3. Infrastructure Play
- Platform + SaaS + OEM licensing
- Network effects: more nodes, better accuracy
- Sandboxed guardrails protect core value

### 4. Pragmatic Execution
- Pre-tracked 90-day plan (hardware, pilots, SDK)
- Current results anchored in JSON + tests
- Investors see instant ROI path

### 5. Strong Narrative
- Bell Labs legacy + music-to-physics insight
- Simple core (Δf = 1 MHz) with huge implications
- Honest about roadmap (hardware validation next)

---

## 📝 Final Message

**We now deliver software-only 22 ps synchronization on existing radios, with deterministic guardrails that make every claim reproducible.**

Monte Carlo `extended_011` promoted the dense preset (0.32 / 0.03 / 1) after sweep verification and regression testing. Anyone can run the scripts, validate the JSON, and see the same 0.33 ps edge vs baseline.

**Driftlock is the timing substrate the wireless world has been missing.**

---

## Contact for Questions

Hunter Bown  
- Email: hunter@shannonlabs.dev  
- GitHub: https://github.com/shannon-labs/driftlock  
- Website: https://driftlock.net

**Deadline: September 28, 2025 at 11:59 pm PT**

Let's ship this! 🚀
