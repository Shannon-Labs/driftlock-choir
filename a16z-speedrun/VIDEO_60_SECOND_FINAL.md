# 60-Second Video Script (Final)

---

## Hook (0-5 seconds)
**[VISUAL: Two radios on bench, oscilloscope showing waves]**

"We turn tiny frequency offsets into time telemetry — with today's radios."

---

## Demo (5-40 seconds)

### Setup (5-10s)
**[VISUAL: Two Feather boards transmitting, RTL-SDR connected to laptop]**

"Two standard transmitters. One RTL-SDR receiver. Watch."

### Beat Visualization (10-20s)
**[VISUAL: Screen showing 1 kHz beat pattern waveform]**

"Offset by 1 kHz. See the beat pattern? That's timing information."

### Movement Demo (20-30s)
**[VISUAL: Move one Feather board 30 cm]**

"Move one node 30 centimeters..."

**[VISUAL: Analyzer screen updates]**
```
Δτ ≈ 1000 picoseconds
```

"One nanosecond. From 30 centimeters of movement."

### SDK Output (30-35s)
**[VISUAL: Terminal showing API calls]**
```python
>>> dl.get_clock_bias()
2.3 ns
>>> dl.get_sync_quality()
98.2%
```

"The SDK reads it instantly."

### Dashboard Flash (35-40s)
**[VISUAL: Grafana panel flashes green, shows time series]**

"Grafana shows it's locked. Sub-10 nanosecond precision."

---

## Close (40-60 seconds)
**[VISUAL: Back to two radios, then code on screen]**

"Works as an overlay on pilots or beacons. Masterless — no single point of failure. Software-first — no new hardware."

**[VISUAL: Logos of private 5G, robotics companies]**

"Ready for private 5G and robotics pilots. From 12 nanoseconds uncalibrated to 2.65 picoseconds with loopback. 4,500× improvement. One software switch."

**[VISUAL: Hunter with radios, Driftlock logo]**

"I'm Hunter Bown. Solo founder. Shipping the timing layer for everything."

**[END CARD: github.com/shannon-labs/driftlock | hunter@shannonlabs.dev]**

---

## Visual Checklist

### Required Equipment
- [ ] 2× Adafruit Feather boards with radios
- [ ] 1× RTL-SDR USB dongle
- [ ] 1× Laptop with dual monitor
- [ ] 1× Ruler or measuring tape
- [ ] Clean bench/desk surface

### Software Setup
- [ ] Oscilloscope app showing beat pattern
- [ ] Terminal with Python REPL ready
- [ ] Grafana dashboard configured
- [ ] Driftlock SDK installed

### Key Shots
1. Wide shot of bench setup
2. Close-up of beat pattern
3. Moving the Feather board
4. Screen capture of Δτ update
5. Terminal showing API calls
6. Grafana dashboard green flash
7. Face shot for closing

### Props/Background
- Clean, professional workspace
- Good lighting (no shadows on screen)
- Driftlock logo visible
- No distracting background

---

## Speaking Notes

### Pace
- Clear, confident, not rushed
- Technical but accessible
- Let visuals breathe

### Emphasis Points
- "tiny frequency offsets"
- "1000 picoseconds"
- "software-first"
- "4,500× improvement"

### Tone
- Excited but not hyperbolic
- Technical credibility
- Solo founder pride

---

## Alternative Takes (if time permits)

### Take 1: Technical Focus
- More emphasis on beat phase math
- Show φ = 2πΔf(t - τ) briefly
- Mention consensus algorithm

### Take 2: Market Focus
- Open with "Private 5G needs timing"
- Mention GPS problems
- Close with customer interest

### Take 3: Solo Founder Story
- "Band director who heard beats differently"
- "No committees, just shipping"
- "One person, one mission"

**Choose best take that feels most natural**