# 60-Second Video Script - Simplified Demo Focus

## Opening (0-5 seconds)
**[VISUAL: Two radios on a desk]**

"Software-only time synchronization. Works on radios you already have."

---

## The Problem (5-15 seconds)
**[VISUAL: GPS satellite with X, indoor warehouse, jammed signal]**

"GPS gives 50 nanoseconds if you can see satellites. Doesn't work indoors. Gets jammed. Every autonomous system, private 5G network, and robot swarm is stuck with bad timing."

---

## Live Demo (15-40 seconds)
**[VISUAL: Screen showing live telemetry dashboard]**

"Watch this. Two standard radios. Creating beat patterns..."

**[VISUAL: Oscilloscope showing beat wave, then phase extraction]**
"See the beats? That's timing information."

**[VISUAL: Terminal showing API calls]**
```
> driftlock.get_clock_bias()
< 2.3 nanoseconds

> driftlock.get_sync_quality()
< 98.2%
```

**[VISUAL: Move one radio, numbers update in real-time]**
"Move the radio. Timing updates. No satellites. No atomic clocks. Just software reading physics that's always been there."

---

## Results (40-50 seconds)
**[VISUAL: Performance graph, pilot customer logos]**

"2 nanoseconds in simulation. 10 nanoseconds in hardware, improving daily. Patent pending. Three pilots starting. Private 5G vendor wants this now."

---

## Close (50-60 seconds)
**[VISUAL: Hunter with radios, Driftlock logo]**

"I'm Hunter Bown. Band director turned engineer. I heard what everyone else was filtering out. This is Driftlock—software-first timing for existing radios."

**[END CARD: driftlock.net | See live demo at booth]**

---

## Director's Notes for 60-Second Version

### Visual Priority
1. **Live demo is key** - Show actual telemetry updating
2. **Simple visuals** - Two radios, one screen
3. **Real-time changes** - Movement → timing update
4. **No complex animations** - Keep it real

### What to Show on Screen
- Beat pattern visualization (simple sine waves)
- Phase extraction (unwrapping graph)
- API terminal with actual calls
- Grafana dashboard with time offset plots
- Move device → watch numbers change

### Audio Design
- Quiet beat pattern (1 Hz) in background during demo
- Terminal typing sounds for API calls
- Clear voiceover, no music

### Key Messages
1. Software-only (no new hardware)
2. Works on existing radios
3. Live demo proves it works
4. Real customers want it

### What NOT to Include
- Family history
- Complex math explanations
- Theoretical claims
- "Revolutionary" language

### The One Visual That Matters
**Moving a radio and watching timing update in real-time**