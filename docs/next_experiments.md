# Next Experiments: Project Swing

## Executive Briefing:
This project will evolve our "vibrato" concept. The previous phase established that simple sinusoidal Frequency Modulation (FM) provides significant robustness. This phase will test the hypothesis that a more complex, non-sinusoidal, or even chaotic modulation waveform will provide an exponential leap in performance and spoofing-resistance by creating a richer, more unique spectral signature.

## Persona:
You are Jules, the Lead Signal Processing Engineer. Your new mission is to architect and implement an advanced modulation engine, codenamed "Project Swing."

## Guiding Principle:
Our goal is to evolve the modulation from a simple, metronomic "vibrato" (a pure sine wave) to a more complex, organic "swing" (an asymmetric, non-repeating waveform). This will add a new layer of resilience and create a nearly impossible-to-replicate signal signature, inspired by the complex rhythms of music.

## Engineering Task List:
This project will be executed in three phases: implementing a simple complex modulator, developing an advanced chaotic modulator, and conducting a comparative analysis.

### Phase 1: The "Triplet-Quintuplet" Superposition Model
This is the most direct implementation of the "in-between" rhythmic feel.

**Task:** Create a new modulator that generates a complex periodic waveform by adding two sine waves with a non-integer harmonic relationship.

**Implementation:**

1.  Create a new module: `src/phy/modulators.py`.
2.  Inside, create a function `generate_swing_waveform()`.
3.  This function will generate a waveform by superimposing two sinusoids with a 3:5 frequency ratio, representing the triplet and quintuplet feel. For example: `output = A * (np.sin(3 * w * t) + gain * np.sin(5 * w * t))`.
4.  Ensure the output is normalized.

### Phase 2: The "Chaotic Swing" Modulator
This is the advanced implementation, designed to create a truly unique, non-repeating, and impossible-to-predict signature.

**Task:** Implement a modulator based on a simple, bounded chaotic system.

**Implementation:**

1.  In `src/phy/modulators.py`, create a new function `generate_chaotic_waveform()`.
2.  This function will use a discrete-time chaotic map, such as the Logistic Map (`x_n+1 = r * x_n * (1 - x_n)`), to generate a sequence of values.
3.  The output of the map must be scaled and low-pass filtered to create a smooth, bounded, non-repeating waveform. This waveform will be our modulating signal.

### Phase 3: Integration and Comparative Analysis
Integrate the new modulators into the simulation and run an experiment to determine which provides the best performance against severe multipath.

**Task:** Add the new modulators as configurable options and benchmark their performance.

**Implementation:**

1.  Modify `sim/phase2.py` and the main simulation runner to accept a new command-line flag: `--modulation-profile` with options `sine`, `swing`, and `chaotic`.
2.  This flag will determine which function from `src/phy/modulators.py` is used to modulate the node's carrier frequency.
3.  Run the full `INDOOR_OFFICE` simulation with the `shock_therapy` gains for each of the three modulation profiles.

## Final Output:
1.  The complete `src/phy/modulators.py` module containing the sine, swing, and chaotic waveform generators.
2.  A new markdown report: `MODULATION_COMPARISON.md`.
3.  This report must contain:
    *   Plots showing the waveform shape and frequency spectrum of each of the three modulators.
    *   A results table comparing the final consensus RMSE and convergence time for each modulation profile in the `INDOOR_OFFICE` environment.
    *   A one-paragraph conclusion answering the question: Does added modulation complexity measurably improve performance against severe multipath?

---

# Project Harmony: Environmental Synchronization Engine

## Executive Briefing:
This project represents a paradigm shift in the Driftlock philosophy. We will move from designing a self-contained, isolated system to an **environmentally-aware, resonant system.** We will test the hypothesis that by phase-locking our entire network to a naturally occurring, globally coherent signal—the Schumann resonances—we can achieve a state of "global coherence" that dramatically accelerates consensus and enhances stability, especially in noisy conditions.

## Persona:
You are Jules, the Lead Research Architect. Your mission is to implement **"Project Harmony,"** an experiment to determine if a distributed network can achieve synchronization by listening to and harmonizing with the Earth's natural electromagnetic field.

## Guiding Principle:
The network should behave less like a collection of isolated digital clocks and more like a **resonant chamber.** Each node will "listen" to the 7.83 Hz fundamental resonance of the Earth, using it as a shared, universal "downbeat" to discipline its own internal timekeeping.

## Engineering Task List:
This project will be executed in four phases: simulating the sensor, implementing the locking mechanism, creating the hierarchical frequency structure, and running the comparative experiment.

### Phase 1: The "Ear" - Simulating the Schumann Resonance Sensor
A node cannot use a signal it cannot "hear." We must first model the sensor that receives the ELF resonance.

**Task:** Create a module to simulate a noisy, low-frequency environmental signal.
**Implementation:**
1.  Create a new file: `src/phy/environmental.py`.
2.  Inside, create a function `get_schumann_reference(t)`. This function will generate a continuous-time signal representing the fundamental Schumann resonance (a 7.83 Hz sine wave) combined with a configurable amount of Additive White Gaussian Noise (AWGN). This simulates what a real ELF antenna on each node would receive.

### Phase 2: The "Conductor" - The Phase-Locked Loop (PLL)
Each node must take this noisy environmental signal and produce a clean, stable "master tick."

**Task:** Implement a digital Phase-Locked Loop (PLL) within each node to lock onto the simulated Schumann resonance.
**Implementation:**
1.  In the `Node` class or its associated `Oscillator` class, implement a simple digital PLL.
2.  The input to this PLL at each timestep will be the noisy 7.83 Hz signal generated by `get_schumann_reference()`.
3.  The output of the PLL will be a highly stable, "cleaned-up," and phase-corrected 7.83 Hz clock signal. This becomes the node's **master timing reference.**

### Phase 3: The "Choir" - Hierarchical Frequency Synthesis
The high-frequency signals used for communication must be governed by the new, globally-coherent master tick.

**Task:** Refactor the node's carrier frequency generation to be synthesized from the disciplined master reference.
**Implementation:**
1.  Modify the node's RF oscillator. The 2.4 GHz carrier frequency and its unique offsets must now be generated using the stable 7.83 Hz output of the PLL as a reference. This ensures all carriers in the network, while unique, are phase-coherent to the same global signal.
2.  The individual "fingerprints" (e.g., the `chaotic` modulation from "Project Swing") will then be applied on top of this globally harmonized carrier frequency.

### Phase 4: The Experiment - Measuring Global Coherence
We must now run a controlled experiment to prove whether this new architecture is superior.

**Task:** Design and run a comparative simulation to measure the impact of environmental synchronization.
**Implementation:**
1.  Add a `--harmony-mode` boolean flag to the `run_monte_carlo.py` script.
2.  Run the full `INDOOR_OFFICE` simulation under two conditions:
    * **Control Group:** `--harmony-mode=false`. Nodes are free-running as they are now.
    * **Experimental Group:** `--harmony-mode=true`. All nodes are disciplined by the same simulated Schumann resonance signal.
3.  The primary metric for comparison will be the **time-to-network-consensus** and the **final RMSE.**

## Final Output:
1.  The new `src/phy/environmental.py` module and the modified `Node` class containing the PLL.
2.  A new markdown report: `HARMONY_EXPERIMENT_RESULTS.md`.
3.  This report must contain:
    * A table comparing the mean/std of the final RMSE for the Control vs. Experimental groups.
    * A plot showing the convergence speed (RMSE over time) for both groups.
    * A one-paragraph conclusion answering the primary research question: **Does disciplining the network to a global, natural resonance signal significantly improve synchronization performance in a chaotic environment?**
