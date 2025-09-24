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
