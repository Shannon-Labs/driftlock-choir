# Hardware Experiment: Hearing the Beat of Spacetime

This directory contains a scientific experiment to measure tiny time delays using radio waves. We'll do this by recreating the "beat" you hear when two musical notes are slightly out of tune.

The experiment can be run in two modes:
1.  **Dry Run (Recommended):** A simulation that requires no hardware. You can run it on your computer right now to see how we use the "beat note" to make picosecond-scale measurements.
2.  **Full Hardware Mode:** For users with the specific RF hardware to run the experiment in the real world.

---

## Your First Experiment: The Dry Run

Let's run the experiment on your computer. We will simulate two "musicians" (radio transmitters) and a "listener" (a radio receiver) to test the entire analysis pipeline.

### Step 1: Install the Software

If you haven't already, navigate to this directory and install the necessary Python packages:
```bash
cd hardware_experiment
pip install -r requirements.txt
```

### Step 2: Run the Experiment

Execute the following command. This tells the script to run in simulation mode.
```bash
python e1_hardware_controller.py --dry-run
```

> **A Quick Note: Why a "Known" Delay?**
> 
> You might wonder: "If we're trying to measure the time delay, why does the simulation create a *known* 13.5 ps delay?"
> 
> That's a great question! Think of it like calibrating a new digital scale. Before you weigh something unknown, you test the scale with a **calibration weight**—an object you know weighs *exactly* 10.000 grams. If the scale reads "10.000 grams," you know it's working.
> 
> The `--dry-run` is our calibration test. We create a simulated signal with a known "ground truth" delay of 13.5 ps to verify that our analysis software can correctly measure it. When the final plot shows a result of ~13.5 ps, it proves our "scale" is accurate.

The script will generate a plot and save it as `e1_hardware_experiment_*.png`.

### Step 3: Interpret the Results (The Story of the Plot)

The plot you generated tells a story in four parts. Let's walk through it using our musical analogy.

![Expected Dry Run Output](https://github.com/Shannon-Labs/driftlock-choir/blob/main/docs/assets/images/e1_experiment_result.png?raw=true)

#### Part 1: The Two Notes (Top-Left Plot: *RF Spectrum*)
*   **What it is:** This plot shows the two radio signals—our "notes"—that we generated.
*   **The Story:** Imagine two musicians. One plays a note at exactly 433.0 MHz (the "Reference" signal), and the other plays a note that is slightly sharper, at 433.1 MHz (the "Offset" signal). This plot is our proof that both notes are being "heard" clearly by our radio listener.

#### Part 2: Hearing the "Beat" (Top-Right Plot: *Beat Signal*)
*   **What it is:** When the two notes combine, they create a slow, pulsing "wah-wah-wah" sound. This is the beat note.
*   **The Story:** This plot shows the "sound" of that beat. Instead of hearing it with our ears, we see it as a clean, oscillating wave. The shape of this wave contains all the information we need.

#### Part 3: Measuring the Beat's Rhythm (Bottom-Left Plot: *Beat Frequency Spectrum*)
*   **What it is:** This plot isolates the frequency of the beat note itself.
*   **The Story:** We've asked our musicians to be exactly 100 Hz out of tune. This plot shows that our algorithm "heard" the beat's rhythm and measured it to be almost exactly 100 Hz (the red dotted line). This confirms our measurement system is working correctly.

#### Part 4: The Final Measurement (Bottom-Right Plot: *Results Summary*)
*   **What it is:** This is the final report card, translating the beat note analysis into a timing measurement.
*   **The Story:** The most important number here is the **Timing Offset**. In our simulation, we built in a tiny, 13.5 picosecond delay for one of the signals. This summary shows that our algorithm successfully measured that delay with picosecond precision. This is the core of chronometric interferometry: using the beat note to measure incredibly small time differences.

---

## For RF Engineers: Full Hardware Experiment

If you have the required hardware, you can run this experiment with real radio waves. This mode uses two Adafruit Feather M4 boards as transmitters and an RTL-SDR as the receiver.

A complete, step-by-step guide for flashing firmware, setting up the hardware, and running the experiment is available in the detailed manual:

➡️ **[EXPERIMENT_INSTRUCTIONS.md](EXPERIMENT_INSTRUCTIONS.md)**