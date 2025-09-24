### Handoff Notes for Driftlock Choir Simulation Debugging

**Objective:**
The primary goal is to run a 50-seed Monte Carlo simulation using the new `INDOOR_OFFICE` channel profile and recently added hardware impairments to generate statistically sound performance benchmarks for the Driftlock Choir platform.

**Current Status: Blocked - Catastrophic Divergence**
The simulation is currently failing. The consensus algorithm is not just converging slowly; it's catastrophically diverging, with the timing error exploding to physically impossible values (on the order of hours). This happens consistently, even in short smoke tests.

**Diagnosis:**
The divergence is a classic feedback loop failure. The new, high-fidelity `INDOOR_OFFICE` channel model is introducing a level of noise and multipath error that the consensus algorithm, in its current tuning, cannot handle. The algorithm overcorrects for these large errors, which amplifies the error in the next state, leading to an uncontrolled spiral.

**Key Debugging Steps & Fixes Implemented:**
I've made significant progress in debugging and improving the simulation script. Here are the key changes I've implemented:

1.  **Parallelization:** The `scripts/run_monte_carlo.py` script now supports parallel execution using Python's `multiprocessing` library. You can control the number of workers with the `--num-workers` flag.
2.  **Smoke Test:** A `--smoke-test` flag has been added to the script. This will run a quick, 2-seed, 16-node simulation to validate the entire pipeline before committing to a long run.
3.  **Profiling:** A `--profile` flag has been added to enable `cProfile` for a single run, which will help identify any performance bottlenecks.
4.  **Shock Therapy (Attempt 1):** I've implemented a two-pronged attack to try to stabilize the system:
    *   **Extreme Gain Reduction:** The Kalman filter gains (`local_kf_clock_gain` and `local_kf_freq_gain`) have been reduced by a factor of 1,000 from their original values.
    *   **Outlier Rejection Clamp:** An outlier clamp has been added to the `_run_local_kf` method in `sim/phase2.py` to discard any measurement with an error greater than 1 microsecond.
5.  **Bug Fixes:** I've fixed several bugs in the simulation code, including:
    *   An incorrect call order in the main simulation loop.
    *   A bug due to an immutable dataclass.
    *   An `AttributeError` due to a renamed attribute.

**Recommendations for Next Steps:**

1.  **Verify the Divergence:** Run the smoke test to observe the divergence firsthand: `python scripts/run_monte_carlo.py --smoke-test --channel-profile INDOOR_OFFICE`
2.  **Investigate the Consensus Algorithm:** The "shock therapy" I implemented was not enough to stabilize the system. The next step should be a deeper dive into the consensus algorithm itself (`src/alg/consensus.py`). It's possible that there are other parameters that can be tuned to dampen the system, or that a more fundamental change to the algorithm is required to handle this level of noise.
3.  **Profile the Code:** Use the `--profile` flag to identify any bottlenecks in the code. This will help to focus optimization efforts.
4.  **Examine the Channel Model:** The `INDOOR_OFFICE` channel profile is the root cause of the divergence. It would be wise to examine the `src/chan/tdl.py` file to see how this profile is implemented. It's possible that there is a bug in this file that is causing the extreme noise.
5.  **Isolate the Problem:** If all else fails, it might be necessary to temporarily disable the `INDOOR_OFFICE` channel profile and see if the simulation converges with a simpler channel model. This would help to isolate the problem and confirm that the consensus algorithm is still fundamentally sound.
