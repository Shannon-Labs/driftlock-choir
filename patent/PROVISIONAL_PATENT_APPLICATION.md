# PROVISIONAL PATENT APPLICATION

Title of Invention

System and Method for Distributed Wireless Synchronization Using Chronometric Interferometry with Intentional Carrier Frequency Offset

Inventor and Assignee

- Inventor: Hunter Bown
- Assignee: Shannon Labs, Inc. (Delaware C‑Corp), Division: Driftlock

Cross-Reference and Lineage

This application draws inspiration from the inventor’s family lineage in communications, including foundational work by Ralph Bown (e.g., US2436376A), and continues a multi‑generational pursuit of precision timing.

Statement Regarding Federally Sponsored Research

Not applicable.

Field of the Invention

The invention relates to wireless time and frequency synchronization. More particularly, it concerns chronometric interferometry that intentionally creates a carrier frequency offset between nodes to produce beat signals from which propagation delay and frequency difference are extracted, enabling masterless, sub‑nanosecond synchronization across distributed wireless networks.

Background of the Invention

For a century, frequency offset between radios has been treated as a defect to estimate and eliminate. Standards (IEEE 1588, 3GPP, ITU, IETF) prescribe offset compensation; PLL design aims to minimize residual error. GPS/GNSS provides ~10–50 ns timing but fails indoors and is vulnerable to interference. Wired protocols (PTP/NTP) degrade in wireless due to variable propagation and jitter. Sub‑nanosecond fiber solutions (e.g., White Rabbit) require specialized wired infrastructure. Master‑based wireless approaches create single points of failure and accumulate estimation errors in networks.

In contrast, prior techniques that do produce beat signals do so for different purposes: frequency‑modulated continuous‑wave (FMCW) radar exploits beat frequencies to estimate range and velocity of targets, not to synchronize independent clocks nor to extract clock bias via two‑way chronometric exchange. Heterodyne receivers form beats to translate spectra to intermediate frequency for demodulation; they do not treat the beat phase as a chronometric observable under intentional offsets. Two‑way time transfer protocols (e.g., satellite and fiber TWSTFT) measure timing asymmetries via message exchange but do not intentionally create and maintain an offset Δf within a wireless channel to generate a chronometric beat for closed‑form τ/Δf estimation or variance‑weighted, masterless network consensus.

Unmet need: a wireless, GPS‑independent, masterless synchronization method achieving <10 ns — and preferably <2 ns — accuracy using commercial oscillators, robust to multipath, mobility, and packet loss, and scalable to large networks.

Summary of the Invention

Paradigm shift — the Bown Driftlock Method intentionally creates and maintains a small carrier frequency offset Δf between communicating nodes. The resulting beat signal at Δf carries chronometric information in its phase evolution that encodes both propagation delay and true frequency difference. By measuring the beat phase over microsecond windows and applying a closed‑form estimator, nodes recover delay τ and Δf without iterative search. Two‑way measurements separate geometric delay from clock bias. A variance‑weighted distributed consensus aligns all nodes without any master reference. Optional Chebyshev acceleration and spectral step‑size selection deliver <5 ms convergence in 50‑node networks.

Key features

- Intentional Δf as a synchronization feature, not an impairment.
- Closed‑form, non‑iterative estimator from beat‑phase linear fit.
- Two‑way bias cancellation (geometric delay and clock bias separation).
- Variance‑weighted (1/σ²) decentralized consensus with spectral ε; optional Chebyshev acceleration.
- Sub‑sampling ADC (~2×Δf) with low‑rate capture of beat tones.
- Robust ambiguity resolution via coarse wideband preambles and multi‑carrier retunes.

Performance (simulation examples; non‑limiting and illustrative)

- Example: ~2 ns RMS timing accuracy in simulation under representative SNR and bandwidth settings, with theoretical capability for sub‑nanosecond precision.
- Network: convergence within milliseconds to high precision using inverse‑variance weighting and spectral step size.
- Hardware: commercial 2–20 ppm TCXOs; no GPS, atomic clocks, or master references required.

Brief Description of the Drawings

- Fig. 1: Two‑node beat generation and handshake. [Figure 1 would show two‑node beat generation with intentional frequency offset Δf.]
- Fig. 2: Phase extraction and closed‑form estimation pipeline. [Figure 2 would show phase extraction and linear fit to obtain τ and Δf.]
- Fig. 3: Network topology and variance‑weighted consensus. [Figure 3 would show a graph of nodes and the consensus update rule.]
- Fig. 4: Convergence vs. iteration (timing RMS). [Figure 4 would show timing RMS decreasing over iterations.]
- Fig. 5: Node/system block diagram (transceiver, sub‑sampling ADC, DSP). [Figure 5 would show RF front‑end, ADC, DSP, and consensus.]

Detailed Description of the Invention

I. Theory of Operation

Consider nodes i and j with carrier frequencies f_i and f_j = f_i + Δf. When both transmit simultaneously and receive each other, each node observes a complex beat signal at frequency Δf. The beat phase

φ_beat(t) = 2π Δf (t − τ_prop) + (θ_i − θ_j) + 2π f_i τ_prop

contains: (i) a linear phase ramp from Δf, (ii) a constant offset from initial phases, and (iii) a delay‑dependent carrier term. Over a window T, unwrapping φ_beat(t) and fitting a line yields slope ≈ 2πΔf and intercept b. A closed‑form estimator recovers

τ = [φ_beat(T) − φ_beat(0) − 2πΔf T + (θ_j − θ_i)]/(2π f_i)

and Δf from the slope. Optional coarse wideband delay estimates and multi‑carrier retunes provide robust unwrapping. Here, f_i denotes the absolute carrier frequency (Hz) of node i; Δf = f_j − f_i is the intentional offset maintained as a measurement feature.

II. Two‑Way Chronometric Handshake

The protocol executes forward (A→B) and reverse (B→A) measurements to separate geometric delay and clock bias:

- Forward: Node A TX at f_1; Node B TX at f_2 = f_1 + Δf; Node B extracts (τ̂_AB, Δf̂_AB).
- Reverse: Roles swapped; Node A extracts (τ̂_BA, Δf̂_BA).
- Bias resolution: δt = 0.5(τ̂_AB − τ̂_BA); geometric delay: τ_geo = 0.5(τ̂_AB + τ̂_BA).

III. Estimation and Signal Processing

Pipeline: band‑limiting at Δf; sub‑sampling ~2×Δf; angle and unwrap; linear fit φ(t) = at + b; extract Δf̂ = a/(2π); compute τ candidate from b and f_carrier; unwrap τ to a coarse hint; average across carriers (weights ∝ f^2). Estimator is non‑iterative and closed‑form.

Ambiguity resolution: (a) coarse preamble yields unambiguous τ hint; (b) retunes form synthetic wavelengths; (c) multi‑carrier Chinese Remainder constructions; (d) continuous phase tracking across cycles.

Hardware considerations: TCXO drift (2–20 ppm) mitigated by short windows and Kalman tracking; IQ imbalance/DC offset mitigated by digital correction; multipath handled by two‑ray modeling and super‑resolution options; sub‑sampling ADC minimizes rate to ~2×Δf.

IV. Distributed Masterless Consensus

For a graph G = (V, E) of N nodes, each edge (i, j) contributes a measurement d_ij = [ΔT_ij, Δf_ij] with covariance Σ_ij. The per‑node state x_i = [ΔT_i, Δf_i] is updated as

x_i(k+1) = x_i(k) + ε Σ_{j∈N_i} W_ij ( d_ij − (x_i(k) − x_j(k)) ),

with W_ij = diag(1/σ^2_τ, 1/σ^2_Δf) derived from measurement variances. A spectral step size ε ≈ c/λ_max(L), 0 < c < 1, ensures stability; optional Chebyshev acceleration further improves convergence rates. Zero‑mean projection enforces observability constraints. Asynchronous updates select a node at random and apply the same weighted correction.

V. Implementation Parameters and Embodiments

- Carrier bands: ISM or licensed (e.g., 2.4 GHz), but broadly applicable from audio to optical.
- Δf range: 1–10 MHz; typical Δf ≈ bandwidth/10; avoid aliases with sampling clocks.
- Measurement window: T = 10–20 µs; sub‑sampling rate: ~2×Δf; beat capture 1 kHz–10 MHz.
- Calibration modes: none, loopback, or perfect; hardware delays tracked as offsets.
- Mobility and loss: operation robust to packet loss (~20%) and mobility (e.g., up to 20 m/s) via repeated updates.

VI. Experimental Validation (Simulation)

- Two‑node Monte Carlo (N=500): 2.081 ns RMS timing error under representative conditions; Δf̂ accuracy within tens of Hz at 20 dB SNR; alias resolution success ≈100% with coarse + retunes.
- 50‑node network: sub‑100 ps RMS network‑wide; <5 ms convergence with variance weighting; O(log N) practical scaling.

VII. Applications

5G/6G base station coordination; distributed radar; high‑frequency trading timestamp alignment; quantum networking; scientific instrumentation; IoT mesh synchronization.

Claims

[Editor’s note] An expanded, consolidated claim set that captures additional implementation details (calibration modes, online variance weighting, multipath phase-bias compensation, MAC scheduling, ADC/ENOB/jitter modeling, etc.) is provided in `patent/CLAIMS_CONSOLIDATED_2025-09-20.md` and is intended for inclusion in the filing. The condensed set below is preserved for readability.

What is claimed is:

Condensed Claim Set — Provisional (25 claims)

1. A method for synchronizing wireless nodes by intentionally generating a frequency offset Δf, forming beat signals at Δf from simultaneous transmission and reception, extracting beat phase over a microsecond window, and recovering propagation delay τ and frequency difference via a closed‑form estimator, wherein said frequency offset Δf is intentionally generated and maintained as a measurement feature rather than being corrected or compensated as an error, contrary to conventional synchronization methods, and wherein bidirectional measurements resolve clock bias and yield geometric delay.

2. A wireless synchronization system comprising: first and second programmable transceivers transmitting at carrier frequencies that differ by an intentional offset Δf; a beat detection path using a sub‑sampling ADC at approximately 2×Δf; and one or more processors configured to extract beat phase and compute τ and Δf from a closed‑form estimator and execute a two‑way handshake to separate geometric delay from clock bias.

3. A distributed synchronization method for a network of nodes, comprising: performing pairwise measurements d_ij = [ΔT_ij, Δf_ij] with variances; and iteratively updating node states x_i = [ΔT_i, Δf_i] via x_i(k+1) = x_i(k) + ε Σ_j W_ij ( d_ij − (x_i(k) − x_j(k)) ), where W_ij = diag(1/σ^2_τ, 1/σ^2_Δf) and ε is selected from a Laplacian spectrum, achieving masterless synchronization without GPS, atomic clocks, or external timing references.

4. A non‑transitory computer‑readable medium storing instructions that, when executed by a processor, cause a device to perform any of the methods of claim 1 or claim 3.

5. A method of wireless synchronization that exploits rather than eliminates frequency offset between nodes, comprising: intentionally generating and maintaining a carrier frequency offset Δf as a measurement feature; forming a beat signal at Δf during simultaneous transmission and reception; extracting beat phase over an observation window; and computing τ and Δf via a closed‑form estimator.

6. The method of claim 1, wherein Δf is between approximately 1 MHz and approximately 10 MHz and is approximately one‑tenth of an available channel bandwidth.

7. The method of claim 1, wherein the measurement window T is between approximately 10 microseconds and approximately 20 microseconds and a sub‑sampling ADC operates at approximately 2×Δf.

8. The method of claim 1, wherein Δf is maintained as a synchronization feature and not corrected as an impairment.

9. The method of claim 1, further comprising coarse wideband delay estimation to obtain an unambiguous delay hint for phase unwrapping.

10. The method of claim 1, further comprising multi‑carrier retunes to provide synthetic wavelengths for robust unwrapping, including Chinese Remainder constructions.

11. The method of claim 1, wherein τ is computed from a closed‑form linear fit of unwrapped beat phase without iterative search.

12. The method of claim 1, wherein bidirectional measurements compute δt = 0.5(τ̂_AB − τ̂_BA) and τ_geo = 0.5(τ̂_AB + τ̂_BA).

13. The method of claim 1, further comprising weighting multi‑carrier τ estimates proportional to carrier frequency squared.

14. The method of claim 1, further comprising multipath mitigation via super‑resolution separation and weighting of direct‑path estimates.

15. The system of claim 2, wherein the sub‑sampling ADC includes anti‑alias filtering and dynamic range sufficient for Δf capture, and a transceiver frequency resolution of at least 1 Hz is provided.

16. The system of claim 2, wherein oscillators are temperature‑compensated crystal oscillators with stability between approximately 2 and approximately 20 parts per million.

17. The method of claim 1, wherein Δf avoids integer relationships with sampling clocks to minimize aliasing.

18. The method of claim 1, wherein Δf is adaptively selected based on signal‑to‑noise ratio and bandwidth to optimize estimator variance.

19. The method of claim 1, further comprising scheduling measurement offsets Δt to average jitter and oscillator phase noise.

20. The method of claim 3, further comprising enforcing zero‑mean constraints across nodes for timing and frequency states; supporting asynchronous updates; and applying Chebyshev polynomial acceleration to reduce convergence time.

21. The method of claim 3, wherein ε = c/λ_max(L) with 0 < c < 1 provides a stability margin and rapid convergence.

22. The method of claim 3, wherein the consensus achieves substantially improved network‑wide precision relative to conventional microsecond‑class methods under representative SNR and connectivity.

23. The system of claim 2, maintaining synchronization performance under up to approximately 20% packet loss and in the presence of node mobility.

24. The system of claim 2, operating in industrial, scientific, and medical bands and/or in licensed cellular spectrum.

25. A method substantially as described herein with reference to the accompanying description and any one of the examples.

Abstract (≤150 words)

The Bown Driftlock Method introduces chronometric interferometry for wireless synchronization by intentionally creating a small carrier frequency offset Δf between nodes to generate a beat signal whose phase encodes propagation delay and true frequency difference. A closed‑form estimator applied to beat‑phase evolution recovers τ and Δf over microsecond windows without iterative search. Bidirectional measurements separate geometric delay from clock bias. A variance‑weighted distributed consensus with spectral step‑size (and optional Chebyshev acceleration) aligns all nodes without a master reference, converging rapidly to high precision. Example results show ~2 ns RMS accuracy in simulation with theoretical capability for sub‑nanosecond precision, enabling GPS‑free, scalable synchronization for 5G/6G, distributed radar, high‑frequency trading, quantum systems, and IoT.

Incorporation by Reference

This application incorporates by reference all materials submitted herewith, and any prior applications to which priority is claimed (if any). Priority details will be provided in the filing submission.

Advantages

- No GPS dependency; masterless operation.
- Works with commercial oscillators (e.g., TCXO); indoor/underground capable.
- Distributed consensus; scalable to many nodes and topologies.

Industrial Applicability

The invention has immediate applicability in 5G/6G base station coordination, distributed radar, high‑frequency trading, scientific instrumentation, and IoT synchronization.

Best Mode and Scope

The best mode contemplated by the inventor for carrying out the invention is described herein. While specific embodiments are described, variations and modifications will be apparent to those skilled in the art. The invention encompasses both methods and systems, as well as computer‑readable media storing instructions for performing the methods. Prior art universally treats frequency offset as an impairment requiring correction; the present approach exploits offset as a measurement tool, representing a fundamental departure from established practice.

Phase 2 Breakthrough — Closed-form τ/Δf Estimators (Geometric / Algebraic / Hybrid)

Summary of advances
- Implemented three complementary closed-form estimators: geometric (complex-plane linear phase fit), algebraic (polynomial constraints across carriers/retunes), and hybrid (selection/fusion by residual consistency).
- Demonstrated substantial reduction of RMSE relative to traditional methods, with frequency diversity improving performance as carrier spacing increases, and with robustness across 100+ Monte Carlo trials and multiple scenarios.
- Established CRLB‑aware design: spacing of Δf and retune offsets selected via sensitivity/CRLB analysis; online monitoring of RMSE/CRLB ratio to trigger adaptive adjustments.

Key implementation elements
- Geometric estimator: unwrap phase φ(t), linear fit to derive Δf̂ and intercept‑based τ candidates; covariance from residuals informs variance weighting.
- Algebraic estimator: formulate polynomial constraints over multi‑carrier observations, solve for τ and Δf, prune roots by physical and residual‑consistency checks.
- Hybrid estimator: compute both; apply robust selection/fusion (e.g., residual‑weighted, M‑estimator) to handle edge cases and multipath bias.
- Frequency diversity: choose retune spacings to maximize unwrapping margin and improve estimator conditioning subject to bandwidth limits.

Claims mapping (non‑limiting)
- Geometric/algebraic/hybrid estimators: see `patent/CLAIMS_CONSOLIDATED_2025-09-20.md` claims 41–45, 49–52.
- CRLB‑aware selection and monitoring: claims 46, 48, 50–51.
- Frequency diversity and spacing optimization: claims 45–46.

Acknowledgment of Lineage

This invention continues a multi‑generational thread of precision communications innovation, building on prior family contributions (e.g., Ralph Bown’s 1940s work) while establishing a new paradigm that treats frequency offset as a chronometric feature rather than a defect.

Preparation and Priority Intent

Prepared to establish a priority date corresponding to September 19, 2025, for conception and reduction to practice in simulation, with intent to pursue non‑provisional and international filings.

Inventor conception date: [insert date of first documentation]; supporting materials (e.g., lab notebooks, drafts, emails) may be incorporated by reference in the filing record.

Figure Reference Numerals

- (100) Node A; (110) Transmitter at f1; (120) Wireless link / propagation delay τ; (130) Node B; (140) Transmitter at f2=f1+Δf; (150) Beat detector at Δf; (160) Phase extraction; (170) Measurement exchange (τ̂_AB, Δf̂_AB, τ̂_BA, Δf̂_BA); (180) Two‑way resolver (τ_geo, δt).
- (200) Beat samples; (210) Filter at Δf; (220) Sub‑sampling ADC (~2×Δf); (230) Angle and unwrap; (240) Linear fit φ(t)=at+b; (250) Δf̂ from slope; (260) τ candidate from intercept and carrier; (270) τ unwrapping to coarse hint/retunes; (280) τ̂ output.
- (300)–(304) Network nodes; (310) Consensus update and weighting rule.
- (400) RF transceiver (f1 / f2=f1+Δf); (410) Beat/Mixer; (420) Sub‑sampling ADC; (430) DSP; (440) Estimator; (450) Consensus engine; (460) Network/MAC.
