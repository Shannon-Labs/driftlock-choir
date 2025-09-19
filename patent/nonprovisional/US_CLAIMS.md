# Claims — US Non‑Provisional (Condensed 25)

Independent Claims

1. A method for synchronizing a pair of wireless communication nodes, comprising: intentionally generating and maintaining a carrier frequency offset Δf between the nodes; simultaneously transmitting and receiving signals at respective carrier frequencies that differ by Δf so as to form a beat signal at frequency Δf; extracting a phase of the beat signal over an observation window; computing, via a closed‑form estimator applied to the extracted phase, a propagation delay τ and a frequency difference between the nodes; and performing bidirectional measurements and combining forward and reverse delays to determine a geometric delay and a clock bias, wherein the frequency offset Δf is intentionally generated and maintained as a measurement feature rather than being corrected or compensated as an error.

2. A wireless synchronization apparatus, comprising: a programmable transceiver configured to emit and receive signals at carrier frequencies that differ by an intentional offset Δf; an analog‑to‑digital converter configured to sub‑sample the beat signal at a rate approximately twice Δf; and one or more processors configured to extract a beat phase, compute τ and Δf via a closed‑form estimator, and execute a two‑way handshake to separate geometric delay from clock bias.

3. A distributed synchronization method for a network of nodes, comprising: for edges (i, j) in a communication graph, performing pairwise measurements that produce delay and frequency‑difference observations d_ij with variances; and iteratively updating per‑node states x_i = [ΔT_i, Δf_i] according to x_i(k+1) = x_i(k) + ε Σ_j W_ij (d_ij − (x_i(k) − x_j(k))), where W_ij = diag(1/σ²_τ, 1/σ²_Δf) are inverse‑variance weights and ε is selected from a Laplacian spectrum of the graph, achieving masterless synchronization without GPS, atomic clocks, or external timing references.

4. A non‑transitory computer‑readable medium storing instructions that, when executed by one or more processors, cause a device to perform any of the methods of claims 1 or 3.

Dependent Claims

5. The method of claim 1, wherein Δf is between approximately 1 MHz and approximately 10 MHz and is approximately one‑tenth of a channel bandwidth.

6. The method of claim 1, wherein an observation window is between approximately 10 microseconds and approximately 20 microseconds, and a sub‑sampling rate is approximately 2×Δf.

7. The method of claim 1, further comprising generating a coarse wideband preamble to obtain an unambiguous delay hint for phase unwrapping.

8. The method of claim 1, further comprising performing multi‑carrier retunes to produce synthetic wavelengths for robust unwrapping, including Chinese Remainder constructions.

9. The method of claim 1, wherein τ is computed from a linear fit of unwrapped beat phase without iterative search.

10. The method of claim 1, wherein bidirectional measurements compute a clock bias δt = 0.5(τ̂_AB − τ̂_BA) and a geometric delay τ_geo = 0.5(τ̂_AB + τ̂_BA).

11. The method of claim 1, further comprising weighting multi‑carrier τ estimates proportional to carrier frequency squared.

12. The apparatus of claim 2, wherein the analog‑to‑digital converter includes anti‑alias filtering and dynamic range sufficient to capture the beat signal at Δf, and a transceiver frequency resolution of at least 1 Hz is provided.

13. The apparatus of claim 2, wherein oscillators are temperature‑compensated crystal oscillators with stability between approximately 2 and approximately 20 parts per million.

14. The method of claim 1, wherein Δf avoids integer relationships with sampling clocks to minimize aliasing.

15. The method of claim 1, wherein Δf is adaptively selected based on signal‑to‑noise ratio and bandwidth to optimize estimator variance.

16. The method of claim 1, further comprising scheduling measurement offsets Δt to average jitter and oscillator phase noise.

17. The method of claim 3, further comprising enforcing zero‑mean constraints across nodes for both timing and frequency states; supporting asynchronous updates; and applying Chebyshev polynomial acceleration to reduce convergence time.

18. The method of claim 3, wherein ε = c/λ_max(L) with 0 < c < 1 provides a stability margin and rapid convergence.

19. The method of claim 3, wherein consensus achieves substantially improved network‑wide precision relative to conventional microsecond‑class methods under representative signal‑to‑noise ratio and connectivity.

20. The apparatus of claim 2, maintaining synchronization performance under up to approximately 20% packet loss and in the presence of node mobility.

21. The apparatus of claim 2, operating in industrial, scientific, and medical bands and/or in licensed cellular spectrum.

22. The method of claim 1, further comprising multipath mitigation via super‑resolution separation and weighting of direct‑path estimates.

23. The method of claim 1, further comprising super‑resolution spectral methods to separate multipath returns.

24. The method of claim 1, comprising estimator variance floors to stabilize unwrapping at low SNR.

25. A method substantially as described herein with reference to the accompanying description and any one of the examples.
