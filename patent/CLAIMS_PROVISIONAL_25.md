# Condensed Claim Set — Provisional (25 claims)

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
