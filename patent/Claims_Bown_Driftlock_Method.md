# Claims — Bown Driftlock Method

This file mirrors the claims section from the updated provisional for quick review.

Independent Claims

1. A method for synchronizing a pair of wireless communication nodes, comprising: intentionally generating a carrier frequency offset Δf between said nodes; forming a beat signal at frequency Δf from simultaneous transmission and reception; extracting beat phase over a measurement window; determining propagation delay τ and frequency offset from a closed-form estimator of said beat phase; and performing bidirectional measurements to resolve clock bias and obtain geometric delay.

2. A wireless synchronization system comprising: first and second transceiver nodes configured to transmit at carrier frequencies differing by an intentional offset Δf; a beat detection path per node including a sub-sampling analog-to-digital converter operating at approximately twice Δf; a digital signal processor configured to extract beat phase and compute τ and Δf via closed-form estimation; and a protocol to execute forward and reverse measurements to separate geometric delay from clock bias.

3. A distributed synchronization method for a network of nodes, comprising: performing pairwise measurements that yield delay and frequency-difference observations; and iteratively updating per-node states x_i = [ΔT_i, Δf_i] via a variance-weighted consensus rule x_i(k+1) = x_i(k) + ε Σ_j W_ij (d_ij − (x_i(k) − x_j(k))), wherein W_ij = diag(1/σ^2_τ, 1/σ^2_Δf) and ε is chosen from the Laplacian spectrum.

4. A computer-readable medium storing instructions that, when executed, cause a device to perform the method of claim 1 or 3.

5. An application-specific method applying any of claims 1–4 to cellular base stations, distributed radar nodes, high-frequency trading timestamping systems, quantum networks, or IoT meshes.

Dependent Claims (enumerated)

6. The method of claim 1, wherein Δf is maintained as a synchronization feature and not corrected as an impairment.

7. The method of claim 1, wherein Δf is between 1 MHz and 10 MHz and selected approximately as channel bandwidth/10.

8. The method of claim 1, wherein the measurement window is between 10 µs and 20 µs and the sub-sampling rate is approximately 2×Δf.

9. The system of claim 2, further comprising a coarse wideband preamble to produce an unambiguous delay hint for phase unwrapping.

10. The system of claim 2, wherein multi-carrier retunes produce a synthetic wavelength enabling robust unwrapping by Chinese Remainder Theorem techniques.

11. The method of claim 3, wherein W_ij uses inverse measurement variances 1/σ^2 and wherein ε = c/λ_max(L) with 0<c<1.

12. The method of claim 3, further comprising Chebyshev polynomial acceleration to reduce convergence time below 5 ms for 50 nodes.

13. The system of claim 2, wherein each node uses a temperature-compensated crystal oscillator (TCXO) with stability between 2–20 ppm and maintains <2 ns RMS synchronization without GPS or atomic clocks.

14. The method of claim 1, wherein the estimator is closed-form and non-iterative based on a linear fit to unwrapped phase.

15. The method of claim 1, wherein bidirectional measurements compute clock bias as one-half the difference between forward and reverse delays and geometric delay as one-half the sum.

16. The method of claim 3, enforcing zero-mean constraints across nodes for both timing and frequency states.

17. The method of claim 1, wherein Δf is adaptively selected based on SNR and bandwidth to optimize estimator variance.

18. The method of claim 1, wherein Δf avoids integer relationships with sampling clocks to minimize aliasing.

19. The method of claim 6, wherein multi-carrier unwrapping uses Chinese Remainder Theorem constructions.

20. The method of claim 1, further comprising scheduling measurement offsets Δt to average jitter and oscillator phase noise.

21. The method of claim 3, wherein asynchronous node selection follows a uniform distribution over nodes.

22. The method of claim 12, wherein the Chebyshev polynomial degree is at least two.

23. The method of claim 3, wherein the spectral step size uses a margin c between 0.6 and 0.9.

24. The method of claim 3, wherein zero-mean projection is enforced on both ΔT and Δf at each iteration.

25. The system of claim 2, maintaining performance under node mobility up to approximately 20 m/s.

26. The system of claim 2, maintaining <2 ns RMS under up to approximately 20% packet loss.

27. The method of claim 1, wherein multipath is modeled by at least a two-ray channel and mitigated by weighting strategies.

28. The method of claim 1, further comprising super-resolution spectral methods to separate multipath returns.

29. The system of claim 2, supporting calibration modes selected from: none, loopback, and perfect.

30. The system of claim 2, wherein the sub-sampling ADC includes anti-alias filtering and dynamic range sufficient for Δf capture.

31. The system of claim 2, wherein the transceiver frequency resolution is at least 1 Hz.

32. The system of claim 2, operating in industrial, scientific, and medical (ISM) bands.

33. The system of claim 2, operating in licensed cellular spectrum.

34. The method of claim 3, applied to random geometric graph topologies.

35. The method of claim 3, exhibiting O(log N) practical convergence scaling with network size.

36. The method of claim 3, further comprising convergence monitoring using Laplacian eigenvalue telemetry.

37. The method of claim 10, wherein synthetic wavelength selection maximizes unwrapping margin subject to bandwidth constraints.

38. The method of claim 1, wherein carrier frequencies span audio through optical regimes, with the same beat-phase principles applied.

39. A computer-readable medium storing instructions that implement any of claims 1–38.

40. The method of claim 3, wherein consensus weights are updated online from measurement residual statistics.

41. The method of claim 1, further comprising coarse estimator variance floors to stabilize unwrapping at low SNR.

42. The method of claim 3, wherein nodes enforce per-iteration bounds on state increments for robustness.

43. The system of claim 2, wherein a MAC schedule deterministically staggers forward/reverse measurements.

44. The method of claim 1, wherein Δf is selected to avoid known interferers and spurs in the receiver chain.

45. The method of claim 3, wherein edge orientation metadata is used to disambiguate directed measurements.

Phase 2 – Closed-form Estimators (Additions)

46. The method of claim 1, wherein a geometric closed-form estimator extracts Δf and τ by linear phase fitting in the complex plane.

47. The method of claim 1, wherein an algebraic closed-form estimator solves polynomial constraints defined across multiple carrier frequencies or retune offsets to recover τ and Δf.

48. The method of claim 1, wherein a hybrid approach computes both geometric and algebraic candidates and selects or fuses them using residual-based consistency checks.

49. The method of claim 1, wherein Δf and retune spacings are selected using CRLB-based sensitivity analysis to minimize estimator variance, with performance improving with larger aggregate spacing subject to bandwidth limits.

50. The method of claim 1, wherein RMSE is maintained within a bounded multiple of the CRLB and monitored online to trigger adaptive Δf or retune adjustments.
