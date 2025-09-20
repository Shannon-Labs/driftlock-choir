# Consolidated Claim Set — Driftlock Chronometric Interferometry (2025-09-20)

This file consolidates and expands the independent and dependent claims to capture the full innovation reflected in the current codebase and design docs. Drafted for direct inclusion in the provisional filing.

## Independent Claims

1. A method for synchronizing a pair of wireless communication nodes, comprising: intentionally generating a carrier frequency offset Δf between said nodes; forming a beat signal at frequency Δf from simultaneous transmission and reception; extracting beat phase over a microsecond-scale window; determining propagation delay τ and frequency difference from a closed-form estimator of said beat phase; and performing forward and reverse measurements to resolve clock bias and obtain geometric delay.

2. A wireless synchronization system comprising: first and second transceiver nodes configured to transmit at carrier frequencies differing by an intentional offset Δf; a beat-detection path per node including a sub-sampling analog-to-digital converter; one or more digital signal processors configured to extract beat phase and compute τ and Δf via closed-form estimation; and a protocol to execute bidirectional measurements to separate geometric delay from clock bias.

3. A distributed synchronization method for a network of nodes, comprising: performing pairwise measurements that yield delay and frequency-difference observations with associated variances; and iteratively updating per-node states x_i = [ΔT_i, Δf_i] via a variance-weighted consensus rule x_i(k+1) = x_i(k) + ε Σ_j W_ij (d_ij − (x_i(k) − x_j(k))), wherein W_ij is a positive-definite 2×2 weight matrix including inverse variances for timing and frequency components, and ε is selected from a graph Laplacian spectrum to ensure stability and rapid convergence.

4. A non-transitory computer-readable medium storing instructions that, when executed by one or more processors, cause a device to perform any of the methods of claims 1 or 3.

5. A calibration method for wireless chronometric interferometry, comprising: estimating per-node transmit and receive hardware delays; computing a reciprocity bias for a link as the difference of residual transmit and receive delays; and compensating measured τ and Δf by subtracting a calibration offset formed from said per-node delays.

## Dependent Claims

6. The method of claim 1, wherein Δf is maintained as a synchronization feature and not corrected as an impairment.

7. The method of claim 1, wherein Δf is between 1 MHz and 10 MHz and selected approximately as channel bandwidth divided by ten.

8. The method of claim 1, wherein the sub-sampling rate is an integer multiple of Δf of at least two, and decimation targets approximately four times Δf to avoid Nyquist ambiguity while minimizing sampling rate.

9. The method of claim 1, further comprising dynamic band-pass filtering centered at |Δf| with a relative bandwidth selected as a function of |Δf| and sampling rate.

10. The method of claim 1, wherein the closed-form estimator unwraps the beat phase, fits a linear model φ(t)=a t + b, determines Δf̂ from slope a/(2π), and computes a τ candidate from intercept b and known carrier frequency, followed by ambiguity resolution.

11. The method of claim 10, further comprising computing an estimator covariance from linear-fit residuals and using the covariance to derive per-parameter variances.

12. The method of claim 1, further comprising generating a coarse wideband preamble, correlating to obtain an unambiguous delay hint, and performing sub-sample peak interpolation; the preamble comprising a Zadoff–Chu sequence optionally windowed (e.g., Kaiser) for spectral shaping.

13. The method of claim 1, further comprising multi-carrier retunes producing a synthetic wavelength for robust unwrapping, including Chinese Remainder Theorem constructions and selection of retune offsets to maximize unwrapping margin subject to bandwidth constraints.

14. The method of claim 13, further comprising weighting multi-carrier τ estimates and reducing effective timing variance by aggregation across carriers.

15. The system of claim 2, wherein the sub-sampling ADC model accounts for finite effective number of bits (ENOB), quantization noise, and aperture jitter.

16. The system of claim 2, wherein each node includes a temperature-compensated crystal oscillator (TCXO) and an unlocked local oscillator model incorporating Allan-deviation-based noise, temperature coefficients, and aging.

17. The system of claim 2, further comprising IQ imbalance and DC offset compensation for quadrature paths using imbalance matrices for transmit and inverse compensation for receive.

18. The method of claim 1, wherein multipath is modeled by at least a two-ray channel or a tapped-delay-line baseband model, and a narrowband channel response at the carrier frequency is used to estimate a phase bias, which is compensated in τ estimation.

19. The method of claim 1, further comprising scheduling measurement offsets Δt across repeated observations to average oscillator phase noise and jitter.

20. The method of claim 3, wherein W_ij includes per-parameter inverse variances diag(1/σ²_τ, 1/σ²_Δf) and optionally off-diagonal terms for parameter coupling.

21. The method of claim 3, wherein ε = c / λ_max(L) with 0 < c < 1 provides a spectral stability margin; λ_max(L) being the largest Laplacian eigenvalue of the communication graph.

22. The method of claim 3, further comprising Chebyshev polynomial acceleration to reduce convergence time below 5 milliseconds for networks of at least 50 nodes.

23. The method of claim 3, further comprising asynchronous node updates selected at random with zero-mean projection enforced on timing and frequency states each iteration.

24. The method of claim 3, further comprising online weight adaptation using running variance estimators computed from measurement residuals per directed edge using a numerically stable algorithm.

25. The method of claim 3, further comprising convergence monitoring and iteration-count prediction using Laplacian spectral quantities including algebraic connectivity.

26. The method of claim 5, wherein calibration modes include: none, loopback measurement-based with stochastic error, and perfect calibration; and wherein a reciprocity bias is computed as the difference between residual transmit and receive hardware delays and applied to measured τ.

27. The system of claim 2, wherein a media access control (MAC) schedule deterministically staggers forward and reverse measurements and employs CSMA/CA backoff, response timeouts, acknowledgments, and prioritization to sustain synchronization traffic.

28. The method of claim 3, applied to random geometric or clustered graph topologies with connectivity enforced by proximity thresholds.

29. The system of claim 2, maintaining <2 ns RMS synchronization accuracy under up to 20% packet loss and node mobility up to 20 meters per second.

30. The system of claim 2, operating in industrial, scientific, and medical (ISM) bands and in licensed cellular spectrum, with Δf chosen to avoid interferers and sampling-clock spurs.

31. The method of claim 12, wherein a variance floor is imposed on the coarse estimator to stabilize unwrapping at low signal-to-noise ratios.

32. The method of claim 10, further comprising edge orientation metadata to disambiguate directed measurements when constructing pairwise observations.

33. The method of claim 1, wherein ADC decimation and baseband sampling rates are adapted dynamically as functions of |Δf| and configuration thresholds to bound computational load while preserving estimator fidelity.

34. The method of claim 1, wherein measurement and fusion are extended across carriers from audio through optical regimes using the same beat-phase principles.

35. The method of claim 3, wherein consensus weights are updated online from residual statistics and bounded per-iteration state increments are enforced for robustness.

36. The system of claim 2, further comprising local two-state Kalman filters at nodes to track clock bias and frequency offset while fusing neighbor pseudo-measurements with associated covariances.

37. The method of claim 18, wherein multipath mitigation includes super-resolution spectral separation and down-weighting of non-line-of-sight components in consensus.

38. The method of claim 1, wherein down-sampled beat samples are limited for visualization and diagnostic trace capture without impacting estimator outputs.

39. The method of claim 3, exhibiting practical O(log N) convergence scaling with network size under representative connectivity and weighting.

40. A computer-readable medium storing instructions that implement any of claims 1–39.

## Phase 2 — Closed-form Estimators (Geometric / Algebraic / Hybrid)

41. The method of claim 1, wherein a geometric closed-form estimator operates in the complex plane by fitting the unwrapped phasor e^{jφ(t)} to a linear phase model, extracting Δf̂ from phase slope and τ candidates from intercept-based constraints at the carrier frequency.

42. The method of claim 1, wherein an algebraic closed-form estimator formulates polynomial constraints across multiple carrier frequencies or retune offsets and solves for τ and Δf by root-finding on said polynomial, with candidate selection based on physical-consistency checks and residual minimization.

43. The method of claim 1, further comprising a hybrid estimator that computes both geometric and algebraic solutions and selects or fuses solutions according to a robust cost function derived from residuals, covariance consistency, and alias-consistency across retunes.

44. The method of claim 43, wherein hybrid fusion employs residual-based weights or M-estimation to downweight outlier candidates and improve robustness at low SNR or under multipath.

45. The method of claim 1, wherein frequency diversity is achieved by selecting multiple carrier spacings or retune offsets and performance improves monotonically with larger aggregate spacing subject to bandwidth constraints.

46. The method of claim 1, wherein Δf and retune spacing are selected via a Cramér–Rao lower bound sensitivity analysis to minimize estimator variance for τ and Δf under given SNR and observation time.

47. The method of claim 1, further comprising estimator selection logic that detects edge cases and gracefully degrades by switching between geometric and algebraic estimators based on residual thresholds, condition numbers, or variance floors.

48. The method of claim 11, wherein covariance from the closed-form fit is used to quantify an RMSE/CRLB ratio and to trigger retune re-measurements or Δf adjustments when ratios exceed configurable thresholds.

49. The method of claim 1, wherein candidate roots from algebraic constraints are pruned using cross-carrier consistency checks and an orientation-aware directed-edge measurement model.

50. The method of claim 1, wherein achievable RMSE is bounded within a constant factor of the CRLB determined by configuration, with said factor reduced by increasing retune spacing, observation time, or SNR.

51. The method of claim 1, wherein estimator performance is improved by joint coupling of τ and Δf in the information model rather than treating Δf as a nuisance parameter to be eliminated.

52. The system of claim 2, wherein a performance monitor visualizes residuals, phase unwrap consistency, RMSE/CRLB ratios, and convergence metrics for operational assurance and online tuning of Δf and retunes.
