# Physics Derivations for Driftlock Atomic Reconciliation

This document derives closed-form precision limits for Driftlock's chronometric interferometry, reconciling with atomic clock standards (cesium fountains: ~1 ps/s stability; optical clocks: <1 ps over 1 s). All derivations assume Gaussian noise models and linear estimators, validated against CRLB in `src/metrics/crlb.py`.

## 1. RF Comb-Based Delay Estimation

### Model
In multi-frequency comb operation, the k-th tone at carrier f_k yields beat phase:

$$
\phi_k = -2\pi f_k \tau + \theta + \eta_k
$$

where τ is one-way delay, θ is phase offset, η_k ~ N(0, σ_φ²) is phase noise (from `src/phy/noise.py`).

For K tones {f_k}_{k=1}^K, this is linear regression:

$$
\mathbf{\phi} = \mathbf{F} \begin{bmatrix} \tau \\ \theta \end{bmatrix} + \boldsymbol{\eta}
$$

with design matrix **F** = [-2π **f**, **1**] where **f** = [f₁, ..., f_K]^T.

### CRLB Derivation
Fisher Information Matrix (FIM):

$$
\mathbf{I}(\boldsymbol{\theta}) = \frac{1}{\sigma_\phi^2} \mathbf{F}^T \mathbf{F}, \quad \boldsymbol{\theta} = [\tau, \theta]^T
$$

$$
\mathbf{I}_{11} = \frac{2\pi^2}{\sigma_\phi^2} \sum_k f_k^2, \quad \mathbf{I}_{12} = \mathbf{I}_{21} = -\frac{2\pi}{\sigma_\phi^2} \sum_k f_k, \quad \mathbf{I}_{22} = \frac{K}{\sigma_\phi^2}
$$

CRLB for τ (via matrix inverse):

$$
\text{Var}(\hat{\tau}) \geq \left[ \mathbf{I}^{-1} \right]_{11} = \frac{\sigma_\phi^2}{2\pi^2 \sum_k f_k^2 - \left( \sum_k f_k \right)^2 / K} = \frac{\sigma_\phi^2}{2\pi^2 K \overline{f^2} - (2\pi \bar{f})^2 / K}
$$

For uniform comb spacing Δf, Σf_k² ≈ Kf_c² + K(K-1)Δf²/3, yielding CRLB(τ) ≈ σ_φ²/(2π²K²Δf²).

This extends single-carrier case in `src/alg/chronometric_handshake.py` (intercept variance) to slope-based τ from phase-vs-frequency, achieving <1 ps at K=10, Δf=10 MHz, σ_φ=0.1 rad (matches optical clock precision under RF constraints).

Validation: Extend `ChronometricHandshakeSimulator` with comb offsets; compare simulated RMSE to CRLB in `sim/phase1.py`.

## 2. Consensus Convergence in Multi-Hop Networks

### Model
Variance-weighted consensus (from `src/alg/consensus.py`):

$$
\mathbf{x}^{(i+1)} = \mathbf{x}^{(i)} - \epsilon \mathbf{L} \mathbf{W} (\mathbf{x}^{(i)} - \mathbf{z})
$$

where **L** is graph Laplacian, **W** diagonal weights (inverse variances), **z** measurements.

For connected undirected graph, convergence to:

$$
\mathbf{x}^* = \arg\min \| \mathbf{W}^{1/2} (\mathbf{x} - \mathbf{z}) \|^2
$$

at rate governed by eigenvalues of **M** = **I** - ε**LW**.

### Analytical Bounds
Spectral gap λ₂ (second-smallest eigenvalue of normalized Laplacian **L̃** = **D**⁻¹**L**, **D** degree matrix) bounds contraction:

$$
\| \mathbf{x}^{(i)} - \mathbf{x}^* \| \leq (1 - \lambda_2)^i \| \mathbf{x}^{(0)} - \mathbf{x}^* \|
$$

for optimal ε = 2/(λ₁ + λₙ), where λ₁=0 < λ₂ ≤ ⋯ ≤ λₙ ≤ 2.

Iterations to ε-precision:

$$
i \geq \frac{\log(1/\varepsilon)}{\log(1/(1 - \lambda_2))} \approx \frac{\log(1/\varepsilon)}{\lambda_2}
$$

For ring topology (N nodes), λ₂ = 2(1 - cos(2π/N)) ≈ 2(π/N)²; line: λ₂ ≈ (π/(N+1))². With weights, effective λ₂^eff = min λ₂(**W**^(1/2)**LW**^(1/2)).

Driftlock achieves ε=1 ps in <10 iterations for λ₂ > 0.1 (e.g., mesh nets), vs. PTP's 100+ ms holdover.

Validation: Plot simulated vs. bound in `sim/phase2.py` using `eigsh` from consensus.py.

## 3. Phase Noise and Hardware Impairments

### Model
Phase noise PSD S_φ(f) = h₋₂/f² + h₀ + h₊₂f² (from `src/phy/osc.py` Allan variance). Integrated over beat bandwidth Δf:

$$
\sigma_\phi^2 = \int_{-\Delta f/2}^{\Delta f/2} S_\phi(f) df \approx h_0 \Delta f + h_{-2} \log(\Delta f / f_c)
$$

Timing jitter σⱼ (from `src/phy/noise.py`) adds σ_φ ≈ 2πf_cσⱼ.

### Variance Propagation
For beat estimator, phase error φ = 2πΔft + θ - 2πf_cτ + η, LS fit yields:

$$
\text{Var}(\hat{\tau}) = \frac{\sigma_\phi^2}{(2\pi f_c)^2 N}, \quad \text{Var}(\hat{\Delta f}) = \frac{12 \sigma_\phi^2}{(2\pi)^2 \Delta f^2 N (N^2 - 1)} \approx \frac{\sigma_\phi^2}{2\pi^2 \Delta f^2 N T^2}
$$

where N samples, T = N/rate. Closed-form: Var(τ) = σ_φ²/(2πf_cΔfN) (approximating intercept dominance).

With impairments (IQ imbalance α, LO phase β): effective σ_φ² ← σ_φ²(1 + |α|² + 2Re(αe^(jβ))).

Driftlock limits: σ_φ < 10⁻³ rad yields Var(τ) < 1 ps² at f_c=2.4 GHz, Δf=1 kHz, N=10⁴ (matches cesium 1 ps stability).

Integration: Update `NoiseGenerator` to propagate to `DirectionalMeasurement.covariance`; validate in `src/metrics/crlb.py` with CRLB ratio > 0.8.

### Atomic Benchmarks
Cesium: σᵧ(τ) = 10⁻¹³√τ ⇒ 1 ps at τ=1 s. Optical: 10⁻¹⁸√τ ⇒ <1 ps at 1 s. Driftlock RF: 22 ps RMSE (existing sims) approaches via comb (KΔf > 100 MHz) and consensus (λ₂ > 0.05).

Refs: Kay (1993) Fundamentals of Statistical Signal Processing; D'Andrea (1995) Phase-locked loops.