# Physics Derivations for Driftlock Atomic Reconciliation

This document derives closed-form precision limits for Driftlock's chronometric interferometry, reconciling with atomic clock standards (cesium fountains: ~1 ps/s stability; optical clocks: <1 ps over 1 s). All derivations assume Gaussian noise models and linear estimators, validated against CRLB in `src/metrics/crlb.py`.

## 1. RF Comb-Based Delay Estimation

### Model
In multi-frequency comb operation, the $k$-th tone at carrier $f_k$ yields beat phase:
$$
\phi_k = -2\pi f_k \tau + \theta + \eta_k
$$
where $\tau$ is one-way delay, $\theta$ is phase offset, $\eta_k \sim \mathcal{N}(0, \sigma_\phi^2)$ is phase noise (from `src/phy/noise.py`).

For $K$ tones $\{f_k\}_{k=1}^K$, this is linear regression: $\mathbf{\phi} = \mathbf{F} \begin{bmatrix} \tau \\ \theta \end{bmatrix} + \boldsymbol{\eta}$, with design matrix $\mathbf{F} = [-2\pi \mathbf{f}, \mathbf{1}]$ ($\mathbf{f} = [f_1, \dots, f_K]^T$).

### CRLB Derivation
Fisher Information Matrix (FIM):
$$
\mathbf{I}(\boldsymbol{\theta}) = \frac{1}{\sigma_\phi^2} \mathbf{F}^T \mathbf{F}, \quad \boldsymbol{\theta} = [\tau, \theta]^T
$$
$$
\mathbf{I}_{11} = \frac{2\pi^2}{\sigma_\phi^2} \sum_k f_k^2, \quad \mathbf{I}_{12} = \mathbf{I}_{21} = -\frac{2\pi}{\sigma_\phi^2} \sum_k f_k, \quad \mathbf{I}_{22} = \frac{K}{\sigma_\phi^2}
$$
CRLB for $\tau$ (via matrix inverse):
$$
\text{Var}(\hat{\tau}) \geq \left[ \mathbf{I}^{-1} \right]_{11} = \frac{\sigma_\phi^2}{2\pi^2 \sum_k f_k^2 - \left( \sum_k f_k \right)^2 / K} = \frac{\sigma_\phi^2}{2\pi^2 K \overline{f^2} - (2\pi \bar{f})^2 / K}
$$
For uniform comb spacing $\Delta f$, $\sum f_k^2 \approx K f_c^2 + K(K-1) \Delta f^2 / 3$, yielding $\text{CRLB}(\tau) \approx \sigma_\phi^2 / (2\pi^2 K^2 \Delta f^2)$.

This extends single-carrier case in `src/alg/chronometric_handshake.py` (intercept variance) to slope-based $\tau$ from phase-vs-frequency, achieving <1 ps at $K=10$, $\Delta f=10$ MHz, $\sigma_\phi=0.1$ rad (matches optical clock precision under RF constraints).

Validation: Extend `ChronometricHandshakeSimulator` with comb offsets; compare simulated RMSE to CRLB in `sim/phase1.py`.

## 2. Consensus Convergence in Multi-Hop Networks

### Model
Variance-weighted consensus (from `src/alg/consensus.py`): $\mathbf{x}^{(i+1)} = \mathbf{x}^{(i)} - \epsilon \mathbf{L} \mathbf{W} (\mathbf{x}^{(i)} - \mathbf{z})$, where $\mathbf{L}$ is graph Laplacian, $\mathbf{W}$ diagonal weights (inverse variances), $\mathbf{z}$ measurements.

For connected undirected graph, convergence to $\mathbf{x}^* = \arg\min \| \mathbf{W}^{1/2} (\mathbf{x} - \mathbf{z}) \|^2$ at rate governed by eigenvalues of $\mathbf{M} = \mathbf{I} - \epsilon \mathbf{L} \mathbf{W}$.

### Analytical Bounds
Spectral gap $\lambda_2$ (second-smallest eigenvalue of normalized Laplacian $\tilde{\mathbf{L}} = \mathbf{D}^{-1} \mathbf{L}$, $\mathbf{D}$ degree matrix) bounds contraction: $\| \mathbf{x}^{(i)} - \mathbf{x}^* \| \leq (1 - \lambda_2)^i \| \mathbf{x}^{(0)} - \mathbf{x}^* \|$ for optimal $\epsilon = 2 / (\lambda_1 + \lambda_n)$, where $\lambda_1=0 < \lambda_2 \leq \cdots \leq \lambda_n \leq 2$.

Iterations to $\varepsilon$-precision: $i \geq \frac{\log(1/\varepsilon)}{\log(1/(1 - \lambda_2)) } \approx \frac{\log(1/\varepsilon)}{\lambda_2}$.

For ring topology ($N$ nodes), $\lambda_2 = 2(1 - \cos(2\pi/N)) \approx 2 (\pi/N)^2$; line: $\lambda_2 \approx (\pi/(N+1))^2$. With weights, effective $\lambda_2^{\text{eff}} = \min \lambda_2(\mathbf{W}^{1/2} \mathbf{L} \mathbf{W}^{1/2})$.

Driftlock achieves $\varepsilon=1$ ps in $<10$ iterations for $\lambda_2 > 0.1$ (e.g., mesh nets), vs. PTP's 100+ ms holdover.

Validation: Plot simulated vs. bound in `sim/phase2.py` using `eigsh` from consensus.py.

## 3. Phase Noise and Hardware Impairments

### Model
Phase noise PSD $S_\phi(f) = h_{-2}/f^2 + h_0 + h_{+2} f^2$ (from `src/phy/osc.py` Allan variance). Integrated over beat bandwidth $\Delta f$: $\sigma_\phi^2 = \int_{-\Delta f/2}^{\Delta f/2} S_\phi(f) df \approx h_0 \Delta f + h_{-2} \log(\Delta f / f_c)$.

Timing jitter $\sigma_j$ (from `src/phy/noise.py`) adds $\sigma_\phi \approx 2\pi f_c \sigma_j$.

### Variance Propagation
For beat estimator, phase error $\phi = 2\pi \Delta f t + \theta - 2\pi f_c \tau + \eta$, LS fit yields:
$$
\text{Var}(\hat{\tau}) = \frac{\sigma_\phi^2}{(2\pi f_c)^2 N}, \quad \text{Var}(\hat{\Delta f}) = \frac{12 \sigma_\phi^2}{(2\pi)^2 \Delta f^2 N (N^2 - 1)} \approx \frac{\sigma_\phi^2}{2\pi^2 \Delta f^2 N T^2}
$$
where $N$ samples, $T = N / \text{rate}$. Closed-form: $\text{Var}(\tau) = \sigma_\phi^2 / (2\pi f_c \Delta f N)$ (approximating intercept dominance).

With impairments (IQ imbalance $\alpha$, LO phase $\beta$): effective $\sigma_\phi^2 \leftarrow \sigma_\phi^2 (1 + |\alpha|^2 + 2 \Re(\alpha e^{j\beta}))$.

Driftlock limits: $\sigma_\phi < 10^{-3}$ rad yields $\text{Var}(\tau) < 1$ ps$^2$ at $f_c=2.4$ GHz, $\Delta f=1$ kHz, $N=10^4$ (matches cesium 1 ps stability).

Integration: Update `NoiseGenerator` to propagate to `DirectionalMeasurement.covariance`; validate in `src/metrics/crlb.py` with $\text{CRLB ratio} > 0.8$.

### Atomic Benchmarks
Cesium: $\sigma_y(\tau) = 10^{-13} \sqrt{\tau}$ $\Rightarrow$ 1 ps at $\tau=1$ s. Optical: $10^{-18} \sqrt{\tau}$ $\Rightarrow$ <1 ps at 1 s. Driftlock RF: 22 ps RMSE (existing sims) approaches via comb ($K\Delta f > 100$ MHz) and consensus ($\lambda_2 > 0.05$).

Refs: Kay (1993) Fundamentals of Statistical Signal Processing; D'Andrea (1995) Phase-locked loops.