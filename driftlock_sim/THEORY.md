# Choir Simulation Theory (Brief)

We synthesize a complex baseband multi‑carrier comb around 0 Hz with spacing
Δf and M carriers. In "missing‑fundamental" mode we omit the DC bin and any
explicit tone at Δf; nevertheless, the envelope e(t)=|x(t)|^2 exhibits
spectral lines at Δf and pairwise differences due to quadratic mixing.

Coherent path: Estimate per‑tone phasors X_k via short FFT/Goertzel, unwrap
phase across f_k, and fit ϕ_k ≈ −2π f_k τ + ϕ0 by weighted least squares
(WLS). The slope gives τ^ = −(1/2π) dϕ/df, with a CI from (XᵀWX)⁻¹.

Aperture path: Form e(t), remove DC, FFT to baseband, locate Δf spike(s)
by matched comb/peak‑search and compute an "env SNR". Both paths should
agree within tolerance across SNR and M.

Channel/impairments: AWGN; tapped delay h(t)=∑α_i δ(t−τ_i); CFO/SCO;
phase noise via discrete‑time Wiener phase; Rapp PA AM/AM nonlinearity;
optional square‑law/envelope branch to emulate an aperture. Cyclostationary
p(t) gating creates spectral correlation replicas at α=±Δf.

CRLB (order‑of‑magnitude): std(τ^) ≳ (2π Beff / (8 SNR))⁻¹. We report
RMSE/CRLB to flag suspect results.

