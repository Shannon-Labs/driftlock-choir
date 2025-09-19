# Drawings Plan and Sources

This file describes the figures to be submitted and provides Mermaid sources (and a convergence plot script). For USPTO filing, export these as black/white line art with reference numerals and captions.

Index

- Fig. 1 — Two-node beat generation and handshake (file: figures/fig1_beat_handshake.svg)
- Fig. 2 — Phase extraction and closed-form estimation pipeline (file: figures/fig2_phase_estimator.svg)
- Fig. 3 — Network topology and variance-weighted consensus (file: figures/fig3_network_consensus.svg)
- Fig. 4 — Convergence vs. iteration (timing RMS) (files: figures/fig4_convergence.png, figures/fig4_convergence.svg)
- Fig. 5 — Node/system block diagram (transceiver, sub-sampling ADC, DSP) (file: figures/fig5_system_block.svg)

Export Guidance

- Mermaid to SVG: Use `mmdc` (mermaid-cli) if preferred; line-art SVGs are included and ready.
- Ensure labels become reference numerals in the USPTO drawings and are described in the specification.

---

## Fig. 1 — Two-node beat generation and handshake

```mermaid
flowchart LR
    A[Node A: TX at f1] -- wireless channel (τ_AB) --> B[Node B: TX at f2=f1+Δf]
    B -- downconvert/mix --> BF1[(Beat @ Δf)]
    A -- downconvert/mix --> BF2[(Beat @ Δf)]
    BF1 -- phase over T --> E1[Phase Extractor]
    BF2 -- phase over T --> E2[Phase Extractor]
    E1 --> P1[τ̂_AB, Δf̂_AB]
    E2 --> P2[τ̂_BA, Δf̂_BA]
    P1 & P2 --> GEO[Two-way Resolver: τ_geo = 0.5(τ̂_AB+τ̂_BA), δt = 0.5(τ̂_AB-τ̂_BA)]
```

## Fig. 2 — Phase extraction and closed-form estimator

```mermaid
flowchart LR
    S[Beat Samples (complex)] --> F[Bandpass/LP @ Δf]
    F --> SS[Sub-sample (~2×Δf)]
    SS --> PH[Angle + Unwrap]
    PH --> LS[Linear Fit φ(t) = 2πΔf t + b]
    LS --> DF[Δf̂ from slope]
    LS --> TAU[τ candidate from intercept and carrier]
    TAU --> UW[Unwrap τ to hint]
    UW --> OUT[τ̂]
```

## Fig. 3 — Network topology and variance-weighted consensus

```mermaid
flowchart TB
    subgraph G[Graph of N Nodes]
      N1((1)) ---|W_12| N2((2))
      N1 ---|W_13| N3((3))
      N2 ---|W_23| N3
      N3 ---|W_34| N4((4))
      N2 ---|W_25| N5((5))
    end
    M[Edge Measurements d_ij = [ΔT_ij, Δf_ij]] --> C[Update: x_i(k+1)=x_i+ε Σ W_ij(d_ij - (x_i-x_j))]
    C --> ZM[Zero-mean projection]
    ZM --> X[State x = [ΔT, Δf] per node]
```

## Fig. 4 — Convergence vs. iteration (timing RMS)

- Source: `figures/generate_fig4_convergence.py` (matplotlib)
- Output: `figures/fig4_convergence.png`

Usage:

```bash
python patent/figures/generate_fig4_convergence.py
```

## Fig. 5 — Node/system block diagram

```mermaid
flowchart TB
    RF[Programmable RF Transceiver]\n(f1 / f2=f1+Δf) --> MIX[Beat Detection / Mixing]
    MIX --> ADC[Sub-sampling ADC (~2×Δf)]
    ADC --> DSP[Digital Signal Processing]
    DSP --> EST[Closed-form τ̂, Δf̂]
    EST --> CONS[Consensus Engine (1/σ² weighting, spectral ε)]
    CONS --> NET[Network Interface / MAC]
```
