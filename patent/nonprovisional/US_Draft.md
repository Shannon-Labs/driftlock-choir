# US Non‑Provisional Patent Application — Draft

Title: System and Method for Distributed Wireless Synchronization Using Chronometric Interferometry with Intentional Carrier Frequency Offset

Applicant/Assignee: Shannon Labs, Inc. (Delaware C‑Corp)
Inventor: Hunter Bown
Priority Claim: This application claims priority to U.S. Provisional Patent Application filed on September 19, 2025, entitled “System and Method for Distributed Wireless Synchronization Using Chronometric Interferometry with Intentional Carrier Frequency Offset,” the entirety of which is incorporated by reference.

1. Title of the Invention

System and Method for Distributed Wireless Synchronization Using Chronometric Interferometry with Intentional Carrier Frequency Offset

2. Cross‑Reference to Related Applications

See priority claim above.

3. Statement Regarding Federally Sponsored Research or Development

Not applicable.

4. Background of the Invention

(Expand from provisional §Background; maintain “teaching‑away” emphasis. No limiting performance language.)

5. Summary of the Invention

(As in provisional §Summary; intentional Δf feature; closed‑form estimator; two‑way bias resolution; variance‑weighted consensus with spectral step size; optional Chebyshev acceleration.)

6. Brief Description of the Drawings

Figs. 1–5 (see patent/figures/*.svg) with reference numerals listed in the specification.

7. Detailed Description of the Invention

7.1 Theory of Operation (beat‑phase equations and closed‑form τ/Δf).  
7.2 Two‑Way Chronometric Handshake (forward/reverse, δt and τ_geo).  
7.3 Estimation Pipeline (filter, sub‑sample, unwrap, linear fit, multi‑carrier unwrapping, coarse preamble, weighting).  
7.4 Hardware Embodiments (sub‑sampling ADC ~2×Δf, TCXO ranges, calibration modes).  
7.5 Distributed Consensus (weights W_ij=diag(1/σ²_τ,1/σ²_Δf), spectral ε, optional Chebyshev, zero‑mean).  
7.6 Applications and Alternatives (cellular, radar, finance, quantum, IoT; carriers audio→optical; spectrum bands; topologies).  

(Use patent/PROVISIONAL_PATENT_APPLICATION.md as the authoritative source text for expansion.)

8. Claims

(Initialize with the 50‑claim set in patent/PROVISIONAL_PATENT_APPLICATION.md; refine during prosecution.)

9. Abstract

(Use patent/ABSTRACT.txt.)

10. Drawings

Include SVG line‑art and PDF render (dist/patent_packet/drawings.pdf); ensure reference numerals match the specification.

Appendix (Optional)

Prior art positioning memo: patent/Prior_Art_Search_Report.md.

Submission Notes

- Convert this draft to USPTO‑compliant format with page numbers, margins, and line spacing.
- Keep performance examples illustrative, not limiting.
- Verify cross‑references and figure numerals.
