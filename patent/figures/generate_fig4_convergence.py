#!/usr/bin/env python3
"""
Generate a convergence plot (timing RMS vs. iteration) for inclusion as Fig. 4.
This is illustrative; replace synthetic data with simulation outputs as needed.
"""
import numpy as np
import matplotlib.pyplot as plt

def main():
    iters = np.arange(0, 101)
    # Synthetic geometric convergence with mild noise
    rms_ps_sync = 5000 * (0.90 ** iters) + np.random.default_rng(7).normal(0, 5, size=len(iters))
    rms_ps_async = 5000 * (0.93 ** iters) + np.random.default_rng(11).normal(0, 6, size=len(iters))

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(iters, rms_ps_sync, label="Synchronous consensus", color="#1f77b4")
    ax.plot(iters, rms_ps_async, label="Asynchronous consensus", color="#ff7f0e", linestyle="--")
    ax.axhline(100, color="#2ca02c", linestyle=":", label="100 ps threshold")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Timing RMS (ps)")
    ax.set_title("Consensus Convergence (Illustrative)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    out_png = "patent/figures/fig4_convergence.png"
    out_svg = "patent/figures/fig4_convergence.svg"
    plt.savefig(out_png, dpi=180)
    plt.savefig(out_svg)
    print(f"Wrote {out_png} and {out_svg}")

if __name__ == "__main__":
    main()
