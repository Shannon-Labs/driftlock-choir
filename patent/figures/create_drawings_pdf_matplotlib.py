#!/usr/bin/env python3
"""Create drawings.pdf from PNG figures using matplotlib."""

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.image as mpimg
import os

def main():
    output_path = "/Volumes/VIXinSSD/driftlock choir/dist/patent_packet/drawings.pdf"
    figures_dir = "/Volumes/VIXinSSD/driftlock choir/dist/patent_packet/figures/png"

    # List of figure files in order
    figure_files = [
        "fig1_beat_handshake.png",
        "fig2_phase_estimator.png",
        "fig3_network_consensus.png",
        "fig4_convergence.png",
        "fig5_system_block.png"
    ]

    # Create multi-page PDF
    with PdfPages(output_path) as pdf:
        for fig_file in figure_files:
            fig_path = os.path.join(figures_dir, fig_file)

            if not os.path.exists(fig_path):
                print(f"Warning: {fig_file} not found, skipping")
                continue

            # Create figure and load image
            fig = plt.figure(figsize=(8.5, 11))  # Letter size
            ax = fig.add_axes([0, 0, 1, 1])  # Full page
            ax.axis('off')

            # Load and display image
            img = mpimg.imread(fig_path)
            ax.imshow(img, aspect='auto')

            # Save page to PDF
            pdf.savefig(fig, dpi=600, bbox_inches='tight')
            plt.close(fig)

            print(f"Added {fig_file} to PDF")

    print(f"\nCreated {output_path}")

    # Get file size
    size_kb = os.path.getsize(output_path) / 1024
    print(f"File size: {size_kb:.1f} KB")

if __name__ == "__main__":
    main()