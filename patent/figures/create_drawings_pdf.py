#!/usr/bin/env python3
"""Create drawings.pdf from cleaned PNG figures for USPTO patent filing."""

from PIL import Image
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import os

def main():
    output_path = "/Volumes/VIXinSSD/driftlock/dist/patent_packet/drawings.pdf"
    figures_dir = "/Volumes/VIXinSSD/driftlock/dist/patent_packet/figures/png"

    # List of figure files in order
    figure_files = [
        "fig1_beat_handshake.png",
        "fig2_phase_estimator.png",
        "fig3_network_consensus.png",
        "fig4_convergence.png",
        "fig5_system_block.png"
    ]

    # Create PDF
    c = canvas.Canvas(output_path, pagesize=letter)
    page_width, page_height = letter

    for fig_file in figure_files:
        fig_path = os.path.join(figures_dir, fig_file)

        if not os.path.exists(fig_path):
            print(f"Warning: {fig_file} not found, skipping")
            continue

        # Open image to get dimensions
        img = Image.open(fig_path)
        img_width, img_height = img.size

        # Calculate scaling to fit on page with margins
        margin = 72  # 1 inch margins
        max_width = page_width - 2 * margin
        max_height = page_height - 2 * margin

        scale_x = max_width / img_width
        scale_y = max_height / img_height
        scale = min(scale_x, scale_y, 1.0)  # Don't enlarge

        # Calculate centered position
        scaled_width = img_width * scale
        scaled_height = img_height * scale
        x = (page_width - scaled_width) / 2
        y = (page_height - scaled_height) / 2

        # Draw image on page
        c.drawImage(fig_path, x, y, width=scaled_width, height=scaled_height)
        c.showPage()  # New page for each figure

        print(f"Added {fig_file} to PDF")

    # Save PDF
    c.save()
    print(f"\nCreated {output_path}")

    # Get file size
    size_kb = os.path.getsize(output_path) / 1024
    print(f"File size: {size_kb:.1f} KB")

if __name__ == "__main__":
    main()