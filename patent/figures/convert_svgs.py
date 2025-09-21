#!/usr/bin/env python3
"""Convert cleaned SVG files to PNG format for USPTO patent filing."""

import os
import sys

# Try different SVG conversion libraries
try:
    import cairosvg
    USE_CAIRO = True
except ImportError:
    USE_CAIRO = False
    try:
        from PIL import Image
        import io
    except ImportError:
        print("Neither cairosvg nor PIL available. Installing cairosvg...")
        os.system("pip install cairosvg")
        import cairosvg
        USE_CAIRO = True

def convert_svg_to_png(svg_path, png_path, dpi=600):
    """Convert SVG to PNG with specified DPI."""
    if USE_CAIRO:
        # Use cairosvg
        with open(svg_path, 'rb') as f:
            svg_data = f.read()
        cairosvg.svg2png(
            bytestring=svg_data,
            write_to=png_path,
            dpi=dpi
        )
    else:
        print(f"Error: No suitable SVG converter available for {svg_path}")
        return False
    return True

def main():
    figures_dir = "/Volumes/VIXinSSD/driftlock choir/patent/figures"
    output_dir = "/Volumes/VIXinSSD/driftlock choir/dist/patent_packet/figures/png"

    # Create output directory if needed
    os.makedirs(output_dir, exist_ok=True)

    # List of SVG files to convert
    svg_files = [
        "fig1_beat_handshake.svg",
        "fig2_phase_estimator.svg",
        "fig3_network_consensus.svg",
        "fig4_convergence.svg",
        "fig5_system_block.svg"
    ]

    for svg_file in svg_files:
        svg_path = os.path.join(figures_dir, svg_file)
        png_file = svg_file.replace('.svg', '.png')
        png_path = os.path.join(output_dir, png_file)

        if os.path.exists(svg_path):
            print(f"Converting {svg_file} to PNG...")
            if convert_svg_to_png(svg_path, png_path, dpi=600):
                print(f"  ✓ Saved to {png_path}")
            else:
                print(f"  ✗ Failed to convert {svg_file}")
        else:
            print(f"  ✗ SVG file not found: {svg_path}")

    print("\nConversion complete!")

if __name__ == "__main__":
    main()