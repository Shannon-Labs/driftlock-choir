
import matplotlib.pyplot as plt
import numpy as np
import os

def create_enhanced_visualization():
    """Create an enhanced, professional visualization for chronometric interferometry."""
    plt.style.use('seaborn-v0_8-darkgrid')

    fig, ax = plt.subplots(figsize=(14, 8))
    fig.patch.set_facecolor('#f0f4f8')
    ax.set_facecolor('#f0f4f8')

    # 1. System Setup
    ax.text(0.1, 0.8, "Node A", ha='center', fontsize=14, fontweight='bold', color='#2c3e50')
    ax.text(0.9, 0.8, "Node B", ha='center', fontsize=14, fontweight='bold', color='#2c3e50')
    ax.plot([0.1, 0.9], [0.78, 0.78], 'o', markersize=15, color='#3498db', alpha=0.7)
    ax.plot([0.1, 0.9], [0.78, 0.78], '--', color='#34495e', alpha=0.5)

    # Arrows for signal exchange
    ax.annotate("", xy=(0.85, 0.75), xytext=(0.15, 0.75),
                arrowprops=dict(arrowstyle="->", color="#e74c3c", lw=2, shrinkA=15, shrinkB=15))
    ax.annotate("", xy=(0.15, 0.81), xytext=(0.85, 0.81),
                arrowprops=dict(arrowstyle="->", color="#2ecc71", lw=2, shrinkA=15, shrinkB=15))
    ax.text(0.5, 0.72, r'Signal with $f_A$, delayed by $\tau$', ha='center', fontsize=11, color='#e74c3c')
    ax.text(0.5, 0.84, r'Signal with $f_B$', ha='center', fontsize=11, color='#2ecc71')

    # 2. Beat Note Visualization
    t = np.linspace(0, 0.1, 400)
    f_beat = 10  # Hz
    beat_wave = np.cos(2 * np.pi * f_beat * t) * np.exp(-t * 20)
    carrier_wave = np.cos(2 * np.pi * f_beat * 10 * t)
    modulated_wave = beat_wave * carrier_wave

    ax.plot(t + 0.25, modulated_wave * 0.05 + 0.5, color='#8e44ad', lw=2.5)
    ax.text(0.5, 0.6, "Beat Note Formation", ha='center', fontsize=14, fontweight='bold', color='#2c3e50')
    ax.text(0.5, 0.55, r'Mixing $f_A$ and $f_B$ produces a beat note at $\Delta f = |f_A - f_B|$',
            ha='center', fontsize=11, color='#34495e')

    # 3. Information Extraction
    ax.add_patch(plt.Rectangle((0.1, 0.1), 0.8, 0.3, facecolor='#ffffff', edgecolor='#bdc3c7', lw=1, alpha=0.7))
    ax.text(0.5, 0.35, "Parameter Extraction", ha='center', fontsize=14, fontweight='bold', color='#2c3e50')

    ax.text(0.3, 0.25, r'Time-of-Flight ($\tau$)', ha='center', fontsize=12, color='#c0392b')
    ax.text(0.3, 0.2, r'$\tau = \frac{\Delta\phi}{2\pi \Delta f}$', ha='center', fontsize=12, bbox=dict(facecolor='white', alpha=0.5, edgecolor='none'))
    ax.text(0.3, 0.15, "Picosecond Precision", ha='center', fontsize=10, color='#7f8c8d')

    ax.text(0.7, 0.25, r'Frequency Offset ($\Delta f$)', ha='center', fontsize=12, color='#2980b9')
    ax.text(0.7, 0.2, r'$\Delta f = \frac{\partial\phi}{\partial t}$', ha='center', fontsize=12, bbox=dict(facecolor='white', alpha=0.5, edgecolor='none'))
    ax.text(0.7, 0.15, "Sub-ppb Accuracy", ha='center', fontsize=10, color='#7f8c8d')

    # Final Touches
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    fig.suptitle("Chronometric Interferometry: From Signal Exchange to Picosecond Precision", fontsize=18, fontweight='bold', color='#1a252f')
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    # Save the figure
    output_dir = "docs/assets/images"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "chronometric_interferometry_enhanced.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    print(f"Enhanced visualization saved to: {output_path}")
    return output_path

if __name__ == '__main__':
    create_enhanced_visualization()
