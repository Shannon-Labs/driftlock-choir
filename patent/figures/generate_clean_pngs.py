#!/usr/bin/env python3
"""Generate clean PNG figures for USPTO patent filing."""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

def create_fig1():
    """Fig 1: Two-node beat generation and two-way handshake"""
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.set_xlim(0, 9)
    ax.set_ylim(0, 5)
    ax.axis('off')

    # Title
    ax.text(0.5, 4.7, 'Fig. 1 — Two-node beat generation and two-way handshake',
            fontsize=12, weight='bold')

    # Node A (100)
    ax.add_patch(FancyBboxPatch((0.5, 3), 1.5, 0.8, boxstyle="round,pad=0.02",
                                 facecolor='white', edgecolor='black', linewidth=1.5))
    ax.text(1.25, 3.4, '(100)', ha='center', fontsize=10, weight='bold')

    # TX (110)
    ax.add_patch(FancyBboxPatch((0.7, 2.3), 1.1, 0.5, boxstyle="round,pad=0.02",
                                 facecolor='white', edgecolor='black', linewidth=1.5))
    ax.text(1.25, 2.55, '(110)', ha='center', fontsize=10, weight='bold')

    # Node B (130)
    ax.add_patch(FancyBboxPatch((7, 3), 1.5, 0.8, boxstyle="round,pad=0.02",
                                 facecolor='white', edgecolor='black', linewidth=1.5))
    ax.text(7.75, 3.4, '(130)', ha='center', fontsize=10, weight='bold')

    # TX (140)
    ax.add_patch(FancyBboxPatch((7.2, 2.3), 1.1, 0.5, boxstyle="round,pad=0.02",
                                 facecolor='white', edgecolor='black', linewidth=1.5))
    ax.text(7.75, 2.55, '(140)', ha='center', fontsize=10, weight='bold')

    # Wireless link (120)
    ax.annotate('', xy=(7, 2.55), xytext=(2, 2.55),
                arrowprops=dict(arrowstyle='->', lw=1.5, linestyle='--'))
    ax.text(4.5, 2.7, '(120)', ha='center', fontsize=10, weight='bold')

    # Beat detectors (150)
    ax.add_patch(FancyBboxPatch((0.5, 1.5), 1.5, 0.5, boxstyle="round,pad=0.02",
                                 facecolor='white', edgecolor='black', linewidth=1.5))
    ax.text(1.25, 1.75, '(150)', ha='center', fontsize=10, weight='bold')

    ax.add_patch(FancyBboxPatch((7, 1.5), 1.5, 0.5, boxstyle="round,pad=0.02",
                                 facecolor='white', edgecolor='black', linewidth=1.5))
    ax.text(7.75, 1.75, '(150)', ha='center', fontsize=10, weight='bold')

    # Phase extraction (160)
    ax.add_patch(FancyBboxPatch((0.5, 0.8), 1.5, 0.5, boxstyle="round,pad=0.02",
                                 facecolor='white', edgecolor='black', linewidth=1.5))
    ax.text(1.25, 1.05, '(160)', ha='center', fontsize=10, weight='bold')

    ax.add_patch(FancyBboxPatch((7, 0.8), 1.5, 0.5, boxstyle="round,pad=0.02",
                                 facecolor='white', edgecolor='black', linewidth=1.5))
    ax.text(7.75, 1.05, '(160)', ha='center', fontsize=10, weight='bold')

    # Central boxes (170, 180)
    ax.add_patch(FancyBboxPatch((3.5, 1.2), 2, 0.4, boxstyle="round,pad=0.02",
                                 facecolor='white', edgecolor='black', linewidth=1.5))
    ax.text(4.5, 1.4, '(170)', ha='center', fontsize=10, weight='bold')

    ax.add_patch(FancyBboxPatch((3.5, 0.5), 2, 0.4, boxstyle="round,pad=0.02",
                                 facecolor='white', edgecolor='black', linewidth=1.5))
    ax.text(4.5, 0.7, '(180)', ha='center', fontsize=10, weight='bold')

    plt.tight_layout()
    return fig

def create_fig2():
    """Fig 2: Phase extraction and closed-form estimator"""
    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.set_xlim(0, 9)
    ax.set_ylim(0, 4.2)
    ax.axis('off')

    ax.text(0.5, 3.9, 'Fig. 2 — Phase extraction and closed-form estimator',
            fontsize=12, weight='bold')

    # Top row boxes
    boxes_top = [(1, '(200)'), (2.8, '(210)'), (4.6, '(220)'), (6.4, '(230)')]
    for x, label in boxes_top:
        ax.add_patch(FancyBboxPatch((x, 2.8), 1.2, 0.5, boxstyle="round,pad=0.02",
                                     facecolor='white', edgecolor='black', linewidth=1.5))
        ax.text(x+0.6, 3.05, label, ha='center', fontsize=10, weight='bold')

    # Middle row boxes
    ax.add_patch(FancyBboxPatch((1.5, 1.8), 1.5, 0.5, boxstyle="round,pad=0.02",
                                 facecolor='white', edgecolor='black', linewidth=1.5))
    ax.text(2.25, 2.05, '(240)', ha='center', fontsize=10, weight='bold')

    ax.add_patch(FancyBboxPatch((3.5, 1.8), 1.2, 0.5, boxstyle="round,pad=0.02",
                                 facecolor='white', edgecolor='black', linewidth=1.5))
    ax.text(4.1, 2.05, '(250)', ha='center', fontsize=10, weight='bold')

    ax.add_patch(FancyBboxPatch((5.2, 1.8), 1.8, 0.5, boxstyle="round,pad=0.02",
                                 facecolor='white', edgecolor='black', linewidth=1.5))
    ax.text(6.1, 2.05, '(260)', ha='center', fontsize=10, weight='bold')

    # Bottom row boxes
    ax.add_patch(FancyBboxPatch((2.5, 0.8), 1.8, 0.5, boxstyle="round,pad=0.02",
                                 facecolor='white', edgecolor='black', linewidth=1.5))
    ax.text(3.4, 1.05, '(270)', ha='center', fontsize=10, weight='bold')

    ax.add_patch(FancyBboxPatch((5, 0.8), 1.2, 0.5, boxstyle="round,pad=0.02",
                                 facecolor='white', edgecolor='black', linewidth=1.5))
    ax.text(5.6, 1.05, '(280)', ha='center', fontsize=10, weight='bold')

    # Arrows
    for i in range(3):
        ax.annotate('', xy=(boxes_top[i+1][0], 3.05), xytext=(boxes_top[i][0]+1.2, 3.05),
                    arrowprops=dict(arrowstyle='->', lw=1.5))

    plt.tight_layout()
    return fig

def create_fig3():
    """Fig 3: Network topology and variance-weighted consensus"""
    fig, ax = plt.subplots(figsize=(9, 5.2))
    ax.set_xlim(0, 9)
    ax.set_ylim(0, 5.2)
    ax.axis('off')

    ax.text(0.5, 4.9, 'Fig. 3 — Network topology and variance-weighted consensus',
            fontsize=12, weight='bold')

    # Network nodes
    nodes = [(1.5, 2.5, '1', '(300)'), (3, 3.5, '2', '(301)'), (3, 1.5, '3', '(302)'),
             (5, 2.5, '4', '(303)'), (6.5, 3.2, '5', '(304)')]

    for x, y, num, ref in nodes:
        circle = plt.Circle((x, y), 0.25, facecolor='white', edgecolor='black', linewidth=1.5)
        ax.add_patch(circle)
        ax.text(x, y, num, ha='center', va='center', fontsize=10)
        ax.text(x, y-0.5, ref, ha='center', fontsize=10, weight='bold')

    # Edges (simplified - just lines)
    edges = [(0, 1), (0, 2), (1, 3), (2, 3), (3, 4)]
    for i, j in edges:
        x1, y1 = nodes[i][0], nodes[i][1]
        x2, y2 = nodes[j][0], nodes[j][1]
        ax.plot([x1, x2], [y1, y2], 'k-', linewidth=1.5)

    # Bottom box (310)
    ax.add_patch(FancyBboxPatch((1, 0.2), 6.5, 0.6, boxstyle="round,pad=0.02",
                                 facecolor='white', edgecolor='black', linewidth=1.5))
    ax.text(4.25, 0.5, '(310)', ha='center', fontsize=10, weight='bold')

    plt.tight_layout()
    return fig

def create_fig5():
    """Fig 5: Node/system block diagram"""
    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.set_xlim(0, 9)
    ax.set_ylim(0, 4.2)
    ax.axis('off')

    ax.text(0.5, 3.9, 'Fig. 5 — Node/system block diagram',
            fontsize=12, weight='bold')

    # Top row blocks
    blocks_top = [(0.5, 1.4, '(400)'), (2.3, 1.2, '(410)'), (4.0, 1.4, '(420)'),
                  (5.9, 1.1, '(430)'), (7.5, 1.0, '(440)')]

    for x, w, label in blocks_top:
        ax.add_patch(FancyBboxPatch((x, 2.5), w, 0.6, boxstyle="round,pad=0.02",
                                     facecolor='white', edgecolor='black', linewidth=1.5))
        ax.text(x + w/2, 2.8, label, ha='center', fontsize=10, weight='bold')

    # Bottom row blocks
    ax.add_patch(FancyBboxPatch((4, 1.2), 1.4, 0.6, boxstyle="round,pad=0.02",
                                 facecolor='white', edgecolor='black', linewidth=1.5))
    ax.text(4.7, 1.5, '(460)', ha='center', fontsize=10, weight='bold')

    ax.add_patch(FancyBboxPatch((5.9, 1.2), 2.6, 0.6, boxstyle="round,pad=0.02",
                                 facecolor='white', edgecolor='black', linewidth=1.5))
    ax.text(7.2, 1.5, '(450)', ha='center', fontsize=10, weight='bold')

    # Arrows between top blocks
    for i in range(4):
        x1 = blocks_top[i][0] + blocks_top[i][1]
        x2 = blocks_top[i+1][0]
        ax.annotate('', xy=(x2, 2.8), xytext=(x1, 2.8),
                    arrowprops=dict(arrowstyle='->', lw=1.5))

    # Down arrows
    ax.annotate('', xy=(4.7, 1.8), xytext=(4.7, 2.5),
                arrowprops=dict(arrowstyle='->', lw=1.5))
    ax.annotate('', xy=(7.2, 1.8), xytext=(7.2, 2.5),
                arrowprops=dict(arrowstyle='->', lw=1.5))

    plt.tight_layout()
    return fig

def main():
    output_dir = "/Volumes/VIXinSSD/driftlock/dist/patent_packet/figures/png"

    figures = [
        (create_fig1, "fig1_beat_handshake.png"),
        (create_fig2, "fig2_phase_estimator.png"),
        (create_fig3, "fig3_network_consensus.png"),
        # Fig 4 already generated
        (create_fig5, "fig5_system_block.png")
    ]

    for create_func, filename in figures:
        fig = create_func()
        filepath = f"{output_dir}/{filename}"
        fig.savefig(filepath, dpi=600, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        print(f"Generated {filename}")

    print("\nAll figures generated successfully!")

if __name__ == "__main__":
    main()