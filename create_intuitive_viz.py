#!/usr/bin/env python3
"""
Create an intuitive visualization for Driftlock Choir synchronization
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, Circle
import matplotlib.patheffects as path_effects

# Set style for clean, modern look
plt.style.use('seaborn-v0_8-whitegrid')
fig = plt.figure(figsize=(14, 10))

# Create subplots
gs = fig.add_gridspec(3, 2, height_ratios=[1, 1.2, 1], width_ratios=[1, 1])
ax1 = fig.add_subplot(gs[0, :])  # Before state
ax2 = fig.add_subplot(gs[1, :])  # Synchronization process
ax3 = fig.add_subplot(gs[2, :])  # After state

fig.patch.set_facecolor('#f8f9fa')
fig.suptitle('Driftlock Choir: Ultra-Precise Synchronization Explained',
             fontsize=18, fontweight='bold', color='#2c3e50', y=0.95)

# 1. BEFORE: Out of sync oscillators
ax1.set_title('❌ BEFORE: Oscillators "Out of Tune"', fontsize=14, fontweight='bold', color='#e74c3c')

# Create oscillator representations
time = np.linspace(0, 2, 1000)
freq1 = 5.0  # 5 Hz
freq2 = 5.3  # 5.3 Hz (out of tune)
phase2 = np.pi/4  # Phase offset

osc1_before = np.sin(2 * np.pi * freq1 * time)
osc2_before = np.sin(2 * np.pi * freq2 * time + phase2)

ax1.plot(time, osc1_before, 'b-', linewidth=2, alpha=0.7, label='Oscillator A (Reference)')
ax1.plot(time, osc2_before, 'r-', linewidth=2, alpha=0.7, label='Oscillator B (Drifting)')

# Add "beat note" visualization
beat_note_before = osc1_before + osc2_before
ax1.fill_between(time, beat_note_before/4, alpha=0.3, color='orange', label='Beat Pattern (Confusion)')

ax1.set_ylabel('Signal Strength', fontsize=11)
ax1.set_ylim(-3, 3)
ax1.legend(loc='upper right', fontsize=10)
ax1.text(0.02, 0.95, 'Different frequencies create confusion\nNo clear timing reference',
         transform=ax1.transAxes, fontsize=10, color='#7f8c8d',
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# 2. DURING: Synchronization process
ax2.set_title('🔄 SYNCHRONIZATION: Finding the Perfect Rhythm', fontsize=14, fontweight='bold', color='#f39c12')

# Simulate convergence
convergence_time = np.linspace(0, 1, 100)
freq_diff = 0.3 * np.exp(-3 * convergence_time)  # Frequency difference decreases
phase_diff = (np.pi/4) * np.exp(-2 * convergence_time)  # Phase difference decreases

# Create convergence visualization
for i, t in enumerate(convergence_time[::20]):
    current_freq2 = freq1 + freq_diff[i*20]
    current_phase2 = phase_diff[i*20]
    osc_converge = np.sin(2 * np.pi * current_freq2 * time[:200] + current_phase2)

    alpha = 0.1 + 0.7 * (i / len(convergence_time[::20]))
    color = plt.cm.RdYlBu_r(i / len(convergence_time[::20]))
    ax2.plot(time[:200], osc_converge, color=color, alpha=alpha, linewidth=1.5)

# Add convergence metrics
ax2_twin = ax2.twinx()
ax2_twin.plot(convergence_time, freq_diff * 1000, 'g--', linewidth=2, label='Frequency Error (mHz)')
ax2_twin.plot(convergence_time, np.abs(phase_diff) * 180 / np.pi, 'purple', linewidth=2, label='Phase Error (degrees)')
ax2_twin.set_ylabel('Synchronization Error', fontsize=11, color='#27ae60')
ax2_twin.tick_params(axis='y', labelcolor='#27ae60')
ax2_twin.legend(loc='upper right', fontsize=10)

ax2.set_xlabel('Time (seconds)', fontsize=11)
ax2.set_ylabel('Oscillator Signals', fontsize=11)
ax2.set_ylim(-2, 2)
ax2.text(0.02, 0.95, 'Algorithm measures differences\nAdjusts to match perfectly',
         transform=ax2.transAxes, fontsize=10, color='#7f8c8d',
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# 3. AFTER: Perfect synchronization
ax3.set_title('✅ AFTER: Perfect Synchronization Achieved', fontsize=14, fontweight='bold', color='#27ae60')

# Show synchronized oscillators
osc1_after = np.sin(2 * np.pi * freq1 * time)
osc2_after = np.sin(2 * np.pi * freq1 * time)  # Now perfectly synchronized

ax3.plot(time, osc1_after, 'b-', linewidth=2.5, alpha=0.8, label='Oscillator A')
ax3.plot(time, osc2_after, 'r--', linewidth=2.5, alpha=0.8, label='Oscillator B (Now Synced)')

# Highlight the perfect match
ax3.fill_between(time, osc1_after - 0.05, osc1_after + 0.05, alpha=0.3, color='green',
                 label='Perfect Match (±2.1 ps precision)')

ax3.set_xlabel('Time (seconds)', fontsize=11)
ax3.set_ylabel('Signal Strength', fontsize=11)
ax3.set_ylim(-2, 2)
ax3.legend(loc='upper right', fontsize=10)
ax3.text(0.02, 0.95, 'Both oscillators in perfect sync\nPicosecond-level timing achieved',
         transform=ax3.transAxes, fontsize=10, color='#27ae60',
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# Add timing precision callout
precision_box = FancyBboxPatch((0.65, 0.02), 0.32, 0.15,
                               boxstyle="round,pad=0.01",
                               facecolor='#e8f5e8', edgecolor='#27ae60', linewidth=2)
ax3.add_patch(precision_box)
ax3.text(0.81, 0.095, '⏱️ 2.1 picoseconds\n   timing precision',
         transform=ax3.transAxes, fontsize=11, fontweight='bold',
         ha='center', va='center', color='#27ae60')

plt.tight_layout()
plt.savefig('docs/assets/images/intuitive_synchronization.png',
            dpi=150, bbox_inches='tight', facecolor='#f8f9fa')
plt.close()

print("✅ Intuitive synchronization visualization created!")
print("📁 Saved to: docs/assets/images/intuitive_synchronization.png")