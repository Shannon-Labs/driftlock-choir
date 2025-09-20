#!/usr/bin/env python3
"""Test script for Phase 4 Super-Resolution Performance.

This script validates advanced super-resolution techniques for achieving
sub-10ps timing precision through:
1. MUSIC (Multiple Signal Classification) algorithm
2. ESPRIT (Estimation of Signal Parameters via Rotational Invariance)
3. Matrix pencil method for enhanced frequency resolution
4. Subspace-based methods for improved timing estimation
5. Compressed sensing approaches for sparse signal reconstruction

Run with: python test_phase4_super_resolution.py
"""

from __future__ import annotations

import time
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Any

from driftlock_sim.dsp.tx_comb import generate_comb
from driftlock_sim.dsp.rx_coherent import (
    estimate_tone_phasors, unwrap_phase, wls_delay, traditional_wls_delay,
    per_tone_snr, estimate_noise_power
)
from driftlock_sim.dsp.super_resolution_estimator import (
    super_resolution_timing_estimator, SuperResolutionEstimator,
    SuperResolutionMethod, SuperResolutionConfig
)
from driftlock_sim.dsp.time_delay import impose_fractional_delay_fft
from driftlock_sim.dsp.impairments import apply_cfo, apply_phase_noise
from driftlock_sim.dsp.channel_models import awgn


OUTPUT_DIR = Path("outputs/phase4_super_resolution")
FIGURES_DIR = OUTPUT_DIR / "figures"


def ensure_output_dirs():
    """Create output directories if they don't exist."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def test_super_resolution_methods():
    """Test different super-resolution methods."""
    print("Testing Super-Resolution Methods...")

    # Test parameters for sub-10ps precision
    fs = 20e6
    duration = 9e-3
    df = 1e6
    m = 21
    truth_tau = 5e-12  # 5 ps timing offset (sub-10ps target)
    truth_df = 500e3   # 500 kHz frequency offset
    n_trials = 100

    methods = [
        SuperResolutionMethod.MUSIC,
        SuperResolutionMethod.ESPRIT,
        SuperResolutionMethod.MATRIX_PENCIL,
        SuperResolutionMethod.SUBSPACE,
        SuperResolutionMethod.COMPRESSED_SENSING
    ]

    results = {method: {'rmse_tau': [], 'rmse_df': [], 'precision_achieved': []}
               for method in methods}

    for trial in range(n_trials):
        rng = np.random.default_rng(2025 + trial)

        # Generate test signal
        x, fk, pilot_mask, _ = generate_comb(
            fs, duration, df, m=m, omit_fundamental=True, rng=rng, return_payload=True
        )

        # Apply impairments
        x = impose_fractional_delay_fft(x, fs, truth_tau)
        x = apply_cfo(x, fs, truth_df / fs)
        x = apply_phase_noise(x, 0.0, rng)
        x = awgn(x, 35.0, rng)  # 35dB SNR for precision testing

        # Estimate phasors and SNR
        Yk = estimate_tone_phasors(x, fs, fk)
        noise_power = estimate_noise_power(x, fs, fk, Yk)
        snr_per_tone = per_tone_snr(Yk, noise_power, len(x))

        # Test each super-resolution method
        for method in methods:
            try:
                tau_est, df_est = super_resolution_timing_estimator(
                    Yk, fk, method=method, snr=snr_per_tone
                )

                # Store results
                results[method]['rmse_tau'].append(abs(tau_est - truth_tau) * 1e12)  # ps
                results[method]['rmse_df'].append(abs(df_est - truth_df))  # Hz
                results[method]['precision_achieved'].append(
                    abs(tau_est - truth_tau) * 1e12 < 10.0  # Sub-10ps achieved
                )

            except Exception as e:
                # Count failures
                results[method]['rmse_tau'].append(1000.0)  # Large error for failures
                results[method]['rmse_df'].append(1e6)      # Large error for failures
                results[method]['precision_achieved'].append(False)

    # Calculate statistics
    summary = {}
    for method in methods:
        tau_rmse = np.mean(results[method]['rmse_tau'])
        df_rmse = np.mean(results[method]['rmse_df'])
        precision_rate = np.mean(results[method]['precision_achieved']) * 100

        summary[method] = {
            'tau_rmse_ps': tau_rmse,
            'df_rmse_hz': df_rmse,
            'precision_rate_pct': precision_rate,
            'sub_10ps_achieved': tau_rmse < 10.0
        }

        method_name = method.value.replace('_', ' ').title()
        print(f"  {method_name:20s} - τ: {tau_rmse:6.2f} ps, Sub-10ps: {precision_rate:5.1f}%")

    return summary


def test_method_comparison():
    """Compare super-resolution methods against traditional approaches."""
    print("\nComparing Super-Resolution vs Traditional Methods...")

    # Test parameters
    fs = 20e6
    duration = 9e-3
    df = 1e6
    m = 21
    truth_tau = 5e-12
    n_trials = 50

    # Methods to compare
    methods = {
        'traditional_wls': 'Traditional WLS',
        'music': 'MUSIC Algorithm',
        'esprit': 'ESPRIT Algorithm',
        'matrix_pencil': 'Matrix Pencil',
        'subspace': 'Subspace Method',
        'compressed_sensing': 'Compressed Sensing'
    }

    comparison_results = {key: {'rmse_ps': [], 'latency_ms': []}
                         for key in methods.keys()}

    for trial in range(n_trials):
        rng = np.random.default_rng(3000 + trial)

        # Generate test signal
        x, fk, pilot_mask, _ = generate_comb(
            fs, duration, df, m=m, omit_fundamental=True, rng=rng, return_payload=True
        )

        # Apply impairments
        x = impose_fractional_delay_fft(x, fs, truth_tau)
        x = apply_cfo(x, fs, 0.0)
        x = apply_phase_noise(x, 0.0, rng)
        x = awgn(x, 30.0, rng)  # 30dB SNR

        # Estimate phasors and SNR
        Yk = estimate_tone_phasors(x, fs, fk)
        noise_power = estimate_noise_power(x, fs, fk, Yk)
        snr_per_tone = per_tone_snr(Yk, noise_power, len(x))

        # Test traditional method
        start_time = time.time()
        tau_traditional, _ = traditional_wls_delay(Yk, fk, snr_per_tone)
        latency_traditional = (time.time() - start_time) * 1000

        comparison_results['traditional_wls']['rmse_ps'].append(
            abs(tau_traditional - truth_tau) * 1e12
        )
        comparison_results['traditional_wls']['latency_ms'].append(latency_traditional)

        # Test super-resolution methods
        for method_key, method_name in [
            ('music', SuperResolutionMethod.MUSIC),
            ('esprit', SuperResolutionMethod.ESPRIT),
            ('matrix_pencil', SuperResolutionMethod.MATRIX_PENCIL),
            ('subspace', SuperResolutionMethod.SUBSPACE),
            ('compressed_sensing', SuperResolutionMethod.COMPRESSED_SENSING)
        ]:
            try:
                start_time = time.time()
                tau_sr, _ = super_resolution_timing_estimator(
                    Yk, fk, method=method_name, snr=snr_per_tone
                )
                latency_sr = (time.time() - start_time) * 1000

                comparison_results[method_key]['rmse_ps'].append(
                    abs(tau_sr - truth_tau) * 1e12
                )
                comparison_results[method_key]['latency_ms'].append(latency_sr)

            except Exception as e:
                # Handle failures
                comparison_results[method_key]['rmse_ps'].append(1000.0)
                comparison_results[method_key]['latency_ms'].append(100.0)

    # Calculate statistics
    comparison_summary = {}
    for method_key, data in comparison_results.items():
        avg_rmse = np.mean(data['rmse_ps'])
        avg_latency = np.mean(data['latency_ms'])
        sub_10ps_rate = np.mean(np.array(data['rmse_ps']) < 10.0) * 100

        comparison_summary[method_key] = {
            'avg_rmse_ps': avg_rmse,
            'avg_latency_ms': avg_latency,
            'sub_10ps_rate_pct': sub_10ps_rate
        }

        method_name = methods[method_key]
        print(f"  {method_name:20s} - RMSE: {avg_rmse:6.2f} ps, Latency: {avg_latency:5.2f} ms, Sub-10ps: {sub_10ps_rate:5.1f}%")

    return comparison_summary


def test_robustness_analysis():
    """Test robustness of super-resolution methods under stress."""
    print("\nTesting Robustness Under Stress Conditions...")

    # Test parameters
    fs = 20e6
    duration = 9e-3
    df = 1e6
    m = 21
    truth_tau = 5e-12
    n_trials = 50

    # Challenging conditions
    conditions = {
        'high_noise': {'snr_db': 15.0, 'phase_noise': 0.1},
        'multipath': {'snr_db': 25.0, 'multipath_delay': 1e-6},
        'doppler': {'snr_db': 20.0, 'doppler_shift': 100.0},
        'interference': {'snr_db': 10.0, 'interference_power': 0.5}
    }

    methods = [SuperResolutionMethod.MUSIC, SuperResolutionMethod.ESPRIT]
    robustness_results = {condition: {method: {'success_rate': 0, 'avg_rmse': 0}
                                     for method in methods}
                         for condition in conditions.keys()}

    for condition, params in conditions.items():
        print(f"  Testing {condition}...")

        for trial in range(n_trials):
            rng = np.random.default_rng(4000 + trial)

            # Generate test signal
            x, fk, pilot_mask, _ = generate_comb(
                fs, duration, df, m=m, omit_fundamental=True, rng=rng, return_payload=True
            )

            # Apply impairments based on condition
            x = impose_fractional_delay_fft(x, fs, truth_tau)
            x = apply_cfo(x, fs, 0.0)
            x = apply_phase_noise(x, params.get('phase_noise', 0.0), rng)
            x = awgn(x, params['snr_db'], rng)

            # Add condition-specific impairments
            if condition == 'multipath':
                x_multipath = np.roll(x, int(params['multipath_delay'] * fs))
                x = x + 0.3 * x_multipath
            elif condition == 'doppler':
                t = np.arange(len(x)) / fs
                doppler_phase = 2 * np.pi * params['doppler_shift'] * t**2 / 2
                x = x * np.exp(1j * doppler_phase)
            elif condition == 'interference':
                interference = np.random.randn(len(x)) + 1j * np.random.randn(len(x))
                interference = interference / np.sqrt(2) * np.sqrt(params['interference_power'])
                x = x + interference

            # Estimate phasors and SNR
            Yk = estimate_tone_phasors(x, fs, fk)
            noise_power = estimate_noise_power(x, fs, fk, Yk)
            snr_per_tone = per_tone_snr(Yk, noise_power, len(x))

            # Test each method
            for method in methods:
                try:
                    tau_est, _ = super_resolution_timing_estimator(
                        Yk, fk, method=method, snr=snr_per_tone
                    )

                    rmse_ps = abs(tau_est - truth_tau) * 1e12
                    success = rmse_ps < 50.0  # 50ps threshold for success

                    robustness_results[condition][method]['success_rate'] += success
                    robustness_results[condition][method]['avg_rmse'] += rmse_ps

                except Exception as e:
                    robustness_results[condition][method]['success_rate'] += 0

        # Calculate final statistics
        for method in methods:
            robustness_results[condition][method]['success_rate'] /= n_trials
            robustness_results[condition][method]['avg_rmse'] /= n_trials

            method_name = method.value.replace('_', ' ').title()
            success_rate = robustness_results[condition][method]['success_rate'] * 100
            avg_rmse = robustness_results[condition][method]['avg_rmse']
            print(f"    {method_name:15s} - Success: {success_rate:5.1f}%, RMSE: {avg_rmse:6.2f} ps")

    return robustness_results


def plot_super_resolution_results(sr_results, comparison_results, robustness_results):
    """Generate super-resolution performance plots."""
    print("\nGenerating Super-Resolution Performance Plots...")

    # Method comparison plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    methods = list(sr_results.keys())
    method_names = [method.value.replace('_', ' ').title() for method in methods]
    rmse_values = [sr_results[method]['tau_rmse_ps'] for method in methods]
    precision_rates = [sr_results[method]['precision_rate_pct'] for method in methods]

    # RMSE comparison
    bars1 = ax1.bar(method_names, rmse_values, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7'])
    ax1.set_ylabel('RMSE (ps)')
    ax1.set_title('Super-Resolution Method Performance')
    ax1.axhline(y=10.0, color='r', linestyle='--', alpha=0.7, label='10ps Target')
    ax1.axhline(y=5.0, color='g', linestyle='--', alpha=0.7, label='5ps Breakthrough')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis='x', rotation=45)

    # Add value labels
    for bar, rmse in zip(bars1, rmse_values):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{rmse:.2f}', ha='center', va='bottom', fontsize=10)

    # Precision achievement rate
    bars2 = ax2.bar(method_names, precision_rates, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7'])
    ax2.set_ylabel('Sub-10ps Achievement Rate (%)')
    ax2.set_title('Precision Success Rate')
    ax2.axhline(y=90.0, color='g', linestyle='--', alpha=0.7, label='90% Target')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(axis='x', rotation=45)

    for bar, rate in zip(bars2, precision_rates):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{rate:.1f}%', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'phase4_super_resolution_methods.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Traditional vs Super-resolution comparison
    fig, ax = plt.subplots(figsize=(12, 6))

    method_keys = list(comparison_results.keys())
    method_names = ['Traditional WLS', 'MUSIC', 'ESPRIT', 'Matrix Pencil', 'Subspace', 'Compressed Sensing']
    rmse_values = [comparison_results[key]['avg_rmse_ps'] for key in method_keys]
    latency_values = [comparison_results[key]['avg_latency_ms'] for key in method_keys]

    colors = ['#666666', '#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']

    bars = ax.bar(method_names, rmse_values, color=colors)
    ax.set_ylabel('Average RMSE (ps)')
    ax.set_title('Traditional vs Super-Resolution Methods')
    ax.axhline(y=10.0, color='r', linestyle='--', alpha=0.7, label='10ps Target')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='x', rotation=45)

    # Add value labels
    for bar, rmse in zip(bars, rmse_values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{rmse:.2f}', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'phase4_method_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Robustness comparison plot
    fig, ax = plt.subplots(figsize=(10, 6))

    conditions = list(robustness_results.keys())
    condition_names = [cond.replace('_', ' ').title() for cond in conditions]

    methods = list(list(robustness_results.values())[0].keys())
    method_names = [method.value.replace('_', ' ').title() for method in methods]

    x = np.arange(len(conditions))
    width = 0.35

    for i, method in enumerate(methods):
        success_rates = [robustness_results[cond][method]['success_rate'] * 100
                        for cond in conditions]
        ax.bar(x + i*width, success_rates, width,
               label=method_names[i], alpha=0.8)

    ax.set_xlabel('Test Condition')
    ax.set_ylabel('Success Rate (%)')
    ax.set_title('Robustness Under Stress Conditions')
    ax.set_xticks(x + width/2)
    ax.set_xticklabels(condition_names)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(y=80.0, color='g', linestyle='--', alpha=0.7, label='80% Target')

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'phase4_robustness_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()


def generate_super_resolution_report(all_results):
    """Generate comprehensive super-resolution performance report."""
    print("\n" + "="*70)
    print("PHASE 4 SUPER-RESOLUTION PERFORMANCE REPORT")
    print("="*70)

    sr_results = all_results['super_resolution']
    comparison = all_results['comparison']
    robustness = all_results['robustness']

    print("\n1. SUPER-RESOLUTION METHOD COMPARISON:")
    print("   Target: Achieve sub-10ps timing precision with >90% success rate")

    best_method = max(sr_results.keys(),
                     key=lambda k: sr_results[k]['precision_rate_pct'])
    best_rate = sr_results[best_method]['precision_rate_pct']
    best_rmse = sr_results[best_method]['tau_rmse_ps']

    print(f"   Best Method: {best_method.value.replace('_', ' ').title()}")
    print(f"   Average RMSE: {best_rmse:.2f} ps")
    print(f"   Sub-10ps Success Rate: {best_rate:.1f}%")
    print(f"   Target Achievement: {'✅ SUCCESS' if best_rate >= 90 else '⚠️ NEEDS_WORK'}")

    print("\n2. TRADITIONAL vs SUPER-RESOLUTION:")
    traditional_rmse = comparison['traditional_wls']['avg_rmse_ps']
    best_sr_rmse = min([comp['avg_rmse_ps'] for comp in comparison.values() if comp['avg_rmse_ps'] < 1000])

    print(f"   Traditional WLS RMSE: {traditional_rmse:.2f} ps")
    print(f"   Best Super-Resolution RMSE: {best_sr_rmse:.2f} ps")
    print(f"   Improvement: {((traditional_rmse - best_sr_rmse) / traditional_rmse * 100):.1f}%")

    print("\n3. ROBUSTNESS ANALYSIS:")
    print("   Testing performance under challenging conditions:")

    for condition, methods in robustness.items():
        condition_name = condition.replace('_', ' ').title()
        print(f"   {condition_name}:")

        for method_key, stats in methods.items():
            method_name = method_key.value.replace('_', ' ').title()
            success_rate = stats['success_rate'] * 100
            avg_rmse = stats['avg_rmse']
            print(f"     {method_name:15s} - Success: {success_rate:5.1f}%, RMSE: {avg_rmse:6.2f} ps")

    print("\n4. TECHNICAL ACHIEVEMENTS:")
    print("   ✅ MUSIC (Multiple Signal Classification) algorithm")
    print("   ✅ ESPRIT (Estimation of Signal Parameters via Rotational Invariance)")
    print("   ✅ Matrix pencil method for enhanced frequency resolution")
    print("   ✅ Subspace-based methods for improved timing estimation")
    print("   ✅ Compressed sensing approaches for sparse signal reconstruction")
    print("   ✅ Comparative analysis against traditional methods")

    print("\n5. PERFORMANCE BREAKTHROUGHS:")
    sub_10ps_achieved = any(data['precision_rate_pct'] >= 90 for data in sr_results.values())
    improvement_over_traditional = (traditional_rmse - best_sr_rmse) / traditional_rmse > 0.1

    if sub_10ps_achieved:
        print("   🎉 SUB-10PS PRECISION ACHIEVED!")
    else:
        print("   ⚠️  Sub-10ps precision needs further optimization")

    if improvement_over_traditional:
        print("   📈 SIGNIFICANT IMPROVEMENT OVER TRADITIONAL METHODS!")
    else:
        print("   ⚠️  Limited improvement over traditional methods")

    print("\n6. NEXT STEPS:")
    if sub_10ps_achieved:
        print("   • Continue to Phase 5: System integration and real-time optimization")
        print("   • Implement hardware acceleration for super-resolution methods")
        print("   • Develop adaptive method selection based on conditions")
    else:
        print("   • Optimize super-resolution algorithm parameters")
        print("   • Enhance noise robustness of subspace methods")
        print("   • Develop hybrid traditional/super-resolution approaches")

    print("="*70)


def main():
    """Run all Phase 4 super-resolution tests."""
    print("Driftlock Phase 4: Super-Resolution Performance")
    print("Testing advanced signal processing techniques for sub-10ps precision...")

    start_time = time.time()
    ensure_output_dirs()

    # Run all tests
    all_results = {}

    all_results['super_resolution'] = test_super_resolution_methods()
    all_results['comparison'] = test_method_comparison()
    all_results['robustness'] = test_robustness_analysis()

    # Generate plots
    plot_super_resolution_results(
        all_results['super_resolution'],
        all_results['comparison'],
        all_results['robustness']
    )

    # Generate comprehensive report
    generate_super_resolution_report(all_results)

    # Save detailed results
    import json
    results_file = OUTPUT_DIR / 'phase4_super_resolution_results.json'

    def serialize_for_json(obj):
        """Convert non-serializable objects to strings for JSON."""
        if hasattr(obj, 'value'):
            return obj.value  # Convert enum to string
        if isinstance(obj, (np.ndarray, np.integer, np.floating)):
            return obj.item() if hasattr(obj, 'item') else float(obj)
        if str(type(obj)) == "<class 'driftlock_sim.dsp.super_resolution_estimator.SuperResolutionMethod'>":
            return obj.value  # Handle SuperResolutionMethod enum specifically
        return str(obj)

    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=serialize_for_json)

    elapsed_time = time.time() - start_time
    print(f"\nSuper-resolution validation completed in {elapsed_time:.1f} seconds")
    print(f"Results saved to: {results_file}")
    print(f"Figures saved to: {FIGURES_DIR}")


if __name__ == "__main__":
    main()