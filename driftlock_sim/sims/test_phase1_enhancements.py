#!/usr/bin/env python3
"""Test script for Phase 1 Choir Lab enhancements.

This script validates the performance improvements achieved through:
1. Enhanced WLS weighting algorithms (RMSE/CRLB < 0.8)
2. Optimized multi-carrier comb synthesis
3. Enhanced aperture detection for sub-0dB SNR
4. Adaptive frequency offset selection

Run with: python test_phase1_enhancements.py
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
    per_tone_snr, estimate_noise_power, adaptive_frequency_offset_selection,
    dynamic_offset_adaptation
)
from driftlock_sim.dsp.rx_aperture import (
    envelope_spectrum, detect_df_peak_robust, choir_health_index
)
from driftlock_sim.dsp.crlb import delay_crlb_rms_bandwidth as delay_crlb_std
from driftlock_sim.dsp.time_delay import impose_fractional_delay_fft
from driftlock_sim.dsp.impairments import apply_cfo, apply_phase_noise
from driftlock_sim.dsp.channel_models import awgn
from driftlock_sim.dsp.metrics import rms_bandwidth_hz, ber_qpsk_awgn


OUTPUT_DIR = Path("driftlock_sim/outputs/phase1_enhancements")
FIGURES_DIR = OUTPUT_DIR / "figures"


def ensure_output_dirs():
    """Create output directories if they don't exist."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def test_enhanced_wls_weighting():
    """Test enhanced WLS weighting for RMSE/CRLB improvement."""
    print("Testing Enhanced WLS Weighting...")

    # Test parameters
    fs = 20e6
    duration = 9e-3
    df = 1e6
    m = 21
    truth_tau = 100e-12
    n_trials = 50

    results = {
        'traditional_rmse': [],
        'enhanced_rmse': [],
        'traditional_crlb': [],
        'enhanced_crlb': [],
        'traditional_ratios': [],
        'enhanced_ratios': []
    }

    for trial in range(n_trials):
        rng = np.random.default_rng(2025 + trial)

        # Generate test signal
        x, fk, pilot_mask, _ = generate_comb(
            fs, duration, df, m=m, omit_fundamental=True, rng=rng, return_payload=True
        )

        # Apply impairments
        x = impose_fractional_delay_fft(x, fs, truth_tau)
        x = apply_cfo(x, fs, 0.0)
        x = apply_phase_noise(x, 0.0, rng)
        x = awgn(x, 30.0, rng)  # 30dB SNR

        # Traditional WLS
        Yk_traditional = estimate_tone_phasors(x, fs, fk)
        ph_traditional = unwrap_phase(np.angle(Yk_traditional), fk)
        noise_power = estimate_noise_power(x, fs, fk, Yk_traditional)
        snr_traditional = per_tone_snr(Yk_traditional, noise_power, len(x))

        tau_traditional, _, crlb_traditional = traditional_wls_delay(
            fk, ph_traditional, snr_traditional, return_stats=True
        )

        # Enhanced WLS (our new implementation)
        tau_enhanced, _, crlb_enhanced = wls_delay(
            fk, ph_traditional, snr_traditional, return_stats=True
        )

        # Store results
        results['traditional_rmse'].append(abs(tau_traditional - truth_tau) * 1e12)
        results['enhanced_rmse'].append(abs(tau_enhanced - truth_tau) * 1e12)
        results['traditional_crlb'].append(crlb_traditional * 1e12)
        results['enhanced_crlb'].append(crlb_enhanced * 1e12)

    # Calculate statistics
    traditional_rmse = np.mean(results['traditional_rmse'])
    enhanced_rmse = np.mean(results['enhanced_rmse'])
    traditional_crlb = np.mean(results['traditional_crlb'])
    enhanced_crlb = np.mean(results['enhanced_crlb'])

    traditional_ratio = traditional_rmse / traditional_crlb
    enhanced_ratio = enhanced_rmse / enhanced_crlb

    improvement = (traditional_ratio - enhanced_ratio) / traditional_ratio * 100

    print(f"  Traditional WLS - RMSE: {traditional_rmse:.1f} ps, CRLB: {traditional_crlb:.1f} ps, Ratio: {traditional_ratio:.3f}")
    print(f"  Enhanced WLS    - RMSE: {enhanced_rmse:.1f} ps, CRLB: {enhanced_crlb:.1f} ps, Ratio: {enhanced_ratio:.3f}")
    print(f"  Improvement: {improvement:.1f}%")

    return {
        'traditional_ratio': traditional_ratio,
        'enhanced_ratio': enhanced_ratio,
        'improvement_pct': improvement,
        'target_achieved': enhanced_ratio < 0.8
    }


def test_optimized_comb_synthesis():
    """Test optimized multi-carrier comb synthesis."""
    print("\nTesting Optimized Comb Synthesis...")

    # Test parameters
    fs = 20e6
    duration = 9e-3
    df = 1e6
    m = 21

    # Generate optimized comb
    x_opt, fk_opt, pilot_mask_opt, meta_opt = generate_comb(
        fs, duration, df, m=m, omit_fundamental=True, return_payload=True
    )

    # Calculate reconstruction SNR
    reconstruction_snr = meta_opt.get('reconstruction_snr_db', 0.0)

    # Calculate RMS bandwidth
    amps = meta_opt.get('amps', np.ones(m))
    weights = amps ** 2
    beff_rms = np.sqrt(np.sum(weights * fk_opt**2) / np.sum(weights))

    # PAPR measurement
    papr_db = 10 * np.log10(np.max(np.abs(x_opt)**2) / np.mean(np.abs(x_opt)**2))

    print(f"  Reconstruction SNR: {reconstruction_snr:.1f} dB")
    print(f"  RMS Bandwidth: {beff_rms/1e6:.1f} MHz")
    print(f"  PAPR: {papr_db:.1f} dB")

    return {
        'reconstruction_snr_db': reconstruction_snr,
        'rms_bandwidth_mhz': beff_rms / 1e6,
        'papr_db': papr_db
    }


def test_enhanced_aperture_detection():
    """Test enhanced aperture detection for sub-0dB SNR."""
    print("\nTesting Enhanced Aperture Detection...")

    # Test parameters
    fs = 5e6
    duration = 0.02
    df = 10e3
    m = 5
    test_snrs = [-5, 0, 5, 10, 15, 20]

    results = {
        'snr_db': [],
        'detection_rate': [],
        'snr_estimate_db': [],
        'freq_accuracy_hz': []
    }

    for snr_db in test_snrs:
        print(f"  Testing SNR = {snr_db} dB...")

        detection_count = 0
        snr_estimates = []
        freq_errors = []

        for trial in range(20):
            rng = np.random.default_rng(3000 + trial)

            # Generate test signal
            x, fk, _, _ = generate_comb(
                fs, duration, df, m=m, omit_fundamental=True, rng=rng, return_payload=True
            )

            # Apply impairments
            x = awgn(x, snr_db, rng)

            # Enhanced aperture detection
            peak_freq, peak_snr, quality_metrics = detect_df_peak_robust(
                x, fs, df, n_averages=3
            )

            # Check detection success
            freq_error = abs(peak_freq - df)
            if freq_error < df * 0.1 and quality_metrics['detection_confidence'] > 0.5:
                detection_count += 1

            snr_estimates.append(peak_snr)
            freq_errors.append(freq_error)

        # Store results
        detection_rate = detection_count / 20.0
        mean_snr_estimate = np.mean(snr_estimates)
        mean_freq_error = np.mean(freq_errors)

        results['snr_db'].append(snr_db)
        results['detection_rate'].append(detection_rate)
        results['snr_estimate_db'].append(mean_snr_estimate)
        results['freq_accuracy_hz'].append(mean_freq_error)

        print(f"    Detection Rate: {detection_rate:.1%}, SNR Est: {mean_snr_estimate:.1f} dB, Freq Err: {mean_freq_error:.0f} Hz")

    return results


def test_adaptive_frequency_offset():
    """Test adaptive frequency offset selection."""
    print("\nTesting Adaptive Frequency Offset Selection...")

    # Test parameters
    base_freq_hz = 2.4e9  # 2.4 GHz
    target_precision_ps = 10.0
    bandwidth_hz = 20e6
    test_snrs = [10, 20, 30]

    results = []

    for snr_db in test_snrs:
        print(f"  Testing SNR = {snr_db} dB...")

        # Get optimal offset
        optimal_offset, expected_precision, metadata = adaptive_frequency_offset_selection(
            base_freq_hz=base_freq_hz,
            target_precision_ps=target_precision_ps,
            bandwidth_hz=bandwidth_hz,
            snr_db=snr_db
        )

        print(f"    Optimal Offset: {optimal_offset/1e3:.1f} kHz")
        print(f"    Expected Precision: {expected_precision:.1f} ps")
        print(f"    Bandwidth Utilization: {metadata['bandwidth_utilization']:.1%}")

        results.append({
            'snr_db': snr_db,
            'optimal_offset_hz': optimal_offset,
            'expected_precision_ps': expected_precision,
            'bandwidth_utilization': metadata['bandwidth_utilization']
        })

    return results


def plot_results(wls_results, aperture_results, offset_results):
    """Generate performance comparison plots."""
    print("\nGenerating Performance Plots...")

    # WLS Performance Comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # RMSE/CRLB Ratio Comparison
    ax1.bar(['Traditional', 'Enhanced'], [wls_results['traditional_ratio'], wls_results['enhanced_ratio']])
    ax1.set_ylabel('RMSE/CRLB Ratio')
    ax1.set_title('WLS Performance Improvement')
    ax1.axhline(y=0.8, color='r', linestyle='--', alpha=0.7, label='Target: 0.8')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Improvement Percentage
    ax2.pie([wls_results['improvement_pct'], 100 - wls_results['improvement_pct']],
            labels=['Improvement', 'Remaining'],
            autopct='%1.1f%%', startangle=90)
    ax2.set_title('Performance Improvement')

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'wls_performance_improvement.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Aperture Detection Performance
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    snr_db = aperture_results['snr_db']
    detection_rate = aperture_results['detection_rate']

    # Detection Rate vs SNR
    ax1.plot(snr_db, detection_rate, 'bo-', linewidth=2, markersize=8)
    ax1.set_xlabel('SNR (dB)')
    ax1.set_ylabel('Detection Rate')
    ax1.set_title('Aperture Detection Performance')
    ax1.grid(True, alpha=0.3)
    ax1.axvline(x=0, color='r', linestyle='--', alpha=0.7, label='0 dB Threshold')
    ax1.legend()

    # SNR Estimation Accuracy
    snr_estimate = aperture_results['snr_estimate_db']
    ax2.plot(snr_db, snr_estimate, 'ro-', linewidth=2, markersize=8, label='Estimated')
    ax2.plot(snr_db, snr_db, 'k--', alpha=0.7, label='Ideal')
    ax2.set_xlabel('True SNR (dB)')
    ax2.set_ylabel('Estimated SNR (dB)')
    ax2.set_title('SNR Estimation Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'aperture_detection_performance.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Adaptive Offset Performance
    fig, ax = plt.subplots(figsize=(8, 6))

    for result in offset_results:
        snr_db = result['snr_db']
        offset_hz = result['optimal_offset_hz']
        precision_ps = result['expected_precision_ps']

        ax.scatter(snr_db, offset_hz/1e3, s=precision_ps*10, alpha=0.7,
                  label=f'SNR {snr_db}dB: {offset_hz/1e3:.0f}kHz')

    ax.set_xlabel('SNR (dB)')
    ax.set_ylabel('Optimal Offset (kHz)')
    ax.set_title('Adaptive Frequency Offset Selection')
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'adaptive_offset_selection.png', dpi=300, bbox_inches='tight')
    plt.close()


def generate_performance_report(all_results):
    """Generate comprehensive performance report."""
    print("\n" + "="*60)
    print("PHASE 1 ENHANCEMENT PERFORMANCE REPORT")
    print("="*60)

    wls_results = all_results['wls']
    aperture_results = all_results['aperture']
    offset_results = all_results['offset']

    print("\n1. ENHANCED WLS WEIGHTING:")
    print(f"   Traditional RMSE/CRLB: {wls_results['traditional_ratio']:.3f}")
    print(f"   Enhanced RMSE/CRLB:    {wls_results['enhanced_ratio']:.3f}")
    print(f"   Improvement:           {wls_results['improvement_pct']:.1f}%")
    print(f"   Target Achieved:       {'✓' if wls_results['target_achieved'] else '✗'}")

    print("\n2. OPTIMIZED COMB SYNTHESIS:")
    comb_results = all_results['comb']
    print(f"   Reconstruction SNR:    {comb_results['reconstruction_snr_db']:.1f} dB")
    print(f"   RMS Bandwidth:         {comb_results['rms_bandwidth_mhz']:.1f} MHz")
    print(f"   PAPR:                  {comb_results['papr_db']:.1f} dB")

    print("\n3. ENHANCED APERTURE DETECTION:")
    sub_zero_db_rate = aperture_results['detection_rate'][1]  # SNR = 0 dB
    print(f"   0 dB SNR Detection:    {sub_zero_db_rate:.1%}")
    print(f"   -5 dB SNR Detection:   {aperture_results['detection_rate'][0]:.1%}")

    print("\n4. ADAPTIVE FREQUENCY OFFSET:")
    for result in offset_results:
        snr = result['snr_db']
        offset = result['optimal_offset_hz']
        precision = result['expected_precision_ps']
        print(f"   SNR {snr:2d}dB: Offset {offset/1e3:4.0f}kHz → {precision:4.1f}ps precision")

    print("\n5. OVERALL ASSESSMENT:")
    targets_met = sum([
        wls_results['target_achieved'],
        comb_results['reconstruction_snr_db'] > 20.0,  # Good reconstruction SNR
        sub_zero_db_rate > 0.5,  # >50% detection at 0dB
    ])

    print(f"   Targets Met: {targets_met}/3")
    print(f"   Overall Status: {'EXCELLENT' if targets_met >= 2 else 'GOOD' if targets_met >= 1 else 'NEEDS_WORK'}")

    print("\n6. RECOMMENDATIONS:")
    if wls_results['enhanced_ratio'] >= 0.8:
        print("   - Consider further WLS optimization for sub-0.8 ratio")
    if sub_zero_db_rate < 0.5:
        print("   - Enhance aperture detection for better sub-0dB performance")
    if comb_results['papr_db'] > 10:
        print("   - Consider PAPR reduction techniques for hardware efficiency")

    print("="*60)


def main():
    """Run all Phase 1 enhancement tests."""
    print("Driftlock Phase 1 Enhancement Validation")
    print("Testing Choir Lab optimizations for breakthrough performance...")

    start_time = time.time()
    ensure_output_dirs()

    # Run all tests
    all_results = {}

    all_results['wls'] = test_enhanced_wls_weighting()
    all_results['comb'] = test_optimized_comb_synthesis()
    all_results['aperture'] = test_enhanced_aperture_detection()
    all_results['offset'] = test_adaptive_frequency_offset()

    # Generate plots
    plot_results(all_results['wls'], all_results['aperture'], all_results['offset'])

    # Generate comprehensive report
    generate_performance_report(all_results)

    # Save detailed results
    import json
    results_file = OUTPUT_DIR / 'phase1_enhancement_results.json'
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    elapsed_time = time.time() - start_time
    print(f"\nValidation completed in {elapsed_time:.1f} seconds")
    print(f"Results saved to: {results_file}")
    print(f"Figures saved to: {FIGURES_DIR}")


if __name__ == "__main__":
    main()