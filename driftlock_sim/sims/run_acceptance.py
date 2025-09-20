from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt

from driftlock_sim.dsp.tx_comb import generate_comb
from driftlock_sim.dsp.channel_models import tapped_delay_channel, awgn
from driftlock_sim.dsp.impairments import apply_cfo, apply_phase_noise, apply_sco
from driftlock_sim.dsp.rx_coherent import (estimate_tone_phasors, unwrap_phase, wls_delay,
                                          per_tone_snr, estimate_noise_power)
from driftlock_sim.dsp.rx_aperture import envelope_spectrum, detect_df_peak
from driftlock_sim.dsp.crlb import delay_crlb_std, delay_crlb_rms_bandwidth
from driftlock_sim.dsp.time_delay import impose_fractional_delay_fft
from driftlock_sim.dsp.metrics import rms_bandwidth_hz, ber_qpsk_awgn


OUT = Path("driftlock_sim/outputs")


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def apply_defaults(x: np.ndarray, fs: float, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    x = apply_cfo(x, fs, 0.0)
    x = apply_sco(x, 0.0)
    x = apply_phase_noise(x, 0.0, rng)
    x = awgn(x, snr_db, rng)
    return x


def coherent_tau(fs: float, x: np.ndarray, fk: np.ndarray, reference_phases: np.ndarray | None = None) -> float:
    """Coherent delay estimate using per-tone SNR weighting."""
    Yk = estimate_tone_phasors(x, fs, fk)
    if reference_phases is not None:
        Yk = Yk * np.exp(-1j * reference_phases)
    phu = unwrap_phase(np.angle(Yk), fk)

    noise_power = estimate_noise_power(x, fs, fk, Yk)
    snr_per_tone = per_tone_snr(Yk, noise_power, len(x))

    tau_hat, _ = wls_delay(fk, phu, snr_per_tone)
    return float(tau_hat)


def apply_fractional_delay(x: np.ndarray, fs: float, tau_s: float) -> np.ndarray:
    if tau_s == 0.0:
        return x
    X = np.fft.fft(x)
    f = np.fft.fftfreq(len(x), 1 / fs)
    Y = X * np.exp(-1j * 2 * np.pi * f * tau_s)
    y = np.fft.ifft(Y)
    return y.astype(complex)


def acceptance_aperture() -> dict:
    seed = 2025
    rng = np.random.default_rng(seed)
    fs = 5e6
    dur = 0.02  # Reduced duration for speed
    df = 10e3
    m = 3
    x, fk, _ = generate_comb(fs, dur, df, m=m, omit_fundamental=True)
    x = apply_defaults(x, fs, snr_db=20.0, rng=rng)
    f, E = envelope_spectrum(x, fs)
    fpk, snr_db = detect_df_peak(f, E, df)
    # Heuristic: detect pairwise differences via second peak near 2*df
    _, snr_db_2 = detect_df_peak(f, E, 2*df)
    # Save a quick figure
    fig = plt.figure(figsize=(8,4))
    plt.plot(f/1e3, 20*np.log10(E+1e-12))
    plt.axvline(df/1e3, color='r', linestyle='--', label='Δf')
    plt.axvline(2*df/1e3, color='g', linestyle=':', label='2Δf')
    plt.legend(); plt.xlabel('Frequency (kHz)'); plt.ylabel('Env |FFT| (dB)'); plt.title('Aperture Acceptance')
    ensure_dir(OUT/"figs").mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT/"figs"/"accept_aperture.png", dpi=160); plt.close(fig)
    return {
        "env_df_snr_dB": snr_db,
        "env_2df_snr_dB": snr_db_2,
        "df_hz": df,
        "passed": (snr_db >= 15.0),
    }



def acceptance_coherent_precision() -> dict:
    seed = 2026
    fs = 20e6  # Nyquist-safe sample rate
    dur = 9e-3  # ~9 ms for fine frequency resolution while staying fast
    df = 1e6
    m = 21
    truth_tau = 100e-12
    trials = 14

    taus: list[float] = []
    cis: list[float] = []
    crlbs: list[float] = []
    validation_results: list[bool] = []

    ref_rng = np.random.default_rng(seed)
    x_ref, fk_ref, _, _ = generate_comb(
        fs,
        dur,
        df,
        m=m,
        omit_fundamental=True,
        rng=ref_rng,
        return_payload=True,
    )
    ref_Yk = estimate_tone_phasors(x_ref, fs, fk_ref)
    ref_phase = np.angle(ref_Yk)

    for i in range(trials):
        r = np.random.default_rng(seed + i)
        x, fk, _, _ = generate_comb(
            fs,
            dur,
            df,
            m=m,
            omit_fundamental=True,
            rng=r,
            return_payload=True,
        )

        x = impose_fractional_delay_fft(x, fs, truth_tau)
        x = apply_defaults(x, fs, snr_db=30.0, rng=r)

        Yk = estimate_tone_phasors(x, fs, fk) * np.exp(-1j * ref_phase)
        phu = unwrap_phase(np.angle(Yk), fk)

        noise_power = estimate_noise_power(x, fs, fk, Yk)
        snr_per_tone = per_tone_snr(Yk, noise_power, len(x))

        tau_hat, ci, crlb_std = wls_delay(fk, phu, snr_per_tone, return_stats=True)
        validation = validate_delay_estimate(tau_hat, fk, phu, ci)
        validation_results.append(validation["unwrap_sanity"])

        taus.append(tau_hat)
        cis.append(ci)
        crlbs.append(crlb_std)

    taus = np.asarray(taus)
    cis = np.asarray(cis)
    crlbs = np.asarray(crlbs)
    rmse = float(np.sqrt(np.mean((taus - truth_tau) ** 2)))

    from driftlock_sim.dsp.phase_schedules import amplitude_taper

    fk_summary = fk_ref
    amps = amplitude_taper(m, "equal")
    weights = amps ** 2
    beff_rms = rms_bandwidth_hz(fk_summary, weights=weights)
    crlb_ref = float(np.mean(crlbs)) if len(crlbs) else float("inf")

    ci_coverage = float(np.mean(np.abs(taus - truth_tau) <= cis)) if len(cis) else 0.0
    unwrap_sanity_rate = float(np.mean(validation_results)) if validation_results else 0.0
    ratio = rmse / crlb_ref if crlb_ref > 0 else np.inf

    return {
        "rmse_ps": rmse * 1e12,
        "crlb_ps": crlb_ref * 1e12,
        "ratio": ratio,
        "ci_coverage": ci_coverage,
        "unwrap_sanity_rate": unwrap_sanity_rate,
        "beff_rms_hz": beff_rms,
        "fs_hz": fs,
        "duration_s": dur,
        "n_carriers": m,
        "df_hz": df,
        "truth_tau_ps": truth_tau * 1e12,
        "n_trials": trials,
        "passed": (
            (rmse * 1e12 <= 120.0)
            and (ratio <= 1.5 + 1e-6)
            and (ci_coverage >= 0.90)
            and (unwrap_sanity_rate >= 0.95)
        ),
    }



def acceptance_robustness() -> dict:
    seed = 2027
    rng = np.random.default_rng(seed)
    fs = 5e6
    dur = 0.02  # Reduced duration for speed
    df = 10e3
    m = 5
    x, fk, _, _ = generate_comb(fs, dur, df, m=m, omit_fundamental=True, return_payload=True)
    reference_phases = np.angle(estimate_tone_phasors(x, fs, fk))
    x = apply_defaults(x, fs, snr_db=0.0, rng=rng)
    f, E = envelope_spectrum(x, fs)
    fpk, snr_db = detect_df_peak(f, E, df)
    tau_hat = coherent_tau(fs, x, fk, reference_phases)
    return {
        "env_df_snr_dB": snr_db,
        "tau_hat_ps": tau_hat * 1e12,
        "passed": (np.isfinite(snr_db) and snr_db > 0.0 and np.isfinite(tau_hat)),
    }



def acceptance_payload() -> dict:
    seed = 2028
    fs = 5e6
    dur = 1.0e-2  # 10 ms for improved averaging while staying within runtime budget
    df = 100e3
    m = 16
    snr_db = 20.0
    trials = 12
    symbol_rate = 5e3

    def run(payload_frac: float) -> Tuple[float, float, float]:
        tau_errs: list[float] = []
        total_bits = 0
        total_bit_errs = 0

        for i in range(trials):
            r = np.random.default_rng(seed + i)
            x, fk, pilot_mask, meta = generate_comb(
                fs,
                dur,
                df,
                m=m,
                omit_fundamental=True,
                payload_qpsk_fraction=payload_frac,
                payload_symbol_rate=symbol_rate,
                rng=r,
                return_payload=True,
            )

            payload_indices = meta["payload_indices"]
            sym_dur = int(meta["samples_per_symbol"])
            tx_symbols_all = meta["payload_symbols"]
            reference_phases_full = np.angle(estimate_tone_phasors(x, fs, fk))
            pilot_indices = np.where(pilot_mask)[0]
            fk_pilots = fk[pilot_indices]
            reference_phases = reference_phases_full[pilot_indices]
            n_full_symbols = len(x) // sym_dur
            tx_syms = tx_symbols_all[:n_full_symbols] if tx_symbols_all.size else np.array([], dtype=complex)

            x_noisy = apply_defaults(x, fs, snr_db, r)
            tau_hat = coherent_tau(fs, x_noisy, fk_pilots, reference_phases)
            tau_errs.append(tau_hat)

            if payload_frac <= 0 or payload_indices.size == 0 or n_full_symbols == 0 or tx_symbols_all.size == 0:
                continue

            t = np.arange(len(x_noisy)) / fs
            for idx in payload_indices:
                carrier_freq = fk[idx]
                baseband = x_noisy * np.exp(-1j * 2 * np.pi * carrier_freq * t)

                symbols_rx = []
                for k in range(n_full_symbols):
                    start = k * sym_dur
                    end = start + sym_dur
                    if end > len(baseband):
                        break
                    symbols_rx.append(np.mean(baseband[start:end]))

                if not symbols_rx:
                    continue

                rx_syms = np.array(symbols_rx)
                tx_syms_aligned = tx_syms[: len(rx_syms)]
                if not len(tx_syms_aligned):
                    continue

                rot = np.mean(rx_syms * np.conj(tx_syms_aligned))
                if np.abs(rot) > 0:
                    rx_syms *= np.exp(-1j * np.angle(rot))
                rx_syms /= np.sqrt(np.mean(np.abs(rx_syms) ** 2) + 1e-18)

                tx_bits = symbols_to_bits(tx_syms_aligned)
                rx_bits = symbols_to_bits(rx_syms[: len(tx_syms_aligned)])
                total_bits += len(tx_bits)
                total_bit_errs += np.sum(tx_bits != rx_bits)

        taus = np.asarray(tau_errs)
        rmse = float(np.sqrt(np.mean((taus - 0.0) ** 2)))
        observed_ber = (total_bit_errs / total_bits) if total_bits else 0.0
        analytic_ber = ber_qpsk_awgn(snr_db)

        return rmse, observed_ber, analytic_ber

    rmse_no, _, _ = run(0.0)
    rmse_pl, obs_ber, ana_ber = run(1.0 / 8.0)
    worsen = (rmse_pl - rmse_no) / (rmse_no + 1e-12)

    return {
        "rmse_no_payload_ps": rmse_no * 1e12,
        "rmse_with_payload_ps": rmse_pl * 1e12,
        "worsening_pct": 100.0 * worsen,
        "observed_ber": obs_ber,
        "analytic_ber": ana_ber,
        "passed": (worsen <= 0.25 + 1e-6) and (obs_ber < 1e-3),
    }



def symbols_to_bits(symbols: np.ndarray) -> np.ndarray:
    """Convert QPSK symbols to bits using hard decisions."""
    mags = np.abs(symbols)
    mags = np.where(mags > 0, mags, 1.0)
    norm_syms = symbols / mags
    norm_syms = np.clip(norm_syms.real, -1, 1) + 1j * np.clip(norm_syms.imag, -1, 1)

    bits = []
    for sym in norm_syms:
        bits.append(1 if sym.real < 0 else 0)
        bits.append(1 if sym.imag < 0 else 0)
    return np.array(bits, dtype=int)



def sanity_check_phasor_unwrap(phases: np.ndarray, frequencies: np.ndarray) -> bool:
    """Check phasor unwrap stability and expected phase slope.

    Args:
        phases: Unwrapped phases in radians
        frequencies: Tone frequencies in Hz

    Returns:
        True if sanity checks pass
    """
    if len(phases) < 2:
        return True  # Can't check with < 2 points

    # Check for phase jumps > π (indicates unwrap failure)
    phase_diffs = np.abs(np.diff(phases))
    max_phase_jump = np.max(phase_diffs)
    if max_phase_jump > np.pi:
        return False  # Unwrap failure detected

    # Check that phase slope is negative for positive frequencies
    # (positive delay should give negative phase slope)
    if len(frequencies) > 1:
        # Fit linear trend
        slope = np.polyfit(frequencies, phases, 1)[0]
        # For positive delay, we expect negative slope (phase decreases with frequency)
        if slope > 0:
            return False  # Unexpected positive slope

    return True


def validate_delay_estimate(tau_hat: float, frequencies: np.ndarray,
                          phases: np.ndarray, ci: float) -> dict:
    """Validate delay estimate with comprehensive checks.

    Returns:
        Dict with validation results
    """
    results = {
        "tau_hat_ps": tau_hat * 1e12,
        "ci_ps": ci * 1e12,
        "unwrap_sanity": sanity_check_phasor_unwrap(phases, frequencies),
        "finite_tau": np.isfinite(tau_hat),
        "reasonable_ci": ci > 0 and ci < 1e-9,  # CI should be positive and < 1ns
        "reasonable_tau": abs(tau_hat) < 1e-9,  # |τ| should be < 1ns
    }

    results["overall_valid"] = all([
        results["unwrap_sanity"],
        results["finite_tau"],
        results["reasonable_ci"],
        results["reasonable_tau"]
    ])

    return results


def executive_summary_pdf(data: dict) -> Path:
    figs_dir = ensure_dir(OUT / "figs")
    pdf_path = figs_dir / "executive_summary.pdf"
    fig = plt.figure(figsize=(8.5, 11))
    ax = fig.add_subplot(111)
    ax.axis('off')

    # Extract data for detailed reporting
    ap = data['aperture']
    coh = data['coherent']
    rob = data['robustness']
    pay = data['payload']

    lines = [
        "Driftlock Choir — Acceptance Summary",
        "=" * 50,
        f"Generated: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "1. APERTURE RECONSTRUCTION",
        f"   Δf SNR: {ap['env_df_snr_dB']:.1f} dB (>=15 dB required)",
        f"   2Δf SNR: {ap['env_2df_snr_dB']:.1f} dB",
        f"   Status: {'PASS' if ap['passed'] else 'FAIL'}",
        "",
        "2. COHERENT DELAY PRECISION",
        f"   RMSE: {coh['rmse_ps']:.1f} ps",
        f"   CRLB: {coh['crlb_ps']:.1f} ps",
        f"   RMSE/CRLB ratio: {coh['ratio']:.2f} (<=1.5 required)",
        f"   CI coverage: {coh['ci_coverage']:.1%} (>=90% required)",
        f"   Unwrap sanity rate: {coh['unwrap_sanity_rate']:.1%} (>=95% required)",
        f"   RMS bandwidth: {coh['beff_rms_hz']/1e6:.1f} MHz",
        f"   Configuration: fs={coh['fs_hz']/1e6:.0f}MHz, dur={coh['duration_s']*1000:.0f}ms",
        f"   Trials: {coh['n_trials']}, Truth τ: {coh['truth_tau_ps']:.0f} ps",
        f"   Status: {'PASS' if coh['passed'] else 'FAIL'}",
        "",
        "3. ROBUSTNESS (0 dB SNR)",
        f"   Δf SNR: {rob['env_df_snr_dB']:.1f} dB",
        f"   τ̂: {rob['tau_hat_ps']:.1f} ps (finite required)",
        f"   Status: {'PASS' if rob['passed'] else 'FAIL'}",
        "",
        "4. PAYLOAD COEXISTENCE",
        f"   RMSE without payload: {pay['rmse_no_payload_ps']:.1f} ps",
        f"   RMSE with payload: {pay['rmse_with_payload_ps']:.1f} ps",
        f"   Worsening: {pay['worsening_pct']:.1f}% (<=25% required)",
        f"   Observed BER: {pay['observed_ber']:.2e} (<1e-3 required)",
        f"   Analytic BER: {pay['analytic_ber']:.2e}",
        f"   Status: {'PASS' if pay['passed'] else 'FAIL'}",
        "",
        "OVERALL STATUS: " + ("ALL TESTS PASSED" if all([
            ap['passed'], coh['passed'], rob['passed'], pay['passed']
        ]) else "SOME TESTS FAILED"),
        "",
        "PERFORMANCE METRICS",
        f"   Total runtime: {data['metadata']['duration_s']:.1f}s (<60s required)",
        f"   Python {data['metadata']['python_version']}, NumPy {data['metadata']['numpy_version']}",
    ]

    ax.text(0.05, 0.95, "\n".join(lines), va='top', fontsize=10, family='monospace')
    fig.savefig(pdf_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return pdf_path


def main() -> None:
    import time
    start_time = time.time()

    ensure_dir(OUT)
    print("Driftlock Choir - Running Acceptance Tests...")

    # Run all acceptance tests
    print("1. Running aperture test...")
    ap = acceptance_aperture()
    print(f"   Completed in {time.time() - start_time:.1f}s")

    print("2. Running coherent precision test...")
    t2 = time.time()
    coh = acceptance_coherent_precision()
    print(f"   Completed in {time.time() - t2:.1f}s")

    print("3. Running robustness test...")
    t3 = time.time()
    rob = acceptance_robustness()
    print(f"   Completed in {time.time() - t3:.1f}s")

    print("4. Running payload test...")
    t4 = time.time()
    pay = acceptance_payload()
    print(f"   Completed in {time.time() - t4:.1f}s")

    res = {
        "aperture": ap,
        "coherent": coh,
        "robustness": rob,
        "payload": pay,
    }

    # Add metadata
    res["metadata"] = {
        "timestamp": time.time(),
        "duration_s": time.time() - start_time,
        "python_version": f"{__import__('sys').version_info.major}.{__import__('sys').version_info.minor}",
        "numpy_version": __import__('numpy').__version__,
    }

    ensure_dir(OUT / "csv")

    def pyify(o):
        if isinstance(o, dict):
            return {k: pyify(v) for k, v in o.items()}
        if isinstance(o, list):
            return [pyify(v) for v in o]
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.bool_,)):
            return bool(o)
        return o

    # Save detailed results
    json_path = OUT / "csv" / "acceptance_summary.json"
    json_path.write_text(json.dumps(pyify(res), indent=2), encoding="utf-8")

    # Generate executive summary PDF
    pdf = executive_summary_pdf(res)

    # Print summary to console
    print(f"\nAcceptance test completed in {time.time() - start_time:.1f}s")
    print(f"Results saved to: {json_path}")
    print(f"Executive summary: {pdf}")

    # Print pass/fail summary
    tests = ["aperture", "coherent", "robustness", "payload"]
    passed = [res[test]["passed"] for test in tests]
    print(f"\nTest Results: {sum(passed)}/{len(passed)} passed")

    for test in tests:
        status = "PASS" if res[test]["passed"] else "FAIL"
        print(f"  {test}: {status}")

    print(json.dumps(pyify({**res, "summary_pdf": str(pdf)}), indent=2))


if __name__ == "__main__":
    main()
