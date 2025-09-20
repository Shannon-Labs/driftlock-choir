from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import yaml
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter

from driftlock_sim.dsp.tx_comb import generate_comb
from driftlock_sim.dsp.channel_models import tapped_delay_channel, awgn
from driftlock_sim.dsp.impairments import apply_cfo, apply_phase_noise, apply_sco, rapp_soft_clip, aperture_branch
from driftlock_sim.dsp.rx_coherent import estimate_tone_phasors, unwrap_phase, wls_delay
from driftlock_sim.dsp.rx_aperture import envelope_spectrum


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    seed = int(cfg.get("seed", 2025))
    rng = np.random.default_rng(seed)
    fs = float(cfg.get("sample_rate_hz", 5e6))
    dur = float(cfg.get("duration_s", 0.05))
    txc = cfg["tx"]
    ch = cfg.get("channel", {})
    truth_tau = float(cfg.get("truth", {}).get("tau_s", 0.0))
    outroot = Path(cfg.get("output", {}).get("dir", "driftlock_sim/outputs"))
    run_id = cfg.get("output", {}).get("run_id", "demo")
    movies = ensure_dir(outroot / "movies")

    df = float(txc.get("df_hz", 10e3))
    m = int(txc.get("m_carriers", 5))

    # Precompute signal
    x, fk, _ = generate_comb(
        fs=fs,
        duration=dur,
        df=df,
        m=m,
        omit_fundamental=bool(txc.get("omit_fundamental", True)),
        amp_mode=str(txc.get("amplitudes", "equal")),
        phase_mode=str(txc.get("phase_schedule", "newman")),
        payload_qpsk_fraction=float(txc.get("payload_qpsk_fraction", 0.0)),
        payload_symbol_rate=float(txc.get("payload_symbol_rate", 1000.0)),
        rng=rng,
    )

    x = apply_cfo(x, fs, float(ch.get("cfo_hz", 0.0)))
    x = apply_sco(x, float(ch.get("sco_ppm", 0.0)))
    x = apply_phase_noise(x, float(ch.get("phase_noise_rad2", 0.0)), rng)
    x = tapped_delay_channel(x, fs, ch.get("taps", []))
    x = rapp_soft_clip(x, p=float(ch.get("rapp_p", 2.0)), sat=float(ch.get("rapp_sat", 1.5)))
    x = aperture_branch(x, alpha=float(ch.get("aperture_alpha", 2.0)), mix=float(ch.get("aperture_mix", 0.0)))
    x = awgn(x, float(ch.get("awgn_snr_db", 20.0)), rng)

    # Prepare movie layout
    fig = plt.figure(figsize=(12, 6))
    gs = fig.add_gridspec(2, 2)
    ax_rf = fig.add_subplot(gs[:, 0])
    ax_env = fig.add_subplot(gs[0, 1])
    ax_tau = fig.add_subplot(gs[1, 1])

    # Precompute spectra
    NFFT = 8192
    X = np.fft.fftshift(np.fft.fft(x, n=NFFT))
    f = np.fft.fftshift(np.fft.fftfreq(NFFT, 1 / fs))
    f_env, E_env = envelope_spectrum(x, fs)

    # Initial plots
    rf_line, = ax_rf.plot(f/1e3, 20*np.log10(np.abs(X)+1e-12))
    ax_rf.set_title("RF Spectrum (Comb)")
    ax_rf.set_xlabel("Frequency (kHz)")
    ax_rf.set_ylabel("Magnitude (dB)")

    env_line, = ax_env.plot(f_env/1e3, 20*np.log10(E_env+1e-12))
    ax_env.axvline(df/1e3, color='r', linestyle='--', label='Δf')
    ax_env.legend()
    ax_env.set_title("Envelope Spectrum")
    ax_env.set_xlabel("Frequency (kHz)")
    ax_env.set_ylabel("Env |FFT| (dB)")

    ax_tau.set_title("Coherent Delay Estimate (ps)")
    ax_tau.set_xlabel("Frame")
    ax_tau.set_ylabel("τ (ps)")
    ax_tau.axhline(truth_tau*1e12, color='g', label='truth')
    tau_scatter = ax_tau.scatter([], [], c='b', s=12)
    ax_tau.legend()

    # Writer
    outfile = movies / f"{run_id}_choir_sim.mp4"
    try:
        writer = FFMpegWriter(fps=int(cfg.get("movie", {}).get("fps", 24)))
    except Exception:
        print("FFMpeg not available; cannot write MP4.")
        return

    frames = int(cfg.get("movie", {}).get("seconds", 6) * cfg.get("movie", {}).get("fps", 24))
    with writer.saving(fig, str(outfile), dpi=120):
        for i in range(frames):
            # Jitter SNR slightly per frame to keep visuals dynamic
            xi = x + np.random.default_rng(i).normal(scale=0.01, size=x.shape)
            Yk = estimate_tone_phasors(xi, fs, fk)
            tau_hat, _ = wls_delay(fk, np.unwrap(np.angle(Yk)), np.ones_like(fk))
            # Update tau plot
            xs = np.arange(i+1)
            ys = np.full(i+1, np.nan)
            ys[-1] = tau_hat*1e12
            ax_tau.scatter(xs, ys, c='b', s=12)
            writer.grab_frame()

    print(f"Wrote movie → {outfile}")


if __name__ == "__main__":
    main()
