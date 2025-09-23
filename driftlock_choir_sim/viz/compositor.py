from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import yaml
from matplotlib.animation import FFMpegWriter, FuncAnimation

# Import simulation modules (assuming PYTHONPATH=.)
from driftlock_choir_sim.dsp.tx_comb import generate_comb
from driftlock_choir_sim.dsp.rx_coherent import estimate_tone_phasors, unwrap_phase, wls_delay, per_tone_snr, estimate_noise_power
from driftlock_choir_sim.dsp.rx_aperture import envelope_spectrum, detect_df_peak
from driftlock_choir_sim.dsp.channel_models import awgn, tapped_delay_channel
from driftlock_choir_sim.dsp.impairments import apply_cfo, apply_phase_noise, apply_sco, aperture_branch, rapp_soft_clip

def load_config(config_path: str) -> dict:
    """Load YAML config."""
    return yaml.safe_load(Path(config_path).read_text())

def simulate_frame(config: dict, frame_idx: int, rng: np.random.Generator, truth_tau: float = 0.0) -> dict:
    """Simulate a single frame for the given config."""
    fs = float(config.get("sample_rate_hz", 5e6))
    dur = float(config.get("duration_s", 0.05))
    txc = config.get("tx", {})
    ch = config.get("channel", {})
    movie_cfg = config.get("movie", {})
    nfft = int(movie_cfg.get("nfft", 4096))

    df = float(txc.get("df_hz", 10e3))
    m = int(txc.get("m_carriers", 5))

    x, fk, _, meta = generate_comb(
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
        return_payload=True,
    )

    x = apply_cfo(x, fs, float(ch.get("cfo_hz", 0.0)))
    x = apply_sco(x, float(ch.get("sco_ppm", 0.0)))
    x = apply_phase_noise(x, float(ch.get("phase_noise_rad2", 0.0)), rng)
    x = tapped_delay_channel(x, fs, ch.get("taps", []))
    x = rapp_soft_clip(x, p=float(ch.get("rapp_p", 2.0)), sat=float(ch.get("rapp_sat", 1.5)))
    x = aperture_branch(x, alpha=float(ch.get("aperture_alpha", 2.0)), mix=float(ch.get("aperture_mix", 0.0)))
    x = awgn(x, float(ch.get("awgn_snr_db", 20.0)), rng)

    # RF Spectrum
    Xi = np.fft.fftshift(np.fft.fft(x, n=nfft))
    f = np.fft.fftshift(np.fft.fftfreq(nfft, 1 / fs))
    rf_db = 20 * np.log10(np.abs(Xi) + 1e-12)

    # Envelope Spectrum
    f_env, E_env = envelope_spectrum(x, fs)
    env_db = 20 * np.log10(E_env + 1e-12)
    df_peak_hz, df_snr_db = detect_df_peak(f_env, E_env, df)

    # Coherent Estimate
    Yk = estimate_tone_phasors(x, fs, fk)
    if meta.get("phases") is not None:
        Yk = Yk * np.exp(-1j * meta["phases"])
    phu = unwrap_phase(np.angle(Yk), fk)
    base_noise_var = estimate_noise_power(x, fs, fk, Yk)
    base_snr_w = per_tone_snr(Yk, base_noise_var, len(x))
    _, base_ci_s, base_crlb_s = wls_delay(fk, phu, base_snr_w, return_stats=True)
    base_ci_ps = base_ci_s * 1e12
    base_crlb_ps = base_crlb_s * 1e12
    tau_hat, _ = wls_delay(fk, phu, base_snr_w)
    tau_ps = tau_hat * 1e12

    return {
        "f_khz": f / 1e3,
        "rf_db": rf_db,
        "f_env_khz": f_env / 1e3,
        "env_db": env_db,
        "df_snr_db": df_snr_db,
        "tau_ps": tau_ps,
        "ci_ps": base_ci_ps,
        "crlb_ps": base_crlb_ps,
        "truth_ps": truth_tau * 1e12,
        "frame": frame_idx
    }

def animate_split_screen(baseline_config: dict, demo_config: dict, num_frames: int = 360, seed: int = 2025, burn_in: bool = True):
    """Create animated split-screen comparison."""
    rng = np.random.default_rng(seed)
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle("Baseline vs Driftlock Split-Screen", fontsize=16, y=0.95)

    baseline_frames = []
    demo_frames = []
    for i in range(num_frames):
        baseline_frame = simulate_frame(baseline_config, i, rng)
        demo_frame = simulate_frame(demo_config, i, rng)
        baseline_frames.append(baseline_frame)
        demo_frames.append(demo_frame)

    def update(frame_idx):
        # Left: Baseline RF
        data = baseline_frames[frame_idx]
        ax_left.clear()
        ax_left.plot(data["f_khz"], data["rf_db"], color="tab:blue")
        ax_left.set_title("Baseline GNSS/PTP - RF Spectrum")
        ax_left.set_xlabel("Frequency (kHz)")
        ax_left.set_ylabel("Magnitude (dB)")
        ax_left.grid(alpha=0.3)
        if burn_in:
            ax_left.text(0.05, 0.95, f"Frame {frame_idx+1}", transform=ax_left.transAxes, fontsize=10,
                         bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

        # Right: Driftlock Envelope
        data = demo_frames[frame_idx]
        ax_right.clear()
        ax_right.plot(data["f_env_khz"], data["env_db"], color="tab:purple")
        ax_right.axvline(data["df_snr_db"], color="r", linestyle="--")
        ax_right.set_title("Driftlock Choir - Aperture Envelope")
        ax_right.set_xlabel("Frequency (kHz)")
        ax_right.set_ylabel("Env |FFT| (dB)")
        ax_right.grid(alpha=0.3)
        if burn_in:
            ax_right.text(0.05, 0.95, f"τ̂: {data['tau_ps']:.1f} ps | SNR: {data['df_snr_db']:.1f} dB", transform=ax_right.transAxes, fontsize=10,
                          bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    ani = FuncAnimation(fig, update, frames=num_frames, interval=1000/24, blit=False)
    output_dir = Path("driftlock_choir_sim/outputs/comparisons/side_by_side")
    output_dir.mkdir(parents=True, exist_ok=True)
    ani.save(output_dir / "comparison.mp4", writer=FFMpegWriter(fps=24))
    print(f"Animated split-screen saved to {output_dir / 'comparison.mp4'}")

    # Spritesheet
    grid_cols = max(1, num_frames // 10)
    fig, axs = plt.subplots(2, grid_cols, figsize=(20, 8))
    if grid_cols == 1:
        axs = np.array([[axs[0]], [axs[1]]])
    frame_indices = np.linspace(0, num_frames - 1, grid_cols, dtype=int)
    for col, idx in enumerate(frame_indices):
        baseline_data = baseline_frames[int(idx)]
        demo_data = demo_frames[int(idx)]
        axs[0, col].plot(baseline_data["f_khz"][:100], baseline_data["rf_db"][:100])  # Subset for size
        axs[1, col].plot(demo_data["f_env_khz"][:100], demo_data["env_db"][:100])
    plt.savefig(output_dir / "spritesheet.png", dpi=100)
    print(f"Spritesheet saved to {output_dir / 'spritesheet.png'}")

def main():
    parser = argparse.ArgumentParser(description="Create split-screen comparison reel.")
    parser.add_argument("--left-config", required=True, help="Path to left config (YAML or MP4)")
    parser.add_argument("--right-config", required=True, help="Path to right config (YAML or MP4)")
    parser.add_argument("--layout", default="side-by-side", choices=["side-by-side"], help="Layout preset")
    parser.add_argument("--burn-in-metrics", action="store_true", help="Add burn-in metrics text")
    parser.add_argument("--num-frames", type=int, default=360, help="Number of frames")
    parser.add_argument("--seed", type=int, default=2025, help="RNG seed")
    parser.add_argument("--output-name", default="comparison", help="Output name")
    args = parser.parse_args()

    # For now, assume YAML; TODO: handle MP4 input with ffmpeg subprocess
    left_config = load_config(args.left_config) if args.left_config.endswith('.yaml') else None
    right_config = load_config(args.right_config) if args.right_config.endswith('.yaml') else None

    if left_config is None or right_config is None:
        print("MP4 input not implemented yet; using YAML generation fallback.")
        left_config = load_config("driftlock_choir_sim/configs/baseline_teaser.yaml")
        right_config = load_config("driftlock_choir_sim/configs/demo_teaser.yaml")

    animate_split_screen(left_config, right_config, args.num_frames, args.seed, args.burn_in_metrics)

if __name__ == "__main__":
    main()
