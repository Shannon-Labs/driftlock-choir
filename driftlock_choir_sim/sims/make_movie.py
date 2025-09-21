from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib.animation import FFMpegWriter

from driftlock_choir_sim.dsp.channel_models import tapped_delay_channel, awgn
from driftlock_choir_sim.dsp.impairments import (
    aperture_branch,
    apply_cfo,
    apply_phase_noise,
    apply_sco,
    rapp_soft_clip,
)
from driftlock_choir_sim.dsp.rx_aperture import envelope_spectrum, detect_df_peak
from driftlock_choir_sim.dsp.rx_coherent import (
    estimate_noise_power,
    estimate_tone_phasors,
    per_tone_snr,
    unwrap_phase,
    wls_delay,
)
from driftlock_choir_sim.dsp.tx_comb import generate_comb
from driftlock_choir_sim.telemetry.collector import TelemetryCollector


EPS = 1e-12


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
    txc = cfg.get("tx", {})
    ch = cfg.get("channel", {})
    truth_tau = float(cfg.get("truth", {}).get("tau_s", 0.0))
    outroot = Path(cfg.get("output", {}).get("dir", "driftlock_choir_sim/outputs"))
    run_id = cfg.get("output", {}).get("run_id", "demo")
    movie_cfg = cfg.get("movie", {})
    telemetry_enabled = movie_cfg.get("telemetry", False)
    if telemetry_enabled:
        telemetry_path = outroot / f"{run_id}_telemetry.jsonl"
        collector = TelemetryCollector(telemetry_path)
    else:
        collector = None

    annotate = bool(movie_cfg.get("annotate", False))
    fps = int(movie_cfg.get("fps", 24))
    seconds = float(movie_cfg.get("seconds", 6.0))
    jitter_std = float(movie_cfg.get("jitter_noise_std", 0.01))
    tau_ylim_ps = movie_cfg.get("tau_ylim_ps")
    history_frames = movie_cfg.get("history_frames")
    nfft = int(movie_cfg.get("nfft", 8192))
    dpi = int(movie_cfg.get("dpi", 120))
    header_spacing = float(movie_cfg.get("header_spacing", 0.028))
    header_fontsize = int(movie_cfg.get("header_fontsize", 10))
    status_fontsize = int(movie_cfg.get("status_fontsize", 9))
    title_fontsize = int(movie_cfg.get("title_fontsize", 13))
    title_weight = movie_cfg.get("title_weight", "bold")
    info_box_cfg = movie_cfg.get("info_box")

    frames = max(1, int(np.ceil(seconds * fps)))

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

    tx_phases = meta.get("phases") if isinstance(meta, dict) else None

    movies_dir = ensure_dir(outroot / "movies")
    outfile = movies_dir / f"{run_id}_choir_sim.mp4"

    fig = plt.figure(figsize=(12.5, 6.5))
    gs = fig.add_gridspec(2, 2, width_ratios=(3, 2))
    ax_rf = fig.add_subplot(gs[:, 0])
    ax_env = fig.add_subplot(gs[0, 1])
    ax_tau = fig.add_subplot(gs[1, 1])

    tau_truth_ps = truth_tau * 1e12

    Xi = np.fft.fftshift(np.fft.fft(x, n=nfft))
    f = np.fft.fftshift(np.fft.fftfreq(nfft, 1 / fs))
    base_rf_db = 20 * np.log10(np.abs(Xi) + EPS)
    rf_line, = ax_rf.plot(f / 1e3, base_rf_db, color="tab:blue")
    ax_rf.set_title("RF Comb (Transmitter)")
    ax_rf.set_xlabel("Frequency (kHz)")
    ax_rf.set_ylabel("Magnitude (dB)")
    ax_rf.grid(alpha=0.2)
    ax_rf.set_ylim(base_rf_db.max() - 80.0, base_rf_db.max() + 5.0)

    f_env, E_env = envelope_spectrum(x, fs)
    base_env_db = 20 * np.log10(E_env + EPS)
    env_line, = ax_env.plot(f_env / 1e3, base_env_db, color="tab:purple")
    ax_env.axvline(df / 1e3, color="r", linestyle="--", label="Δf target")
    ax_env.axvline(2 * df / 1e3, color="g", linestyle=":", label="2Δf harmonic")
    ax_env.legend(loc="upper right", fontsize=9)
    ax_env.set_title("Aperture Envelope Spectrum")
    ax_env.set_xlabel("Frequency (kHz)")
    ax_env.set_ylabel("Env |FFT| (dB)")
    ax_env.grid(alpha=0.2)
    ax_env.set_xlim(f_env[0] / 1e3, f_env[-1] / 1e3)
    ax_env.set_ylim(base_env_db.max() - 80.0, base_env_db.max() + 5.0)

    df_peak_hz, df_snr_db = detect_df_peak(f_env, E_env, df)
    _, df2_snr_db = detect_df_peak(f_env, E_env, 2 * df)
    env_peak_hz = float(df_peak_hz)

    ax_tau.set_title("Coherent Delay Estimate (τ̂)")
    ax_tau.set_xlabel("Frame")
    ax_tau.set_ylabel("τ̂ (ps)")
    tau_line, = ax_tau.plot([], [], color="tab:blue", lw=1.5, label="τ̂ running")
    tau_marker, = ax_tau.plot([], [], marker="o", color="tab:orange", markersize=5, linestyle="none", label="Latest τ̂")
    truth_line = ax_tau.axhline(tau_truth_ps, color="tab:green", linestyle="--", lw=1.2, label="Truth τ")
    ax_tau.grid(alpha=0.2)

    base_Yk = estimate_tone_phasors(x, fs, fk)
    if tx_phases is not None:
        base_Yk = base_Yk * np.exp(-1j * tx_phases)
    phu_base = unwrap_phase(np.angle(base_Yk), fk)
    base_noise_var = estimate_noise_power(x, fs, fk, base_Yk)
    base_snr_w = per_tone_snr(base_Yk, base_noise_var, len(x))
    _, base_ci_s, base_crlb_s = wls_delay(fk, phu_base, base_snr_w, return_stats=True)
    base_ci_ps = base_ci_s * 1e12
    base_crlb_ps = base_crlb_s * 1e12

    tau_conf_band = ax_tau.axhspan(
        tau_truth_ps - base_ci_ps,
        tau_truth_ps + base_ci_ps,
        color="tab:green",
        alpha=0.08,
        label="95% CI band",
    )
    tau_crlb_line = ax_tau.axhline(
        tau_truth_ps + base_crlb_ps,
        color="tab:red",
        linestyle=":",
        lw=1.1,
        label="CRLB ±σ",
    )
    ax_tau.axhline(tau_truth_ps - base_crlb_ps, color="tab:red", linestyle=":", lw=1.1)

    for item in (truth_line, tau_conf_band, tau_crlb_line):
        item.set_zorder(1)

    ax_tau.legend(loc="upper right", fontsize=9)

    if tau_ylim_ps is not None:
        span = float(tau_ylim_ps)
    else:
        max_span = max(abs(base_ci_ps) * 3.0, abs(base_crlb_ps) * 6.0)
        span = max(50.0, abs(tau_truth_ps) * 0.2 + 50.0, max_span)
    ax_tau.set_ylim(tau_truth_ps - span, tau_truth_ps + span)

    if history_frames is None or int(history_frames) <= 0:
        history_frames = frames
    else:
        history_frames = int(history_frames)
    ax_tau.set_xlim(0, history_frames)

    status_text = None
    tau_overlay_text = None
    env_annotation = None
    legend_note = None

    if annotate:
        title_text = str(movie_cfg.get("title", f"Driftlock Choir Acceptance — {run_id}"))
        default_headers = [
            (
                f"Δf {df / 1e3:.1f} kHz • m={m} carriers • payload "
                f"{float(txc.get('payload_qpsk_fraction', 0.0)) * 100:.0f}% QPSK"
            ),
            (
                f"SNR {float(ch.get('awgn_snr_db', 20.0)):.1f} dB • CFO {float(ch.get('cfo_hz', 0.0)) / 1e3:.2f} kHz • "
                f"Aperture mix {float(ch.get('aperture_mix', 0.0)):.2f}"
            ),
            (
                f"Truth τ {tau_truth_ps:.1f} ps • 95% CI ±{base_ci_ps:.1f} ps • CRLB σ {base_crlb_ps:.1f} ps"
            ),
        ]
        header_lines = movie_cfg.get("header_lines", default_headers)
        if isinstance(header_lines, str):
            header_lines = [header_lines]
        y_pos = float(movie_cfg.get("title_y", 0.965))
        fig.text(
            0.02,
            y_pos,
            title_text,
            fontsize=title_fontsize,
            weight=title_weight,
            ha="left",
            va="top",
        )
        y_pos -= header_spacing
        for line in header_lines:
            fig.text(
                0.02,
                y_pos,
                str(line),
                fontsize=header_fontsize,
                ha="left",
                va="top",
            )
            y_pos -= header_spacing
        status_y = float(movie_cfg.get("status_y", y_pos))
        status_text = fig.text(0.02, status_y, "", fontsize=status_fontsize, ha="left", va="top")
        if info_box_cfg:
            fig.text(
                float(info_box_cfg.get("x", 0.62)),
                float(info_box_cfg.get("y", 0.88)),
                str(info_box_cfg.get("text", "")),
                fontsize=int(info_box_cfg.get("fontsize", 9)),
                ha=info_box_cfg.get("ha", "left"),
                va=info_box_cfg.get("va", "top"),
                color=info_box_cfg.get("color", "#1a1a1a"),
                bbox={
                    "boxstyle": info_box_cfg.get("boxstyle", "round"),
                    "facecolor": info_box_cfg.get("facecolor", "#f4f4f4"),
                    "alpha": float(info_box_cfg.get("alpha", 0.85)),
                    "edgecolor": info_box_cfg.get("edgecolor", "#a1a1a1"),
                },
            )
        tau_overlay_text = ax_tau.text(
            0.02,
            0.92,
            "",
            transform=ax_tau.transAxes,
            fontsize=10,
            ha="left",
            va="top",
            color="tab:blue",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.75, edgecolor="tab:blue"),
        )
        env_annotation = ax_env.text(
            0.98,
            0.92,
            (
                f"Δf target {df / 1e3:.1f} kHz\npeak {env_peak_hz / 1e3:.1f} kHz\nΔf SNR {df_snr_db:.1f} dB"
                f"\n2Δf SNR {df2_snr_db:.1f} dB"
            ),
            transform=ax_env.transAxes,
            fontsize=9,
            ha="right",
            va="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.75, edgecolor="tab:red"),
        )
        legend_note = fig.text(
            0.02,
            0.05,
            (
                "How to read: left → transmitter comb amplitude; top-right → aperture detects the missing fundamental Δf; "
                "bottom-right → coherent τ̂ hugging the truth line while tracking CI and CRLB."
            ),
            fontsize=9,
            ha="left",
            va="bottom",
            color="#333333",
        )
        fig.subplots_adjust(
            left=float(movie_cfg.get("layout_left", 0.08)),
            right=float(movie_cfg.get("layout_right", 0.97)),
            bottom=float(movie_cfg.get("layout_bottom", 0.10)),
            top=float(movie_cfg.get("layout_top", 0.78)),
            wspace=float(movie_cfg.get("layout_wspace", 0.28)),
            hspace=float(movie_cfg.get("layout_hspace", 0.35)),
        )
    else:
        fig.tight_layout()

    frame_rng = np.random.default_rng(seed + 7919)

    try:
        writer = FFMpegWriter(fps=fps)
    except Exception:  # pragma: no cover - ffmpeg availability varies
        print("FFMpeg not available; cannot write MP4.")
        return

    full_history_ps: list[float] = []

    print(f"Rendering {frames} frames @ {fps} fps → {outfile}")
    with writer.saving(fig, str(outfile), dpi=dpi):
        for i in range(frames):
            rf_line.set_ydata(base_rf_db + frame_rng.normal(scale=0.15, size=base_rf_db.shape))
            env_line.set_ydata(base_env_db + frame_rng.normal(scale=0.15, size=base_env_db.shape))

            measurement_noise = (
                frame_rng.normal(scale=jitter_std, size=base_Yk.shape)
                + 1j * frame_rng.normal(scale=jitter_std, size=base_Yk.shape)
            )
            Yk = base_Yk + measurement_noise

            phu = unwrap_phase(np.angle(Yk), fk)
            tau_hat, _ = wls_delay(fk, phu, base_snr_w)

            tau_ps = float(tau_hat * 1e12)
            full_history_ps.append(tau_ps)

            frame_indices = np.arange(len(full_history_ps))
            start_idx = max(0, len(full_history_ps) - history_frames)
            tau_window = full_history_ps[start_idx:]
            tau_line.set_data(frame_indices[start_idx:], tau_window)
            tau_marker.set_data([frame_indices[-1]], [tau_ps])
            ax_tau.set_xlim(frame_indices[start_idx], max(frame_indices[-1], frame_indices[start_idx] + history_frames))

            if annotate:
                err_ps = tau_ps - tau_truth_ps
                rmse_ps = float(
                    np.sqrt(np.mean((np.asarray(full_history_ps) - tau_truth_ps) ** 2))
                )
                crlb_ratio = rmse_ps / max(base_crlb_ps, 1e-9)
                if telemetry_enabled:
                    collector.add_frame(i, tau_ps, base_ci_ps, base_crlb_ps, df_snr_db, rmse_ps, tau_truth_ps)
                status_prefix = movie_cfg.get("status_prefix", "Frame")
                status_text.set_text(
                    (
                        f"{status_prefix} {i + 1:04d}/{frames} • τ̂ {tau_ps:.1f} ps • error {err_ps:+.1f} ps"
                        f" • RMSE {rmse_ps:.1f} ps (/{base_crlb_ps:.1f} ps = {crlb_ratio:.2f}× CRLB)"
                    )
                )
                tau_overlay_text.set_text(
                    (
                        f"Latest τ̂ {tau_ps:.1f} ps\n|τ̂−τ| {abs(err_ps):.1f} ps\n"
                        f"95% CI ±{base_ci_ps:.1f} ps\nCRLB σ {base_crlb_ps:.1f} ps"
                    )
                )

            writer.grab_frame()

    if telemetry_enabled:
        collector.save()
        print(f"Telemetry saved to {telemetry_path}")
    plt.close(fig)
    print(f"Wrote movie → {outfile}")


if __name__ == "__main__":
    main()
