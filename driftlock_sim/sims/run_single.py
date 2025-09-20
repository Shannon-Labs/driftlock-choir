from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml
import matplotlib.pyplot as plt

from driftlock_sim.dsp.tx_comb import generate_comb
from driftlock_sim.dsp.channel_models import tapped_delay_channel, awgn
from driftlock_sim.dsp.impairments import (
    apply_cfo,
    apply_phase_noise,
    apply_sco,
    rapp_soft_clip,
    aperture_branch,
    cyclo_gate,
)
from driftlock_sim.dsp.rx_coherent import estimate_tone_phasors, unwrap_phase, wls_delay, per_tone_snr, estimate_noise_power
from driftlock_sim.dsp.rx_aperture import envelope_spectrum, detect_df_peak, choir_health_index
from driftlock_sim.dsp.metrics import papr_db
from driftlock_sim.dsp.crlb import delay_crlb_std


def ensure_dirs(root: Path) -> dict[str, Path]:
    out = {
        "csv": root / "csv",
        "figs": root / "figs",
        "movies": root / "movies",
        "logs": root / "logs",
    }
    for p in out.values():
        p.mkdir(parents=True, exist_ok=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    seed = int(cfg.get("seed", 2025))
    rng = np.random.default_rng(seed)
    fs = float(cfg["sample_rate_hz"]) 
    dur = float(cfg["duration_s"]) 
    txc = cfg["tx"]
    ch = cfg.get("channel", {})
    truth_tau = float(cfg.get("truth", {}).get("tau_s", 0.0))
    outroot = Path(cfg.get("output", {}).get("dir", "driftlock_sim/outputs"))
    run_id = cfg.get("output", {}).get("run_id", "demo")
    dirs = ensure_dirs(outroot)

    df = float(txc.get("df_hz", 10e3))
    m = int(txc.get("m_carriers", 5))

    x, fk, pilot_mask, meta = generate_comb(
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

    # Impairments and channel
    x = cyclo_gate(x, fs, float(ch.get("cyclo_gate_rate_hz", 0.0)))
    x = apply_cfo(x, fs, float(ch.get("cfo_hz", 0.0)))
    x = apply_sco(x, float(ch.get("sco_ppm", 0.0)))
    x = apply_phase_noise(x, float(ch.get("phase_noise_rad2", 0.0)), rng)
    x = tapped_delay_channel(x, fs, ch.get("taps", []))
    x = rapp_soft_clip(x, p=float(ch.get("rapp_p", 2.0)), sat=float(ch.get("rapp_sat", 1.5)))
    x = aperture_branch(x, alpha=float(ch.get("aperture_alpha", 2.0)), mix=float(ch.get("aperture_mix", 0.0)))
    x = awgn(x, float(ch.get("awgn_snr_db", 20.0)), rng)

    # Coherent path
    tx_phases = meta.get("phases") if isinstance(meta, dict) else None
    Yk = estimate_tone_phasors(x, fs, fk)
    if tx_phases is not None:
        Yk = Yk * np.exp(-1j * tx_phases)
    phu = unwrap_phase(np.angle(Yk), fk)
    noise_var = estimate_noise_power(x, fs, fk, Yk)
    snr_w = per_tone_snr(Yk, noise_var, len(x))
    tau_hat, ci95 = wls_delay(fk, phu, snr_w)

    # Aperture path
    f_env, E_env = envelope_spectrum(x, fs, alpha=float(ch.get("aperture_alpha", 2.0)))
    fpk, env_snr_db = detect_df_peak(f_env, E_env, df_hz=df)
    health = choir_health_index(f_env, E_env, df_hz=df)

    # Metrics
    beff = (np.max(fk) - np.min(fk)) if len(fk) > 1 else df
    snr_lin = 10 ** (float(ch.get("awgn_snr_db", 20.0)) / 10.0)
    crlb = delay_crlb_std(beff_hz=beff, snr_lin=snr_lin)
    papr = papr_db(x)
    # Approximate payload BER (if any): use SNR as Es/N0 proxy here
    from driftlock_sim.dsp.metrics import ber_qpsk_awgn
    ber = ber_qpsk_awgn(float(ch.get("awgn_snr_db", 20.0)))
    resid_ps = (tau_hat - truth_tau) * 1e12

    # Save figures
    # RF spectrum
    fig1 = plt.figure(figsize=(8, 4))
    X = np.fft.fftshift(np.fft.fft(x, n=8192))
    f = np.fft.fftshift(np.fft.fftfreq(8192, 1 / fs))
    plt.plot(f / 1e3, 20 * np.log10(np.abs(X) + 1e-12))
    plt.xlabel("Frequency (kHz)")
    plt.ylabel("Magnitude (dB)")
    plt.title("RF Spectrum (Comb)")
    fig1.tight_layout()
    fig1.savefig(dirs["figs"] / f"{run_id}_rf_spectrum.png", dpi=160)
    plt.close(fig1)

    # Envelope spectrum
    fig2 = plt.figure(figsize=(8, 4))
    plt.plot(f_env / 1e3, 20 * np.log10(E_env + 1e-12))
    plt.axvline(df / 1e3, color="r", linestyle="--", label="Δf")
    plt.legend()
    plt.xlabel("Frequency (kHz)")
    plt.ylabel("Env |FFT| (dB)")
    plt.title("Envelope Spectrum")
    fig2.tight_layout()
    fig2.savefig(dirs["figs"] / f"{run_id}_env_spectrum.png", dpi=160)
    plt.close(fig2)

    # Tau estimate strip (single point demo)
    fig3 = plt.figure(figsize=(6, 3))
    plt.axhline(truth_tau * 1e12, color="g", label="truth")
    plt.scatter([0], [tau_hat * 1e12], color="b", label="τ^ (ps)")
    plt.fill_between([-0.5, 0.5], (tau_hat - ci95) * 1e12, (tau_hat + ci95) * 1e12, color="b", alpha=0.2)
    plt.ylabel("τ (ps)")
    plt.title("Coherent Delay Estimate")
    plt.legend()
    plt.xticks([])
    fig3.tight_layout()
    fig3.savefig(dirs["figs"] / f"{run_id}_tau_estimate.png", dpi=160)
    plt.close(fig3)

    # CSV row
    row = {
        "seed": seed,
        "df_hz": df,
        "m": m,
        "beff_hz": beff,
        "snr_db": float(ch.get("awgn_snr_db", 20.0)),
        "tau_true_ps": truth_tau * 1e12,
        "tau_hat_ps": tau_hat * 1e12,
        "ci95_ps": ci95 * 1e12,
        "residual_ps": resid_ps,
        "env_df_snr_dB": env_snr_db,
        "health": health,
        "BER": ber,
        "PAPR_dB": papr,
        "CRLB_ps": crlb * 1e12,
    }
    (dirs["csv"] / f"{run_id}_single.json").write_text(json.dumps(row, indent=2), encoding="utf-8")
    print(json.dumps(row, indent=2))


if __name__ == "__main__":
    main()
