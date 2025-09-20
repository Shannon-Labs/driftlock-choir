from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from driftlock_sim.dsp.tx_comb import generate_comb
from driftlock_sim.dsp.channel_models import tapped_delay_channel, awgn
from driftlock_sim.dsp.impairments import apply_cfo, apply_phase_noise, apply_sco, rapp_soft_clip, aperture_branch
from driftlock_sim.dsp.rx_coherent import estimate_tone_phasors, unwrap_phase, wls_delay, per_tone_snr, estimate_noise_power
from driftlock_sim.dsp.rx_aperture import envelope_spectrum, detect_df_peak
from driftlock_sim.dsp.metrics import papr_db
from driftlock_sim.dsp.crlb import delay_crlb_std


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())

    outroot = Path(cfg.get("output", {}).get("dir", "driftlock_sim/outputs"))
    run_id = cfg.get("output", {}).get("run_id", "sweep")
    csv_path = ensure_dir(outroot / "csv") / f"{run_id}.csv"

    fs = float(cfg.get("sample_rate_hz", 5e6))
    dur = float(cfg.get("duration_s", 0.02))
    base_seed = int(cfg.get("seed", 2025))

    grid = cfg.get("sweep", {})
    snr_list = [float(x) for x in grid.get("snr_db", [20])]
    m_list = [int(x) for x in grid.get("m_carriers", [5])]
    df_list = [float(x) for x in grid.get("df_hz", [1e4])]
    beff_list = [float(x) * 1e6 for x in grid.get("beff_mhz", [10])]

    txc = cfg.get("tx", {})
    ch = cfg.get("channel", {})
    truth_tau = float(cfg.get("truth", {}).get("tau_s", 0.0))

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "seed","snr_db","m","df_hz","beff_hz","tau_true_ps","tau_hat_ps","ci95_ps","residual_ps","env_df_snr_dB","PAPR_dB","CRLB_ps"
            ],
        )
        writer.writeheader()

        for snr in snr_list:
            for m in m_list:
                for df in df_list:
                    for beff in beff_list:
                        rng = np.random.default_rng(base_seed + int(snr) + m + int(df) + int(beff))
                        x, fk, _, meta = generate_comb(
                            fs=fs,
                            duration=dur,
                            df=df,
                            m=m,
                            omit_fundamental=bool(txc.get("omit_fundamental", True)),
                            amp_mode=str(txc.get("amplitudes", "equal")),
                            phase_mode=str(txc.get("phase_schedule", "newman")),
                            rng=rng,
                            return_payload=True,
                        )
                        # Scale fk to desired Beff by spacing adjustment if needed
                        tx_phases = meta.get("phases")
                        if len(fk) > 1:
                            scale = beff / (np.max(fk) - np.min(fk))
                            fk_scaled = fk * scale
                            # Re-synthesize quickly by re-mixing with zero phases
                            n = len(x)
                            t = np.arange(n) / fs
                            x = np.sum(np.exp(1j * (2 * np.pi * fk_scaled[:, None] * t)), axis=0)
                            x /= np.sqrt(np.mean(np.abs(x) ** 2) + 1e-12)
                            fk = fk_scaled
                            tx_phases = np.zeros_like(fk)

                        x = apply_cfo(x, fs, float(ch.get("cfo_hz", 0.0)))
                        x = apply_sco(x, float(ch.get("sco_ppm", 0.0)))
                        x = apply_phase_noise(x, float(ch.get("phase_noise_rad2", 0.0)), rng)
                        x = tapped_delay_channel(x, fs, ch.get("taps", []))
                        x = rapp_soft_clip(x, p=float(ch.get("rapp_p", 2.0)), sat=float(ch.get("rapp_sat", 1.5)))
                        x = aperture_branch(x, alpha=float(ch.get("aperture_alpha", 2.0)), mix=float(ch.get("aperture_mix", 0.0)))
                        x = awgn(x, snr, rng)

                        Yk = estimate_tone_phasors(x, fs, fk)
                        if tx_phases is not None:
                            Yk = Yk * np.exp(-1j * tx_phases)
                        phu = unwrap_phase(np.angle(Yk), fk)
                        noise_var = estimate_noise_power(x, fs, fk, Yk)
                        snr_w = per_tone_snr(Yk, noise_var, len(x))
                        tau_hat, ci95 = wls_delay(fk, phu, snr_w)
                        f_env, E_env = envelope_spectrum(x, fs)
                        _, env_snr = detect_df_peak(f_env, E_env, df)
                        crlb = delay_crlb_std(beff, 10 ** (snr / 10.0))
                        # Approximate BER for QPSK in AWGN at the provided SNR
                        from driftlock_sim.dsp.metrics import ber_qpsk_awgn
                        row = {
                            "seed": base_seed,
                            "snr_db": snr,
                            "m": m,
                            "df_hz": df,
                            "beff_hz": beff,
                            "tau_true_ps": truth_tau * 1e12,
                            "tau_hat_ps": tau_hat * 1e12,
                            "ci95_ps": ci95 * 1e12,
                            "residual_ps": (tau_hat - truth_tau) * 1e12,
                            "env_df_snr_dB": env_snr,
                            "BER": ber_qpsk_awgn(snr),
                            "PAPR_dB": papr_db(x),
                            "CRLB_ps": crlb * 1e12,
                        }
                        writer.writerow(row)

    print(f"Wrote sweep CSV → {csv_path}")


if __name__ == "__main__":
    main()
