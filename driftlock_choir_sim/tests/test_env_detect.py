import numpy as np
from driftlock_choir_sim.dsp.tx_comb import generate_comb
from driftlock_choir_sim.dsp.rx_aperture import envelope_spectrum, detect_df_peak


def test_env_detect_df_peak():
    fs = 1e6
    dur = 0.02
    df = 10e3
    x, fk, _ = generate_comb(fs, dur, df, m=3, omit_fundamental=True)
    f, E = envelope_spectrum(x, fs)
    fpk, _ = detect_df_peak(f, E, df)
    assert abs(fpk - df) / df < 0.02

