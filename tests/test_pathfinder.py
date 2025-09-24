import numpy as np

from phy.preamble import build_preamble
from phy.pathfinder import PathfinderConfig, find_first_arrival
from chan.tdl import TappedDelayLine


def test_pathfinder_detects_first_path_above_stronger_reflection():
    sample_rate = 20e6
    preamble, _ = build_preamble(length=256, sample_rate=sample_rate, bandwidth_hz=5e6)
    transmitted = preamble.samples
    n = transmitted.size

    total_len = n + 200
    received = np.zeros(total_len, dtype=np.complex128)

    first_idx = 40
    strong_idx = 140

    received[first_idx:first_idx + n] += 0.35 * transmitted
    received[strong_idx:strong_idx + n] += 1.0 * transmitted

    rng = np.random.default_rng(123)
    noise = (rng.normal(scale=0.02, size=total_len) + 1j * rng.normal(scale=0.02, size=total_len))
    received += noise

    cfg = PathfinderConfig(relative_threshold_db=-15.0, noise_guard_multiplier=4.0, smoothing_kernel=5)
    result = find_first_arrival(received, preamble, sample_rate, cfg)

    tol = 1.0 / sample_rate
    assert abs(result.first_path_s - first_idx / sample_rate) < tol
    assert abs(result.peak_path_s - strong_idx / sample_rate) < tol
    assert result.first_path_amplitude < result.peak_path_amplitude
    assert 0.0 < result.peak_to_first_ratio < 1.0


def test_tdl_window_respects_guard_interval():
    delays = np.array([0.0, 40e-9, 120e-9])
    gains = np.array([1.0, 0.5, 0.25], dtype=np.complex128)
    tdl = TappedDelayLine(delays_s=delays, gains_c=gains)

    trimmed = tdl.window(60e-9)
    assert np.allclose(trimmed.delays_s, delays[:2])
    assert np.allclose(trimmed.gains_c, gains[:2])

    trimmed_small = tdl.window(-10e-9)
    assert trimmed_small.delays_s.size == 1
    assert np.isclose(trimmed_small.delays_s[0], 0.0)
