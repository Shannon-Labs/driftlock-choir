import numpy as np

from phy.noise import PowerLawPhaseNoiseGenerator, NoiseGenerator, NoiseParams
from phy.impairments import generate_phase_noise


def _estimate_log_slope(signal: np.ndarray, sample_rate: float) -> float:
    spectrum = np.fft.rfft(signal)
    freqs = np.fft.rfftfreq(signal.size, d=1.0 / sample_rate)
    # Ignore DC and highest bin for slope estimate
    valid = (freqs > 0) & (freqs < sample_rate / 4.0)
    freqs = freqs[valid]
    psd = np.abs(spectrum[valid]) ** 2
    log_f = np.log(freqs)
    log_psd = np.log(psd + 1e-24)
    slope, _ = np.polyfit(log_f, log_psd, 1)
    return float(slope)


def test_power_law_generator_reproducible() -> None:
    coeffs = {-1: 1e-4, 0: 1e-6}
    rng_seed = 1234
    gen_a = PowerLawPhaseNoiseGenerator(coeffs, sample_rate=5e6, rng=np.random.default_rng(rng_seed))
    gen_b = PowerLawPhaseNoiseGenerator(coeffs, sample_rate=5e6, rng=np.random.default_rng(rng_seed))
    out_a = gen_a.generate(2048)
    out_b = gen_b.generate(2048)
    assert np.allclose(out_a, out_b)


def test_power_law_generator_slope_matches_alpha() -> None:
    coeffs = {-2: 1e3}
    sample_rate = 2e6
    rng = np.random.default_rng(2025)
    generator = PowerLawPhaseNoiseGenerator(coeffs, sample_rate=sample_rate, rng=rng)
    values = generator.generate(8192)
    slope = _estimate_log_slope(values, sample_rate)
    # Expect slope roughly -2 (allowing loose tolerance due to stochastic estimate)
    assert -2.4 < slope < -1.6


def test_noise_generator_legacy_path_uses_rng() -> None:
    params = NoiseParams(snr_db=30.0, phase_noise_psd=-90.0, jitter_rms=1e-12)
    rng = np.random.default_rng(99)
    generator = NoiseGenerator(params, sample_rate=1e6, rng=rng)
    seq = generator.generate_phase_noise(10)
    # Calling again without resetting rng should produce different sequence
    seq2 = generator.generate_phase_noise(10)
    assert not np.allclose(seq, seq2)


def test_generate_phase_noise_impairments_wrapper() -> None:
    rng = np.random.default_rng(777)
    values = generate_phase_noise(4096, 10e6, {-1: 5e-5}, rng)
    assert values.size == 4096
    assert np.var(values) > 0.0
