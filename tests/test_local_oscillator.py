import numpy as np

from hw.lo import LocalOscillator, LOConfig, ThermalConfig
from phy.osc import OscillatorParams


def _make_lo(**kwargs) -> LocalOscillator:
    params = OscillatorParams(
        allan_dev_1s=0.0,
        drift_rate=0.0,
        flicker_corner=0.0,
        white_noise_level=0.0,
        phase_noise_h={}
    )
    rng = np.random.default_rng(123)
    return LocalOscillator(2.45e9, params, rng=rng, **kwargs)


def test_lo_temperature_rc_relaxes_toward_ambient() -> None:
    thermal = ThermalConfig(ambient_c=30.0, initial_c=40.0, time_constant_s=5.0)
    lo = _make_lo(thermal=thermal)
    lo.update_state(0.0)
    lo.update_state(5.0)
    temp = lo.current_temperature()
    assert thermal.ambient_c < temp < thermal.initial_c
    lo.update_state(25.0)
    assert abs(lo.current_temperature() - (thermal.ambient_c + thermal.steady_state_rise_c)) < 0.5


def test_lo_external_temperature_trace_overrides_model() -> None:
    lo = _make_lo(thermal=ThermalConfig(ambient_c=20.0, initial_c=20.0, time_constant_s=10.0))
    times = [0.0, 10.0, 20.0]
    temps = [20.0, 35.0, 25.0]
    lo.set_temperature_trace(times, temps)
    lo.update_state(5.0)
    assert abs(lo.current_temperature() - 27.5) < 0.6
    lo.update_state(15.0)
    assert abs(lo.current_temperature() - 30.0) < 0.6


def test_lo_temperature_polynomial_applied() -> None:
    config = LOConfig(nominal_freq=2.4e9, temp_coeff=1.0, temp_poly_ppm=(1.0, 0.5, 0.0))
    lo = _make_lo(config=config, thermal=ThermalConfig(ambient_c=25.0, initial_c=25.0, time_constant_s=1.0))
    # Force a positive temperature step via override
    lo.update_state(0.0)
    lo.update_state(1.0, temperature=35.0)
    # A 10°C delta with coeffs (1, 0.5, 0) gives ppm = 1*10 + 0.5*100 = 60 ppm
    expected_offset = 60e-6
    measured_ppm = lo.get_frequency_offset() / lo.nominal_freq
    assert abs(measured_ppm - expected_offset) < 5e-6
