import numpy as np
from driftlock_sim.dsp.rx_coherent import unwrap_phase


def test_unwrap_continuity():
    ph = np.linspace(0, 10*np.pi, 100)
    ph_wrapped = np.angle(np.exp(1j*ph))
    ph_un = unwrap_phase(ph_wrapped)
    # Endpoints should be close to original trend
    assert abs(ph_un[-1] - ph[-1]) < 1.0

