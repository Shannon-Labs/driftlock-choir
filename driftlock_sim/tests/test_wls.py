import numpy as np
from driftlock_sim.dsp.rx_coherent import wls_delay


def test_wls_slope_sign():
    fk = np.array([-1000.0, 0.0, 1000.0])
    tau_true = 123e-12
    ph = -2*np.pi*fk*tau_true + 0.1
    w = np.ones_like(fk)
    tau_hat, _ = wls_delay(fk, ph, w)
    assert tau_hat > 0
    assert abs(tau_hat - tau_true) < 1e-12

