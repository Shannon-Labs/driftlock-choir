import numpy as np

from alg.kalman_local import LocalKFConfig, LocalTwoStateKF


def test_local_kf_reduces_error() -> None:
    dt = 1e-3
    cfg = LocalKFConfig(dt=dt, sigma_T=5e-12, sigma_f=0.5)
    init_var_T = (1e-9) ** 2
    init_var_f = (5.0) ** 2
    kf_a = LocalTwoStateKF(cfg, x0=np.zeros(2), P0=np.diag([init_var_T, init_var_f]))
    kf_b = LocalTwoStateKF(cfg, x0=np.zeros(2), P0=np.diag([init_var_T, init_var_f]))

    true_a = np.array([2.5e-9, -0.8])
    true_b = np.array([-1.5e-9, 1.2])

    kf_a.predict()
    kf_b.predict()

    measurement = (true_b - true_a) + np.array([0.3e-9, 0.2])
    sigma_tau_sq = (0.6e-9) ** 2
    sigma_df_sq = 0.5 ** 2

    mu_b, P_b = kf_b.get_posterior()
    R_a = np.diag([sigma_tau_sq + P_b[0, 0], sigma_df_sq + P_b[1, 1]])
    z_a = mu_b - measurement
    kf_a.update_with_neighbor(z_a, R_a)

    mu_a, P_a = kf_a.get_posterior()
    R_b = np.diag([sigma_tau_sq + P_a[0, 0], sigma_df_sq + P_a[1, 1]])
    z_b = mu_a + measurement
    kf_b.update_with_neighbor(z_b, R_b)

    pre_error = np.linalg.norm(true_a) + np.linalg.norm(true_b)
    post_error = np.linalg.norm(true_a - mu_a) + np.linalg.norm(true_b - kf_b.get_posterior()[0])
    assert post_error < pre_error
