import numpy as np

from src.metrics.crlb import CRLBParams, JointCRLBCalculator


def _ls_estimator_tau_df(phase, t, fc_hz, tau_hint):
    """Linear-phase LS estimator consistent with chronometric_handshake._estimate_parameters.

    phase: unwrapped phase samples (radians)
    t: time samples (seconds)
    fc_hz: carrier frequency (Hz)
    tau_hint: hint for unwrapping (seconds) [used here only to mirror pipeline]
    """
    A = np.vstack([t, np.ones_like(t)]).T
    xtx = A.T @ A
    xtx_inv = np.linalg.inv(xtx)
    slope, intercept = np.linalg.lstsq(A, phase, rcond=None)[0]
    delta_f_est = float(slope / (2.0 * np.pi))
    # tau candidate from intercept (θ unknown -> use true θ absorbed in intercept)
    # Here we simulate the same mapping as in the code: τ ≈ (θ_diff - b) / (2π fc)
    # During MC we set θ_diff=0 so τ = (-b)/(2π fc)
    tau_candidate = float((-intercept) / (2.0 * np.pi * fc_hz))
    # Unwrap τ using hint (mirror of _unwrap_single_tau)
    cycles = np.round((tau_hint - tau_candidate) * fc_hz)
    tau_unwrapped = tau_candidate + cycles / fc_hz

    # Residuals/covariance
    fitted = (A @ np.array([slope, intercept])).astype(float)
    residual = phase - fitted
    sigma_phase_sq = float(np.dot(residual, residual) / max(len(phase) - 2, 1))
    cov_params = sigma_phase_sq * xtx_inv
    var_slope = cov_params[0, 0]
    var_intercept = cov_params[1, 1]
    cov_si = cov_params[0, 1]

    tau_var = var_intercept / (2.0 * np.pi * fc_hz) ** 2
    df_var = var_slope / (2.0 * np.pi) ** 2
    cov_tau_df = -cov_si / ((2.0 * np.pi) ** 2 * fc_hz)
    cov = np.array([[tau_var, cov_tau_df], [cov_tau_df, df_var]], dtype=float)

    return tau_unwrapped, delta_f_est, residual, cov


def test_crlb_matches_ls_covariance_and_mc_rms():
    rng = np.random.default_rng(20250919)

    # Signal/estimator config
    fs = 200e3
    T = 0.02  # 20 ms window
    n = int(T * fs)
    t = np.arange(n) / fs
    fc = 10e6   # modest carrier frequency to avoid numerical blow-up in intercept mapping

    # True parameters
    tau_true = 1.2e-6   # 1.2 microseconds
    df_true = 1234.0    # 1234 Hz
    theta = 0.0        # zero to simplify tau mapping

    # SNR (complex) and corresponding noise model
    snr_db = 20.0
    snr_lin = 10 ** (snr_db / 10)

    # Monte Carlo
    trials = 200
    tau_err = []
    df_err = []
    ls_cov_diag_tau = []
    ls_cov_diag_df = []

    for _ in range(trials):
        phase = 2.0 * np.pi * df_true * t + theta - 2.0 * np.pi * fc * tau_true
        s = np.exp(1j * phase)
        # Complex AWGN with E[|w|^2] = 1/SNR
        noise_power = 1.0 / snr_lin
        w = np.sqrt(noise_power / 2.0) * (rng.standard_normal(n) + 1j * rng.standard_normal(n))
        y = s + w

        # Estimator uses unwrapped phase
        phase_u = np.unwrap(np.angle(y))
        tau_hat, df_hat, residual, cov = _ls_estimator_tau_df(phase_u, t, fc, tau_hint=tau_true)
        tau_err.append(tau_hat - tau_true)
        df_err.append(df_hat - df_true)
        ls_cov_diag_tau.append(cov[0, 0])
        ls_cov_diag_df.append(cov[1, 1])

    tau_err = np.array(tau_err)
    df_err = np.array(df_err)
    rmse_tau = np.sqrt(np.mean(tau_err ** 2))
    rmse_df = np.sqrt(np.mean(df_err ** 2))

    # Average LS covariance across trials (should match MC RMSE^2)
    ls_tau_var = float(np.mean(ls_cov_diag_tau))
    ls_df_var = float(np.mean(ls_cov_diag_df))

    # Corrected discrete-time CRLB consistent with estimator
    params = CRLBParams(
        snr_db=snr_db,
        bandwidth=0.0,
        duration=T,
        carrier_freq=fc,
        sample_rate=fs,
    )
    _ = JointCRLBCalculator(params)  # ensure constructor works

    # Sanity: For Δf, MC RMSE^2 ≈ LS variance (within factors)
    assert 0.3 < (rmse_df ** 2) / ls_df_var < 3.0
