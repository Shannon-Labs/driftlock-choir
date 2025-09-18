"""
Extended Kalman Filter (EKF) for temporal fusion of synchronization parameters.

This module implements an EKF for tracking time-varying synchronization
parameters (delay and frequency offset) with temporal correlation.
"""

import numpy as np
from typing import Tuple, Optional, Dict, Any, List
from dataclasses import dataclass
from scipy.linalg import cholesky, solve_triangular


@dataclass
class EKFParams:
    """Parameters for Extended Kalman Filter."""
    process_noise_delay: float = 1e-12    # Process noise for delay (s²/s)
    process_noise_freq: float = 1e-18     # Process noise for frequency (Hz²/s)
    measurement_noise_delay: float = 1e-9  # Measurement noise for delay (s²)
    measurement_noise_freq: float = 1e-12  # Measurement noise for frequency (Hz²)
    initial_uncertainty_delay: float = 1e-6  # Initial delay uncertainty (s²)
    initial_uncertainty_freq: float = 1e-9   # Initial frequency uncertainty (Hz²)


class ExtendedKalmanFilter:
    """Extended Kalman Filter for synchronization parameter tracking."""
    
    def __init__(self, params: EKFParams, dt: float = 1.0):
        """
        Initialize EKF.
        
        Args:
            params: EKF parameters
            dt: Time step between updates (s)
        """
        self.params = params
        self.dt = dt
        
        # State vector: [delay, frequency_offset, delay_rate, freq_drift]
        self.state_dim = 4
        self.measurement_dim = 2
        
        # Initialize state and covariance
        self.x = np.zeros(self.state_dim)  # [τ, Δf, τ̇, Δḟ]
        self.P = np.diag([
            params.initial_uncertainty_delay,
            params.initial_uncertainty_freq,
            params.process_noise_delay,
            params.process_noise_freq
        ])
        
        # Process noise covariance
        self.Q = self._create_process_noise_matrix()
        
        # Measurement noise covariance
        self.R = np.diag([
            params.measurement_noise_delay,
            params.measurement_noise_freq
        ])
        
        # History tracking
        self.state_history = []
        self.covariance_history = []
        self.innovation_history = []
        
    def _create_process_noise_matrix(self) -> np.ndarray:
        """Create process noise covariance matrix."""
        # Continuous-time process noise
        q_delay = self.params.process_noise_delay
        q_freq = self.params.process_noise_freq
        
        # Discretize using Van Loan method
        dt = self.dt
        Q = np.array([
            [q_delay * dt**3 / 3, 0, q_delay * dt**2 / 2, 0],
            [0, q_freq * dt**3 / 3, 0, q_freq * dt**2 / 2],
            [q_delay * dt**2 / 2, 0, q_delay * dt, 0],
            [0, q_freq * dt**2 / 2, 0, q_freq * dt]
        ])
        
        return Q
        
    def _state_transition_model(self, x: np.ndarray) -> np.ndarray:
        """State transition model: x[k+1] = f(x[k])."""
        dt = self.dt
        
        # Linear state transition for synchronization parameters
        F = np.array([
            [1, 0, dt, 0],    # delay = delay + delay_rate * dt
            [0, 1, 0, dt],    # freq = freq + freq_drift * dt
            [0, 0, 1, 0],     # delay_rate (constant)
            [0, 0, 0, 1]      # freq_drift (constant)
        ])
        
        return F @ x
        
    def _state_transition_jacobian(self, x: np.ndarray) -> np.ndarray:
        """Jacobian of state transition model."""
        dt = self.dt
        
        F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        
        return F
        
    def _measurement_model(self, x: np.ndarray) -> np.ndarray:
        """Measurement model: z = h(x)."""
        # Direct measurement of delay and frequency
        return x[:2]  # [delay, frequency]
        
    def _measurement_jacobian(self, x: np.ndarray) -> np.ndarray:
        """Jacobian of measurement model."""
        H = np.array([
            [1, 0, 0, 0],  # delay measurement
            [0, 1, 0, 0]   # frequency measurement
        ])
        
        return H
        
    def predict(self) -> Tuple[np.ndarray, np.ndarray]:
        """Prediction step of EKF."""
        # Predict state
        x_pred = self._state_transition_model(self.x)
        
        # Predict covariance
        F = self._state_transition_jacobian(self.x)
        P_pred = F @ self.P @ F.T + self.Q
        
        return x_pred, P_pred
        
    def update(self, measurement: np.ndarray, 
               measurement_covariance: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Update step of EKF.
        
        Args:
            measurement: [delay, frequency] measurement
            measurement_covariance: Optional measurement covariance override
            
        Returns:
            Dictionary with update results and metrics
        """
        # Use provided covariance or default
        R = measurement_covariance if measurement_covariance is not None else self.R
        
        # Prediction
        x_pred, P_pred = self.predict()
        
        # Measurement prediction
        z_pred = self._measurement_model(x_pred)
        H = self._measurement_jacobian(x_pred)
        
        # Innovation
        innovation = measurement - z_pred
        S = H @ P_pred @ H.T + R
        
        # Kalman gain
        K = P_pred @ H.T @ np.linalg.inv(S)
        
        # State update
        self.x = x_pred + K @ innovation
        
        # Covariance update (Joseph form for numerical stability)
        I_KH = np.eye(self.state_dim) - K @ H
        self.P = I_KH @ P_pred @ I_KH.T + K @ R @ K.T
        
        # Store history
        self.state_history.append(self.x.copy())
        self.covariance_history.append(self.P.copy())
        self.innovation_history.append(innovation)
        
        # Compute metrics
        innovation_covariance = S
        mahalanobis_distance = innovation.T @ np.linalg.inv(S) @ innovation
        log_likelihood = -0.5 * (mahalanobis_distance + np.log(np.linalg.det(2 * np.pi * S)))
        
        return {
            'state': self.x.copy(),
            'covariance': self.P.copy(),
            'innovation': innovation,
            'innovation_covariance': innovation_covariance,
            'mahalanobis_distance': mahalanobis_distance,
            'log_likelihood': log_likelihood,
            'kalman_gain': K
        }
        
    def get_current_estimates(self) -> Dict[str, Any]:
        """Get current parameter estimates with uncertainties."""
        return {
            'delay': {
                'estimate': self.x[0],
                'uncertainty': np.sqrt(self.P[0, 0])
            },
            'frequency': {
                'estimate': self.x[1],
                'uncertainty': np.sqrt(self.P[1, 1])
            },
            'delay_rate': {
                'estimate': self.x[2],
                'uncertainty': np.sqrt(self.P[2, 2])
            },
            'frequency_drift': {
                'estimate': self.x[3],
                'uncertainty': np.sqrt(self.P[3, 3])
            }
        }
        
    def predict_future(self, time_ahead: float) -> Dict[str, Any]:
        """Predict synchronization parameters at future time."""
        # Propagate state forward
        n_steps = int(time_ahead / self.dt)
        x_future = self.x.copy()
        P_future = self.P.copy()
        
        F = self._state_transition_jacobian(x_future)
        Q_total = self.Q * n_steps  # Accumulated process noise
        
        for _ in range(n_steps):
            x_future = self._state_transition_model(x_future)
            P_future = F @ P_future @ F.T + self.Q
            
        return {
            'predicted_delay': x_future[0],
            'predicted_frequency': x_future[1],
            'delay_uncertainty': np.sqrt(P_future[0, 0]),
            'frequency_uncertainty': np.sqrt(P_future[1, 1]),
            'prediction_horizon': time_ahead
        }
        
    def get_convergence_metrics(self) -> Dict[str, Any]:
        """Analyze filter convergence and performance."""
        if len(self.state_history) < 2:
            return {'error': 'Insufficient history for analysis'}
            
        # Trace of covariance (total uncertainty)
        trace_history = [np.trace(P) for P in self.covariance_history]
        
        # Innovation sequence analysis
        innovations = np.array(self.innovation_history)
        
        # Normalized innovation squared (should be chi-squared distributed)
        nis_sequence = []
        for i, innov in enumerate(innovations):
            if i < len(self.covariance_history):
                H = self._measurement_jacobian(self.state_history[i])
                S = H @ self.covariance_history[i] @ H.T + self.R
                nis = innov.T @ np.linalg.inv(S) @ innov
                nis_sequence.append(nis)
                
        return {
            'trace_history': trace_history,
            'final_trace': trace_history[-1] if trace_history else 0,
            'innovation_mean': np.mean(innovations, axis=0),
            'innovation_covariance': np.cov(innovations.T),
            'nis_sequence': nis_sequence,
            'nis_mean': np.mean(nis_sequence) if nis_sequence else 0,
            'filter_consistent': np.abs(np.mean(nis_sequence) - self.measurement_dim) < 0.1 if nis_sequence else False
        }
