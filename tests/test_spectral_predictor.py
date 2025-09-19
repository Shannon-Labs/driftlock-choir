from math import inf

from alg.spectral_predictor import predict_iterations_to_rmse


def test_predict_iterations_monotonic_lambda() -> None:
    base = predict_iterations_to_rmse(100.0, 1000.0, lambda2=0.1, epsilon=0.2)
    improved = predict_iterations_to_rmse(100.0, 1000.0, lambda2=0.2, epsilon=0.2)
    assert improved < base


def test_predict_iterations_monotonic_epsilon() -> None:
    slow = predict_iterations_to_rmse(50.0, 500.0, lambda2=0.15, epsilon=0.1)
    fast = predict_iterations_to_rmse(50.0, 500.0, lambda2=0.15, epsilon=0.2)
    assert fast < slow


def test_predict_iterations_handles_degenerate() -> None:
    assert predict_iterations_to_rmse(100.0, 100.0, lambda2=0.2, epsilon=0.1) == 0.0
    assert predict_iterations_to_rmse(100.0, 0.0, lambda2=0.2, epsilon=0.1) == 0.0
    assert predict_iterations_to_rmse(100.0, 1000.0, lambda2=0.0, epsilon=0.1) == inf
