import numpy as np
import pytest
from src.utils.fisher import (
    compute_fisher_matrix,
    fisher_estimator_pipeline,
    compute_cramer_rao_bounds,
)


def test_compute_fisher_matrix_matches_hand_calculation():
    # Two params with constant derivatives over a 4-element "data" array, snr=1
    # (sigma^2 = 1e-2 / snr^2 = 1e-2 here, constant everywhere).
    jacobian_dict = {
        "a": np.array([1.0, 1.0, 1.0, 1.0]),
        "b": np.array([2.0, 0.0, 0.0, 0.0]),
    }
    fisher_matrix = compute_fisher_matrix(jacobian_dict, ["a", "b"], snr=1.0)

    sigma_squared = 1e-2
    expected = np.array([
        [np.sum(jacobian_dict["a"] ** 2) / sigma_squared, np.sum(jacobian_dict["a"] * jacobian_dict["b"]) / sigma_squared],
        [np.sum(jacobian_dict["b"] * jacobian_dict["a"]) / sigma_squared, np.sum(jacobian_dict["b"] ** 2) / sigma_squared],
    ])
    assert np.allclose(fisher_matrix, expected)


def test_compute_fisher_matrix_scales_inversely_with_snr_squared():
    jacobian_dict = {"a": np.array([1.0, 2.0, 3.0])}
    f_low_snr = compute_fisher_matrix(jacobian_dict, ["a"], snr=1.0)
    f_high_snr = compute_fisher_matrix(jacobian_dict, ["a"], snr=10.0)
    assert np.isclose(f_high_snr[0, 0], f_low_snr[0, 0] * 100)  # F ~ snr^2


def test_fisher_estimator_pipeline_selects_correct_params_per_mode():
    random_param = {"E_break": 1e3, "B": 1e-5, "L": 5000.0, "g_ag": 1e-10, "eqg": 1e18, "unused": 42}

    assert fisher_estimator_pipeline(random_param, "ssc") == {"E_break": 1e3, "B": 1e-5, "L": 5000.0}
    assert fisher_estimator_pipeline(random_param, "alp") == {
        "E_break": 1e3, "g_ag": 1e-10, "B": 1e-5, "L": 5000.0,
    }
    assert fisher_estimator_pipeline(random_param, "liv") == {
        "E_break": 1e3, "eqg": 1e18, "B": 1e-5, "L": 5000.0,
    }


def test_fisher_estimator_pipeline_missing_key_returns_not_found():
    result = fisher_estimator_pipeline({"E_break": 1e3}, "ssc")
    assert result["B"] == "Not Found"


def test_compute_cramer_rao_bounds_matches_hand_calculation_for_diagonal_matrix():
    fisher_matrix = np.array([[4.0, 0.0], [0.0, 9.0]])
    error_bounds, covariance = compute_cramer_rao_bounds(fisher_matrix, ["a", "b"])

    assert np.allclose(covariance, np.array([[0.25, 0.0], [0.0, 1 / 9]]))
    assert np.allclose(error_bounds, np.array([0.5, 1 / 3]))


def test_compute_cramer_rao_bounds_rejects_wrong_sized_active_params():
    fisher_matrix = np.eye(2)
    with pytest.raises(AssertionError):
        compute_cramer_rao_bounds(fisher_matrix, ["a", "b", "c"])


def test_compute_cramer_rao_bounds_stabilizes_ill_conditioned_matrix():
    # Two nearly-identical rows -> huge condition number, should trigger ridge damping
    # instead of crashing, and still return finite, positive error bounds.
    fisher_matrix = np.array([[1.0, 1.0 - 1e-14], [1.0 - 1e-14, 1.0]])
    error_bounds, covariance = compute_cramer_rao_bounds(fisher_matrix, ["a", "b"])

    assert np.all(np.isfinite(error_bounds))
    assert np.all(error_bounds > 0)
    assert covariance.shape == (2, 2)
