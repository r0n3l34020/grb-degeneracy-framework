import numpy as np
from src.utils.fisher import (
    numerical_jacobian,
    compute_fisher_matrix,
    invert_fisher_matrix,
    plot_error_ellipse,
    liv_spectral_index_fisher_ellipse,
)


def test_numerical_jacobian_matches_known_linear_model():
    # model(theta) = A @ theta -> Jacobian should just be A, independent of theta.
    A = np.array([[2.0, 0.0], [0.0, 3.0], [1.0, 1.0]])
    model_func = lambda theta: A @ theta

    jac = numerical_jacobian(model_func, np.array([1.0, 1.0]), eps=1e-4)
    assert np.allclose(jac, A, atol=1e-6)


def test_fisher_matrix_matches_analytic_linear_gaussian_case():
    # For a linear Gaussian model y = A @ theta + noise(sigma), the closed-form
    # Fisher matrix is F = A^T A / sigma^2 — use this to validate compute_fisher_matrix.
    A = np.array([[2.0, 0.0], [0.0, 3.0], [1.0, 1.0]])
    sigma = 2.0
    model_func = lambda theta: A @ theta

    fisher_matrix = compute_fisher_matrix(model_func, np.array([1.0, 1.0]), sigma, eps=1e-4)
    expected = (A.T @ A) / sigma ** 2
    assert np.allclose(fisher_matrix, expected, atol=1e-4)


def test_invert_fisher_matrix_cramer_rao_bound():
    fisher_matrix = np.array([[4.0, 0.0], [0.0, 9.0]])  # diagonal -> trivial to check by hand
    covariance, variances = invert_fisher_matrix(fisher_matrix)

    assert np.allclose(covariance, np.array([[0.25, 0.0], [0.0, 1 / 9]]))
    assert np.allclose(variances, np.array([0.25, 1 / 9]))


def test_plot_error_ellipse_produces_positive_dimensions():
    covariance = np.array([[0.25, 0.05], [0.05, 1 / 9]])
    _, ellipse = plot_error_ellipse(covariance, center=(1.0, 2.0), n_std=1.0)
    assert ellipse.width > 0
    assert ellipse.height > 0


def test_liv_spectral_index_fisher_ellipse_is_positive_definite():
    result = liv_spectral_index_fisher_ellipse(snr=15.0)
    covariance = result["covariance"]

    assert covariance.shape == (2, 2)
    eigenvalues = np.linalg.eigvalsh(covariance)
    assert np.all(eigenvalues > 0), "Covariance from a valid Fisher matrix must be positive definite"
    assert np.all(np.isfinite(result["sigma_theta"]))
