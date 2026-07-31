import numpy
import math
from src.physics.main import generate_grb_event

def compute_flux_jacobian(active_params, mode, obs_mode):
    epsilon = 1e-4
    jacobian_dict = {}
    for key in active_params:
        val = active_params[key]
        final_val = val if val != 0 else 1e-8
        delta = epsilon * final_val
        param_up = active_params.copy()
        param_up[key] += delta
        flux_up = generate_grb_event(param_up, mode)
        flux_down = generate_grb_event(param_down, mode)
        if obs_mode == "spectral_only":
            new_flux_up = numpy.mean(flux_up, axis=(0, 2))
            new_flux_down = numpy.mean(flux_down, axis=(0, 2))
        elif obs_mode == "spectral_temporal":
            new_flux_up = flux_up[0, :, :]
            new_flux_down = flux_down[0, :, :]
        elif obs_mode == "full_polarimetric":
            new_flux_up = flux_up
            new_flux_down = flux_down
        param_down = active_params.copy()
        param_down[key] -= delta
        derivative_matrix = (new_flux_up - new_flux_down) / (2 * delta)
        jacobian_dict[key] = derivative_matrix
    return jacobian_dict

def compute_fisher_matrix(jacobian_dict, active_params, snr):
    noise_level_value = 1e-2
    noise_variance_scalar = noise_level_value / (snr ** 2)
    data_shape = next(iter(jacobian_dict.values())).shape
    sigma_squared = numpy.full((data_shape), noise_variance_scalar)
    fisher_matrix = numpy.zeros((len(active_params), len(active_params)))
    for i, key_i in enumerate(active_params):
        deriv_i = jacobian_dict[key_i]
        for j, key_j in enumerate(active_params):
            deriv_j = jacobian_dict[key_j]
            fisher_matrix[i, j] = numpy.sum((deriv_i * deriv_j) / sigma_squared)
    return fisher_matrix

def fisher_estimator_pipeline(random_param, mode):
    target_keys = []
    if mode == "ssc":
        target_keys = ["E_break", "B", "L"]
    elif mode == "alp":
        target_keys = ["E_break", "g_ag", "B", "L"]
    elif mode == "liv":
        target_keys = ["E_break", "eqg", "B", "L"]
    else:
        print("mode is not coming through")
    active_params = {key: random_param.get(key, "Not Found") for key in target_keys}

    return active_params

def compute_cramer_rao_bounds(fisher_matrix, active_params):
    rank = numpy.linalg.matrix_rank(fisher_matrix)
    condition_number = numpy.linalg.cond(fisher_matrix)
    assert fisher_matrix.shape[0] == fisher_matrix.shape[1], "Error, Fisher matrix is not square"
    assert fisher_matrix.shape[0] == len(active_params), "Error, Fisher matrix does not match target key lengths"
    assert condition_number < 1e12, "Error, Fisher matrix is ill-conditioned"
    assert rank == len(active_params), "Error, Fisher matrix is rank-deficient"
    covariance_matrix = numpy.linalg.pinv(fisher_matrix)
    variances = numpy.diagonal(covariance_matrix)
    extracted_error_bounds = numpy.sqrt(variances)

    return extracted_error_bounds, covariance_matrix