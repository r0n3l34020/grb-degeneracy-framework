import numpy
import math

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
    assert fisher_matrix.shape[0] == fisher_matrix.shape[1], "Error, Fisher matrix is not square"
    assert fisher_matrix.shape[0] == len(active_params), "Error, Fisher matrix does not match target key lengths"

    condition_number = numpy.linalg.cond(fisher_matrix)

    if condition_number >= 1e12:
        print(f"[WARNING] Ill-conditioned Fisher matrix detected (cond: {condition_number:.2e}). Applying ridge stabilization...")
        lambda_damping = 1e-6 * numpy.trace(fisher_matrix) / fisher_matrix.shape[0]
        fisher_matrix = fisher_matrix + lambda_damping * numpy.eye(fisher_matrix.shape[0])
        
        condition_number = numpy.linalg.cond(fisher_matrix)
        assert condition_number < 1e12, f"Error, Fisher matrix remains ill-conditioned after damping (cond: {condition_number:.2e})"

    rank = numpy.linalg.matrix_rank(fisher_matrix)
    assert rank == len(active_params), "Error, Fisher matrix is rank-deficient"

    covariance_matrix = numpy.linalg.pinv(fisher_matrix)
    variances = numpy.diagonal(covariance_matrix)
    extracted_error_bounds = numpy.sqrt(numpy.abs(variances))

    return extracted_error_bounds, covariance_matrix