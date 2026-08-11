import numpy
import math
from ..physics.constants import FISHER_NOISE_LEVEL, FISHER_COND_THRESHOLD

def compute_fisher_matrix(jacobian_dict, active_params, snr):
    noise_level_value = FISHER_NOISE_LEVEL
    noise_variance_scalar = noise_level_value / (snr ** 2)
    data_shape = next(iter(jacobian_dict.values())).shape
    sigma_squared = numpy.full((data_shape), noise_variance_scalar)
    fisher_matrix = numpy.zeros((len(active_params), len(active_params)))
    n_params = len(active_params)
    fisher_matrix = numpy.zeros((n_params, n_params))

    for i, key1 in enumerate(active_params):
        for j, key2 in enumerate(active_params):
            if key1 in jacobian_dict and key2 in jacobian_dict:
                d1 = jacobian_dict[key1]
                d2 = jacobian_dict[key2]    
                fisher_matrix[i, j] = (snr ** 2) * numpy.sum(d1 * d2)
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

    if condition_number >= FISHER_COND_THRESHOLD:
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