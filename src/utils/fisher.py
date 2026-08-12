import numpy
import math
from ..physics.constants import FISHER_NOISE_LEVEL, FISHER_COND_THRESHOLD

def compute_fisher_matrix(jacobian_dict, active_params, snr):
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

    ridge_lambda = 1e-8 * numpy.trace(fisher_matrix) * numpy.eye(fisher_matrix.shape[0])
    stabilized_fisher = fisher_matrix + ridge_lambda

    try:
        covariance_matrix = numpy.linalg.inv(stabilized_fisher)
    except numpy.linalg.LinAlgError:
        covariance_matrix = numpy.linalg.pinv(stabilized_fisher)

    assert not numpy.any(numpy.isnan(covariance_matrix)), "Error, covariance matrix contains NaN values"
    assert not numpy.any(numpy.isinf(covariance_matrix)), "Error, covariance matrix contains infinite values"

    cramer_rao_bounds = {
        param: float(numpy.sqrt(numpy.maximum(0.0, covariance_matrix[idx, idx])))
        for idx, param in enumerate(active_params)
    }

    return cramer_rao_bounds, covariance_matrix