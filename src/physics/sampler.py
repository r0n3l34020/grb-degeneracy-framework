from scipy.stats import qmc
import numpy as np

def default_f_e_func(E):
    return 1.0

def generate_lhs_parameter_batch(num_samples, parameter_bounds):
    print("[DEBUG] Entering Latin Hypercube sampling routine...")
    num_dimensions = 10
    parameters = [
        "E_break",
        "eqg",
        "g_ag",
        "B",
        "L",
        "p",
        "t_rise",
        "t_decay",
        "R_v",
        "E_BV",
    ]
    log_indices = {0, 1, 2, 3, 4}

    print("[DEBUG] Initializing qmc.LatinHypercube...")
    sampler = qmc.LatinHypercube(d=num_dimensions)

    print("[DEBUG] Generating raw samples...")
    raw_samples = sampler.random(n=num_samples)
    scaled_parameter_batch = np.zeros_like(raw_samples)

    print("[DEBUG] Scaling parameters...")
    for i, key in enumerate(parameters):
        min_val, max_val = parameter_bounds[key]
        unit_column = raw_samples[:, i]

        if i in log_indices:
            log_min = np.log10(min_val)
            log_max = np.log10(max_val)
            log_value = log_min + unit_column * (log_max - log_min)
            scaled_parameter_batch[:, i] = 10.0 ** log_value
        else:
            scaled_parameter_batch[:, i] = min_val + unit_column * (max_val - min_val)

    print("[DEBUG] Constructing parameter dictionaries...")
    params = []

    for row in scaled_parameter_batch:
        E_break, eqg, g_ag, B, L, p, t_rise, t_decay, R_v, E_BV= row

        alpha = (p - 1) / 2
        beta = alpha + 0.5

        parameter_batch = {
            "p": p,
            "alpha": alpha,
            "beta": beta,
            "E_break": E_break,
            "t_rise": t_rise,
            "t_decay": t_decay,
            "g_ag": g_ag,
            "B": B,
            "L": L,
            "eqg": eqg,
            "f_break": 1.0,
            "A": 5.0,
            "f_E_func": default_f_e_func,
            "R_v": R_v,
            "E_BV": E_BV,
        }

        params.append(parameter_batch)

    print("[DEBUG] LHS parameter batch successfully generated.")
    return params