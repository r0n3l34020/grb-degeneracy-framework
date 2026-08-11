import numpy as np
from .constants import ALP_PHASE_CONVERSION

def photon_survival_probability(initial_energy_grid, g_ag, B, L, f_E_func):
    oscillation_phase = ((B * L * g_ag) * ALP_PHASE_CONVERSION * (10 ** 13)) / initial_energy_grid
    sin_oscillation = np.sin(oscillation_phase)**2
    
    f_E = f_E_func(initial_energy_grid) if callable(f_E_func) else f_E_func
    
    photon_survival = 1.0 - ((1/3) * sin_oscillation * f_E)
    return photon_survival

def apply_alp_to_spectrum(matrix_2d, photon_survival):
    expanded_1d_matrix = photon_survival[:, np.newaxis]
    alp_matrix_2d = matrix_2d * expanded_1d_matrix
    return alp_matrix_2d