import numpy as np
from .constants import GRB_REDSHIFT, LIV_C_FACTOR

def calculate_time_delay(energies, redshift=GRB_REDSHIFT, eqg=1e19):
    c_factor = LIV_C_FACTOR
    proper_distance_factor = redshift * c_factor

    time_delay = (1 + redshift) * (energies / eqg) * proper_distance_factor

    return time_delay

def apply_liv_to_spectrum(initial_time_grid, initial_energy_grid, eqg, flux_matrix):
    shifted_matrix = np.zeros_like(flux_matrix)
    delays = calculate_time_delay(initial_energy_grid, eqg=eqg)

    for i, delay in enumerate(delays):
        shifted_matrix[i, :] = np.interp(
            initial_time_grid - delay, initial_time_grid, flux_matrix[i, :], left=0.0, right=0.0
        )

    return shifted_matrix