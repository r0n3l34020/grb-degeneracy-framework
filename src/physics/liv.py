import numpy as np

def calculate_time_delay(energies, redshift=0.151, eqg=1e19):
    c_factor = 3.33e-19
    proper_distance_factor = redshift * c_factor

    time_delay = (1 + redshift) * (energies / eqg) * proper_distance_factor

    return time_delay

def apply_liv_to_spectrum(time_grid, energy_grid, eqg, flux_matrix):
    shifted_matrix = np.zeros_like(flux_matrix)
    delays = calculate_time_delay(energy_grid, eqg=eqg)

    for i, delay in enumerate(delays):
        shifted_matrix[i, :] = np.interp(
            time_grid - delay, time_grid, flux_matrix[i, :], left=0.0, right=0.0
        )

    return shifted_matrix