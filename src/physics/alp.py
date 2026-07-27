import numpy as np

def photon_survival_probability(initial_energy_grid, g_ag, B, L, f_E_func):
    oscillation_phase = ((B * L * g_ag) * 1.52 * (10 ** 13)) / initial_energy_grid
    sin_oscillation = np.sin(oscillation_phase)**2
    
    f_E = f_E_func(initial_energy_grid) if callable(f_E_func) else f_E_func
    
    photon_survival = 1.0 - ((1/3) * sin_oscillation * f_E)
    return photon_survival

def apply_alp_to_spectrum(matrix_2d, photon_survival):
    expanded_1d_matrix = photon_survival[:, np.newaxis]
    alp_matrix_2d = expanded_1d_matrix * matrix_2d
    return alp_matrix_2d

"""
photon_survival_list = []

for i in range(20):
    g_ag = np.random.uniform(1e-10, 1e-9)
    B = np.random.uniform(1e-6, 1e-5) 
    L = np.random.uniform(1000.0, 10000.0)
    
    f_E_func = lambda E: 1.0 
    energy_grid = np.logspace(0, 6, 500)

    get_photon_survival = photon_survival_probability(energy_grid, g_ag, B, L, f_E_func)
    photon_survival_list.append(get_photon_survival)

photon_survival_array = np.array(photon_survival_list)


print("Minimum survival probability:", np.min(photon_survival_array))
print("Maximum survival probability:", np.max(photon_survival_array))
"""