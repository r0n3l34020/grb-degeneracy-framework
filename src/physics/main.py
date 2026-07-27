import numpy
from pathlib import Path
import shutil
from synchrotron import simulate_grb_emission
from liv import apply_liv_to_spectrum
from alp import apply_alp_to_spectrum, photon_survival_probability
from operator import itemgetter

scenario_mode =["ssc", "alp", "liv"]

# All bounds converted to match the 'energy_grid' scale
GRB221009A_PARAMETER_BOUNDS = {
    "E_break_bounds": (1e+5, 1e+6),
    "eqg_bounds": (1e+26, 1e+28),
    "g_ag_bounds": (1e-21, 1e-19),
    "B_bounds": (1e-6, 1e-5),
    "L_bounds": (1000.0, 10000.0),
    "p_bounds": (1.5, 3.0),
    "t_rise_bounds": (1.0, 5.0),
    "t_decay_bounds": (5.0, 30.0)
}

def compute_polarization_profile(energy_grid, time_grid, mode, g_ag, B, L):
    base_pd = numpy.random.uniform(0.10, 0.20)

    if mode in ("ssc", "liv"):
        pd_matrix = base_pd + (0 * energy_grid) + (0 * time_grid)
        pa_matrix = numpy.zeros_like(energy_grid)
    else:
        oscillation_phase = (B * L * g_ag * (1.52 * (10 ** 13))) / energy_grid
        pd_1d = base_pd * (1.0 - (0.5 * oscillation_phase * (numpy.sin(oscillation_phase))**2))
        pa_1d = (numpy.pi / 4) * numpy.sin(2 * oscillation_phase)
        pd_matrix = pd_1d + numpy.zeros_like(time_grid)
        pa_matrix = pa_1d + numpy.zeros_like(time_grid)

    return pd_matrix, pa_matrix

def generate_grb_event(random_param, mode):
    energy_grid, time_grid = itemgetter("energy_grid", "time_grid")(random_param)
    E_break, alpha, beta, f_break, t_decay, t_rise, A = itemgetter("E_break", "alpha", "beta", "f_break", "t_decay", "t_rise", "A")(random_param)
    g_ag, B, L, f_E_func = itemgetter("g_ag", "B", "L", "f_E_func")(random_param)
    intensity_matrix = simulate_grb_emission(energy_grid, time_grid, E_break, alpha, beta, f_break, t_decay, t_rise, A)

    if mode == "ssc":
        spectrum = intensity_matrix
    elif mode == "liv":
        eqg = itemgetter("eqg")(random_param)
        spectrum = apply_liv_to_spectrum(time_grid, energy_grid, eqg, intensity_matrix)
    else:
        g_ag, B, L, f_E_func = itemgetter("g_ag", "B", "L", "f_E_func")(random_param)
        photon_survival = photon_survival_probability(energy_grid, g_ag, B, L, f_E_func)
        spectrum = apply_alp_to_spectrum(intensity_matrix, photon_survival)

    pd_matrix, pa_matrix = compute_polarization_profile(energy_grid, time_grid, mode, g_ag, B, L)
    observation_matrix_3d = numpy.stack([spectrum, pa_matrix, pd_matrix], axis = 0)
    assert observation_matrix_3d.shape == (3, 500, 500), f"Error, expected shape (3, 500, 500), but got {observation_matrix_3d.shape}"

    return observation_matrix_3d

if __name__ == "__main__":
    target_folder = Path('C:/Users/Ronel/Desktop/multi-modal-grb-classifier/grb-degeneracy-framework/grb-degeneracy-framework/data/synthetic/test_datasets')
    if target_folder.exists():
        shutil.rmtree(target_folder)
    target_folder.mkdir(parents=True, exist_ok=True)

    for i in range(60):
        mode = scenario_mode[i % len(scenario_mode)]

        random_param = {
            "energy_grid": energy_grid,
            "time_grid": time_grid,
            "f_break": f_break,
            "A": A,
            "alpha": alpha,
            "beta": beta,
            "E_break": E_break,
            "t_rise": t_rise,
            "t_decay": t_decay,
            "g_ag": g_ag,
            "B": B,
            "L": L,
            "eqg": 1e19,
            "f_E_func": f_E_func,
        }

        f_break = 1.0
        A = 5.0
        initial_time_grid = numpy.linspace(0, 50, 500)
        initial_energy_grid = numpy.logspace(0, 6, 500)
        energy_grid, time_grid = numpy.meshgrid(initial_energy_grid, initial_time_grid)
        p = numpy.random.uniform(1.5, 3.0)
        alpha = (p-1)/2
        beta = alpha + 0.5
        log_E_break = numpy.random.uniform(0.0, 6.0)
        E_break = 10.0 ** log_E_break
        t_rise = numpy.random.uniform(1.0, 5.0)
        t_decay = numpy.random.uniform(5.0, 30.0)
        eqg=1e19
        g_ag = numpy.random.uniform(1e-10, 1e-9)
        B = numpy.random.uniform(1e-6, 1e-5) 
        L = numpy.random.uniform(1000.0, 10000.0)
        f_E_func = lambda E: 1.0

        observation_matrix_3d = generate_grb_event(random_param, mode)
        assert observation_matrix_3d.shape == (3, 500, 500), f"Error, expected shape (3, 500, 500) got {observation_matrix_3d.shape}"
        assert not numpy.any(numpy.isnan(observation_matrix_3d)), "Error, matrix contains NaN values"
        assert not numpy.any(numpy.isinf(observation_matrix_3d)), "Error, matrix contains infinite values"

        final_filename = f"simulation_{i}.npy"
        numpy.save(target_folder / final_filename, observation_matrix_3d)

        print(f"Successfully saved No.{int(i)+1} 3D observation matrix")