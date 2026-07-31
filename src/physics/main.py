import numpy
from pathlib import Path
import math
import shutil
from synchrotron import simulate_grb_emission
from liv import apply_liv_to_spectrum
from alp import apply_alp_to_spectrum, photon_survival_probability
from sampler import generate_lhs_parameter_batch
from operator import itemgetter
from src.utils.fisher import compute_cramer_rao_bounds, compute_fisher_matrix, compute_flux_jacobian, fisher_estimator_pipeline
import concurrent.futures

scenario_mode = ["ssc", "alp", "liv"]
target_folder = Path('C:/Users/Ronel/Desktop/multi-modal-grb-classifier/grb-degeneracy-framework/grb-degeneracy-framework/data/synthetic/test_datasets')
# All bounds converted to match the 'energy_grid' scale
parameter_bounds = {
    "E_break": (1e+5, 1e+6),
    "eqg": (1e+26, 1e+28),
    "g_ag": (1e-21, 1e-19),
    "B": (1e-6, 1e-5),
    "L": (1000.0, 10000.0),
    "p": (1.5, 3.0),
    "t_rise": (1.0, 5.0),
    "t_decay": (5.0, 30.0)
}

target_params = [
    "E_break",
    "g_ag",
    "B",
    "L",
    "eqg",
]

snr_choices = [5, 15, 50]
repeats = 20
snr_pool = numpy.repeat(snr_choices, repeats)
snr_range = numpy.random.default_rng()
snr_range.shuffle(snr_pool)

obs_mode_choices = ["spectral_only", "spectral_temporal", "full_polarimetric"]
obs_mode_pool = numpy.repeat(obs_mode_choices, repeats)
obs_mode_range = numpy.random.default_rng()
obs_mode_range.shuffle(obs_mode_pool)

def compute_polarization_profile(initial_energy_grid, time_grid, mode, g_ag, B, L):
    base_pd = 0.15

    if mode in ("ssc", "liv"):
        pd_matrix = base_pd + (0 * initial_energy_grid[None, :]) + (0 * time_grid)
        pa_matrix = numpy.zeros_like(time_grid)
    else:
        oscillation_phase = (B * L * g_ag * (1.52 * (10 ** 13))) / initial_energy_grid
        pd_1d = base_pd * (1.0 - (0.5 * oscillation_phase * (numpy.sin(oscillation_phase))**2))
        pa_1d = (numpy.pi / 4) * numpy.sin(2 * oscillation_phase)
        pd_matrix = pd_1d[None, :] + numpy.zeros_like(time_grid)
        pa_matrix = pa_1d[None, :] + numpy.zeros_like(time_grid)

    return pd_matrix, pa_matrix

def generate_grb_event(random_param, mode):
    initial_energy_grid, initial_time_grid = itemgetter("initial_energy_grid", "initial_time_grid")(random_param)
    energy_grid, time_grid = itemgetter("energy_grid", "time_grid")(random_param)
    
    E_break, alpha, beta, f_break, t_decay, t_rise, A = itemgetter("E_break", "alpha", "beta", "f_break", "t_decay", "t_rise", "A")(random_param)
    g_ag, B, L, f_E_func = itemgetter("g_ag", "B", "L", "f_E_func")(random_param)

    intensity_matrix = simulate_grb_emission(initial_energy_grid, initial_time_grid, E_break, alpha, beta, f_break, t_decay, t_rise, A)

    if mode == "ssc":
        spectrum = intensity_matrix
    elif mode == "liv":
        eqg = itemgetter("eqg")(random_param)
        spectrum = apply_liv_to_spectrum(initial_time_grid, initial_energy_grid, eqg, intensity_matrix)
    else:
        photon_survival = photon_survival_probability(initial_energy_grid, g_ag, B, L, f_E_func)
        spectrum = apply_alp_to_spectrum(intensity_matrix, photon_survival)

    pd_matrix, pa_matrix = compute_polarization_profile(initial_energy_grid, time_grid, mode, g_ag, B, L)

    if not (spectrum.shape == pa_matrix.shape == pd_matrix.shape == (500, 500)):
        print(f"[DEBUG Mode '{mode}'] spectrum: {spectrum.shape}, pa_matrix: {pa_matrix.shape}, pd_matrix: {pd_matrix.shape}")

        if spectrum.shape == (500, 500, 1):
            spectrum = spectrum.squeeze(-1)
        elif spectrum.shape != (500, 500):
            spectrum = numpy.atleast_2d(spectrum).reshape(500, 500)

    observation_matrix_3d = numpy.stack([spectrum, pa_matrix, pd_matrix], axis=0)
    assert observation_matrix_3d.shape == (3, 500, 500), f"Error, expected shape (3, 500, 500), but got {observation_matrix_3d.shape}"

    return observation_matrix_3d

def process_single_simulation(args):
    i, random_param, mode, snr, obs_mode = args

    observation_matrix_3d = generate_grb_event(random_param, mode)
    assert observation_matrix_3d.shape == (3, 500, 500), f"Error, expected shape (3, 500, 500) got {observation_matrix_3d.shape}"
    assert not numpy.any(numpy.isnan(observation_matrix_3d)), "Error, matrix contains NaN values"
    assert not numpy.any(numpy.isinf(observation_matrix_3d)), "Error, matrix contains infinite values"

    print(f"Successfully created {int(i)+1} GRB event(s)")

    active_params_matrix = fisher_estimator_pipeline(random_param, mode)

    jacobian_observation_matrix_3d = compute_flux_jacobian(active_params_matrix, mode, obs_mode)

    fisher_jacobian_observation_matrix_3d = compute_fisher_matrix(jacobian_observation_matrix_3d, active_params_matrix, snr)

    cramer_rao_extraction, covariance_matrix_final = compute_cramer_rao_bounds(fisher_jacobian_observation_matrix_3d, active_params_matrix)

    final_filename = f"simulation_{i}_{obs_mode}.npy"
    numpy.savez(target_folder / final_filename, fisher_jacobian_observation_matrix_3d, covariance_matrix_final, cramer_rao_extraction)

    print(f"Successfully saved No.{int(i)+1} for all 3 outputs")

# Main Generation Script

if __name__ == "__main__":
    if target_folder.exists():
        shutil.rmtree(target_folder)
    target_folder.mkdir(parents=True, exist_ok=True)

    initial_time_grid = numpy.linspace(0, 50, 500)
    initial_energy_grid = numpy.logspace(0, 6, 500)
    energy_grid, time_grid = numpy.meshgrid(initial_energy_grid, initial_time_grid)

    sampled_batch = generate_lhs_parameter_batch(num_samples=60, parameter_bounds=parameter_bounds)

    tasks = []
    for i, random_param in enumerate(sampled_batch):
        mode = scenario_mode[i % len(scenario_mode)]
        snr = snr_pool[i]
        obs_mode = obs_mode_pool[i]

        random_param["initial_energy_grid"] = initial_energy_grid
        random_param["initial_time_grid"] = initial_time_grid
        random_param["energy_grid"] = energy_grid
        random_param["time_grid"] = time_grid

        tasks.append((i, random_param, mode, snr, obs_mode))
    with concurrent.futures.ProcessPoolExecutor() as executor:
        executor.map(process_single_simulation, tasks)