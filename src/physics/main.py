import numpy
from pathlib import Path
import shutil
from .constants import ALP_PHASE_CONVERSION, EV_TO_ANGSTORMS, DEFAULT_BASE_PD, IXPE_MODULATION_FACTOR_MU
from .synchrotron import simulate_grb_emission
from .liv import apply_liv_to_spectrum
from .alp import apply_alp_to_spectrum, photon_survival_probability
from .sampler import generate_lhs_parameter_batch
from operator import itemgetter
from ..utils.fisher import compute_cramer_rao_bounds, compute_fisher_matrix, fisher_estimator_pipeline
import concurrent.futures

scenario_mode = ["ssc", "alp", "liv"]
target_folder = Path('C:/Users/Ronel/Desktop/grb-degeneracy-framework/data/synthetic/test_datasets')

parameter_bounds = {
    "E_break": (1e+5, 1e+6),
    "eqg": (1e+26, 1e+28),
    "g_ag": (1e-21, 1e-19),
    "B": (1e-6, 1e-5),
    "L": (1000.0, 10000.0),
    "p": (1.5, 3.0),
    "t_rise": (1.0, 5.0),
    "t_decay": (5.0, 30.0),
    "R_v": (2.1, 5.5),
    "E_BV": (0.01, 0.50),
}

def compute_polarization_profile(initial_energy_grid, time_grid, mode, g_ag, B, L):
    base_pd = DEFAULT_BASE_PD

    if mode in ("ssc", "liv"):
        pd_matrix = base_pd + (0 * initial_energy_grid[None, :]) + (0 * time_grid)
        pa_matrix = numpy.zeros_like(time_grid)
    else:
        oscillation_phase = (B * L * g_ag * (ALP_PHASE_CONVERSION * (10 ** 13))) / initial_energy_grid
        pd_1d = base_pd * (1.0 - (0.5 * oscillation_phase * (numpy.sin(oscillation_phase))**2))
        pa_1d = (numpy.pi / 4) * numpy.sin(2 * oscillation_phase)
        pd_matrix = pd_1d[:, None] + numpy.zeros_like(time_grid)
        pa_matrix = pa_1d[:, None] + numpy.zeros_like(time_grid)

    return pd_matrix, pa_matrix

def generate_grb_event(random_param, mode):
    initial_energy_grid, initial_time_grid = itemgetter("initial_energy_grid", "initial_time_grid")(random_param)
    energy_grid, time_grid = itemgetter("energy_grid", "time_grid")(random_param)
    
    E_break = random_param.get("E_break", 5e5)
    p = random_param.get("p", 2.2)
    alpha = (p - 1) / 2
    beta = alpha + 0.5
    
    t_rise = random_param.get("t_rise", 2.0)
    t_decay = random_param.get("t_decay", 15.0)
    f_break = random_param.get("f_break", 1.0)
    A = random_param.get("A", 5.0)
    g_ag, B, L, f_E_func = itemgetter("g_ag", "B", "L", "f_E_func")(random_param)
    R_v = random_param.get("R_v", 3.1)
    E_BV = random_param.get("E_BV", 0.1)

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

    if pa_matrix.shape != spectrum.shape:
        pa_matrix = numpy.broadcast_to(pa_matrix, spectrum.shape)

    if pd_matrix.shape != spectrum.shape:
        pd_matrix = numpy.broadcast_to(pd_matrix, spectrum.shape)
    if not (spectrum.shape == pa_matrix.shape == pd_matrix.shape == (500, 500)):
        print(f"[DEBUG Mode '{mode}'] spectrum: {spectrum.shape}, pa_matrix: {pa_matrix.shape}, pd_matrix: {pd_matrix.shape}")

        if spectrum.shape == (500, 500, 1):
            spectrum = spectrum.squeeze(-1)
        elif spectrum.shape != (500, 500):
            spectrum = numpy.atleast_2d(spectrum).reshape(500, 500)

    observation_matrix_3d = numpy.stack([spectrum, pa_matrix, pd_matrix], axis=0)
    assert observation_matrix_3d.shape == (3, 500, 500), f"Error, expected shape (3, 500, 500), but got {observation_matrix_3d.shape}"

    wavelength = EV_TO_ANGSTORMS / initial_energy_grid
    x = 1 / wavelength

    a_x = numpy.zeros_like(x)
    b_x = numpy.zeros_like(x)

    ir_mask = (x >= 0.3) & (x < 1.1)
    opt_mask = (x >= 1.1) & (x < 3.3)

    if numpy.any(ir_mask):
        a_x[ir_mask] = 0.574 * (x[ir_mask] ** 1.61)
        b_x[ir_mask] = -0.527 * (x[ir_mask] ** 1.61)

    if numpy.any(opt_mask):
        y = x[opt_mask] - 1.82
        a_x[opt_mask] = (
            1.0 
            + (0.17699 * y) 
            - (0.50447 * (y**2)) 
            - (0.02427 * (y**3)) 
            + (0.72085 * (y**4)) 
            + (0.01979 * (y**5)) 
            - (0.77530 * (y**6)) 
            + (0.32999 * (y**7))
        )
        b_x[opt_mask] = (
            (1.41338 * y) 
            + (2.28305 * (y**2)) 
            + (1.07233 * (y**3)) 
            - (5.38434 * (y**4)) 
            - (0.62251 * (y**5)) 
            + (5.30260 * (y**6)) 
            - (2.09002 * (y**7))
        )

    extinction_ratio = a_x + (b_x / R_v) 

    absolute_extinction = E_BV * extinction_ratio
    transmission = 10.0 ** (-0.4 * absolute_extinction)

    if observation_matrix_3d.shape[1] == len(transmission):
        observation_matrix_3d[0] = observation_matrix_3d[0] * transmission[:, numpy.newaxis]
    else:
        observation_matrix_3d[0] = observation_matrix_3d[0] * transmission[numpy.newaxis, :]

    return observation_matrix_3d

def compute_flux_jacobian(active_params, mode, obs_mode, base_param):
    epsilon = 1e-4
    jacobian_dict = {}

    init_en = base_param.get("initial_energy_grid")
    init_tm = base_param.get("initial_time_grid")
    en_grid = base_param.get("energy_grid")
    tm_grid = base_param.get("time_grid")

    base_flux_raw = generate_grb_event(base_param, mode)

    if obs_mode == "spectral_only":
        base_flux = numpy.mean(base_flux_raw, axis=(0, 2))
    elif obs_mode == "spectral_temporal":
        base_flux = base_flux_raw[0, :, :]
    elif obs_mode == "full_polarimetric":
        base_flux = base_flux_raw

    for key in active_params:
        val = base_param[key]
        final_val = val if val != 0 else 1e-8
        delta = epsilon * final_val

        param_up = base_param.copy()
        param_up[key] += delta
        param_up["initial_energy_grid"] = init_en
        param_up["initial_time_grid"] = init_tm
        param_up["energy_grid"] = en_grid
        param_up["time_grid"] = tm_grid
        flux_up_raw = generate_grb_event(param_up, mode)

        param_down = base_param.copy()
        param_down[key] -= delta
        param_down["initial_energy_grid"] = init_en
        param_down["initial_time_grid"] = init_tm
        param_down["energy_grid"] = en_grid
        param_down["time_grid"] = tm_grid
        flux_down_raw = generate_grb_event(param_down, mode)

        if obs_mode == "spectral_only":
            new_flux_up = numpy.mean(flux_up_raw, axis=(0, 2))
            new_flux_down = numpy.mean(flux_down_raw, axis=(0, 2))
        elif obs_mode == "spectral_temporal":
            new_flux_up = flux_up_raw[0, :, :]
            new_flux_down = flux_down_raw[0, :, :]
        elif obs_mode == "full_polarimetric":
            new_flux_up = flux_up_raw
            new_flux_down = flux_down_raw
        
        derivative_matrix = (new_flux_up - new_flux_down) / (2 * epsilon * (base_flux + 1e-30))
        jacobian_dict[key] = derivative_matrix

        grad_std = numpy.std(derivative_matrix)
        if grad_std < 1e-12:
            continue
    return jacobian_dict

def get_effective_params(active_params, mode, obs_mode):
    effective = list(active_params)

    if obs_mode == "spectral_only":
        for inactive_key in ["liv_scale", "alp_coupling"]:
            if inactive_key in effective:
                effective.remove(inactive_key)
    return effective

def scale_polarimetric_noise(flux_matrix, mode="full_polarimetric", mu=IXPE_MODULATION_FACTOR_MU):
    sigma_flux = numpy.sqrt(numpy.abs(flux_matrix) + 1.0)

    if mode == "full_polarimetric":
        sigma_pol = sigma_flux / (mu * numpy.sqrt(2.0))
        return sigma_flux, sigma_pol

    return sigma_flux, None

def process_single_simulation(args):
    try:
        i, random_param, mode, snr, obs_mode = args

        print(f"[INFO] Worker started for event {int(i)+1} under mode '{mode}', SNR {snr}, Obs: {obs_mode}...")
        observation_matrix_3d = generate_grb_event(random_param, mode)
        assert observation_matrix_3d.shape == (3, 500, 500), f"Error, expected shape (3, 500, 500) got {observation_matrix_3d.shape}"
        assert not numpy.any(numpy.isnan(observation_matrix_3d)), "Error, matrix contains NaN values"
        assert not numpy.any(numpy.isinf(observation_matrix_3d)), "Error, matrix contains infinite values"

        print(f"[DEBUG] Event {int(i)+1} generated successfully. Running fisher pipeline...")
        active_params_dict = fisher_estimator_pipeline(random_param, mode)
        active_keys = list(active_params_dict.keys())

        print(f"[DEBUG] Computing effective parameters for event {int(i)+1}...")
        effective_params = get_effective_params(active_params_dict, mode, obs_mode)

        print(f"[DEBUG] Computing Jacobian for event {int(i)+1}...")
        jacobian_observation_matrix_3d = compute_flux_jacobian(effective_params, mode, obs_mode, random_param)

        print(f"[DEBUG] Computing Fisher Matrix for event {int(i)+1}...")
        fisher_jacobian_observation_matrix_3d = compute_fisher_matrix(jacobian_observation_matrix_3d, effective_params, snr)

        print(f"[DEBUG] Extracting Cramer-Rao bounds for event {int(i)+1}...")
        cramer_rao_extraction, covariance_matrix_final = compute_cramer_rao_bounds(fisher_jacobian_observation_matrix_3d, active_keys)

        """
        batch_folder = target_folder / f"batch_{i // 1000}"
        if batch_folder.exists():
            shutil.rmtree(batch_folder)
        batch_folder.mkdir(parents=True, exist_ok=True)
        """

        final_filename = f"simulation_{i}_{obs_mode}.npz"
        numpy.savez(
            target_folder / final_filename,
            observation=observation_matrix_3d,
            label=mode,
            snr=snr,
            obs_mode=obs_mode,
            fisher=fisher_jacobian_observation_matrix_3d, 
            covariance=covariance_matrix_final, 
            cramer=cramer_rao_extraction
            )

        print(f"[SUCCESS] Saved No.{int(i)+1} for all 3 outputs")
    except Exception as e:
        print(f"[WORKER ERROR] Failed on event {i}: {str(e)}")
        raise e

# Main Generation Script

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()

    print("[INFO] Starting GRB simulation pipeline...")
    if target_folder.exists():
        shutil.rmtree(target_folder)
    target_folder.mkdir(parents=True, exist_ok=True)

    initial_time_grid = numpy.linspace(0, 50, 500)
    initial_energy_grid = numpy.logspace(0, 6, 500)
    energy_grid, time_grid = numpy.meshgrid(initial_energy_grid, initial_time_grid)

    num_samples = 60
    print("[INFO] Generating LHS parameter batch...")
    sampled_batch = generate_lhs_parameter_batch(num_samples=num_samples, parameter_bounds=parameter_bounds)

    rng = numpy.random.default_rng(42)

    snr_choices = [5, 15, 50]
    obs_mode_choices = ["spectral_only", "spectral_temporal", "full_polarimetric"]
    repeats = num_samples // len(snr_choices)
    snr_pool = numpy.repeat(snr_choices, repeats)
    obs_mode_pool = numpy.repeat(obs_mode_choices, repeats)

    rng.shuffle(snr_pool)
    rng.shuffle(obs_mode_pool)

    tasks = []
    for i, random_param in enumerate(sampled_batch):
        mode = scenario_mode[i % len(scenario_mode)]
        snr = int(snr_pool[i])
        obs_mode = str(obs_mode_pool[i])

        random_param["initial_energy_grid"] = initial_energy_grid
        random_param["initial_time_grid"] = initial_time_grid
        random_param["energy_grid"] = energy_grid
        random_param["time_grid"] = time_grid

        tasks.append((i, random_param, mode, snr, obs_mode))

    with concurrent.futures.ProcessPoolExecutor() as executor:
        executor.map(process_single_simulation, tasks)