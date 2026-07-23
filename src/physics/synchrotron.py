import numpy
# import matplotlib.pyplot as plt
import shutil
from pathlib import Path
from liv import apply_liv_to_spectrum
from alp import apply_alp_to_spectrum, photon_survival_probability

def synchrotron_signature(E_break, energy_grid, alpha, beta, f_break):
    low_energy = f_break * (energy_grid / E_break) ** (-alpha)
            
    high_energy = f_break * (energy_grid / E_break) ** (-beta)
    
    return numpy.where(energy_grid <= E_break, low_energy, high_energy)

def fred_light_curve(time_grid, t_decay, t_rise, A):
    t = numpy.maximum(time_grid, 1e-5)
    result = A * numpy.exp( - (t/t_decay + t_rise/t))
    return result

def simulate_grb_emission(energy_grid, time_grid, E_break, alpha, beta, f_break, t_decay, t_rise, A):

    final_flux = synchrotron_signature(E_break, energy_grid, alpha, beta, f_break)

    final_light_curve = fred_light_curve(time_grid, t_decay, t_rise, A)

    matrix_2d = numpy.outer(final_flux, final_light_curve)
    return matrix_2d

if __name__ == "__main__":

    f_break = 1.0
    energy_grid = numpy.logspace(0, 6, 500)

    A = 5.0
    time_grid = numpy.linspace(0, 50, 500)
    target_folder = Path('C:/Users/Ronel/Desktop/multi-modal-grb-classifier/grb-degeneracy-framework/grb-degeneracy-framework/data/synthetic/test_datasets')
    if target_folder.exists():
        shutil.rmtree(target_folder)
    target_folder.mkdir(parents=True, exist_ok=True)
    ssc_folder = target_folder / 'ssc'
    liv_folder = target_folder / 'liv'
    alp_folder = target_folder / 'alp'
    ssc_folder.mkdir(parents=True, exist_ok=True)
    liv_folder.mkdir(parents=True, exist_ok=True)
    alp_folder.mkdir(parents=True, exist_ok=True)

    for i in range(20):
        p = numpy.random.uniform(1.5, 3.0)
        alpha = (p-1)/2
        beta = alpha + 0.5
        log_E_break = numpy.random.uniform(0.0, 6.0)
        E_break = 10.0 ** log_E_break
        t_rise = numpy.random.uniform(1.0, 5.0)
        t_decay = numpy.random.uniform(5.0, 30.0)

    # Generate Baseline Synchrotron Matrix
        matrix_2d = simulate_grb_emission(energy_grid, time_grid, E_break, alpha, beta, f_break, t_decay, t_rise, A)

    # Sanity checks to ensure there are no errors when generating data
        assert matrix_2d.shape == (500, 500), f"Error, expected shape (500, 500) got {matrix_2d.shape}"
        assert not numpy.any(numpy.isnan(matrix_2d)), "Error, matrix contains NaN values"
        assert not numpy.any(numpy.isinf(matrix_2d)), "Error, matrix contains infinite values"

    # Saving SSC simulation matrices to [test_datasets] folder
        ssc_filename = f"ssc_simulation_{i}_{round(p, 2)}_{E_break:.1e}.npy"
        numpy.save(ssc_folder / ssc_filename, matrix_2d)

    # Generating and Saving (to [test_datasets] folder) LIV-modified Spectrum
        eqg=1e19
        liv_spectrum = apply_liv_to_spectrum(time_grid, energy_grid, eqg, matrix_2d)
        liv_filename = f"liv_simulation_{i}_{round(p, 2)}_{E_break:.1e}.npy"
        numpy.save(liv_folder / liv_filename, liv_spectrum)

    # Generating and Saving (to [test_datasets] folder) ALP-modified Spectrum
        g_ag = numpy.random.uniform(1e-10, 1e-9)
        B = numpy.random.uniform(1e-6, 1e-5) 
        L = numpy.random.uniform(1000.0, 10000.0)
        f_E_func = lambda E: 1.0 
        get_photon_survival = photon_survival_probability(energy_grid, g_ag, B, L, f_E_func)
        alp_spectrum = apply_alp_to_spectrum(matrix_2d, get_photon_survival)
        alp_filename = f"alp_simulation_{i}_{round(p, 2)}_{E_break:.1e}.npy"
        numpy.save(alp_folder / alp_filename, alp_spectrum)

        print(f"Successfully saved No.{int(i)+1} for all three synthetic simulation matrices!")
    

    # Visual De-bugging with Heatmap
    """
    plt.figure(figsize=(8, 5))
    plt.pcolormesh(time_grid, numpy.log10(energy_grid), matrix_2d, shading='auto', cmap='inferno')
    plt.colorbar(label="Flux Intensity")
    plt.xlabel("Time (s)")
    plt.ylabel("Log10 Energy (eV)")
    plt.show()
    """
