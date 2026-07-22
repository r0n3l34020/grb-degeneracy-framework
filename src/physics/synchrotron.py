import numpy
import matplotlib.pyplot as plt
from pathlib import Path

def synchrotron_signature(E_break, energy_grid, alpha, beta, f_break):

    """
    Calculates a broken power-law synchrotron spectrum for GRB afterglows.

    Parameters:
    alpha (float): Low-energy photon spectral index.
    beta (float): High-energy photon spectral index.
    E_break (float): Break energy threshold
    f_break (float): Flux normalization at the break energy

    """

    low_energy = f_break * (energy_grid / E_break) ** (-alpha)
            
    high_energy = f_break * (energy_grid / E_break) ** (-beta)
    
    return numpy.where(energy_grid <= E_break, low_energy, high_energy)

def fred_light_curve(time_grid, t_decay, t_rise, A):
    """
    Calculates light curve intensity across time for GRB afterglows:

    Parameters:
    t_decay: time taken for the afterglow to dim
    t_rise: time taken for the explosion to reach its peak
    A: amplitude of the explosion (how bright the explosion is)
    """
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
    energy_grid = numpy.logspace(1, 19, 500)

    A = 5.0
    time_grid = numpy.linspace(0, 50, 500)

    for i in range(20):
        p = numpy.random.uniform(1.5, 3.0)
        alpha = (p-1)/2
        beta = alpha + 0.5
        log_E_break = numpy.random.uniform(1.0, 19.0)
        E_break = 10.0 ** log_E_break
        t_rise = numpy.random.uniform(1.0, 5.0)
        t_decay = numpy.random.uniform(50.0, 120.0)
        matrix_2d = simulate_grb_emission(energy_grid, time_grid, E_break, alpha, beta, f_break, t_decay, t_rise, A)

    # Sanity checks to ensure there are no errors when generating data
        assert matrix_2d.shape == (500, 500), f"Error! Expected shape (500, 500), but got {matrix_2d.shape}"
        assert not numpy.any(numpy.isnan(matrix_2d)), "Error! Matrix contains NaN values (division by zero or math failure)."
        assert not numpy.any(numpy.isinf(matrix_2d)), "Error! Matrix contains infinite values."
        print("✅ All validation tests passed successfully!")

    # Saving simulation matrices to [physics] folder
        target_folder = Path('C:/Users/Ronel/Desktop/multi-modal-grb-classifier/grb-degeneracy-framework/grb-degeneracy-framework/data/synthetic/test_datasets')
        filename = f"grb_simulation_{i}_{round(p, 2)}_{E_break:.1e}.npy"
        numpy.save(target_folder / filename, matrix_2d)
        print(f"Successfully saved simulation matrix to {filename}!")
    

    # Visual De-bugging with Heatmap
    """
    plt.figure(figsize=(8, 5))
    plt.pcolormesh(time_grid, numpy.log10(energy_grid), matrix_2d, shading='auto', cmap='inferno')
    plt.colorbar(label="Flux Intensity")
    plt.xlabel("Time (s)")
    plt.ylabel("Log10 Energy (eV)")
    plt.show()
    """
