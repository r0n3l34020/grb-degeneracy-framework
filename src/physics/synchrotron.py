import numpy
import matplotlib.pyplot as plt

def synchrotron_signature(E_break, energy_grid, alpha, beta, f_break):

    """
    Calculates a broken power-law synchrotron spectrum for GRB afterglows.

    Parameters:
    alpha (float): Low-energy photon spectral index.
    beta (float): High-energy photon spectral index.
    E_break (float): Break energy threshold
    f_break (float): Flux normalization at the break energy
    """
    flux_list = []

    low_energy = f_break * (energy_grid / E_break) ** (-alpha)
            
    high_energy = f_break * (energy_grid / E_break) ** (-beta)
    
    return numpy.where(energy_grid <= E_break, low_energy, high_energy)

if __name__ == "__main__":
    f_break = 1.0
    E_break = 50.0
    energy_grid = numpy.logspace(1, 19, 500)

    p = float(input("What is the electron distribution index? "))
    alpha = (p-1)/2
    beta = alpha + 0.5

    final_flux = synchrotron_signature(E_break, energy_grid, alpha, beta, f_break)

    plt.figure(figsize=(8, 5))
    plt.loglog(energy_grid, final_flux)
    plt.title(f"Synchrotron Spectrum (p={p})")
    plt.xlabel("Energy (eV)")
    plt.ylabel("Flux")
    plt.grid(True, which="both", ls="-")
    plt.show()