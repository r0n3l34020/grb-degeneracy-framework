import numpy

def synchrotron_signature(E_break, energy_grid, alpha, beta, f_break):

    """
    Calculates a broken power-law synchrotron spectrum for GRB afterglows.

    Parameters:
    E (array): Photon energy grid (eV to TeV).
    alpha (float): Low-energy photon spectral index.
    beta (float): High-energy photon spectral index.
    E_break (float): Break energy threshold
    f_break (float): Flux normalization at the break energy
    """
    flux_list = []

    for E in energy_grid:
        if E <= E_break:
            result = f_break * (E / E_break) ** (-alpha)
            flux_list.append(result)
        
        else:
            result = f_break * (E / E_break) ** (-beta)
            flux_list.append(result)

    return(flux_list)

if __name__ == "__main__":
    f_break = 1.0
    E_break = 50.0
    energy_grid = numpy.logspace(1, 19, 500)

    p = float(input("What is the electron distribution index? "))
    alpha = (p-1)/2
    beta = alpha + 0.5

    final_flux = synchrotron_signature(E_break, energy_grid, alpha, beta, f_break)

print(final_flux)