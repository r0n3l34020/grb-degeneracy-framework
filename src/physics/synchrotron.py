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

    return final_flux, final_light_curve

if __name__ == "__main__":

    f_break = 1.0
    E_break = 50.0
    energy_grid = numpy.logspace(1, 19, 500)
    p = float(input("What is the electron distribution index? "))
    alpha = (p-1)/2
    beta = alpha + 0.5

    A = 1.0
    t_rise = 10.5
    t_decay = 2.0
    time_grid = numpy.linspace(0, 50, 500)

    flux, light_curve = simulate_grb_emission(energy_grid, time_grid, E_break, alpha, beta, f_break, t_decay, t_rise, A)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))
    ax1.loglog(energy_grid, flux)
    ax1.set_xlabel("Energy (ev)")
    ax1.set_ylabel("Flux")
    ax1.grid(True, which="both", ls="-")
    ax2.plot(time_grid, light_curve)
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Light Curve Intensity")
    ax2.grid(True)
    plt.show()

    """
    A = 1.0
    t_rise = 10.5
    t_decay = 2.0
    time_grid = numpy.linspace(0, 50, 500)

    final_light_curve = fred_light_curve(time_grid, t_decay, t_rise, A)

    plt.figure(figsize=(8, 5))
    plt.plot(time_grid, final_light_curve)
    plt.title(f"FRED")
    plt.xlabel("Time (s)")
    plt.ylabel("Light Curve")
    plt.grid(True, which="both", ls="-")
    plt.show()
    
    f_break = 1.0
    E_break = 50.0
    energy_grid = numpy.logspace(1, 19, 500)

    p = float(input("What is the electron distribution index? "))
    alpha = (p-1)/2
    beta = alpha + 0.5

    
    final_flux = synchrotron_signature(E_break, energy_grid, alpha, beta, f_break)
    """
    