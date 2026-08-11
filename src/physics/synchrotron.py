import numpy
from .constants import FRED_MIN_TIME_EPSILON

def synchrotron_signature(E_break, initial_energy_grid, alpha, beta, f_break):
    low_energy = f_break * (initial_energy_grid / E_break) ** (-alpha)
            
    high_energy = f_break * (initial_energy_grid / E_break) ** (-beta)
    
    return numpy.where(initial_energy_grid <= E_break, low_energy, high_energy)

def fred_light_curve(initial_time_grid, t_decay, t_rise, A):
    t = numpy.maximum(initial_time_grid, FRED_MIN_TIME_EPSILON)
    result = A * numpy.exp( - (t/t_decay + t_rise/t))
    return result

def simulate_grb_emission(initial_energy_grid, initial_time_grid, E_break, alpha, beta, f_break, t_decay, t_rise, A):

    final_flux = synchrotron_signature(E_break, initial_energy_grid, alpha, beta, f_break)

    final_light_curve = fred_light_curve(initial_time_grid, t_decay, t_rise, A)

    matrix_2d = numpy.outer(final_flux, final_light_curve)
    return matrix_2d