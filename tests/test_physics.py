import numpy
from src.physics.synchrotron import simulate_grb_emission

def test_synchrotron_output():
    energy_grid = numpy.logspace(1, 19, 500)
    time_grid = numpy.linspace(0, 50, 500)
    matrix_2d = simulate_grb_emission(energy_grid, time_grid, 100.0, 1.0, 1.5, 1.0, 50.0, 2.0, 5.0)
    assert matrix_2d.shape == (500, 500), f"Error! Expected shape (500, 500), but got {matrix_2d.shape}"
    assert not numpy.any(numpy.isnan(matrix_2d)), "Error! Matrix contains NaN values (division by zero or math failure)."
    assert not numpy.any(numpy.isinf(matrix_2d)), "Error! Matrix contains infinite values."
print("✅ All validation tests passed successfully!")