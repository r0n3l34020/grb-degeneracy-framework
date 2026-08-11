import numpy as np
from pathlib import Path

target_folder = Path('C:/Users/Ronel/Desktop/grb-degeneracy-framework/data/synthetic/test_datasets')

data = numpy.load(target_folder / "simulation_0_full_polarimetric.npz")
print("Saved keys:", list(data.keys()))
# Expected output: ['observation', 'label', 'snr', 'obs_mode', 'fisher', 'covariance', 'cramer_rao']

print("Observation tensor shape:", data['observation'].shape)
# Expected output: (3, 500, 500)

print("Label:", data['label'])
# Expected output: 'ssc' (or 'alp', 'liv')

"""
ssc_samples = list((target_folder / 'ssc').glob('*.npy'))

if ssc_samples:
    sample_path = ssc_samples[0]
    loaded_matrix = np.load(sample_path)
    
    print(f"Inspecting file: {sample_path.name}")
    print(f"Matrix Shape: {loaded_matrix.shape}")
    print(f"Data Type: {loaded_matrix.dtype}")
    print(f"Min Value: {np.min(loaded_matrix)}")
    print(f"Max Value: {np.max(loaded_matrix)}")
    print(f"Contains NaNs: {np.any(np.isnan(loaded_matrix))}")
    print(f"Contains Infs: {np.any(np.isinf(loaded_matrix))}")
else:
    print("No files found in the ssc directory.")
"""