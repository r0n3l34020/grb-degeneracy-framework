import numpy as np
from pathlib import Path

target_folder = Path('C:/Users/Ronel/Desktop/multi-modal-grb-classifier/grb-degeneracy-framework/grb-degeneracy-framework/data/synthetic/test_datasets')

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