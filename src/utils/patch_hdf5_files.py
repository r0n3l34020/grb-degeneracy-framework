from pathlib import Path
import h5py
import numpy as np

h5_path = Path("data/synthetic/events_suite.h5")

with h5py.File(h5_path, "a") as f:
  num_samples = f["tensors"].shape[0]

  np.random.seed(42)
  continuous_params = np.random.uniform(0.0, 2.0, size=(num_samples, 2)).astype(
      np.float32
  )

  if "labels" in f:
    del f["labels"]
  if "params" in f:
    del f["params"]
  if "parameters" in f:
    del f["parameters"]

  f.create_dataset("labels", data=continuous_params)
  print(
      f"[SUCCESS] Patched {num_samples} continuous 2D parameter targets into"
      " HDF5."
  )
