import h5py

with h5py.File("data/synthetic/events_suite.h5", "r") as f:
  print("Keys in HDF5:", list(f.keys()))
  print("Tensors shape:", f["tensors"].shape)
  for key in f.keys():
    if key != "tensors":
      print(f"{key} shape:", f[key].shape)
      print(f"{key} sample:", f[key][:5])
