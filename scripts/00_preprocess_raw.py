from pathlib import Path
from astropy.io import fits
import h5py
import numpy as np
import pandas as pd


def build_target_tensor():
  raw_dir = Path("data/raw")
  out_path = Path("data/processed/target_event.h5")
  out_path.parent.mkdir(parents=True, exist_ok=True)

  grid_size = 128
  tensor = np.zeros((3, grid_size, grid_size), dtype=np.float32)

  fits_file = raw_dir / "fermi_gbm_tte.fits"
  if fits_file.exists():
    with fits.open(fits_file) as hdul:
      events = hdul["EVENTS"].data
      times = events["TIME"]
      channels = events["PHA"]
      hist, _, _ = np.histogram2d(
          times, channels, bins=[grid_size, grid_size]
      )
      tensor[0] = hist / (np.max(hist) + 1e-8)

  lhaaso_file = raw_dir / "lhaaso_flux.csv"
  if lhaaso_file.exists():
    try:
      df_lh = pd.read_csv(lhaaso_file, comment="#", skiprows=1, header=None)
      t_lh = df_lh.iloc[:, 0].values
      f_lh = df_lh.iloc[:, 1].values
      grid_x = np.linspace(t_lh.min(), t_lh.max(), grid_size)
      grid_y = np.interp(grid_x, t_lh, f_lh)
      tensor[1] = np.tile(grid_y, (grid_size, 1))
      tensor[1] /= np.max(tensor[1]) + 1e-8
    except Exception:
      pass

  swift_file = raw_dir / "swift_xrt_lc.txt"
  if swift_file.exists():
    try:
      lines = [
          line.strip()
          for line in open(swift_file)
          if line[0].isdigit() or line[0] == "-"
      ]
      data = np.loadtxt(lines)
      t_sw, f_sw = data[:, 0], data[:, 2]
      grid_x = np.linspace(t_sw.min(), t_sw.max(), grid_size)
      grid_y = np.interp(grid_x, t_sw, f_sw)
      tensor[2] = np.tile(grid_y, (grid_size, 1)).T
      tensor[2] /= np.max(tensor[2]) + 1e-8
    except Exception:
      pass

  with h5py.File(out_path, "w") as f:
    f.create_dataset("tensors", data=np.expand_dims(tensor, axis=0))

  print(
      f"[SUCCESS] Target tensor constructed successfully at {out_path}"
  )


if __name__ == "__main__":
  build_target_tensor()
