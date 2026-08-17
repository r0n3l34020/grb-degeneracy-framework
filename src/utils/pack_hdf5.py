import json
from pathlib import Path
import h5py
import numpy as np
from tqdm import tqdm


def pack_npz_to_hdf5(
    input_dir: Path,
    output_h5_path: Path,
    chunk_size: int = 500
) -> Path:

    input_dir = Path(input_dir)
    output_h5_path = Path(output_h5_path)
    output_h5_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Scanning for simulation files in {input_dir}...")
    npz_files = sorted(list(input_dir.rglob("*.npz")))
    total_files = len(npz_files)

    if total_files == 0:
        raise FileNotFoundError(f"No .npz simulation files found in {input_dir}")

    print(f"[INFO] Found {total_files} simulation files. Reading initial metadata...")
    with np.load(npz_files[0]) as sample:
        obs_shape = sample["observation"].shape
        has_params = "parameters" in sample
        if has_params:
            param_shape = sample["parameters"].shape
        else:
            param_shape = None

    unique_labels = set()
    for f in npz_files:

        with np.load(f) as data:
            unique_labels.add(str(data["label"]))

    label_map = {name: i for i, name in enumerate(sorted(unique_labels))}
    print(f"[INFO] Discovered labels: {label_map}")

    all_parameters = np.zeros((total_files, param_shape[0]), dtype=np.float32) if has_params else None

    with h5py.File(output_h5_path, "w") as h5f:
        tensors_ds = h5f.create_dataset(
            "tensors",
            shape=(total_files, *obs_shape),
            dtype=np.float32,
            chunks=(1, *obs_shape),
            compression="gzip",
            compression_opts=4
        )

        labels_ds = h5f.create_dataset(
            "labels",
            shape=(total_files,),
            dtype=np.int64
        )

        if has_params:
            params_ds = h5f.create_dataset(
                "parameters",
                shape=(total_files, *param_shape),
                dtype=np.float32,
                compression="gzip",
                compression_opts=4
            )

        print(f"[INFO] Packaging {total_files} events into {output_h5_path.name}...")
        
        for start_idx in tqdm(range(0, total_files, chunk_size), desc="Packaging HDF5 Archive"):
            end_idx = min(start_idx + chunk_size, total_files)
            chunk_files = npz_files[start_idx:end_idx]

            chunk_tensors = []
            chunk_labels = []
            chunk_params = []

            for file_path in chunk_files:
                with np.load(file_path) as data:
                    obs = data["observation"].astype(np.float32)
                    lbl = label_map[str(data["label"])]
                    
                    chunk_tensors.append(obs)
                    chunk_labels.append(lbl)

                    if has_params:
                        p_vec = data["parameters"].astype(np.float32)
                        chunk_params.append(p_vec)
            tensors_ds[start_idx:end_idx] = np.array(chunk_tensors, dtype=np.float32)
            labels_ds[start_idx:end_idx] = np.array(chunk_labels, dtype=np.int64)

            if has_params:
                chunk_p_arr = np.array(chunk_params, dtype=np.float32)
                params_ds[start_idx:end_idx] = chunk_p_arr
                all_parameters[start_idx:end_idx] = chunk_p_arr

        h5f.attrs["label_map"] = json.dumps(label_map)

        if has_params:
            param_mean = np.mean(all_parameters, axis=0)
            param_std = np.std(all_parameters, axis=0)
            param_std[param_std == 0.0] = 1.0

            h5f.attrs["param_mean"] = json.dumps(param_mean.tolist())
            h5f.attrs["param_std"] = json.dumps(param_std.tolist())

            print("\n[SUMMARY] Parameter Normalization Statistics Computed:")
            for idx, (m, s) in enumerate(zip(param_mean, param_std)):
                print(f"  Param {idx:02d} -> Mean: {m:12.4e} | Std: {s:12.4e}")

    print(f"\n[SUCCESS] Dataset successfully archived to: {output_h5_path.resolve()}")
    return output_h5_path


if __name__ == "__main__":
    import sys
    root_dir = Path(__file__).resolve().parents[2]
    input_datasets = root_dir / "data" / "synthetic" / "test_datasets"
    output_hdf5 = root_dir / "data" / "synthetic" / "events_suite.h5"

    if not input_datasets.exists():
        print(f"[ERROR] Directory not found: {input_datasets}")
        sys.exit(1)

    pack_npz_to_hdf5(input_dir=input_datasets, output_h5_path=output_hdf5)
