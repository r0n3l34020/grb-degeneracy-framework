import sys
from pathlib import Path
import numpy as np
from tqdm import tqdm

def repair_and_validate_dataset(dataset_dir: str, eps: float = 1e-10):
    target_path = Path(dataset_dir)
    if not target_path.exists():
        sys.stderr.write(f"[ERROR] Directory '{dataset_dir}' does not exist.\n")
        sys.exit(1)

    file_paths = sorted(list(target_path.rglob("simulation_*.npz")))
    if not file_paths:
        sys.stderr.write("[ERROR] No simulation files found in target batch folders.\n")
        sys.exit(1)

    valid_count = 0
    repaired_count = 0
    unrecoverable_count = 0

    pbar = tqdm(file_paths, desc="Auditing & Repairing Batches", unit="file", mininterval=0.5)

    for file_path in pbar:
        try:
            with np.load(file_path, allow_pickle=True) as data:
                payload = {k: data[k] for k in data.files}

            needs_save = False

            obs = payload.get("observation")
            if obs is not None:

                if np.isnan(obs).any() or np.isinf(obs).any():
                    payload["observation"] = np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=0.0)
                    needs_save = True

            fisher = payload.get("fisher")
            if fisher is not None:
                if np.isnan(fisher).any() or np.isinf(fisher).any():
                    fisher_clean = np.nan_to_num(fisher, nan=0.0, posinf=0.0, neginf=0.0)

                    diag_indices = np.diag_indices_from(fisher_clean)
                    fisher_clean[diag_indices] += eps

                    covariance_clean = np.linalg.pinv(fisher_clean)

                    payload["fisher"] = fisher_clean
                    payload["covariance"] = covariance_clean
                    needs_save = True

            if needs_save:
                np.savez_compressed(file_path, **payload)
                repaired_count += 1
            else:
                valid_count += 1

        except Exception as e:
            unrecoverable_count += 1

        pbar.set_postfix(valid=valid_count, fixed=repaired_count, err=unrecoverable_count)

    print(f"\n[SUMMARY] Audit & Repair Complete, Sir.")
    print(f"Total Evaluated : {len(file_paths)}")
    print(f"Clean Files     : {valid_count}")
    print(f"Repaired Files  : {repaired_count}")
    print(f"Unrecoverable   : {unrecoverable_count}")

if __name__ == "__main__":
    DATASET_ROOT = "C:/Users/Ronel/Desktop/grb-degeneracy-framework/data/synthetic/test_datasets"
    repair_and_validate_dataset(DATASET_ROOT)
