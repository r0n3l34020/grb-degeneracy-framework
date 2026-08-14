import sys
import gc
from pathlib import Path
import numpy as np
from tqdm import tqdm

def validate_dataset_safely(dataset_dir: str):
    target_path = Path(dataset_dir)
    if not target_path.exists():
        sys.stderr.write(f"[ERROR] Directory '{dataset_dir}' does not exist.\n")
        sys.exit(1)

    file_paths = sorted(list(target_path.rglob("simulation_*.npz")))
    if not file_paths:
        sys.stderr.write("[ERROR] No simulation files found.\n")
        sys.exit(1)

    valid_count = 0
    error_count = 0
    max_errors = 50

    pbar = tqdm(file_paths, desc="Memory-Safe Validation", unit="file", mininterval=0.5)

    for file_path in pbar:
        try:
            with np.load(file_path, allow_pickle=True) as data:
                for key in data.files:
                    _ = data[key]
            
            valid_count += 1

        except Exception as e:
            error_count += 1
            sys.stderr.write(f"\n[SCHEMA ERROR] {file_path.relative_to(target_path.parent)}: {e}\n")
            
            if error_count >= max_errors:
                sys.stderr.write(f"\n[CIRCUIT BREAKER] Threshold reached ({max_errors} errors). Aborting sweep.\n")
                break

        finally:
            gc.collect()

        pbar.set_postfix(valid=valid_count, errors=error_count)

    print(f"\n[SUMMARY] Validation complete, Sir.")
    print(f"Total Valid Files : {valid_count}")
    print(f"Total Errors      : {error_count}")

if __name__ == "__main__":
    DATASET_ROOT = "C:/Users/Ronel/Desktop/grb-degeneracy-framework/data/synthetic/test_datasets"
    validate_dataset_safely(DATASET_ROOT)
