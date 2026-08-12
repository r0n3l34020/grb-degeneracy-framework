import numpy as np
from pathlib import Path

target_folder = Path('C:/Users/Ronel/Desktop/grb-degeneracy-framework/data/synthetic/test_datasets')

def test_and_audit_dataset(dataset_folder):
    dataset_path = Path(dataset_folder)
    npz_files = list(dataset_path.glob("*.npz"))
    
    total_files = len(npz_files)
    if total_files == 0:
        print(f"[ERROR] No .npz files found in {dataset_folder}")
        return

    print(f"[INFO] Auditing and inspecting {total_files} simulation files in {dataset_folder}...\n")

    ill_conditioned_count = 0
    mode_counts = {}
    obs_mode_counts = {}

    for file in npz_files:
        data = np.load(file, allow_pickle=True)
        
        fisher = data["fisher"]
        obs_mode = str(data["obs_mode"])
        mode = str(data["label"])

        # Track dataset distributions
        mode_counts[mode] = mode_counts.get(mode, 0) + 1
        obs_mode_counts[obs_mode] = obs_mode_counts.get(obs_mode, 0) + 1

        # Audit condition number offline
        cond = np.linalg.cond(fisher)
        if cond > 1e12 or np.isinf(cond):
            ill_conditioned_count += 1
            print(f"[AUDIT WARNING] High ill-conditioning in {file.name:<32} | Mode: {mode:<4} | Obs: {obs_mode:<18} | Cond: {cond:.2e}")

    print("\n" + "="*80)
    print(f"[AUDIT SUMMARY] {ill_conditioned_count}/{total_files} matrices required ridge stabilization.")
    print(f"[DATASET BREAKDOWN] Scenario Modes: {mode_counts}")
    print(f"[DATASET BREAKDOWN] Observation Modes: {obs_mode_counts}")
    print("="*80 + "\n")

    # Sample File Sanity Verification
    sample_file = npz_files[0]
    print(f"[SANITY CHECK] Inspecting sample file: {sample_file.name}")
    sample_data = np.load(sample_file, allow_pickle=True)
    
    print(f"  * Saved keys: {list(sample_data.keys())}")
    print(f"  * Observation Tensor Shape: {sample_data['observation'].shape}")
    print(f"  * Fisher Matrix Shape: {sample_data['fisher'].shape}")
    print(f"  * Covariance Matrix Shape: {sample_data['covariance'].shape}")
    
    # Unpack stored Cramer-Rao dictionary
    cramer_dict = sample_data['cramer'].item() if sample_data['cramer'].dtype == object else sample_data['cramer']
    print(f"  * Extracted Cramer-Rao Bounds: {cramer_dict}")
    
    # Matrix integrity verification
    obs_matrix = sample_data['observation']
    print(f"  * Contains NaNs: {np.any(np.isnan(obs_matrix))}")
    print(f"  * Contains Infs: {np.any(np.isinf(obs_matrix))}")
    print("[SUCCESS] Dataset inspection and audit completed successfully.")

if __name__ == "__main__":
    test_and_audit_dataset(target_folder)